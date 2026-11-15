"""Measure Marin 32B MLP neuron activations on a small GPU (a Colab T4) by loading only layers 0..L.

The activation is the atlas's and the runner's: silu(gate) * up, read as the input to mlp.down_proj, in bf16,
with the same tokenizer, BOS and [[target]] rules as runner/prompts.py. The forward pass stops at the deepest
requested layer, so layers above it are never needed.

Weights come from the Hub by HTTP range request, tensor by tensor (about 1 GB per layer instead of whole
5-9 GB shards), and are cached per layer under --cache.

    python scripts/colab/measure.py --neuron L6-N9321 --check scripts/colab/atlas_check.json
    python scripts/colab/measure.py --neuron L10-N18895 --prompts requests/L10-N18895-001.yaml
    python scripts/colab/measure.py --neuron L6-N9321 --prompts probe.yaml --out /content/out/probe.json

A prompts file is a request YAML (activation-test, contrast) or a probe YAML: {groups: {name: [prompts]}}.
"""

import argparse
import json
import os
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from runner.prompts import MODEL_REPO, MODEL_SHA, PromptError, encode  # noqa: E402

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}
CHUNK = 64 << 20


class _Stop(Exception):
    pass


def parse_neuron(s):
    m = re.fullmatch(r"L(\d+)-N(\d+)", s)
    if not m:
        raise SystemExit(f"bad neuron {s!r}; use L6-N9321")
    return int(m[1]), int(m[2])


# ---------------------------------------------------------------- weights

def _session():
    import requests

    s = requests.Session()
    tok = os.environ.get("HF_TOKEN")
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    return s


def _get_range(sess, url, start, end):
    for attempt in range(5):
        try:
            r = sess.get(url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
            r.raise_for_status()
            if len(r.content) != end - start:
                raise IOError(f"short read {len(r.content)} != {end - start}")
            return r.content
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


def _shard_header(sess, url):
    n = struct.unpack("<Q", _get_range(sess, url, 0, 8))[0]
    return 8 + n, json.loads(_get_range(sess, url, 8, 8 + n))


def _fetch_group(keys, weight_map, out_path):
    """Download the named tensors by range request and save them as one safetensors file."""
    from huggingface_hub import hf_hub_url
    from safetensors.torch import save_file

    sess = _session()
    tensors = {}
    by_shard = {}
    for k in keys:
        by_shard.setdefault(weight_map[k], []).append(k)
    for shard, ks in by_shard.items():
        url = hf_hub_url(MODEL_REPO, shard, revision=MODEL_SHA)
        base, header = _shard_header(sess, url)
        for k in ks:
            meta = header[k]
            a, b = meta["data_offsets"]
            buf = bytearray(b - a)
            spans = [(s, min(s + CHUNK, b - a)) for s in range(0, b - a, CHUNK)]
            with ThreadPoolExecutor(8) as ex:
                for (s, e), data in zip(spans, ex.map(lambda se: _get_range(sess, url, base + a + se[0], base + a + se[1]), spans)):
                    buf[s:e] = data
            tensors[k] = torch.frombuffer(buf, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"]).clone()
    tmp = out_path.with_suffix(".tmp")
    save_file(tensors, str(tmp))
    tmp.rename(out_path)


def ensure_cache(max_layer, cache):
    from huggingface_hub import hf_hub_download

    cache.mkdir(parents=True, exist_ok=True)
    wm = json.load(open(hf_hub_download(MODEL_REPO, "model.safetensors.index.json", revision=MODEL_SHA)))["weight_map"]
    groups = {"embed": [k for k in wm if k.startswith("model.embed_tokens.")]}
    for i in range(max_layer + 1):
        groups[f"layer{i:02d}"] = [k for k in wm if k.startswith(f"model.layers.{i}.")]
    for name, keys in groups.items():
        path = cache / f"{name}.safetensors"
        if not path.exists():
            t = time.time()
            _fetch_group(keys, wm, path)
            print(f"[fetch] {name}: {len(keys)} tensors in {time.time() - t:.0f}s", flush=True)
    return [cache / f"{n}.safetensors" for n in groups]


def load_model(max_layer, cache):
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model, Qwen3RotaryEmbedding

    tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_SHA)
    cfg = AutoConfig.from_pretrained(MODEL_REPO, revision=MODEL_SHA)
    cfg.num_hidden_layers = max_layer + 1
    if getattr(cfg, "layer_types", None):
        cfg.layer_types = cfg.layer_types[: max_layer + 1]
    cfg._attn_implementation = "sdpa"
    files = ensure_cache(max_layer, cache)
    t = time.time()
    with torch.device("meta"):
        model = Qwen3Model(cfg)
    sd = {}
    for f in files:
        for k, v in load_file(str(f), device="cuda").items():
            sd[k[len("model."):]] = v.to(torch.bfloat16)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if unexpected or set(missing) - {"norm.weight"}:
        raise SystemExit(f"weight mismatch: missing={missing} unexpected={unexpected}")
    model.rotary_emb = Qwen3RotaryEmbedding(config=cfg, device="cuda")
    model.eval()
    print(f"[load] layers 0-{max_layer} on {torch.cuda.get_device_name(0)} in {time.time() - t:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return model, tok


# ---------------------------------------------------------------- measuring

@torch.no_grad()
def activations(model, ids, pos, neurons):
    """Activation of every (layer, neuron) at token index pos of one sequence."""
    out = {}
    last = max(l for l, _ in neurons)
    by_layer = {}
    for l, n in neurons:
        by_layer.setdefault(l, []).append(n)

    def hook(layer):
        def fn(module, args):
            x = args[0]
            for n in by_layer[layer]:
                out[(layer, n)] = float(x[0, pos, n].float())
            if layer == last:
                raise _Stop
        return fn

    hs = [model.layers[l].mlp.down_proj.register_forward_pre_hook(hook(l)) for l in by_layer]
    try:
        model(input_ids=torch.tensor([ids], device="cuda"))
    except _Stop:
        pass
    finally:
        for h in hs:
            h.remove()
    return out


def load_items(path):
    """(group, text) pairs from a request or probe YAML."""
    import yaml

    req = yaml.safe_load(open(path))
    t = req.get("type")
    if t == "activation-test":
        return [("should_fire", p) for p in req["should_fire"]] + [("should_not_fire", p) for p in req["should_not_fire"]]
    if t == "contrast":
        items = []
        for i, p in enumerate(req["pairs"]):
            items += [(f"pair{i:02d}.high", p["high"]), (f"pair{i:02d}.low", p["low"])]
        return items
    if "groups" in req:
        return [(g, p) for g, ps in req["groups"].items() for p in ps]
    raise SystemExit(f"{path}: not an activation-test, contrast or probe file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--neuron", action="append", required=True, help="L6-N9321; repeatable")
    ap.add_argument("--prompts", help="request or probe YAML")
    ap.add_argument("--check", help="atlas_check.json: exact atlas token ids for exemplars")
    ap.add_argument("--cache", default="/content/marin_cache")
    ap.add_argument("--out", help="write results as JSON here")
    args = ap.parse_args()

    neurons = [parse_neuron(n) for n in args.neuron]
    model, tok = load_model(max(l for l, _ in neurons), Path(args.cache))
    rows = []
    t = time.time()
    if args.check:
        for ex in json.load(open(args.check)):
            key = (ex["layer"], ex["neuron"])
            if key not in neurons:
                continue
            got = activations(model, ex["ids"], len(ex["ids"]) - 1, [key])[key]
            rows.append({**{k: ex[k] for k in ("layer", "neuron", "pole", "rank", "act")}, "got": got})
            print(f"L{key[0]}-N{key[1]} pole {ex['pole']:+d} rank {ex['rank']:2d}  atlas {ex['act']:+7.3f}  "
                  f"t4 {got:+7.3f}  diff {got - ex['act']:+.3f}", flush=True)
    if args.prompts:
        for group, text in load_items(args.prompts):
            try:
                e = encode(tok, text)
            except PromptError as err:
                print(f"[skip] {err}", flush=True)
                rows.append({"group": group, "text": text, "error": str(err)})
                continue
            acts = activations(model, e.ids, e.pos, neurons)
            rows.append({"group": group, "text": text, "token": e.token,
                         "acts": {f"L{l}-N{n}": v for (l, n), v in acts.items()}})
            vals = "  ".join(f"{v:+7.3f}" for v in acts.values())
            print(f"{vals}  {group:<16} {text}", flush=True)
    print(f"[done] {len(rows)} rows in {time.time() - t:.0f}s", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
