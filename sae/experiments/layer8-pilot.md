# Layer 8 pilot: vanilla vs Matryoshka BatchTopK SAE

- Goal: compare a vanilla and a Matryoshka SAE on the residual stream after layer 8 of marin-32b-base.
- Both SAEs train on the same activations in one run, so the LLM forward pass is paid once.
- Budget: 300M tokens, one A100 80GB on Colab, checkpoints tarred to Drive.
- This log is append-only. Failures stay in, because they are the useful part.

## Decisions before any code ran

- **Site:** resid_post of decoder layer 8, 0-indexed, 5120 dims. The BOS position is dropped.
- **Why not the sparse tier:** the atlas keeps only the top 64 of 27,648 neurons per token. An SAE needs dense activations.
- **Mix:** FineWeb for normal text, plus the non-web parts of the cooldown mix.
  - Nemotron-CC-v2 is gated and the access request was rejected, so the cooldown's web share has no exact source.
  - A DCLM-based mix was planned first, then dropped. FineWeb replaces it.
  - The Marin-specific sets get public stand-ins: arxiv_markdownified -> common-pile arxiv.
  - The mix is frozen in `configs/layer8_mix_300m.yaml` before any activation is computed.
- **Pure FineWeb run:** dropped. It can be added later for about 6 more GPU hours.

## Log

### Setup on Colab

- **Machine:** A100-SXM4-80GB, 167 GB RAM, 253 GB disk, torch 2.14.1+cu130, transformers 5.22.1.
- **Model:** Qwen3 architecture, 64 layers, d_model 5120, d_ff 27648. Only layers 0 to 8 are needed.
- **Mistake: a launch cell was added but never run.**
  - Symptom: the status cell showed "no download alive" and `/content/dl.log` did not exist.
  - Cause: `add_code_cell` only inserts a cell. `run_code_cell` has to be called separately.
  - Fix: ran the cell. Lesson: after adding a launch cell, check that its log file exists.
- **Dataset access probe:**
  - Open: FineWeb, FineMath-3plus, StarCoderData, Wikipedia, common-pile arxiv, Dolmino.
  - `allenai/peS2o` and `EleutherAI/proof-pile-2` fail: "Dataset scripts are no longer supported". The Dolmino copy of pes2o replaces peS2o.
  - `nvidia/Nemotron-CC-v2` is gated and the request was rejected.
- **Probe helper bug:** `probe(name, **kw)` clashed with the `name=` dataset config kwarg. Renamed the first argument to `repo`.

### Code, in six stacked PRs on the fork

- **Order:** config, data stream, activations, SAE and losses, training loop, post-hoc evaluation.
- **Tests:** 44 CPU tests pass. They do not run the real model, so they cannot catch GPU-only bugs.
- **Layer tap check:** on a tiny Qwen3, the tap output equals the full model's hidden state after the same layer.
- **Rope config:** Marin uses llama3 rope scaling (factor 8, original 8192) with max positions 4096.
  - Transformers warns about this, but it still builds the scaled rotary embedding from the config.
  - The prefix follows the config. A clean-loss check on the full model is planned to confirm it.

### Source probe on Colab

- **Fields and sizes:** eight of nine sources gave real text with sane token counts.
  - Mean tokens per document: fineweb 387, pes2o 5034, arxiv 9634, finemath 780, stackexchange 460, flan 245, code 980, wikipedia 679.
- **Math failed:** the glob `data/math/dolmino_math_synth/*` matched no supported files.
  - The files are nested `.jsonl`. The set is templated arithmetic, so it is also too narrow.
  - Fix: use `data/math/mathcoder2-synthmath/ajibawa-2023/*.jsonl`, which has college-level `text`.
- **Slow start:** pes2o took 231 s, arxiv 83 s and finemath 56 s for 20 documents.
  - Cause: each source fills a 10,000-document shuffle buffer before the first row, and HF reads time out and retry.
  - Fix: `data.shuffle_buffer`, default 1000.

### Smoke run on Colab (4M tokens, real model, real SAE size)

- **Three bugs, each found within a minute of its launch:**
  - `run.py` imported `checkpoint` from the wrong module.
    - No unit test imports `run.py`, because it needs the real model.
    - Fix: correct the import, and add `tests/test_imports.py`.
  - The normalisation fit sent CPU token batches to a model on the GPU.
    - Symptom: `index is on cpu, different from other tensors on cuda`.
    - Cause: the tests run on CPU only, so they never saw a device mismatch.
    - Fix: the tap and `residual_rows` move tokens to the model's device.
  - After the crash the process stayed alive.
    - Cause: Hugging Face `datasets` keeps non-daemon download threads running.
    - A crash then looks like a run that is still going. A clean finish also waits for them.
    - Fix: `__main__` ends the process with `os._exit` after flushing.
- **Mistake in my own tooling:** I broke a patch script with nested triple quotes, so the fix did not apply.
  - The smoke run then restarted without it. I killed the run and applied the edits inline.
- **Normalisation:** scale 0.159, and the mean vector has norm 70.97.
  - A typical row has norm about 71.6 (the square root of 5120).
  - So the layer 8 residual stream is dominated by a large constant component.
  - Consequence: `b_dec` starts at the mean, and FVU is measured against the centred variance.
- **Throughput:** about 9,500 rows each second with both variants, so 300M tokens takes about 9 hours.
  - Peak GPU memory allocated: 39.5 GiB.
- **Loss at step 180:** FVU 0.359 (vanilla) and 0.350 (Matryoshka, full dictionary). Both were still falling.
  - The Matryoshka total loss is higher because it averages four prefixes.
  - Learning rate was still in warmup, so this says nothing about the final result.

### The VM was lost

- **What happened:** my Colab tool calls began to fail with "Unknown tool", then the reconnect returned false.
  - You then saw the VM was gone.
- **Cause:** not known. A guess is that Colab recycled an idle runtime after the browser link dropped.
- **Lost:** the 118 GB of shards, the VM-only patches, and any checkpoint that was not on Drive yet.
- **Not lost:** the code and the fixes. The fixes are in this stack and in local fixup commits.
- **Never seen:** the first in-run evaluation, the first checkpoint, and the Drive upload speed.
- **Before the next run:** decide how to keep the session alive for about 9 hours.
  - Options: Colab Pro+ background execution, or a keepalive cell.
  - Check that `resume.tar` reaches Drive, so a lost VM costs at most one checkpoint interval.
