"""Print a compact per-group summary of a measure.py --out JSON, against a side's atlas thresholds.

    python scripts/colab/summarize.py /content/out/L6_r1.json L6-N9321
"""

import json
import sys

# fineweb-10m-sparse pole_stats: (pos q99, q999, q9999), (neg q99, q999, q9999)
LEVELS = {
    "L6-N9321": ((0.074, 0.229, 0.835), (0.080, 0.269, 0.982)),
    "L10-N18895": ((0.130, 0.372, 0.982), (0.130, 0.343, 0.770)),
}


def band(v, pos, neg):
    if v >= 0:
        return "+" * sum(v > t for t in pos) or "."
    return "-" * sum(-v > t for t in neg) or "."


def main():
    path, neuron = sys.argv[1], sys.argv[2]
    pos, neg = LEVELS[neuron]
    rows = json.load(open(path))
    groups = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    print(f"{neuron}: + q99/q999/q9999 {pos}  - {neg}   bands: + > q99, ++ > q999, +++ > q9999")
    for g, rs in groups.items():
        vals = [r["acts"][neuron] for r in rs if "acts" in r]
        if vals:
            print(f"\n## {g}  n={len(vals)} mean {sum(vals) / len(vals):+.2f}  min {min(vals):+.2f}  max {max(vals):+.2f}")
        for r in rs:
            if "error" in r:
                print(f"   ERR  {r['text']}  ({r['error'][-60:]})")
            else:
                v = r["acts"][neuron]
                print(f"  {v:+6.2f} {band(v, pos, neg):<3} {r['text']}")


if __name__ == "__main__":
    main()
