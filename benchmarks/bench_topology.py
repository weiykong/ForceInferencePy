"""Benchmark for label-driven topology extraction (full pipeline).

Measures extract_topology_label end-to-end on the bundled sample images.
Baseline ("before" the algorithmic optimisation) is preserved in
baseline_python.json; current numbers are written to after_algofix.json.
"""
import json, time, platform, sys
from datetime import datetime, timezone
import numpy as np

from force_inference.segmentation import segment_grayscale
from force_inference.topology_label import extract_topology_label

IMAGES = ["data/example.tif", "data/test.tif"]
REPEATS = 5


def median_time(fn, repeats=REPEATS):
    fn()  # warm-up
    s = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    s.sort()
    return s[len(s) // 2]


def main():
    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "images": {},
    }
    try:
        baseline = json.load(open("benchmarks/baseline_python.json"))
    except FileNotFoundError:
        baseline = None

    for img in IMAGES:
        labels, _ = segment_grayscale(img)
        labels = labels.astype(np.int64)
        for hw in (1, 2):
            t = median_time(lambda: extract_topology_label(labels, junction_window=hw))
            key = f"{img}|hw={hw}"
            entry = {"median_s": t}
            if baseline and key in baseline["images"]:
                before = baseline["images"][key]["extract_topology_full"]["median_s"]
                entry["baseline_s"] = before
                entry["speedup"] = before / t
            out["images"][key] = entry
            tag = f" ({entry['speedup']:.1f}x)" if "speedup" in entry else ""
            print(f"{key}: {t*1e3:8.1f} ms{tag}")

    json.dump(out, open("benchmarks/after_algofix.json", "w"), indent=2)
    print("\nSaved -> benchmarks/after_algofix.json")


if __name__ == "__main__":
    main()
