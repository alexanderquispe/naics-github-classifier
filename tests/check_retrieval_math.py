"""
Verify the 1200/60 retrieval count makes sense.

For each NAICS sector:
  n          = number of subindustry phrases
  k_per_sub  = max(min_k, ceil(base_k / n))
  pre_dedup  = n * k_per_sub        (upper bound: every query returns k distinct hits)
  actual     = rows in the candidates file for that sector
  dedup%     = 1 - actual / pre_dedup

Then compares totals and cross-checks against the earlier 400/20 run.
"""

import json
import math
from pathlib import Path

import polars as pl

REPO = Path(r"C:\Users\Alexander\Documents\GitHub\naics-github-classifier")
NAICS = REPO / "data" / "raw" / "naics_titles_by_group_6digit_clean.json"
CAND = REPO / "data" / "processed" / "retrieval_candidates.parquet"

BASE_K = 1200
MIN_K = 60


def k_per_sub(n, base_k, min_k):
    return max(min_k, math.ceil(base_k / n))


def main():
    naics = json.loads(NAICS.read_text(encoding="utf-8"))
    cand = pl.read_parquet(CAND)
    actual_by_sector = dict(
        cand.group_by("naics_code").len().rename({"len": "n"}).iter_rows()
    )

    print(f"{'sector':<8}{'#sub':>6}{'k/sub':>7}{'pre_dedup':>11}{'actual':>9}{'dedup%':>9}")
    print("-" * 50)

    total_sub = 0
    total_pre = 0
    total_actual = 0
    for code, definition in naics.items():
        subs = [s.strip() for s in definition.split(";") if s.strip()]
        n = len(subs)
        if n == 0:
            continue
        k = k_per_sub(n, BASE_K, MIN_K)
        pre = n * k
        actual = actual_by_sector.get(code, 0)
        dedup = (1 - actual / pre) * 100 if pre else 0
        total_sub += n
        total_pre += pre
        total_actual += actual
        print(f"{code:<8}{n:>6}{k:>7}{pre:>11,}{actual:>9,}{dedup:>8.0f}%")

    print("-" * 50)
    overall_dedup = (1 - total_actual / total_pre) * 100 if total_pre else 0
    print(f"{'TOTAL':<8}{total_sub:>6}{'':>7}{total_pre:>11,}{total_actual:>9,}{overall_dedup:>8.0f}%")

    print()
    print(f"Subindustry phrases total : {total_sub:,}")
    print(f"Pre-dedup upper bound      : {total_pre:,}  (sum of n*k_per_sub)")
    print(f"Actual unique (nwo,naics)  : {total_actual:,}")
    print(f"Overall within-sector dedup: {overall_dedup:.0f}%")

    # Cross-check vs the 400/20 run (which produced 7,357)
    pre_400 = 0
    for code, definition in naics.items():
        subs = [s.strip() for s in definition.split(";") if s.strip()]
        n = len(subs)
        if n:
            pre_400 += n * k_per_sub(n, 400, 20)
    print()
    print("Cross-check vs the earlier 400/20 run:")
    print(f"  400/20 pre-dedup bound : {pre_400:,}")
    print(f"  400/20 actual (seen)   : 7,357")
    print(f"  400/20 implied dedup   : {(1 - 7357 / pre_400) * 100:.0f}%")
    print(f"  1200/60 pre-dedup bound: {total_pre:,}")
    print(f"  1200/60 actual         : {total_actual:,}")
    print(f"  depth ratio (k 20->60) : 3.0x   |   actual count ratio: "
          f"{total_actual / 7357:.2f}x")


if __name__ == "__main__":
    main()
