"""
Phase 1 — Validation statistics for the NAICS-GH paper.

Reads the manually checked gold set (n=2,421) from data/raw/gold_set_manual_check.xlsx
and computes the headline numbers we need for the paper:

1. Overall precision (fraction of GPT labels marked Correct by RAs)
2. Per-sector precision with Wilson 95% CIs and sample sizes
3. Precision by GPT score (8, 9, 10)
4. Sector x score cross-tab

Outputs are saved as CSV under paper/validation/output/ so the LaTeX can
include them as tables.
"""

from pathlib import Path
import math
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = REPO_ROOT / "data" / "raw" / "gold_set_manual_check.xlsx"
OUT_DIR = REPO_ROOT / "paper" / "validation" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wilson_ci(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson score interval — better than normal approx for small n or extreme p."""
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> None:
    df = pd.read_excel(GOLD_PATH)

    df["manual_check"] = df["manual_check"].astype(str).str.strip()
    df["is_correct"] = (df["manual_check"].str.lower() == "correct").astype(int)

    print(f"Loaded {len(df)} rows from {GOLD_PATH.name}")
    print(f"manual_check value counts:")
    print(df["manual_check"].value_counts(dropna=False))
    print()

    # 1. Overall precision
    n = len(df)
    k = int(df["is_correct"].sum())
    p = k / n
    lo, hi = wilson_ci(k, n)
    overall = pd.DataFrame(
        [{"metric": "precision", "k_correct": k, "n_total": n, "estimate": p, "ci_low": lo, "ci_high": hi}]
    )
    overall.to_csv(OUT_DIR / "overall_precision.csv", index=False)
    print("=== 1. Overall precision ===")
    print(f"  Correct: {k}/{n}  =  {p:.4f}   (95% CI: [{lo:.4f}, {hi:.4f}])")
    print()

    # 2. Per-sector precision
    rows = []
    for (code, sector_name), g in df.groupby(["code", "sector_name"], dropna=False):
        nn = len(g)
        kk = int(g["is_correct"].sum())
        pp = kk / nn if nn else float("nan")
        lo_s, hi_s = wilson_ci(kk, nn)
        rows.append({
            "code": code,
            "sector_name": sector_name,
            "n": nn,
            "n_correct": kk,
            "precision": pp,
            "ci_low": lo_s,
            "ci_high": hi_s,
        })
    per_sector = pd.DataFrame(rows).sort_values("precision", ascending=False)
    per_sector.to_csv(OUT_DIR / "per_sector_precision.csv", index=False)
    print("=== 2. Per-sector precision (sorted by precision) ===")
    print(per_sector.to_string(index=False,
        formatters={"precision": "{:.4f}".format, "ci_low": "{:.4f}".format, "ci_high": "{:.4f}".format}))
    print()

    # 3. Precision by GPT score
    rows = []
    for score, g in df.groupby("score"):
        nn = len(g)
        kk = int(g["is_correct"].sum())
        pp = kk / nn if nn else float("nan")
        lo_s, hi_s = wilson_ci(kk, nn)
        rows.append({
            "score": int(score),
            "n": nn,
            "n_correct": kk,
            "precision": pp,
            "ci_low": lo_s,
            "ci_high": hi_s,
        })
    by_score = pd.DataFrame(rows).sort_values("score")
    by_score.to_csv(OUT_DIR / "precision_by_score.csv", index=False)
    print("=== 3. Precision by GPT score ===")
    print(by_score.to_string(index=False,
        formatters={"precision": "{:.4f}".format, "ci_low": "{:.4f}".format, "ci_high": "{:.4f}".format}))
    print()

    # 4. Sector x score cross-tab (counts of "Correct" / total)
    crosstab_total = df.pivot_table(index="code", columns="score", values="is_correct", aggfunc="count", fill_value=0)
    crosstab_corr = df.pivot_table(index="code", columns="score", values="is_correct", aggfunc="sum", fill_value=0)
    crosstab_prec = (crosstab_corr / crosstab_total.where(crosstab_total > 0)).round(4)
    crosstab_total.to_csv(OUT_DIR / "sector_score_counts.csv")
    crosstab_corr.to_csv(OUT_DIR / "sector_score_correct.csv")
    crosstab_prec.to_csv(OUT_DIR / "sector_score_precision.csv")
    print("=== 4. Sector x score precision table ===")
    print(crosstab_prec.to_string())
    print()

    # 5. Coverage check vs training file
    train_path = REPO_ROOT.parent / "naics-github-train" / "data" / "raw" / "train_data_gpt_ab8_score_with_code.parquet"
    if train_path.exists():
        train = pd.read_parquet(train_path)
        gold_in_train = df["nwo"].isin(train["name_repo"]) if "nwo" not in train.columns else df["nwo"].isin(train["nwo"])
        if "nwo" not in train.columns:
            print(f"=== 5. Sample frame check ===")
            print(f"  Training file has no 'nwo' column; cannot join directly.")
            print(f"  Training file columns: {list(train.columns)}")
            print(f"  Training file rows: {len(train)}")
        else:
            in_train = int(gold_in_train.sum())
            print(f"=== 5. Sample frame check ===")
            print(f"  Gold-set repos found in training file: {in_train}/{len(df)}")
    print()
    print(f"All output tables saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
