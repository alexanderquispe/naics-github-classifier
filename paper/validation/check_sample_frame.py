"""
Phase 1 - Sample frame check.

Verifies the gold set (n=2,421) is a strict subset of the training-ready file
(n=6,588 in naics-github-train/data/raw/train_data_gpt_ab8_score_with_code.parquet).

Reports:
  - Match coverage on (name_repo, code)
  - Duplicates in either file
  - Mismatched code assignments (same repo, different code)
  - Repos in gold that are NOT in training
"""

from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = REPO_ROOT / "data" / "raw" / "gold_set_manual_check.xlsx"
TRAIN_PATH = (
    REPO_ROOT.parent
    / "naics-github-train"
    / "data"
    / "raw"
    / "train_data_gpt_ab8_score_with_code.parquet"
)
FULL_PATH = (
    REPO_ROOT.parent
    / "naics-github-train"
    / "data"
    / "raw"
    / "train_data_naics_github.parquet"
)


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    gold = pd.read_excel(GOLD_PATH)
    train = pd.read_parquet(TRAIN_PATH)
    full = pd.read_parquet(FULL_PATH)

    print(f"Gold:         {len(gold):>6} rows  (cols: {list(gold.columns)})")
    print(f"Training:     {len(train):>6} rows  (cols: {list(train.columns)})")
    print(f"Full merged:  {len(full):>6} rows  (cols: {list(full.columns)})")

    # ------------------------------------------------------------------
    section("1. Duplicates within each file")
    print(f"Gold     duplicates on (name_repo, code): {gold.duplicated(['name_repo', 'code']).sum()}")
    print(f"Training duplicates on (name_repo, code): {train.duplicated(['name_repo', 'code']).sum()}")
    print(f"Full     duplicates on (name_repo, code): {full.duplicated(['name_repo', 'code']).sum()}")

    if "nwo" in full.columns:
        print(f"Full     duplicates on (nwo, code):       {full.duplicated(['nwo', 'code']).sum()}")
    if "nwo" in gold.columns:
        print(f"Gold     duplicates on (nwo, code):       {gold.duplicated(['nwo', 'code']).sum()}")

    # ------------------------------------------------------------------
    section("2. Coverage: gold-set rows found in training file")
    # Composite-key match
    train_keys = set(zip(train["name_repo"].astype(str), train["code"].astype(str)))
    gold_keys = list(zip(gold["name_repo"].astype(str), gold["code"].astype(str)))
    matched = [k in train_keys for k in gold_keys]
    n_matched = sum(matched)
    print(f"Matched on (name_repo, code):  {n_matched}/{len(gold)}  ({n_matched/len(gold)*100:.2f}%)")

    # Match by name_repo alone (looser)
    train_names = set(train["name_repo"].astype(str))
    name_matched = sum(n in train_names for n in gold["name_repo"].astype(str))
    print(f"Matched on name_repo only:     {name_matched}/{len(gold)}  ({name_matched/len(gold)*100:.2f}%)")

    # ------------------------------------------------------------------
    section("3. Coverage in the full merged file (nwo + code)")
    if "nwo" in full.columns and "nwo" in gold.columns:
        full_nwo_keys = set(zip(full["nwo"].astype(str), full["code"].astype(str)))
        gold_nwo_keys = list(zip(gold["nwo"].astype(str), gold["code"].astype(str)))
        nwo_matched = sum(k in full_nwo_keys for k in gold_nwo_keys)
        print(f"Matched on (nwo, code):        {nwo_matched}/{len(gold)}  ({nwo_matched/len(gold)*100:.2f}%)")

        # Repos in gold not in full
        unmatched_mask = [k not in full_nwo_keys for k in gold_nwo_keys]
        if any(unmatched_mask):
            print(f"\nGold rows NOT in full-merged file ({sum(unmatched_mask)}):")
            print(gold.loc[unmatched_mask, ["nwo", "code", "manual_check", "score"]].to_string(index=False))

    # ------------------------------------------------------------------
    section("4. Code-mismatch check (same repo, different NAICS sector)")
    # Repos that appear in both gold and training, but with different code assignments
    if "nwo" in gold.columns and "nwo" in full.columns:
        gold_code_by_nwo = gold.set_index("nwo")["code"]
        full_code_by_nwo = full.set_index("nwo")["code"]
        common = gold_code_by_nwo.index.intersection(full_code_by_nwo.index)
        # Drop duplicates first to avoid one-to-many noise
        gold_unique = gold_code_by_nwo[~gold_code_by_nwo.index.duplicated(keep="first")]
        full_unique = full_code_by_nwo[~full_code_by_nwo.index.duplicated(keep="first")]
        common = gold_unique.index.intersection(full_unique.index)
        mismatches = (gold_unique.loc[common].astype(str) != full_unique.loc[common].astype(str)).sum()
        print(f"Repos with mismatched codes: {mismatches}/{len(common)}")

    # ------------------------------------------------------------------
    section("5. Stratification check: gold sector distribution vs training")
    gold_dist = gold["code"].value_counts(normalize=True).sort_index()
    train_dist = train["code"].value_counts(normalize=True).sort_index()
    compare = pd.DataFrame({"gold_pct": gold_dist, "train_pct": train_dist}).fillna(0)
    compare["delta"] = compare["gold_pct"] - compare["train_pct"]
    compare = (compare * 100).round(2)
    compare.columns = ["gold_%", "train_%", "delta_pp"]
    print(compare.to_string())

    print()
    print(f"Output saved to: {REPO_ROOT / 'paper' / 'validation' / 'output'}")

    out_dir = REPO_ROOT / "paper" / "validation" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    compare.to_csv(out_dir / "sector_distribution_gold_vs_train.csv")


if __name__ == "__main__":
    main()
