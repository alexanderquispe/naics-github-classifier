"""
Generate LaTeX table bodies (tabular rows only) from the validation CSVs
and the released parquet, so main.tex can \\input them directly.

Outputs (under paper/validation/output/):
  - sector_counts_rows.tex          (Table 1 body: code, sector_name, n)
  - per_sector_precision_rows.tex   (Table 2 body: code, sector_name, n, precision, 95% CI)
"""

from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "paper" / "validation" / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Released training corpus — this is the canonical n=6,588 file.
FULL_PARQUET = (
    REPO_ROOT.parent
    / "naics-github-train"
    / "data"
    / "raw"
    / "train_data_gpt_ab8_score_with_code.parquet"
)
GOLD_XLSX = REPO_ROOT / "data" / "raw" / "gold_set_manual_check.xlsx"
PER_SECTOR_CSV = OUT_DIR / "per_sector_precision.csv"


def latex_escape(s: str) -> str:
    """Escape characters that LaTeX needs escaped in tabular text."""
    if not isinstance(s, str):
        return ""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


# Some NAICS sector titles are too long for a single-column NeurIPS table.
# Map them to compact forms; the full names live in the appendix.
SECTOR_NAME_OVERRIDES = {
    "56": "Administrative and Support Services",
    "81": "Other Services",
    "54": "Professional and Technical Services",
    "62": "Health Care and Social Assistance",
    "71": "Arts, Entertainment, and Recreation",
}


def compact_sector_name(code: str, name: str) -> str:
    return SECTOR_NAME_OVERRIDES.get(str(code), name)


def get_sector_names_from_gold() -> dict[str, str]:
    gold = pd.read_excel(GOLD_XLSX)
    name_map = (
        gold.dropna(subset=["code", "sector_name"])
        .drop_duplicates("code")
        .set_index("code")["sector_name"]
        .to_dict()
    )
    name_map.setdefault("55", "Management of Companies and Enterprises")
    return name_map


def gen_sector_counts() -> None:
    full = pd.read_parquet(FULL_PARQUET)
    counts = full["code"].value_counts().sort_index()
    name_map = get_sector_names_from_gold()
    total = int(counts.sum())

    lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llr}",
        r"\toprule",
        r"Code & Sector & $n$ \\",
        r"\midrule",
    ]
    for code, n in counts.items():
        raw_name = name_map.get(code, "")
        sector = latex_escape(compact_sector_name(code, raw_name))
        lines.append(f"{code} & {sector} & {n:,} \\\\")
    lines += [
        r"\midrule",
        f"\\multicolumn{{2}}{{l}}{{\\textbf{{Total}}}} & \\textbf{{{total:,}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
    ]

    out_path = OUT_DIR / "sector_counts_table.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}  ({len(counts)} sectors, {total:,} repos)")


def gen_per_sector_precision() -> None:
    df = pd.read_csv(PER_SECTOR_CSV)
    df = df.sort_values("precision", ascending=False)

    lines = [
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llrrl}",
        r"\toprule",
        r"Code & Sector & $n$ & Precision & 95\% CI \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        code = str(row["code"])
        raw_name = str(row["sector_name"])
        sector = latex_escape(compact_sector_name(code, raw_name))
        n = int(row["n"])
        prec = row["precision"]
        lo = row["ci_low"]
        hi = row["ci_high"]
        lines.append(
            f"{code} & {sector} & {n} & {prec:.3f} & [{lo:.3f}, {hi:.3f}] \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}%", r"}"]

    out_path = OUT_DIR / "per_sector_precision_table.tex"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}  ({len(df)} sectors)")


def main() -> None:
    gen_sector_counts()
    gen_per_sector_precision()


if __name__ == "__main__":
    main()
