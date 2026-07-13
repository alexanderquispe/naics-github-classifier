"""
Generate Appendix E's gold-sample stratification table.

Reads:
  - data/raw/gold_set_manual_check.xlsx                                 (gold, n=2,421)
  - ../naics-github-train/data/raw/
      train_data_gpt_ab8_score_with_code.parquet                       (released, n=6,588)

Writes:
  - paper/appendix/output/stratification_appendix.tex
"""

from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD = REPO_ROOT / "data" / "raw" / "gold_set_manual_check.xlsx"
CORPUS = (
    REPO_ROOT.parent
    / "naics-github-train"
    / "data"
    / "raw"
    / "train_data_gpt_ab8_score_with_code.parquet"
)
OUT_PATH = (
    REPO_ROOT / "paper" / "appendix" / "output" / "stratification_appendix.tex"
)


def latex_escape(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    repl = {
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
    return "".join(repl.get(ch, ch) for ch in s)


# Compact sector names (same scheme as validation/generate_tex_tables.py)
SECTOR_NAME_OVERRIDES = {
    "56": "Administrative and Support Services",
    "81": "Other Services",
    "54": "Professional and Technical Services",
}


def main() -> None:
    gold = pd.read_excel(GOLD)
    corpus = pd.read_parquet(CORPUS)

    gold["code"] = gold["code"].astype(str)
    corpus["code"] = corpus["code"].astype(str)

    # Per-sector counts and percentages
    g = gold["code"].value_counts().sort_index()
    u = corpus["code"].value_counts().sort_index()
    n_g, n_u = len(gold), len(corpus)

    # Build name map from gold (it carries sector_name)
    name_map = (
        gold.dropna(subset=["code", "sector_name"])
        .drop_duplicates("code")
        .set_index("code")["sector_name"]
        .to_dict()
    )

    cmp = pd.DataFrame({"gold_n": g, "corpus_n": u}).fillna(0).astype(int)
    cmp["gold_pct"] = cmp["gold_n"] / n_g * 100
    cmp["corpus_pct"] = cmp["corpus_n"] / n_u * 100
    cmp["delta_pp"] = cmp["gold_pct"] - cmp["corpus_pct"]
    cmp = cmp.sort_index()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    parts = []

    max_abs_delta = cmp["delta_pp"].abs().max()
    pp_under_1 = int((cmp["delta_pp"].abs() < 1.0).sum())

    parts.append(
        r"This appendix tabulates the per-sector distribution of the "
        r"validation gold sample (\S\ref{sec:validation}; $n = "
        rf"{n_g:,}$ rows) against the released NAICS-GH training corpus "
        rf"($n = {n_u:,}$ rows). The largest signed delta is "
        rf"$\pm {max_abs_delta:.2f}$ percentage points; "
        rf"{pp_under_1} of {len(cmp)} sectors fall within "
        rf"$\pm 1$ pp."
    )
    parts.append("")
    parts.append(
        r"Reasonable agreement supports our claim that the headline "
        r"96.98\% precision is not biased by an over- or "
        r"under-representation of any single NAICS sector in the gold "
        r"sample."
    )
    parts.append("")

    parts.append(r"\begin{table}[ht]")
    parts.append(r"\centering")
    parts.append(
        r"\caption{Per-sector distribution: validation gold sample "
        r"vs.\ the released NAICS-GH corpus. \emph{Gold \%} is the "
        rf"share of the {n_g:,} gold rows in each sector; "
        rf"\emph{{Corpus \%}} is the share of the {n_u:,} released rows; "
        r"\emph{$\Delta$} is the signed difference in percentage "
        r"points (gold $-$ corpus).}"
    )
    parts.append(r"\label{tab:stratification}")
    parts.append(r"\small")
    parts.append(r"\begin{tabular}{llrrrrr}")
    parts.append(r"\toprule")
    parts.append(
        r"Code & Sector & Gold $n$ & Corpus $n$ & Gold \% & Corpus \% & $\Delta$ (pp) \\"
    )
    parts.append(r"\midrule")
    for code in cmp.index:
        raw_name = name_map.get(code, "")
        short_name = SECTOR_NAME_OVERRIDES.get(code, raw_name)
        sector = latex_escape(short_name)
        gn = int(cmp.loc[code, "gold_n"])
        un = int(cmp.loc[code, "corpus_n"])
        gp = cmp.loc[code, "gold_pct"]
        up = cmp.loc[code, "corpus_pct"]
        d = cmp.loc[code, "delta_pp"]
        parts.append(
            rf"{code} & {sector} & {gn} & {un} & {gp:.2f} & {up:.2f} & "
            rf"{d:+.2f} \\"
        )
    parts.append(r"\midrule")
    parts.append(
        rf"\textbf{{Total}} & & \textbf{{{n_g:,}}} & \textbf{{{n_u:,}}} & "
        rf"\textbf{{100.00}} & \textbf{{100.00}} & \textbf{{0.00}} \\"
    )
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")

    OUT_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(
        f"Wrote {OUT_PATH}  ({len(cmp)} sectors, "
        f"max abs(delta) = {max_abs_delta:.2f} pp)"
    )


if __name__ == "__main__":
    main()
