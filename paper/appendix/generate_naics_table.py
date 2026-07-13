"""
Generate Appendix B's NAICS-sector reference table as a LaTeX longtable.

Reads:  data/raw/naics_titles_by_group_6digit_clean.json
Writes: paper/appendix/output/naics_appendix_table.tex
"""

from pathlib import Path
import json


REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "data" / "raw" / "naics_titles_by_group_6digit_clean.json"
OUT_PATH = REPO_ROOT / "paper" / "appendix" / "output" / "naics_appendix_table.tex"

# Canonical 2-digit NAICS 2022 sector names (U.S. Census Bureau).
NAICS_2022_NAMES = {
    "11":    "Agriculture, Forestry, Fishing and Hunting",
    "21":    "Mining, Quarrying, and Oil and Gas Extraction",
    "22":    "Utilities",
    "23":    "Construction",
    "31-33": "Manufacturing",
    "42":    "Wholesale Trade",
    "44-45": "Retail Trade",
    "48-49": "Transportation and Warehousing",
    "51":    "Information",
    "52":    "Finance and Insurance",
    "53":    "Real Estate and Rental and Leasing",
    "54":    "Professional, Scientific, and Technical Services",
    "55":    "Management of Companies and Enterprises",
    "56":    "Administrative and Support and Waste Management "
             "and Remediation Services",
    "61":    "Educational Services",
    "62":    "Health Care and Social Assistance",
    "71":    "Arts, Entertainment, and Recreation",
    "72":    "Accommodation and Food Services",
    "81":    "Other Services (except Public Administration)",
    "92":    "Public Administration",
}

NUM_EXAMPLES = 3  # How many subindustry phrases to show per sector


def latex_escape(s: str) -> str:
    """Escape characters that need escaping in LaTeX tabular text."""
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


def main() -> None:
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{longtable}{@{}lp{4.3cm}rp{6.0cm}@{}}",
        r"\caption{NAICS 2022 sectors used by the pipeline. The $n$ "
        r"column gives the number of 6-digit subindustry phrases under "
        r"each 2-digit sector in our taxonomy file; the full list of "
        r"1{,}029 phrases is in "
        r"\texttt{naics\_titles\_by\_group\_6digit\_clean.json} in the "
        r"released code repository. The Example column shows the first "
        rf"{NUM_EXAMPLES} subindustry phrases as they appear in the "
        r"file (verbatim, including any minor whitespace). Sector~55 "
        r"is absent from the released training corpus because too few "
        r"candidates passed verification under the 80-sample "
        r"minimum-class-size filter; it is retained here for taxonomic "
        r"completeness.}\\",
        r"\label{tab:naics-appendix}\\",
        r"\toprule",
        r"Code & Sector & $n$ & Example subindustry titles \\",
        r"\midrule",
        r"\endfirsthead",
        r"\multicolumn{4}{c}{\textit{(continued from previous page)}}\\",
        r"\toprule",
        r"Code & Sector & $n$ & Example subindustry titles \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    for code in [
        "11", "21", "22", "23", "31-33", "42", "44-45", "48-49",
        "51", "52", "53", "54", "55", "56", "61", "62",
        "71", "72", "81", "92",
    ]:
        raw_string = taxonomy.get(code, "")
        phrases = [p.strip() for p in raw_string.split(";") if p.strip()]
        n = len(phrases)
        examples = "; ".join(phrases[:NUM_EXAMPLES])
        if n > NUM_EXAMPLES:
            examples = examples + r"; \dots"
        sector_name = NAICS_2022_NAMES.get(code, "")
        lines.append(
            f"{code} & {latex_escape(sector_name)} & {n} & "
            f"{latex_escape(examples)} \\\\"
        )

    lines.append(r"\end{longtable}")

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    total = sum(
        len([p for p in taxonomy.get(c, "").split(";") if p.strip()])
        for c in NAICS_2022_NAMES
    )
    print(f"Wrote {OUT_PATH}  (20 sectors, {total:,} subindustries)")


if __name__ == "__main__":
    main()
