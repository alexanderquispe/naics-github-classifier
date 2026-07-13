"""
Generate Appendix D's schema reference for the released training corpus.

Reads:  naics-github-train/data/raw/train_data_gpt_ab8_score_with_code.parquet
Writes: paper/appendix/output/schema_appendix.tex
"""

from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET = (
    REPO_ROOT.parent
    / "naics-github-train"
    / "data"
    / "raw"
    / "train_data_gpt_ab8_score_with_code.parquet"
)
OUT_PATH = REPO_ROOT / "paper" / "appendix" / "output" / "schema_appendix.tex"


COL_DESCRIPTIONS = {
    "name_repo": "Repository short name (no owner prefix).",
    "description": "Repository description from GitHub.",
    "topics": (
        "Topic tags joined with ``\\texttt{;\\space}'', empty string "
        "if the repository declares no tags."
    ),
    "readme_content": (
        "Cleaned README text (Markdown code blocks, HTML tags, and "
        "URLs stripped; whitespace normalized)."
    ),
    "label": "Integer class encoding, $0\\ldots 18$, monotonic in \\texttt{code}.",
    "code": "2-digit NAICS sector code as a string.",
}

DTYPE_DISPLAY = {
    "object": "string",
    "int64": "int64",
    "int32": "int32",
    "float64": "float64",
}


def latex_escape(s) -> str:
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


def truncate(s: str, n: int = 80) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = s.replace("\n", " ").strip()
    # Strip any non-ASCII characters (emoji, etc.) — they don't render
    # in LaTeX verbatim blocks without extra setup.
    s = s.encode("ascii", "ignore").decode("ascii")
    # Collapse repeated spaces that may result from the stripping.
    s = " ".join(s.split())
    if len(s) > n:
        return s[: n - 1].rstrip() + "..."
    return s


def main() -> None:
    df = pd.read_parquet(PARQUET)

    n_rows = len(df)
    cols = list(df.columns)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    parts = []

    parts.append(
        rf"The released file \texttt{{train\_data\_gpt\_ab8\_score\_with\_code.parquet}} "
        rf"contains \textbf{{{n_rows:,} rows}} and \textbf{{{len(cols)} columns}}. "
        r"Every cell is non-null. Table~\ref{tab:schema-fields} gives the "
        r"per-column schema; Table~\ref{tab:schema-stats} gives "
        r"character-length and uniqueness statistics for the four text "
        r"columns; the listing at the end shows one full row (with the "
        r"\texttt{readme\_content} field truncated for readability)."
    )
    parts.append("")

    # --- Table D.1: schema ---
    parts.append(r"\begin{table}[ht]")
    parts.append(r"\centering")
    parts.append(
        r"\caption{Schema of the released training corpus. The order "
        r"shown matches the column order in the parquet file.}"
    )
    parts.append(r"\label{tab:schema-fields}")
    parts.append(r"\small")
    parts.append(r"\begin{tabular}{lll}")
    parts.append(r"\toprule")
    parts.append(r"Column & Type & Description \\")
    parts.append(r"\midrule")
    for col in cols:
        dtype = str(df[col].dtype)
        type_disp = DTYPE_DISPLAY.get(dtype, dtype)
        desc = COL_DESCRIPTIONS.get(col, "")
        parts.append(
            rf"\texttt{{{latex_escape(col)}}} & {type_disp} & {desc} \\"
        )
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")
    parts.append("")

    # --- Table D.2: stats for the four free-form text columns ---
    # We exclude `code` here (also string-typed) because it is a
    # categorical label with only ~19 unique values, not a free-form
    # text field; per-character-length stats are not informative for it.
    text_cols = ["name_repo", "description", "topics", "readme_content"]
    parts.append(r"\begin{table}[ht]")
    parts.append(r"\centering")
    parts.append(
        r"\caption{Per-column character-length and uniqueness statistics "
        r"for the four string columns. Empty strings (length 0) count as "
        r"present, not missing.}"
    )
    parts.append(r"\label{tab:schema-stats}")
    parts.append(r"\small")
    parts.append(r"\begin{tabular}{lrrrrrr}")
    parts.append(r"\toprule")
    parts.append(
        r"Column & Min & Median & Mean & Max & Empty rows & Unique \\"
    )
    parts.append(r"\midrule")
    for col in text_cols:
        s = df[col].astype(str)
        lens = s.str.len()
        empties = int((lens == 0).sum())
        unique = int(df[col].nunique(dropna=False))
        parts.append(
            rf"\texttt{{{latex_escape(col)}}} & "
            rf"{int(lens.min())} & "
            rf"{int(lens.median())} & "
            rf"{lens.mean():.0f} & "
            rf"{int(lens.max()):,} & "
            rf"{empties:,} ({empties / n_rows * 100:.1f}\%) & "
            rf"{unique:,} \\"
        )
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")
    parts.append("")

    # --- Example row ---
    # Pick a row with non-empty topics so all four text fields are populated.
    candidates = df[df["topics"].astype(str).str.len() > 0]
    sample = candidates.iloc[0] if len(candidates) > 0 else df.iloc[0]

    parts.append(r"\paragraph{Example row.}")
    parts.append(
        r"One representative row from the released file (the "
        r"\texttt{readme\_content} field is truncated here to 240 "
        r"characters for readability; the actual stored value may be "
        r"much longer)."
    )
    parts.append("")
    parts.append(
        r"\begin{Verbatim}[fontsize=\footnotesize,breaklines=true,"
        r"breakanywhere=true,breakindent=0pt,"
        r"breaksymbolleft={\textcolor{gray}{\tiny$\hookrightarrow$}},"
        r"frame=single,framesep=2mm,xleftmargin=0pt,xrightmargin=0pt]"
    )
    parts.append(f"name_repo:      {truncate(sample['name_repo'], 200)}")
    parts.append(f"description:    {truncate(sample['description'], 200)}")
    parts.append(f"topics:         {truncate(sample['topics'], 200)}")
    parts.append(f"readme_content: {truncate(sample['readme_content'], 240)}")
    parts.append(f"label:          {int(sample['label'])}")
    parts.append(f"code:           {sample['code']}")
    parts.append(r"\end{Verbatim}")

    OUT_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(
        f"Wrote {OUT_PATH}  ({len(cols)} columns, {len(text_cols)} text columns, "
        f"{n_rows:,} rows)"
    )


if __name__ == "__main__":
    main()
