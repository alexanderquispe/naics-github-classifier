"""
Generate Appendix F's training-details reference.

The training hyperparameters live in setup_training_arguments() of
naics_training_gptdata.ipynb. They are constants here so the appendix
stays a single source of truth alongside the paper.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "paper" / "appendix" / "output" / "training_appendix.tex"


# Six base models with HF identifiers + parameter counts (verified from
# naics_training_gptdata.ipynb).
MODELS = [
    ("roberta-base",                  "RoBERTa-base",     "125M"),
    ("roberta-large",                 "RoBERTa-large",    "355M"),
    ("answerdotai/ModernBERT-base",   "ModernBERT-base",  "139M"),
    ("answerdotai/ModernBERT-large",  "ModernBERT-large", "395M"),
    ("microsoft/deberta-v3-base",     "DeBERTa-v3-base",  "183M"),
    ("microsoft/deberta-v3-large",    "DeBERTa-v3-large", "400M"),
]

# Hyperparameters from setup_training_arguments() — applied identically
# to all six runs.
HYPERPARAMS = [
    (r"\texttt{num\_train\_epochs}",            "8"),
    (r"\texttt{learning\_rate}",                r"$1.5\times 10^{-5}$"),
    (r"\texttt{lr\_scheduler\_type}",           r"\texttt{polynomial}"),
    (r"\texttt{warmup\_ratio}",                 "0.15"),
    (r"\texttt{weight\_decay}",                 "0.02"),
    (r"\texttt{per\_device\_train\_batch\_size}", "8"),
    (r"\texttt{per\_device\_eval\_batch\_size}", "16"),
    (r"\texttt{gradient\_accumulation\_steps}", "2"),
    (r"Effective batch size",                   "16"),
    (r"\texttt{max\_grad\_norm}",               "1.0"),
    (r"\texttt{bf16}",                          r"\texttt{True}"),
    (r"\texttt{optim}",                         r"\texttt{adamw\_torch\_fused}"),
    (r"\texttt{eval\_strategy}",                r"\texttt{steps} (every 100)"),
    (r"\texttt{save\_strategy}",                r"\texttt{steps} (every 100)"),
    (r"\texttt{save\_total\_limit}",            "5"),
    (r"\texttt{load\_best\_model\_at\_end}",    r"\texttt{True}"),
    (r"\texttt{metric\_for\_best\_model}",      r"\texttt{f1} (weighted)"),
    (r"\texttt{early\_stopping\_patience}",     "2"),
    (r"\texttt{early\_stopping\_threshold}",    "0.001"),
    (r"\texttt{seed}",                          "42"),
    (r"Tokenizer \texttt{max\_length}",         "512 WordPiece tokens (uniform)"),
]


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    parts = []

    parts.append(
        r"This appendix documents the full benchmark setup behind "
        r"Section~\ref{sec:benchmark}: the input cleaning, the six base "
        r"models and their parameter counts, the complete list of "
        r"training hyperparameters (identical across all six runs), and "
        r"reproducibility notes. The canonical source is "
        r"\texttt{naics\_training\_gptdata.ipynb} in the Colab "
        r"\texttt{naics\_github} folder; a re-runnable local refactor is "
        r"\texttt{naics-github-train/scripts/train.py}."
    )
    parts.append("")

    # --- F.1 Input cleaning ---
    parts.append(r"\subsection*{F.1\quad Input construction and cleaning}")
    parts.append(
        r"Each row is serialized as "
        r"\texttt{Repository: \{name\_repo\} | Description: "
        r"\{description\} | Topics: \{topics\} | README: "
        r"\{readme\_content\}} and then passed through the helper "
        r"\texttt{clean\_repository\_text(text)} which performs nine "
        r"normalization steps:"
    )
    parts.append(r"\begin{enumerate}")
    parts.append(
        r"  \item Strip Markdown badges and shields "
        r"(\texttt{!{[}\dots{]}({[}url{]})} and "
        r"\texttt{{[}!{[}\dots{]}(\dots){]}(\dots)});"
    )
    parts.append(
        r"  \item strip license/copyright headers "
        r"(\texttt{MIT License}, \texttt{Apache License}, \texttt{GPL}, "
        r"\texttt{BSD}, \texttt{Copyright \dots});"
    )
    parts.append(
        r"  \item collapse URLs to the bare domain "
        r"(\texttt{https?://(domain)/path?...} $\rightarrow$ "
        r"\texttt{domain});"
    )
    parts.append(
        r"  \item strip Markdown headers (\texttt{\#}\dots\texttt{\#\#\#\#\#\#}) "
        r"and bold/italic/code markers (\texttt{*}, \texttt{\_}, \texttt{\textasciitilde}, "
        r"\texttt{\textasciigrave});"
    )
    parts.append(
        r"  \item replace fenced code blocks with a "
        r"\texttt{code-\{lang\}} placeholder; strip inline backticks "
        r"but keep the content;"
    )
    parts.append(
        r"  \item normalize tech-stack mentions "
        r"(\texttt{js} $\rightarrow$ \texttt{javascript}, "
        r"\texttt{py} $\rightarrow$ \texttt{python}, "
        r"\texttt{reactjs} $\rightarrow$ \texttt{react}, "
        r"\texttt{nodejs} $\rightarrow$ \texttt{nodejs});"
    )
    parts.append(
        r"  \item normalize excessive punctuation "
        r"(\texttt{!!} $\rightarrow$ \texttt{!}, "
        r"\texttt{??} $\rightarrow$ \texttt{?}, "
        r"\texttt{....} $\rightarrow$ \texttt{\dots});"
    )
    parts.append(
        r"  \item normalize whitespace (collapse multiple newlines and "
        r"spaces);"
    )
    parts.append(
        r"  \item strip installation-command noise "
        r"(\texttt{npm install}, \texttt{pip install}, "
        r"\texttt{git clone}, and the rest of those lines)."
    )
    parts.append(r"\end{enumerate}")
    parts.append(
        r"The cleaned string is the \texttt{text} column passed to the "
        r"tokenizer."
    )
    parts.append("")

    # --- F.2 Models ---
    parts.append(r"\subsection*{F.2\quad Base models and parameter counts}")
    parts.append(
        r"All six runs share data, splits, and hyperparameters; only the "
        r"base model differs."
    )
    parts.append(r"\begin{table}[ht]")
    parts.append(r"\centering")
    parts.append(
        r"\caption{Pretrained encoders fine-tuned in "
        r"\S\ref{sec:benchmark}.}"
    )
    parts.append(r"\label{tab:training-models}")
    parts.append(r"\small")
    parts.append(r"\begin{tabular}{lll}")
    parts.append(r"\toprule")
    parts.append(r"Hugging Face identifier & Label in paper & Parameters \\")
    parts.append(r"\midrule")
    for hf_id, label, params in MODELS:
        hf_id_escaped = hf_id.replace("_", r"\_")
        parts.append(
            rf"\texttt{{{hf_id_escaped}}} & {label} & {params} \\"
        )
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")
    parts.append("")

    # --- F.3 Hyperparameters ---
    parts.append(r"\subsection*{F.3\quad Training hyperparameters}")
    parts.append(
        r"All values below are set in "
        r"\texttt{setup\_training\_arguments()} and were applied "
        r"identically to every model run."
    )
    parts.append(r"\begin{table}[ht]")
    parts.append(r"\centering")
    parts.append(r"\caption{Full hyperparameter sheet for the fine-tuning runs.}")
    parts.append(r"\label{tab:training-hp}")
    parts.append(r"\small")
    parts.append(r"\begin{tabular}{ll}")
    parts.append(r"\toprule")
    parts.append(r"Argument & Value \\")
    parts.append(r"\midrule")
    for arg, val in HYPERPARAMS:
        parts.append(f"{arg} & {val} \\\\")
    parts.append(r"\bottomrule")
    parts.append(r"\end{tabular}")
    parts.append(r"\end{table}")
    parts.append("")

    # --- F.4 Splits + metric ---
    parts.append(r"\subsection*{F.4\quad Splits and evaluation metric}")
    parts.append(
        r"The 6{,}588-row corpus is divided into train / validation / "
        r"test sets of \textbf{4{,}611 / 659 / 1{,}318} via "
        r"\texttt{sklearn.model\_selection.train\_test\_split} with "
        r"\texttt{test\_size = 0.2}, \texttt{val\_size = 0.1}, "
        r"\texttt{random\_state = 42}, and stratification on the "
        r"\texttt{label} column. All 19 NAICS classes appear in every "
        r"split. The evaluation metric for both "
        r"\texttt{metric\_for\_best\_model} and the headline test-set "
        r"result is the weighted F1 over the 19 classes; accuracy, "
        r"weighted precision, and weighted recall are reported alongside."
    )
    parts.append("")

    # --- F.5 Hardware + runtime ---
    parts.append(r"\subsection*{F.5\quad Hardware and runtime}")
    parts.append(
        r"All six runs were performed on a Colab Pro+ instance with a "
        r"single NVIDIA A100 (40~GB). BF16 mixed precision and the "
        r"fused \texttt{adamw\_torch\_fused} optimizer keep memory use "
        r"comfortable for the largest model (DeBERTa-v3-large, 400~M "
        r"parameters) at the per-device batch size of~8. End-to-end "
        r"runtime per model was on the order of 8--15 minutes including "
        r"checkpointing, with early stopping commonly firing between "
        r"epochs 4 and 7."
    )
    parts.append("")

    # --- F.6 Reproducibility ---
    parts.append(r"\subsection*{F.6\quad Reproducibility}")
    parts.append(
        r"The published RoBERTa-large checkpoint at "
        r"\url{https://huggingface.co/alexanderquispe/naics-github-classifier} "
        r"is the artifact produced by the above pipeline with "
        r"\texttt{model\_id = \"roberta-large\"}. To reproduce, clone "
        r"\texttt{naics-github-train}, place the released training file "
        r"\texttt{train\_data\_gpt\_ab8\_score\_with\_code.parquet} "
        r"under \texttt{data/raw/}, and run "
        r"\texttt{python scripts/train.py --model roberta-large "
        r"--batch-size 32 --epochs 8} (the script's defaults match "
        r"the hyperparameters listed in Table~\ref{tab:training-hp}). "
        r"With the same \texttt{seed = 42}, the F1 figure of "
        r"86.45\% should reproduce within $\pm 0.5$ pp of stochastic "
        r"variation."
    )

    OUT_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(
        f"Wrote {OUT_PATH}  ({len(MODELS)} models, {len(HYPERPARAMS)} hyperparameters)"
    )


if __name__ == "__main__":
    main()
