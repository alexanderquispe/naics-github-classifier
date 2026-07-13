"""
Extract the verbatim production LLM prompt from the EU classification
notebook's Python mirror and emit a LaTeX appendix fragment.

Reads:  Dropbox/naics_github/Colab Notebooks/3_eu_repo_gpt_classif.py
Writes: paper/appendix/output/prompt_appendix.tex
"""

from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(
    r"C:\Users\Alexander\Dropbox\naics_github\Colab Notebooks"
    r"\3_eu_repo_gpt_classif.py"
)
OUT_PATH = REPO_ROOT / "paper" / "appendix" / "output" / "prompt_appendix.tex"


PROMPT_RE = re.compile(
    r'def build_prompt\([^)]*\):.*?prompt\s*=\s*f"""(?P<body>.*?)"""',
    re.DOTALL,
)

SYSTEM_MSG_RE = re.compile(
    r'system_msg\s*=\s*f?"(?P<msg>[^"]+)"'
)


def f_string_unescape(s: str) -> str:
    """Turn the f-string {{ / }} escapes back into literal braces, matching
    what GPT actually receives after Python substitutes the placeholders."""
    return s.replace("{{", "{").replace("}}", "}")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Production notebook mirror not found at {SOURCE}.\n"
            f"Update SOURCE in this script if the path has changed."
        )

    text = SOURCE.read_text(encoding="utf-8")

    m = PROMPT_RE.search(text)
    if not m:
        raise RuntimeError(
            "Could not locate the build_prompt body; the regex needs an update."
        )
    prompt_body = f_string_unescape(m.group("body")).strip("\n")

    sm = SYSTEM_MSG_RE.search(text)
    if not sm:
        raise RuntimeError("Could not locate the system_msg assignment.")
    system_msg = sm.group("msg")

    # Sanity check: the prompt should contain the section headers we expect.
    expected = [
        "TASK: GitHub Repository NAICS Sector Classification",
        "REPOSITORY README CONTENT:",
        "TARGET NAICS SECTOR:",
        "CLASSIFICATION FRAMEWORK:",
        "SCORING GUIDE:",
        "Provide your response in this exact JSON format:",
    ]
    for s in expected:
        assert s in prompt_body, f"Missing expected header: {s!r}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    parts.append(
        r"This appendix reproduces verbatim the prompt sent to GPT-4.1 for "
        r"every candidate (repository, NAICS sector) pair. The three "
        r"placeholders \texttt{\{naics\_code\}}, "
        r"\texttt{\{readme\_text.strip()\}}, and "
        r"\texttt{\{sector\_industries.strip()\}} are substituted per row "
        r"before the request is sent. Source: "
        r"\texttt{3\_eu\_repo\_gpt\_classif.ipynb} cell 22 "
        r"(\texttt{build\_prompt} function) and cell 26 (system message and "
        r"API call). The corresponding Python mirror is "
        r"\texttt{3\_eu\_repo\_gpt\_classif.py}."
    )
    parts.append("")

    vbox_opts = (
        r"fontsize=\footnotesize,"
        r"breaklines=true,"
        r"breakanywhere=true,"
        r"breakindent=0pt,"
        r"breaksymbolleft={\textcolor{gray}{\tiny$\hookrightarrow$}},"
        r"frame=single,"
        r"framesep=2mm,"
        r"xleftmargin=0pt,"
        r"xrightmargin=0pt"
    )

    # System message
    parts.append(r"\paragraph{System message.}")
    parts.append(r"\begin{Verbatim}[" + vbox_opts + r"]")
    parts.append(system_msg)
    parts.append(r"\end{Verbatim}")
    parts.append("")

    # User message (the prompt)
    parts.append(r"\paragraph{User message.}")
    parts.append(r"\begin{Verbatim}[" + vbox_opts + r"]")
    parts.append(prompt_body)
    parts.append(r"\end{Verbatim}")
    parts.append("")

    # API parameters
    parts.append(r"\paragraph{API parameters.}")
    parts.append(r"\begin{itemize}")
    parts.append(
        r"  \item \textbf{Endpoint:} "
        r"\url{https://api-model-lab.githubcopilot.com/chat/completions}"
    )
    parts.append(
        r"  \item \textbf{Model alias:} \texttt{gpt-4.1} "
        r"(resolves to snapshot \texttt{gpt-4.1-2025-04-14} at the "
        r"GitHub Copilot LLM-lab endpoint)."
    )
    parts.append(r"  \item \textbf{\texttt{temperature}:} 0")
    parts.append(
        r"  \item \textbf{\texttt{max\_tokens} (output):} "
        r"$\min(3500,\; 128000 - n_{\text{input}})$, where "
        r"$n_{\text{input}}$ is the token count of the system "
        r"plus user message under "
        r"\texttt{tiktoken.encoding\_for\_model(\"gpt-4.1\")}."
    )
    parts.append(
        r"  \item \textbf{Retry policy:} up to 5 attempts on any non-200 "
        r"response, with linear back-off "
        r"\texttt{sleep = 5 $\times$ attempt} seconds."
    )
    parts.append(
        r"  \item \textbf{Response parsing:} the model's reply is "
        r"searched with \texttt{re.search(r'\textbackslash\{[\textbackslash s\textbackslash S]*\textbackslash\}', reply)} "
        r"and the matched substring passed to \texttt{json.loads}. The "
        r"expected output is nested under the outer key "
        r"\texttt{\"NAICS \{code\}\"} with string-valued "
        r"\texttt{match} (\texttt{\"Yes\"} or \texttt{\"No\"}), "
        r"string-valued \texttt{score} (\texttt{\"1\"}--\texttt{\"10\"}), "
        r"and free-form \texttt{rationale}."
    )
    parts.append(r"\end{itemize}")

    OUT_PATH.write_text("\n".join(parts) + "\n", encoding="utf-8")
    n_lines = len(prompt_body.splitlines())
    print(
        f"Wrote {OUT_PATH}  (system message + {n_lines}-line user prompt + "
        f"API parameter block)"
    )


if __name__ == "__main__":
    main()
