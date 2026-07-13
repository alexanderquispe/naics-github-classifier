"""
Offline smoke test for scripts/05_filter_results.py.

Creates synthetic batch CSVs (with STRING score/match, as the patched
step-4 classifier emits), a synthetic candidates parquet (with a `region`
column to exercise the Tier-1 path), and a synthetic source pool, then runs
the filter script via subprocess and asserts the output corpus is correct.

No API calls, no GPT spend.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
SCRIPT = REPO / "scripts" / "05_filter_results.py"


def main():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        gpt_dir = td / "batch_csvs"
        gpt_dir.mkdir()

        # --- synthetic GPT batch outputs (string score/match) -------------
        # batch_0000: mix of pass/fail/duplicate
        pl.DataFrame({
            "repo_url": [
                "https://github.com/a/one",   # score 9 -> keep
                "https://github.com/b/two",   # score 5 -> drop
                "https://github.com/c/three", # score 8 -> keep
                "https://github.com/a/one",   # dup of first, score 7 (lower)
            ],
            "naics_code": ["51", "51", "11", "51"],
            "match": ["Yes", "No", "Yes", "Yes"],
            "score": ["9", "5", "8", "7"],
            "rationale": ["r1", "r2", "r3", "r4"],
        }).write_csv(gpt_dir / "batch_0000.csv")

        # batch_0001: another file, all-"No" (forces score col to be str-only)
        pl.DataFrame({
            "repo_url": ["https://github.com/d/four", "https://github.com/e/five"],
            "naics_code": ["62", "11"],
            "match": ["No", "Yes"],
            "score": ["2", "10"],
            "rationale": ["r5", "r6"],
        }).write_csv(gpt_dir / "batch_0001.csv")

        # --- synthetic candidates (with region for Tier-1 path) -----------
        candidates = td / "candidates.parquet"
        pl.DataFrame({
            "nwo": ["a/one", "b/two", "c/three", "d/four", "e/five"],
            "naics_code": ["51", "51", "11", "62", "11"],
            "description": ["d-a", "d-b", "d-c", "d-d", "d-e"],
            "topics": ["t-a", "t-b", "t-c", "t-d", "t-e"],
            "naics_definition": ["def51", "def51", "def11", "def62", "def11"],
            "region": ["USA", "EU", "AU", "USA", "EU"],
        }).write_parquet(candidates)

        # --- synthetic source pool (readme_content) -----------------------
        source = td / "source.parquet"
        pl.DataFrame({
            "nwo": ["a/one", "c/three", "e/five", "zzz/missing"],
            "readme_content": ["readme-a", "readme-c", "readme-e", "x"],
        }).write_parquet(source)

        out = td / "corpus.parquet"

        # --- run the filter script (min-score 8, min-samples 0) -----------
        r = subprocess.run(
            [PY, str(SCRIPT),
             "--gpt-output-dir", str(gpt_dir),
             "--candidates", str(candidates),
             "--source-pool", str(source),
             "--output", str(out),
             "--min-score", "8"],
            capture_output=True, text=True,
        )
        print(r.stdout)
        if r.returncode != 0:
            print("STDERR:\n", r.stderr)
            raise SystemExit("[FAIL] filter script exited non-zero")

        corpus = pl.read_parquet(out)

        # Expected to keep: a/one(9), c/three(8), e/five(10) -> 3 rows
        # b/two(5) dropped, d/four(2) dropped, a/one(7) is the dropped dup
        assert corpus.height == 3, f"Expected 3 rows, got {corpus.height}"
        print("[PASS] Row count after score>=8 + dedup = 3")

        # a/one should have score 9 (highest of the dup pair), not 7
        a_one = corpus.filter(pl.col("nwo") == "a/one")
        assert a_one.height == 1 and a_one["score"][0] == 9.0, \
            f"a/one should keep score 9; got {a_one['score'].to_list()}"
        print("[PASS] Dedup kept the higher score (9, not 7)")

        # region preserved
        assert "region" in corpus.columns, "region column missing"
        regions = set(corpus["region"].to_list())
        assert regions == {"USA", "AU", "EU"}, f"Unexpected regions: {regions}"
        print("[PASS] Region column preserved for cross-country analysis")

        # readme_content joined
        assert "readme_content" in corpus.columns, "readme_content missing"
        a_readme = corpus.filter(pl.col("nwo") == "a/one")["readme_content"][0]
        assert a_readme == "readme-a", f"readme join wrong: {a_readme}"
        print("[PASS] readme_content joined from source pool")

        # `code` alias present and equals naics_code
        assert "code" in corpus.columns, "code column missing"
        assert corpus["code"].to_list() == corpus["naics_code"].to_list()
        print("[PASS] `code` label column present")

        # --- run again with min-samples-per-class=2 -----------------------
        out2 = td / "corpus_minsamples.parquet"
        r2 = subprocess.run(
            [PY, str(SCRIPT),
             "--gpt-output-dir", str(gpt_dir),
             "--candidates", str(candidates),
             "--output", str(out2),
             "--min-score", "8",
             "--min-samples-per-class", "2"],
            capture_output=True, text=True,
        )
        print(r2.stdout)
        if r2.returncode != 0:
            print("STDERR:\n", r2.stderr)
            raise SystemExit("[FAIL] filter script (min-samples) exited non-zero")
        corpus2 = pl.read_parquet(out2)
        # Sector 11 has 2 rows (c/three, e/five); 51 has 1 (a/one) -> dropped
        codes = sorted(set(corpus2["code"].to_list()))
        assert codes == ["11"], f"min-samples=2 should keep only sector 11; got {codes}"
        print("[PASS] min-samples-per-class drops under-populated sectors")

        print("\nAll step-5 filter checks passed.")


if __name__ == "__main__":
    main()
