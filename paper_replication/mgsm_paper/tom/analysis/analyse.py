"""Point-estimate analysis of a run: the paper's Table 3 layout, compliance, diagnostics.

    python -m analysis.analyse results/kaggle/<run>.jsonl [more.jsonl ...]

Pass-2 rows (non-native exemplar sets) appear as their own table rows, labelled
"<cell>+<exemplar_set>", so a pass-1 and a pass-2 file for the same model can be
analysed together. Tables and per-row scores are written to analysis/out/<run>/.
Decided 2026-11-02: point estimates first; extracted answers trusted as stored.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.compliance import language_profile
from src.data import HIGH_RESOURCE, LANGUAGES, UNDER_REPRESENTED

# The paper's Table 3 (PaLM-540B, native exemplars), for side-by-side reading.
# Column order matches LANGUAGES.
PAPER_PALM_540B = {
    "direct":     [22.0, 18.8, 19.6, 20.0, 22.0, 19.2, 16.0, 16.8, 17.6, 17.2, 15.6],
    "native_cot": [62.4, 49.2, 46.4, 56.8, 48.4, 46.8, 40.0, 52.8, 45.6, 46.0, 35.2],
    "en_cot":     [62.4, 53.6, 51.2, 58.0, 55.6, 46.0, 49.6, 49.6, 46.8, 46.4, 44.4],
}
CELL_ORDER = ["direct", "native_cot", "en_cot",
              "direct+english", "en_cot+english",
              "direct+multilingual", "native_cot+multilingual", "en_cot+multilingual"]


# -----------------------------------------------------------------------------
# 1. Loading. pandas reads JSON Lines directly. We add the solution language,
#    which is what compliance is measured against, and a boolean `scored_correct`
#    where an unparsed answer (correct is None) counts as wrong. That is the
#    intention-to-treat headline from the plan, section 5.
# -----------------------------------------------------------------------------
def load(paths: list[str]) -> pd.DataFrame:
    # json.loads per line rather than pd.read_json: a model once wrote a number too big
    # for a 64-bit integer, and pandas' JSON reader refuses it; Python ints have no limit.
    rows = [json.loads(line) for p in paths for line in open(p, encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["solution_lang"] = df["lang"].where(df["cell"] != "en_cot", "en")
    # Pass-2 rows: label the cell with its exemplar set so they get their own table rows.
    non_native = df["exemplar_set"] != "native"
    df.loc[non_native, "cell"] = df.loc[non_native, "cell"] + "+" + df.loc[non_native, "exemplar_set"]
    df["scored_correct"] = df["correct"].fillna(False).astype(bool)
    df["parsed"] = df["extracted_answer"].notna()
    return df


# -----------------------------------------------------------------------------
# 2. Compliance. Only meaningful for chain-of-thought rows; DIRECT answers are a
#    bare number. language_profile returns a dict per row; we keep three fields.
#    8,000 rows take about a minute, which is why it is one pass, done once.
# -----------------------------------------------------------------------------
def add_compliance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["complied"] = pd.NA
    df["target_frac"] = pd.NA
    df["dominant_lang"] = pd.NA
    cot = ~df["cell"].str.startswith("direct")
    for idx in df.index[cot]:
        profile = language_profile(df.at[idx, "completion"], df.at[idx, "solution_lang"])
        df.at[idx, "complied"] = profile["complied"]
        df.at[idx, "target_frac"] = profile["target_frac"]
        df.at[idx, "dominant_lang"] = profile["dominant_lang"]
    return df


# -----------------------------------------------------------------------------
# 3. Tables. One helper lays out any per-(cell, lang) statistic in the paper's
#    column order with AVG / HRL / URL on the left, as Table 3 does.
# -----------------------------------------------------------------------------
def table3_layout(per_cell_lang: pd.Series) -> pd.DataFrame:
    """per_cell_lang: a Series indexed by (cell, lang) holding a percentage.
    Returns rows = cells, columns = AVG, HRL, URL, then the 11 languages."""
    present = [c for c in CELL_ORDER if c in per_cell_lang.index.get_level_values("cell")]
    wide = per_cell_lang.unstack("lang").reindex(index=present, columns=list(LANGUAGES))
    out = pd.DataFrame(index=wide.index)
    out["AVG"] = wide.mean(axis=1)                       # average over all 11 languages
    out["HRL"] = wide[list(HIGH_RESOURCE)].mean(axis=1)
    out["URL"] = wide[list(UNDER_REPRESENTED)].mean(axis=1)
    return pd.concat([out, wide], axis=1)


def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Percent correct per cell x language; unparsed counts as wrong."""
    acc = df.groupby(["cell", "lang"])["scored_correct"].mean() * 100
    return table3_layout(acc)


def compliant_only_table(df: pd.DataFrame) -> pd.DataFrame:
    """Percent correct among rows whose solution was in the requested language."""
    sub = df[df["complied"] == True]  # noqa: E712  (pd.NA makes `is True` unusable)
    acc = sub.groupby(["cell", "lang"])["scored_correct"].mean() * 100
    return table3_layout(acc).drop(index=[c for c in ("direct", "direct+english", "direct+multilingual")], errors="ignore")


def compliance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Percent of chain-of-thought rows whose solution was in the requested language."""
    cot = df[~df["cell"].str.startswith("direct")]
    rate = cot.groupby(["cell", "lang"])["complied"].apply(lambda s: (s == True).mean() * 100)  # noqa: E712
    return table3_layout(rate).drop(index=[c for c in ("direct", "direct+english", "direct+multilingual")], errors="ignore")


def diagnostics_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per cell: rows, parse failures, truncations, and how answers were extracted."""
    g = df.groupby("cell")
    out = pd.DataFrame({
        "rows": g.size(),
        "unparsed %": g["parsed"].apply(lambda s: (~s).mean() * 100),
        "truncated %": g["finish_reason"].apply(lambda s: (s == "length").mean() * 100),
        "mean tokens": g["n_tokens"].mean(),
    })
    methods = df.pivot_table(index="cell", columns="extraction_method", values="item_id",
                             aggfunc="count", fill_value=0)
    methods = methods.div(methods.sum(axis=1), axis=0) * 100
    methods.columns = [f"{c} %" for c in methods.columns]
    return pd.concat([out, methods], axis=1).reindex([c for c in CELL_ORDER if c in out.index])


def paper_table() -> pd.DataFrame:
    s = pd.Series({(cell, lang): v for cell, vals in PAPER_PALM_540B.items()
                   for lang, v in zip(LANGUAGES, vals)})
    s.index.names = ["cell", "lang"]
    return table3_layout(s)


# -----------------------------------------------------------------------------
# 4. Output. A small Markdown formatter (one decimal, blanks for missing cells),
#    so no extra dependency, and CSVs of the tables and the per-row scores.
# -----------------------------------------------------------------------------
def md(df: pd.DataFrame, decimals: int = 1) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([""] + [str(c) for c in cols]) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = ["" if pd.isna(v) else f"{v:.{decimals}f}" for v in row]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit("usage: python -m analysis.analyse results/<file>.jsonl [more.jsonl ...]")
    df = load(argv)
    run_name = df["run"].iloc[0]
    out_dir = Path("analysis/out") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"# {run_name}: {len(df)} rows, model {df['model'].iloc[0]}\n")
    df = add_compliance(df)

    tables = {
        "accuracy": accuracy_table(df),
        "accuracy_compliant_only": compliant_only_table(df),
        "compliance_rate": compliance_table(df),
        "diagnostics": diagnostics_table(df),
        "paper_palm540b": paper_table(),
    }
    titles = {
        "accuracy": "Accuracy % (all items; unparsed = wrong). Paper Table 3 layout.",
        "accuracy_compliant_only": "Accuracy % on rows whose solution was in the requested language",
        "compliance_rate": "Compliance % (solution in the requested language), chain-of-thought cells",
        "diagnostics": "Diagnostics per cell",
        "paper_palm540b": "For reference: the paper's PaLM-540B, native exemplars (Table 3)",
    }
    for key, table in tables.items():
        print(f"## {titles[key]}\n")
        print(md(table))
        print()
        table.to_csv(out_dir / f"{key}.csv")

    keep = ["run", "model", "lang", "cell", "item_id", "gold", "extracted_answer", "extraction_method",
            "correct", "scored_correct", "parsed", "finish_reason", "n_tokens", "n_prompt_tokens",
            "solution_lang", "complied", "target_frac", "dominant_lang"]
    df[keep].to_csv(out_dir / "rows.csv", index=False)
    print(f"wrote {len(tables)} tables and rows.csv to {out_dir}/")


if __name__ == "__main__":
    main(sys.argv[1:])
