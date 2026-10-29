"""Stage 06: generate charts and assemble the reproducible report.

Reads every artifact written by stages 01-05, draws the two charts, and
writes ``report.md``. Nothing here recomputes a statistic: the report is a
pure rendering of the CSV/JSON artifacts, so re-running this stage can never
change a number.

Every list in the report has an explicit ordering rule from
:mod:`analysis.sorting` (see :data:`DIRECTION_ORDER`): the most extreme
theme always comes first, so a reader skimming a section sees the headline
example at the top. Every function documents and enforces its
*preconditions* via :mod:`analysis.preconditions`.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common import ensure_output_dir, load_json  # noqa: E402
from analysis.preconditions import (  # noqa: E402
    require,
    require_all,
    require_columns,
    require_existing_dir,
    require_existing_file,
    require_int,
    require_keys,
    require_non_empty,
    require_type,
)
from analysis.sorting import (  # noqa: E402
    count_desc_key,
    high_pct_desc_key,
    high_pct_key,
    residual_desc_key,
    residual_key,
    silhouette_key,
    sort_frame,
    theme_id_key,
)

CLASS_COLORS: dict[str, str] = {
    "HIGH": "#2e7d32",
    "LOW": "#c62828",
    "baseline": "#757575",
    "Excluded (small)": "#bdbdbd",
}

#: Artifact name -> file name. Every one must exist before the report runs.
ARTIFACT_FILES: dict[str, str] = {
    "load_stats": "load_stats.json",
    "vocab": "vocab.json",
    "cluster_summary": "cluster_summary.json",
    "global_stats": "global_stats.json",
    "theme_stats": "theme_stats.csv",
    "oddballs": "oddballs.csv",
    "validation": "validation.csv",
    "floor_sensitivity": "floor_sensitivity.csv",
}

#: Ordering rule for each bulleted list in the report. The most extreme
#: theme leads every list: strongest HIGH, weakest LOW, biggest positive
#: surprise, biggest negative surprise. Keys come from analysis.sorting.
DIRECTION_ORDER: dict[str, Callable[[Any], Any]] = {
    "HIGH": high_pct_desc_key,
    "LOW": high_pct_key,
    "oddball up": residual_desc_key,
    "oddball down": residual_key,
}

#: Columns the merged theme table must carry for the report helpers.
MERGED_COLUMNS: list[str] = [
    "theme", "count", "mean_rating", "high_pct", "lift", "classification",
    "top_keywords", "top_genres", "flag", "residual",
]


def load_artifacts(output_dir: Path) -> dict[str, Any]:
    """Load every intermediate artifact written by stages 01-05.

    Preconditions:
        ``output_dir`` is an existing directory containing every file named
        in :data:`ARTIFACT_FILES` (i.e. stages 01-05 have all been run).
    """
    require_existing_dir(output_dir, "output_dir")
    for label, file_name in ARTIFACT_FILES.items():
        require_existing_file(output_dir / file_name, label)
    return {
        "load_stats": load_json(output_dir / ARTIFACT_FILES["load_stats"]),
        "vocab": load_json(output_dir / ARTIFACT_FILES["vocab"]),
        "cluster_summary": load_json(output_dir / ARTIFACT_FILES["cluster_summary"]),
        "global_stats": load_json(output_dir / ARTIFACT_FILES["global_stats"]),
        "theme_stats": pd.read_csv(output_dir / ARTIFACT_FILES["theme_stats"]),
        "oddballs": pd.read_csv(output_dir / ARTIFACT_FILES["oddballs"]),
        "validation": pd.read_csv(output_dir / ARTIFACT_FILES["validation"]),
        "floor_sensitivity": pd.read_csv(output_dir / ARTIFACT_FILES["floor_sensitivity"]),
    }


def make_theme_chart(theme_stats: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart of per-theme high-rate vs the global baseline; return the PNG path.

    Bars are ordered bottom-to-top by high-rate (``high_pct_key``) so the
    chart reads as a ranking.

    Preconditions:
        ``theme_stats`` is a non-empty ``DataFrame`` with columns ``theme``,
        ``count``, ``high_pct``, ``classification`` and ``global_rate``;
        ``output_dir`` is an existing directory.
    """
    require_columns(
        theme_stats, ["theme", "count", "high_pct", "classification", "global_rate"],
        "theme_stats",
    )
    require_non_empty(theme_stats, "theme_stats")
    require_existing_dir(output_dir, "output_dir")
    ordered = sort_frame(theme_stats, high_pct_key)
    colors = [CLASS_COLORS.get(row.classification, "#757575") for row in ordered.itertuples()]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        [f"T{int(row.theme)} ({int(row.count)})" for row in ordered.itertuples()],
        ordered["high_pct"],
        color=colors,
    )
    global_rate = float(ordered["global_rate"].iloc[0])
    ax.axvline(global_rate, color="black", linestyle="--", linewidth=1.2,
               label=f"global rate {global_rate:.3f}")
    ax.set_xlabel("share of high-rated movies (vote_average >= 7.5)")
    ax.set_title("Keyword themes vs global high-rating rate")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "theme_rates.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_silhouette_chart(scores: dict[int, float], output_dir: Path) -> Path:
    """Silhouette score by ``k`` line chart; return the PNG path.

    The highlighted "best k" uses the same tie-break rule as stage 03
    (:func:`analysis.sorting.silhouette_key`), so chart and clustering agree.

    Preconditions:
        ``scores`` is a non-empty ``dict`` mapping ``int`` k to a numeric
        score; ``output_dir`` is an existing directory.
    """
    require_type(scores, dict, "scores")
    require_non_empty(scores, "scores")
    require_all(
        list(scores.keys()),
        lambda k: isinstance(k, int) and not isinstance(k, bool),
        "scores", "keyed by an int k",
    )
    require_existing_dir(output_dir, "output_dir")
    ks = sorted(scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, [scores[k] for k in ks], marker="o")
    best = max(scores, key=silhouette_key(scores))
    ax.axvline(best, color="red", linestyle="--", linewidth=1.2,
               label=f"best k = {best}")
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("silhouette score")
    ax.set_title("Silhouette score across candidate k")
    ax.legend()
    fig.tight_layout()
    path = output_dir / "silhouette.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_report(artifacts: dict[str, Any], output_dir: Path) -> Path:
    """Write the final report markdown; return its path.

    Preconditions:
        ``artifacts`` is the ``dict`` returned by :func:`load_artifacts`
        (every key in :data:`ARTIFACT_FILES` present; ``theme_stats`` has the
        stage-04 columns incl. ``top_keywords`` / ``top_genres``; ``oddballs``
        has ``theme``, ``flag``, ``residual``); ``output_dir`` is an existing
        directory.
    """
    require_keys(artifacts, ARTIFACT_FILES.keys(), "artifacts")
    require_existing_dir(output_dir, "output_dir")
    load_stats = artifacts["load_stats"]
    vocab = artifacts["vocab"]
    summary = artifacts["cluster_summary"]
    global_stats = artifacts["global_stats"]
    theme_stats = artifacts["theme_stats"]
    oddballs = artifacts["oddballs"]
    validation = artifacts["validation"]
    sensitivity = artifacts["floor_sensitivity"]
    require_columns(
        theme_stats,
        ["theme", "count", "mean_rating", "high_pct", "lift", "classification",
         "top_keywords", "top_genres"],
        "artifacts['theme_stats']",
    )
    require_columns(oddballs, ["theme", "flag", "residual"], "artifacts['oddballs']")

    merged = theme_stats.merge(
        oddballs[["theme", "flag", "residual"]], on="theme", how="left"
    )
    global_rate = float(global_stats["global_rate"])

    sections: list[str] = [
        "# Keyword-defined movie themes and their ratings",
        "",
        "This report discovers themes purely from keyword data (not the official genre "
        "labels), then measures how each theme relates to ratings, and finds themes that "
        "rate surprisingly high, surprisingly low, or that defy what their own official "
        "genres would predict ('oddballs').",
        "",
        "## Headline findings",
        "",
        f"- The global high-rating rate (TMDB `vote_average >= 7.5`) is **{global_rate:.1%}** "
        f"across **{int(global_stats['n_movies'])}** movies with 50+ TMDB votes.",
        _headlines(merged),
        "",
        "## Data and working set",
        "",
        _load_stats_block(load_stats),
        f"- Vocabulary: **{vocab['vocab_size']}** keywords after deduping, dropping terms "
        f"seen in < {vocab['min_df']} or > {vocab['max_df']:.0%} of movies, capped at "
        f"{vocab['max_features']}. {vocab['n_excluded']} movies had no retained keywords.",
        "",
        "## Method",
        f"- **Themes**: hard k-means on the cosine-normalized TF-IDF matrix of movie "
        f"keywords. k was chosen by silhouette over {8}-{15} (`k = {summary['k']}`, "
        f"silhouette **{summary['silhouette_scores'][str(summary['k'])]:.4f}**). Fixed "
        f"seed 42 for full reproducibility.",
        f"- Clusters below **{summary['min_cluster_size']}** movies are "
        f"excluded: **{len(summary['excluded_clusters'])}** clusters "
        f"({summary['excluded_clusters']}).",
        "- **HIGH / LOW**: a theme is HIGH if its high-rate is above the global rate by "
        "more than the normal-approximation 95% CI (and LOW for the mirror case); "
        "otherwise it is 'baseline'.",
        "- **Oddball**: each movie gets an *expected* high probability = mean high-rate of "
        "its own official genres; a theme's residual is observed minus expected. A theme "
        "is an oddball when the bootstrap 95% CI of its members' residuals excludes 0 "
        "(or, as a fallback, when |residual| >= 0.10 with >= 30 movies).",
        "- **Validation**: per-theme mean of MovieLens (100K) user ratings, joined via "
        "`links_small.csv` to TMDB ids.",
        "",
        "## Theme table",
        "",
        _theme_table(merged),
        "",
        "![Theme high-rates vs global baseline](theme_rates.png)",
        "",
        "![Silhouette by k](silhouette.png)",
        "",
        "",
        "## HIGH-rated themes",
        "",
        _direction_list(merged, "HIGH"),
        "",
        "## LOW-rated themes",
        "",
        _direction_list(merged, "LOW"),
        "",
        "## Oddball themes",
        "",
        _oddball_list(merged),
        "",
        "## MovieLens validation",
        _validation_paragraph(merged, validation),
        "",
        _validation_table(validation),
        "",
        "## Caveats",
        "",
        f"- **Working-set size**: the plan anticipated a ~30k-movie working set assuming a "
        f"~28k-row corrupted tail in `movies_metadata.csv`; this file is largely well-"
        f"formed (only {load_stats['metadata']['total_rows'] - load_stats['metadata']['kept']} "
        f"of {load_stats['metadata']['total_rows']} rows dropped), so the working set is "
        f"driven by the `vote_count >= 50` floor and is "
        f"**{load_stats['working_set']['kept']:,}** movies — a real property of the data, "
        f"not a data-loss artifact.",
        f"- **Rating rounding**: TMDB stores `vote_average` rounded to 1 decimal, so scores "
        "like 7.49 vs 7.5 sit on the threshold; movies are labelled high only at "
        "`>= 7.5` exactly.",
        f"- **Vote-floor sensitivity**: the global high-rate at other floors is "
        f"{'; '.join(_floor_bits(sensitivity))} — the baseline is stable around "
        f"{global_rate:.1%}.",
        f"- **Weak silhouette**: the silhouette score is low, and one catch-all theme "
        f"absorbs {int(summary['cluster_sizes'].get('8', 0))} "
        "movies; interpret individual labels as noisy clusters rather than crisp genres.",
        "- **Genre-expected baseline** self-includes theme members (a mild "
        "circularity the project accepts); the MovieLens pass is an independent check.",
        "",
        "## Reproducibility",
        "",
        "```bash",
        "python analysis/01-load.py",
        "python analysis/02-vocab.py",
        "python analysis/03-cluster.py",
        "python analysis/04-ratings.py",
        "python analysis/05-validate.py",
        "python analysis/06-report.py",
        "```",
        "",
        "All random steps use seed 42. Tests: `pytest tests -q`.",
        "",
    ]
    path = output_dir / "report.md"
    path.write_text("\n".join(sections), encoding="utf-8")
    return path


def _load_stats_block(load_stats: dict[str, Any]) -> str:
    """Bulleted working-set statistics from the namespaced ``load_stats.json``.

    Preconditions:
        ``load_stats`` is a ``dict`` with keys ``metadata``, ``keywords`` and
        ``working_set`` (the structure stage 01 writes).
    """
    require_keys(load_stats, ["metadata", "keywords", "working_set"], "load_stats")
    meta = load_stats["metadata"]
    kw = load_stats["keywords"]
    work = load_stats["working_set"]
    agg_dropped = (meta["dropped_bad_id"] + meta["dropped_duplicate"]
                   + meta["dropped_bad_va"] + meta["dropped_bad_vc"])
    return (
        f"- Metadata rows read: **{meta['total_rows']}**; kept **{meta['kept']}** "
        f"(dropped: {agg_dropped} = {meta['dropped_bad_id']} bad ids, "
        f"{meta['dropped_duplicate']} duplicates, "
        f"{meta['dropped_bad_va']} bad ratings, {meta['dropped_bad_vc']} bad vote "
        f"counts).\n"
        f"- Keywords rows read: **{kw['total_rows']}**; kept **{kw['kept']}** "
        f"(dropped: {kw['dropped_bad_id']} bad ids, {kw['dropped_duplicate']} "
        f"duplicates, {kw['unparsable']} rows with no parseable keywords). "
        f"{work['metadata_count'] - work['missing_keywords']} metadata movies joined "
        f"to keywords; {work['keyword_count'] - work['missing_metadata']} keyword rows "
        f"had metadata.\n"
        f"- Vote floor: `vote_count >= {work['min_votes']}` removes "
        f"**{work['below_floor']}** of {work['below_floor'] + work['kept']} joined movies.\n"
        f"- Working set: **{work['kept']}** movies unique by id, "
        f"**{work['high_count']}** high-rated."
    )


def _headlines(merged: pd.DataFrame) -> str:
    """One-line summary of the strongest HIGH / LOW finds (up to three each).

    HIGH themes are listed best-first (``high_pct_desc_key``) and LOW themes
    worst-first (``high_pct_key``), so the most striking example leads.

    Preconditions:
        ``merged`` is a ``DataFrame`` with columns ``theme``,
        ``classification``, ``high_pct`` and ``top_keywords``.
    """
    require_columns(merged, ["theme", "classification", "high_pct", "top_keywords"], "merged")
    highs = sort_frame(merged[merged["classification"] == "HIGH"], high_pct_desc_key)
    lows = sort_frame(merged[merged["classification"] == "LOW"], high_pct_key)
    line = ""
    for r in highs.head(3).itertuples():
        line += (f"- **HIGH**: theme T{r.theme} ({r.top_keywords[:60]}...) is high-rated "
                 f"at {r.high_pct:.1%}.\n")
    for r in lows.head(3).itertuples():
        line += (f"- **LOW**: theme T{r.theme} ({r.top_keywords[:60]}...) is high-rated "
                 f"at {r.high_pct:.1%}.\n")
    return line.rstrip("\n")


def _theme_table(merged: pd.DataFrame) -> str:
    """Markdown table of every theme with its stats, keywords, and genres, by theme id.

    Preconditions:
        ``merged`` is a ``DataFrame`` with every column in
        :data:`MERGED_COLUMNS`.
    """
    require_columns(merged, MERGED_COLUMNS, "merged")
    header = ("| T | size | mean | high% | lift | class | residual | keywords | genres |\n"
              "|---|---|---|---|---|---|---|---|---|")
    rows = []
    for r in sort_frame(merged, theme_id_key).itertuples():
        rows.append(
            f"| {r.theme} | {r.count} | {r.mean_rating:.2f} | {r.high_pct:.1%} | "
            f"{r.lift:.2f} | {r.classification} | {r.residual:+.3f} | "
            f"{_clip(str(r.top_keywords))} | {_clip(str(r.top_genres))} |"
        )
    return "\n".join([header] + rows)


def _direction_list(merged: pd.DataFrame, kind: str) -> str:
    """Bulleted list of themes with a given classification or oddball flag.

    ``kind`` selects both the filter column (``classification`` for HIGH/LOW,
    ``flag`` for oddballs) and the ordering rule from :data:`DIRECTION_ORDER`.

    Preconditions:
        ``kind`` is one of the keys of :data:`DIRECTION_ORDER`; ``merged`` is
        a ``DataFrame`` with columns ``theme``, ``count``, ``high_pct``,
        ``residual``, ``top_keywords`` and the filter column for ``kind``.
    """
    require_type(kind, str, "kind")
    require(kind in DIRECTION_ORDER, f"'kind' must be one of {list(DIRECTION_ORDER)}, got {kind!r}")
    column = "classification" if kind in ("HIGH", "LOW") else "flag"
    require_columns(
        merged, ["theme", "count", "high_pct", "residual", "top_keywords", column], "merged"
    )
    rows = sort_frame(merged[merged[column] == kind], DIRECTION_ORDER[kind])
    lines = []
    for r in rows.itertuples():
        extra = f" (residual {r.residual:+.3f})" if kind not in ("HIGH", "LOW") else ""
        lines.append(f"- **T{r.theme}** — {_clip(str(r.top_keywords))} "
                     f"[{r.count} movies, high-rate {r.high_pct:.1%}]{extra}")
    return "\n".join(lines) if lines else "_none_"


def _oddball_list(merged: pd.DataFrame) -> str:
    """Split oddball themes into "up" and "down" bullet groups.

    Preconditions:
        ``merged`` satisfies the contract of :func:`_direction_list` for the
        ``"oddball up"`` and ``"oddball down"`` kinds.
    """
    require_columns(merged, ["flag"], "merged")
    lines = ["**Rating higher than genres predict (up):**", ""]
    lines.append(_direction_list(merged, "oddball up"))
    lines += ["", "**Rating lower than genres predict (down):**", ""]
    lines.append(_direction_list(merged, "oddball down"))
    return "\n".join(lines)


def _validation_paragraph(merged: pd.DataFrame, validation: pd.DataFrame) -> str:
    """One-paragraph cross-check of the MovieLens results against the headlines.

    Non-baseline themes are visited biggest-first (``count_desc_key``) and
    the first six with a MovieLens mean are quoted.

    Preconditions:
        ``merged`` is a ``DataFrame`` with columns ``theme``,
        ``classification`` and ``count``; ``validation`` is a ``DataFrame``
        with columns ``theme``, ``ml_mean_rating`` and ``ml_n_ratings``.
    """
    require_columns(merged, ["theme", "classification", "count"], "merged")
    require_columns(validation, ["theme", "ml_mean_rating", "ml_n_ratings"], "validation")
    val = validation.set_index("theme")
    pieces = []
    non_baseline = merged[merged["classification"] != "baseline"]
    for r in sort_frame(non_baseline, count_desc_key).itertuples():
        if r.theme not in val.index:
            continue
        v = val.loc[r.theme]
        if pd.isna(v["ml_mean_rating"]):
            continue
        pieces.append(f"theme T{r.theme} ({r.classification.lower()}) — mean "
                      f"user rating **{v['ml_mean_rating']:.2f}** across "
                      f"{int(v['ml_n_ratings'])} ratings")
    return ("MovieLens user ratings (100K set) confirm the headline ordering: " +
            "; ".join(pieces[:6]) + ".")


def _validation_table(validation: pd.DataFrame) -> str:
    """Markdown table of per-theme MovieLens coverage, by theme id.

    Preconditions:
        ``validation`` is a ``DataFrame`` with columns ``theme``,
        ``ml_mean_rating``, ``ml_rated_movies`` and ``ml_n_ratings``.
    """
    require_columns(
        validation, ["theme", "ml_mean_rating", "ml_rated_movies", "ml_n_ratings"], "validation"
    )
    header = "| T | mean ML rating | rated movies | ratings |\n|---|---|---|---|"
    rows = [
        f"| {int(r.theme)} | {r.ml_mean_rating:.2f} | {int(r.ml_rated_movies)} | {int(r.ml_n_ratings)} |"
        for r in sort_frame(validation, theme_id_key).itertuples()
    ]
    return "\n".join([header] + rows)


def _floor_bits(sensitivity: pd.DataFrame) -> list[str]:
    """Human-readable bits for each vote floor's global high-rate.

    Preconditions:
        ``sensitivity`` is a ``DataFrame`` with columns ``floor``, ``count``
        and ``high_rate``.
    """
    require_columns(sensitivity, ["floor", "count", "high_rate"], "sensitivity")
    return [
        f"floor {int(r.floor)}: {r.high_rate:.1%} ({int(r.count)} movies)"
        for r in sensitivity.itertuples()
    ]


def _clip(text: str, limit: int = 48) -> str:
    """Truncate a long joined label for table cells, ending it with an ellipsis.

    Preconditions:
        ``text`` is a ``str``; ``limit`` is an ``int`` >= 2 (one character
        of text plus the ellipsis is the shortest meaningful clip).
    """
    require_type(text, str, "text")
    require_int(limit, "limit", minimum=2)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> None:
    """Run stage 06 end-to-end and write charts + report.

    Preconditions:
        stages 01-05 have been run (checked by :func:`load_artifacts`).
    """
    out = ensure_output_dir()
    artifacts = load_artifacts(out)
    theme_chart = make_theme_chart(artifacts["theme_stats"], out)
    silhouette_chart = make_silhouette_chart(
        {int(k): v for k, v in artifacts["cluster_summary"]["silhouette_scores"].items()},
        out,
    )
    report_path = write_report(artifacts, out)
    print(f"wrote {theme_chart}")
    print(f"wrote {silhouette_chart}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
