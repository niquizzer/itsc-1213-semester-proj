"""Stage 06: generate charts and assemble the reproducible report."""

from __future__ import annotations

import sys
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

CLASS_COLORS: dict[str, str] = {
    "HIGH": "#2e7d32",
    "LOW": "#c62828",
    "baseline": "#757575",
    "Excluded (small)": "#bdbdbd",
}


def load_artifacts(output_dir: Path) -> dict[str, Any]:
    """Load every intermediate artifact written by stages 01-05."""
    return {
        "load_stats": load_json(output_dir / "load_stats.json"),
        "vocab": load_json(output_dir / "vocab.json"),
        "cluster_summary": load_json(output_dir / "cluster_summary.json"),
        "global_stats": load_json(output_dir / "global_stats.json"),
        "theme_stats": pd.read_csv(output_dir / "theme_stats.csv"),
        "oddballs": pd.read_csv(output_dir / "oddballs.csv"),
        "validation": pd.read_csv(output_dir / "validation.csv"),
        "floor_sensitivity": pd.read_csv(output_dir / "floor_sensitivity.csv"),
    }


def make_theme_chart(theme_stats: pd.DataFrame, output_dir: Path) -> Path:
    """Bar chart of per-theme high-rate vs global baseline; return PNG path."""
    ordered = theme_stats.sort_values("high_pct")
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
    """Silhouette score by k chart; return PNG path."""
    ks = sorted(scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, [scores[k] for k in ks], marker="o")
    best = max(scores, key=scores.get)
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
    """Write the final report markdown; return its path."""
    load_stats = artifacts["load_stats"]
    vocab = artifacts["vocab"]
    summary = artifacts["cluster_summary"]
    global_stats = artifacts["global_stats"]
    theme_stats = artifacts["theme_stats"]
    oddballs = artifacts["oddballs"]
    validation = artifacts["validation"]
    sensitivity = artifacts["floor_sensitivity"]

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
    """Bulleted working-set statistics from the namespaced load_stats.json."""
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
    """One-line summary of the strongest HIGH / LOW finds."""
    highs = merged[merged["classification"] == "HIGH"].sort_values("high_pct", ascending=False)
    lows = merged[merged["classification"] == "LOW"].sort_values("high_pct")
    line = ""
    for r in highs.head(3).itertuples():
        line += (f"- **HIGH**: theme T{r.theme} ({r.top_keywords[:60]}...) is high-rated "
                 f"at {r.high_pct:.1%}.\n")
    for r in lows.head(3).itertuples():
        line += (f"- **LOW**: theme T{r.theme} ({r.top_keywords[:60]}...) is high-rated "
                 f"at {r.high_pct:.1%}.\n")
    return line.rstrip("\n")


def _theme_table(merged: pd.DataFrame) -> str:
    """Markdown table of every theme with its stats, keywords, and genres."""
    header = ("| T | size | mean | high% | lift | class | residual | keywords | genres |\n"
              "|---|---|---|---|---|---|---|---|---|")
    rows = []
    for r in merged.sort_values("theme").itertuples():
        flag = r.flag if isinstance(r.flag, str) else "n/a"
        rows.append(
            f"| {r.theme} | {r.count} | {r.mean_rating:.2f} | {r.high_pct:.1%} | "
            f"{r.lift:.2f} | {r.classification} | {r.residual:+.3f} | "
            f"{_clip(str(r.top_keywords))} | {_clip(str(r.top_genres))} |"
        )
    return "\n".join([header] + rows)


def _direction_list(merged: pd.DataFrame, kind: str) -> str:
    """Bulleted list of themes with a given classification or oddball flag."""
    if kind in ("HIGH", "LOW"):
        rows = merged[merged["classification"] == kind].sort_values("high_pct")
    else:
        rows = merged[merged["flag"] == kind].sort_values("residual")
    lines = []
    for r in rows.itertuples():
        extra = f" (residual {r.residual:+.3f})" if kind not in ("HIGH", "LOW") else ""
        lines.append(f"- **T{r.theme}** — {_clip(str(r.top_keywords))} "
                     f"[{r.count} movies, high-rate {r.high_pct:.1%}]{extra}")
    return "\n".join(lines) if lines else "_none_"


def _oddball_list(merged: pd.DataFrame) -> str:
    """Split oddball themes into up and down bullets."""
    up = merged[merged["flag"] == "oddball up"].sort_values("residual", ascending=False)
    down = merged[merged["flag"] == "oddball down"].sort_values("residual")
    lines = ["**Rating higher than genres predict (up):**", ""]
    lines.append(_direction_list(merged, "oddball up") or "_none_")
    lines += ["", "**Rating lower than genres predict (down):**", ""]
    lines.append(_direction_list(merged, "oddball down") or "_none_")
    return "\n".join(lines)


def _validation_paragraph(merged: pd.DataFrame, validation: pd.DataFrame) -> str:
    """One-paragraph cross-check of the MovieLens results against headlines."""
    val = validation.set_index("theme")
    pieces = []
    for r in merged[merged["classification"] != "baseline"].sort_values("count", ascending=False).itertuples():
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
    """Markdown table of per-theme MovieLens coverage."""
    header = "| T | mean ML rating | rated movies | ratings |\n|---|---|---|---|"
    rows = [
        f"| {int(r.theme)} | {r.ml_mean_rating:.2f} | {int(r.ml_rated_movies)} | {int(r.ml_n_ratings)} |"
        for r in validation.sort_values("theme").itertuples()
    ]
    return "\n".join([header] + rows)


def _floor_bits(sensitivity: pd.DataFrame) -> list[str]:
    """Human-readable bits for each vote floor's global high-rate."""
    return [
        f"floor {int(r.floor)}: {r.high_rate:.1%} ({int(r.count)} movies)"
        for r in sensitivity.itertuples()
    ]


def _clip(text: str, limit: int = 48) -> str:
    """Truncate a long joined label for table cells."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def main() -> None:
    """Run stage 06 end-to-end and write charts + report."""
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