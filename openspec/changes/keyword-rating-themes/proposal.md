# Proposal

## Why

For an ITSC 1213 data science project, we want to know whether keyword-defined themes in a movie dataset are associated with high or low ratings. The dataset (TMDB metadata + keywords + MovieLens ratings in `data/raw/`) already has official genres; the insight this project adds is discovering *themes* purely from keywords and finding themes that rate surprisingly high, surprisingly low, or that defy what their own genres would predict ("oddballs").

## What Changes

- Add an analysis pipeline that reads the raw files in `data/raw/` and produces a clean working set of movies (join metadata/keywords on TMDB id, drop malformed rows, apply a `vote_count` floor).
- Define "high" rating as TMDB `vote_average >= 7.5`, matching the dataset's 0-10 scale.
- Build a curated keyword vocabulary (dedupe, drop ultra-common and ultra-rare terms) and a movie x keyword TF-IDF matrix.
- Cluster movies into 8-15 keyword-driven **themes** (hard clustering), label each theme with its top keywords and official-genre make-up.
- Produce per-theme rating statistics: mean rating, percentage rated high, and lift vs. the global baseline, classified into HIGH / LOW themes.
- Detect **oddball** themes: themes whose high-rating rate deviates from the rate their official genres would predict.
- Validate the top/bottom findings against the MovieLens user-rating data.
- Emit a class-report-shaped set of outputs: stats, theme table, charts, oddball list, caveats.
- Deliver the project with a README that documents setup, project layout, run instructions, and how to interpret the outputs.
- Adopt a test-first workflow: an extensive pytest suite is written before the modules it covers, and every method carries a docstring and full type hints.

## Capabilities

### New Capabilities

- `keyword-theme-analysis`: Discover keyword-defined themes in a movie dataset, associate them with ratings, classify high/low themes, surface oddball themes, and validate against user ratings.

### Modified Capabilities

(none - this project has no existing specs)

## Impact

- **Data**: read-only use of `data/raw/` (`movies_metadata.csv`, `keywords.csv`, `links*.csv`, `ratings*.csv`). No changes to raw files.
- **Environment**: Python (3.14 available) plus pandas, numpy, scikit-learn, matplotlib, and pytest to be installed. No pandas is currently installed.
- **New artifacts**: analysis scripts, a pytest suite under `tests/`, a root `README.md`, and generated report outputs (theme table, charts, oddball list).
- **No production system or API changes** - standalone class-project analysis.