# Design

## Context

See proposal.md - Why for motivation. All raw data sits in `data/raw/` (TMDB `movies_metadata.csv` + `keywords.csv`, MovieLens `ratings*.csv` bridged by `links*.csv`). The environment is Windows with Python 3.14 available but no pandas installed yet. Findings that shape the design:

- ~9.2% of a clean sample of movies rate `>= 7.5`, so "high" is a meaningful minority.
- Popularity bias is real: the 3,000 fewest-vote movies average 5.3 vs 6.53 for the 3,000 most-voted.
- `movies_metadata.csv` has a known dirty tail of malformed rows that shift columns (observed live during exploration).

## Goals / Non-Goals

**Goals:**
- A reproducible pipeline from raw files to a class-report-ready analysis.
- Interpretable themes discovered from keywords (hard clustering, one theme per movie).
- HIGH / LOW / oddball classification grounded in statistics, not eyeballing.
- A MovieLens validation pass on the headline findings.
- A test-first, documented codebase: pytest suite authored ahead of implementation; docstrings and type hints on every function; a README that makes the project runnable and interpretable.

**Non-Goals:**
- Predicting ratings or building a recommendation model.
- Making the official 19 genres the subject of the analysis (they serve only as the oddball baseline).
- Loading the full 709MB `ratings.csv` as a requirement (the 100K `ratings_small.csv` is the default validation source; the full file is an optional extension).
- Any UI, service, or deployment.

## Decisions

**D1. Rating source: TMDB `vote_average`, threshold `>= 7.5`, vote floor 50.**
The 0-10 TMDB scale is the only one where 7.5 is meaningful, and it is one value per movie, which clusters cleanly. The floor kills the measured popularity bias. Sensitivity checks at floors 25 and 100 are run to show results do not hinge on the cutoff.
*Alternatives:* MovieLens aggregated mean (0.5-5 scale, would force a different threshold like `>= 4.0`), or both side by side. Rejected because the user's 7.5 definition matches TMDB; MovieLens is kept as validation instead.

**D2. Hard clustering with k-means on TF-IDF, cosine-normalized.**
Movies are vectors over the curated keyword vocabulary; rows are L2-normalized and k is chosen in the 8-15 range by silhouette/elbow. Each movie lands in exactly one theme.
*Alternatives:* agglomerative clustering (more expensive, similar interpretability), HDBSCAN (handles noise better but is harder to explain and can leave movies unclustered). k-means is the class-project-friendly, explainable pick.

**D3. Keyword vocabulary curation.**
Lowercase, dedupe within a movie, drop keywords present in fewer than 3 movies or more than ~40% of them, cap at roughly the 1000 most frequent. This removes both noise terms and theme-less filler ("woman", "murder").
*Alternatives:* no frequency caps (too noisy), or tokenizing `overview` text instead of using curated keywords (out of scope - keywords are the requested signal).

**D4. Theme labeling: top TF-IDF keywords + official-genre make-up.**
Each theme is labeled by its top weighted keywords (the words with the highest average TF-IDF within the cluster) plus a breakdown of its members' official genres, so every theme reads as a sentence.

**D5. Oddball scoring: observed minus genre-expected rate.**
For each movie, expected P(high) is the mean of its own genres' global high-rates; a theme's expected rate is the mean over its members. Residual = observed - expected. Themes with residual magnitude beyond a bootstrap 95% CI (or, fallback, `>= 10` percentage points with theme size `>= 30`) are flagged as oddballs, direction up/down.
*Alternatives:* plain lift vs. the global 9.2% baseline (that yields HIGH/LOW but cannot find surprises - a horror theme with 20% high looks ordinary vs. 9.2% baseline yet is extraordinary vs. horror's own ~7%).

**D6. Validation from MovieLens 100K by default.**
Aggregate per-theme mean user rating via `links_small.csv` -> `ratings_small.csv`. Report next to TMDB findings. Optional: repeat with the full 20M if runtime allows.
*Alternatives:* skipping validation (weakens the story) or requiring full 20M (adds runtime for marginal benefit at 100K scale).

**D7. Data hygiene around `movies_metadata.csv`.**
Parse defensively (field-count recovery, numeric coercion with try/except); record the drop count and final working-set size in the report. Expect roughly 30k+ clean movies.

**D8. Tooling and layout.**
- venv + `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, pytest).
- Code in `analysis/` as short numbered scripts (01-load, 02-vocab, 03-cluster, 04-ratings, 05-validate, 06-report), runnable top-to-bottom.
- Tests in `tests/` mirroring the pipeline stages; fixtures built from tiny hand-written toy datasets.
- Report outputs to `analysis/output/` (theme table CSV, charts PNG, summary markdown).
- Root `README.md` as the entry point for setup and interpretation.

**D9. Test-first development (TDD).**
Each pipeline stage's pytest suite is written before the module it covers and drives its implementation. Tests run against small hand-built toy datasets (fixtures in `tests/conftest.py`), never the messy real files, so they are deterministic, fast, and pin exact expected values (e.g., a fixture movie with `vote_average = 7.5` and `vote_count = 50` is labeled high; one with 49 votes is excluded). Only the final end-to-end task reads `data/raw/`.
*Alternatives:* write tests after implementation (weakens the "tests before modules" requirement), or test directly on real data (slow and non-deterministic — rejected).

**D10. Docstrings and type hints on every function.**
Every public function/method has a docstring and full type hints (parameter and return types). A lightweight check (AST scan or review pass) in each module's verification confirms neither is missing.
*Alternatives:* docstrings only for public API (unclear boundary in a script-based project), or external doc tooling (overkill at this scope).

**D11. README scope.**
`README.md` at the repo root covers: what the project does, data attribution (TMDB + MovieLens), setup (venv + `pip install -r requirements.txt`), how to run (`analysis/01-load.py` through `06-report.py`), the layout tree, and how to interpret the theme table, HIGH/LOW lists, oddballs, and caveats. It is finalized and verified against a clean run in the final task group.

**D12. Preconditions on every function.**
Every pipeline function states its preconditions in a `Preconditions:` docstring section and enforces them at the top of its body with the `require_*` helpers in `analysis/preconditions.py` (column presence, integer/fraction ranges, parallel-list lengths, existing files, sparse-matrix shape). Violations raise `PreconditionError` (a `ValueError`; `TypePreconditionError` is also a `TypeError`) naming the argument and the value received, so a wrong call fails at the call site rather than deep inside pandas/scikit-learn. Functions with no preconditions say `Preconditions: none` explicitly, and the style test enforces the section's presence.
*Alternatives:* bare `assert` statements (stripped under `python -O`, and messages are rarely informative), or no checks (errors surface as opaque `KeyError`/`ValueError` from library code). Rejected in favour of explicit, documented contracts.

**D13. Custom sort keys instead of ad-hoc sorts.**
Every ordering rule is a named key function in `analysis/sorting.py` (`theme_id_key`, `high_pct_desc_key`, `genre_count_key`, `classification_rank_key`, `keyword_weight_key`, `silhouette_key`, ...) applied through `sorted(..., key=...)` or the stable `sort_frame(frame, key)` helper. Keys return tuples so ties are deterministic, negate a component for descending order, use rank tables for categorical order, and map `NaN` to the end. Report lists are "most extreme first"; k-selection ties resolve to the smaller k.
*Alternatives:* scattered `sort_values(...)` calls with implicit (and, for ties, non-stable) ordering. Rejected because the rule is then invisible, untestable, and inconsistent between the CSV, the chart, and the report.

## Risks / Trade-offs

- **Popularity bias** -> vote floor (D1) plus floor sensitivity checks; measured numbers quoted in the report.
- **Dirty tail of metadata CSV** -> defensive parse (D7) and reported drop count; a healthy final size (~30k+) acts as a sanity gate.
- **Lumpy ratings from one-decimal rounding** -> stated as a caveat; threshold still usable.
- **Small or keyword-poor clusters** -> minimum cluster-size filter; too-small clusters excluded from HIGH/LOW/oddball claims and reported separately.
- **k-means instability across seeds/runs** -> fixed seed, silhouette-based k, optionally repeated runs to confirm stability; recorded in the report.
- **Idiosyncratic TMDB keyword phrases** -> frequency caps (D3) plus manual spot-check of theme labels during the labeling pass.
- **Tests erode on messy real data** -> fixtures stay small and deterministic (D9); the only real-data read is the end-to-end task.
- **README drifts from the actual layout/commands** -> README is verified against a clean run in the final group (D11) before the report is finalized.

## Migration Plan

Not applicable - standalone analysis, no deployment or rollback surface. Re-running the pipeline top-to-bottom is the only "deploy" step.

## Open Questions

- Exact vocabulary cap (500 vs 1000) and the oddball residual threshold tuning. Both are analysis-time parameters that do not change the spec or the approach; the chosen values will be recorded in the report as parameters.