# ITSC 1213 — Keyword-Defined Movie Themes and Their Ratings

A reproducible, test-first data-science pipeline that discovers **themes** from movie
keywords — without using the official genre labels — and then measures how each theme
relates to ratings. It answers three questions:

1. **Which keyword themes are HIGH or LOW rated?** (relative to the global baseline)
2. **Which themes are oddballs?** (they rate much higher or lower than what their own
   official genres would predict)
3. **Do the findings hold up against real user ratings?** (MovieLens cross-check)

A full write-up, theme table, charts, oddball list, and caveats are generated into
`analysis/output/report.md`.

## Data

All raw input lives under `data/raw/` and is **read-only** — the pipeline never
modifies it. The four files the pipeline actually reads (`movies_metadata.csv`,
`keywords.csv`, `links_small.csv`, `ratings_small.csv`, ~41 MB) **are committed**,
so a fresh clone can reproduce every result. The three larger files the pipeline
does *not* read (`ratings.csv` ~677 MB, `credits.csv` ~181 MB, `links.csv`) are not
versioned — fetch them yourself if you want the full dataset.

| File | Contents | Used by |
|---|---|---|
| `movies_metadata.csv` | TMDB movie metadata (title, genres, vote counts/averages) | stage 01 |
| `keywords.csv` | Per-movie keyword tags | stages 01, 02, 03 |
| `credits.csv` | Crew/cast (not used by this pipeline) | — |
| `links.csv` / `ratings.csv` | Full MovieLens link + rating tables | — (not read by stages) |
| `links_small.csv` / `ratings_small.csv` | MovieLens links + user ratings used to validate themes | stage 05 |

**Attribution:** the TMDB files (`movies_metadata.csv`, `keywords.csv`, `credits.csv`)
are from [The Movie Database (TMDB)](https://www.themoviedb.org/). The MovieLens files
(`links*.csv`, `ratings*.csv`) are from the
[MovieLens datasets](https://grouplens.org/datasets/movielens/), GroupLens, University
of Minnesota. See the course handout for the original download sources.

## Setup

A Python 3 virtual environment with all dependencies is assumed. From the project root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`requirements.txt` installs pandas, numpy, scipy, scikit-learn, matplotlib, and pytest.
On macOS/Linux, replace `.venv\Scripts\python` with `.venv/bin/python`.

## How to run

Run the stages in order from the project root (each reads the previous stage's outputs —
they are *not* idempotent out of order):

```powershell
.venv\Scripts\python analysis\01-load.py
.venv\Scripts\python analysis\02-vocab.py
.venv\Scripts\python analysis\03-cluster.py
.venv\Scripts\python analysis\04-ratings.py
.venv\Scripts\python analysis\05-validate.py
.venv\Scripts\python analysis\06-report.py
```

Equivalent unix forms use `/` and `.venv/bin/python`. Or run the beat-for-beat
sequence documented in `analysis/output/report.md` → *Reproducibility*.

Run the test suite with:

```powershell
.venv\Scripts\python -m pytest tests -q
```

## Project layout

```
.
├── analysis/
│   ├── common.py          # shared helpers (output dir, JSON/sparse io, dashed-name loader)
│   ├── 01-load.py         # load & clean metadata/keywords, join, vote floor, high label
│   ├── 02-vocab.py        # keyword cleaning, vocab filter, TF-IDF matrix
│   ├── 03-cluster.py      # k selection (silhouette) + k-means themes
│   ├── 04-ratings.py      # theme labels, genre make-up, HIGH/LOW statistics
│   ├── 05-validate.py     # oddball detection + MovieLens validation
│   ├── 06-report.py       # charts + report markdown
│   └── output/            # all generated artifacts (never hand-edited)
├── tests/
│   ├── conftest.py        # toy fixtures
│   ├── test_smoke.py, test_working_set.py, test_vocab.py, test_cluster.py,
│   ├── test_labeling.py, test_ratings.py, test_oddballs.py,
│   ├── test_validation.py, test_style.py   # TDD suite (docstring/hint checks too)
├── data/raw/              # read-only raw datasets (do not modify)
├── requirements.txt
└── README.md
```

## How to interpret the outputs

- **`analysis/output/working_set.csv`** — clean movies with 50+ TMDB votes, their
  keyword lists, and a `high_rating` flag (`vote_average >= 7.5`).
- **`analysis/output/theme_stats.csv`** — one row per theme: size, mean rating,
  % high-rated, lift vs the global rate, **classification** (`HIGH` / `LOW` / `baseline`),
  and its top keywords and genres.
- **`analysis/output/oddballs.csv`** — per-theme residual (observed high-rate minus the
  rate its own genres predict) and a flag: `oddball up`, `oddball down`, or
  `not an oddball` (themes below 30 movies are `Excluded (small)`).
- **`analysis/output/validation.csv`** — per-theme mean *user* rating from the
  MovieLens small set (a check that is independent of TMDB's own scores).
- **`analysis/output/report.md`** — the class-report: headline findings, method,
  full theme table, HIGH/LOW/oddball lists, validation paragraph, caveats, and the
  exact reproduce commands. Charts `theme_rates.png` and `silhouette.png` are embedded.
- **`analysis/output/cluster_summary.json` / `global_stats.json` / `load_stats.json` /
  `vocab.json` / `floor_sensitivity.csv`** — the numbers behind the report
  (chosen k, silhouette, global high-rate, vote-floor sensitivity).

Thresholds and rules (all fixed, documented in the report):

- high rating = `vote_average >= 7.5`, vote floor = 50;
- themes = k-means on the TF-IDF keyword matrix, `k` chosen by silhouette in 8–15;
- HIGH/LOW = beyond a 95% CI around the global rate; oddball = bootstrap 95% CI on
  genre-expected residuals (fallback `|residual| >= 0.10`);
- all random steps use **seed 42** so the whole pipeline reproduces exactly.

## Findings (summary)

The generated report is the authoritative result; the short version:

- **HIGH-rated themes**: WWII / war drama (26.3% high) and romance / relationship drama
  (18.4% high).
- **LOW-rated themes**: NYC indie dramas, teen / high-school movies, blockbuster
  sequel/3D comedies, and wedding/christmas dramas.
- **Oddballs**: war and romance dramas rate higher than their genres predict; teen,
  NYC-indie, wedding, and sequel themes rate lower.
- **Validation**: MovieLens users agree — those themes top and bottom the mean rating
  table the same way (e.g. WWII-drama 3.86 vs NYC-indie 3.36).