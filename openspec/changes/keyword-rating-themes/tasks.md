# Tasks

## 1. Environment and scaffolding

- [x] 1.1 Create a Python virtual environment, add `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, pytest), install it, and verify each import plus `pytest --collect-only` succeeds from the venv
- [x] 1.2 Scaffold `tests/conftest.py` with tiny hand-built toy fixtures (movies spanning clean, malformed, low-vote, high-vote, and duplicate-id cases; matching keyword and rating data) and a smoke test, and verify `pytest` collects and passes the smoke test
- [x] 1.3 Scaffold the pipeline script files (`analysis/01-load.py` through `06-report.py`) as stubs with docstrings and type hints, and verify these files exist and collect without import errors

## 2. Working set (tests first)

- [x] 2.1 Write `tests/test_working_set.py` covering: defensive parse drops malformed rows, join on TMDB id works, `vote_count < 50` movies are excluded, `vote_average >= 7.5` labels high, boundary values (49 vs 50 votes, 7.49 vs 7.5) behave exactly, and drop counts are recorded; verify the tests run and fail against the stubs before implementation
- [x] 2.2 Implement `analysis/01-load.py` so every function has a docstring and full type hints, and verify the 2.1 tests pass and the docstring/type-hint check passes on this module
- [x] 2.3 Run `01-load.py` on `data/raw/` and verify it reports row counts matching the known file sizes (metadata ~45,466 rows, keywords ~45k rows, ratings_small 100k) and a final working-set size of roughly 30k+ with the malformed-row drop count recorded

## 3. Vocabulary and features (tests first)

- [x] 3.1 Write `tests/test_vocab.py` covering: lowercase and per-movie dedupe, keywords in fewer than 3 movies and more than 40% of movies are dropped, vocabulary size cap, TF-IDF matrix shape, and no-keyword movies excluded; verify the tests fail against the stubs before implementation
- [x] 3.2 Implement `analysis/02-vocab.py` (docstrings + type hints on every function) and verify the 3.1 tests pass and the docstring/type-hint check passes
- [x] 3.3 Run `02-vocab.py` on the working set and verify the recorded vocabulary size and matrix shape match the design's expectations

## 4. Theme discovery (tests first)

- [x] 4.1 Write `tests/test_cluster.py` covering: k selection lands in the 8-15 range with a recorded silhouette score, every clustered movie gets exactly one theme, cluster sizes are logged, and clusters below minimum size are flagged; verify the tests fail against the stubs before implementation
- [x] 4.2 Implement `analysis/03-cluster.py` (docstrings + type hints) with a fixed seed and verify the 4.1 tests pass and the docstring/type-hint check passes
- [x] 4.3 Run `03-cluster.py` and verify the final k, silhouette score, cluster sizes, and any size-excluded clusters are logged

## 5. Theme labeling (tests first)

- [x] 5.1 Write `tests/test_labeling.py` covering: each theme gets top TF-IDF weighted keywords and an official-genre distribution, and themes without genre data are handled gracefully; verify the tests fail against the stubs before implementation
- [x] 5.2 Implement the labeling pass in `analysis/04-ratings.py` (docstrings + type hints) and verify the 5.1 tests pass and the docstring/type-hint check passes
- [x] 5.3 Spot-check the generated theme labels by hand and verify every theme reads as an interpretable label with its genre make-up present

## 6. Rating outcomes (tests first)

- [x] 6.1 Write `tests/test_ratings.py` covering: per-theme count/mean/high-percentage/lift are computed correctly on fixtures, the global high-rate is reproduced, and the HIGH/LOW significance rule classifies or explicitly excludes every theme; verify the tests fail against the stubs before implementation
- [x] 6.2 Implement the rating statistics and classification in `analysis/04-ratings.py` (docstrings + type hints) and verify the 6.1 tests pass and the docstring/type-hint check passes
- [x] 6.3 Run the rating pass and verify the theme statistics CSV is written and reproduces the ~9.2% global baseline

## 7. Oddball detection (tests first)

- [x] 7.1 Write `tests/test_oddballs.py` covering: genre-expected high-rate per movie and per theme, observed-minus-expected residuals, and flagging above the bootstrap 95% CI or the fallback magnitude rule with direction (up/down); verify the tests fail against the stubs before implementation
- [x] 7.2 Implement oddball scoring in `analysis/05-validate.py` (docstrings + type hints) and verify the 7.1 tests pass and the docstring/type-hint check passes
- [x] 7.3 Run the oddball pass and verify the oddball list with directions is written

## 8. MovieLens validation (tests first)

- [x] 8.1 Write `tests/test_validation.py` covering: joining working-set movies to `links_small.csv` and `ratings_small.csv`, per-theme mean user rating and rated-movie counts, and handling of movies with no user ratings; verify the tests fail against the stubs before implementation
- [x] 8.2 Implement the MovieLens validation in `analysis/05-validate.py` (docstrings + type hints) and verify the 8.1 tests pass and the docstring/type-hint check passes
- [x] 8.3 Run the validation and verify the validation table has entries for the HIGH/LOW/oddball headlines

## 9. Report, README, and reproducibility

- [x] 9.1 Generate the report charts (theme rating vs baseline, cluster silhouette) into `analysis/output/` and verify the files exist
- [x] 9.2 Write the report markdown to `analysis/output/report.md` covering working-set stats, theme table, HIGH/LOW/oddball lists, validation paragraph, and caveats (dirty-tail drop rate, rating rounding, vote-floor sensitivity), and verify all sections are present
- [x] 9.3 Write the root `README.md` covering what the project does, data attribution, setup, how to run `01-load.py` through `06-report.py`, the layout tree, and how to interpret the outputs, and verify every command it documents runs as written from a clean shell
- [x] 9.4 Re-run the pipeline end-to-end 01-06 and verify the report regenerates identically from scratch with no manual steps

## Workflow follow-up

- Archive the change after the class project review and required deliverables are satisfied.
- Verify the archived result.