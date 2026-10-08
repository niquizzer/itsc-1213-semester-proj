# Spec Delta

## Purpose

Discovers keyword-defined themes in a movie dataset and characterizes how each theme relates to movie ratings, surfacing high-rated themes, low-rated themes, and surprising "oddball" themes.

## ADDED Requirements

### Requirement: Build a clean working set
The system SHALL build a working set of movies by joining movie metadata with keyword data on the TMDB movie id, filtering to movies with at least a configured minimum vote count, and labeling each movie as high-rated when its TMDB vote_average is at least 7.5.

#### Scenario: Joined and filtered working set
- **WHEN** the system processes the metadata and keywords files
- **THEN** every movie in the working set has keyword data, a vote_count at or above the configured floor, and a high-rated label derived from vote_average >= 7.5

#### Scenario: Malformed rows are dropped
- **WHEN** a raw metadata row cannot be parsed cleanly (for example, shifted columns in the file's dirty tail)
- **THEN** the row is excluded from the working set and the report records how many rows were dropped

### Requirement: Curate keyword vocabulary and features
The system SHALL curate a keyword vocabulary that excludes keywords appearing in too few or too many movies and SHALL represent each movie as a TF-IDF weighted keyword vector.

#### Scenario: Frequency filters applied
- **WHEN** a keyword appears in fewer than the configured minimum or more than the configured maximum share of movies
- **THEN** the keyword is dropped from the vocabulary

#### Scenario: Movies without retained keywords are excluded
- **WHEN** a movie has no keywords left after vocabulary curation
- **THEN** the movie is excluded from clustering and is reported as excluded

### Requirement: Discover keyword themes
The system SHALL cluster movies by keyword similarity into a small number of themes, assigning each clustered movie to exactly one theme.

#### Scenario: Hard clustering
- **WHEN** movies are clustered
- **THEN** every clustered movie belongs to exactly one theme

#### Scenario: Interpretable theme count
- **WHEN** the number of themes is chosen
- **THEN** it falls in a small interpretable range and is justified by a clustering quality metric such as silhouette score

### Requirement: Label themes
The system SHALL label each theme with its most characteristic keywords and the distribution of its movies' official genres.

#### Scenario: Human-readable theme labels
- **WHEN** a theme has been produced
- **THEN** its top characteristic keywords and genre distribution are emitted so the theme can be described in plain words

### Requirement: Compare rating outcomes by theme
The system SHALL report, for each theme, its movie count, mean rating, percentage of high-rated movies, and lift versus the global high-rating rate, and SHALL classify each theme as HIGH or LOW when it departs from the global rate.

#### Scenario: Per-theme statistics
- **WHEN** the working set is grouped by theme
- **THEN** each theme reports movie count, mean rating, high-rated percentage, and lift relative to the global high-rate

#### Scenario: High and low classification
- **WHEN** a theme's high-rated percentage significantly exceeds the global rate
- **THEN** the theme is classified HIGH, and when it significantly falls below, it is classified LOW, using a stated significance rule

### Requirement: Detect oddball themes
The system SHALL compute, for each theme, the difference between its observed high-rated percentage and the percentage expected given its movies' official genres, and SHALL flag themes whose difference exceeds a configured magnitude.

#### Scenario: Residual score per theme
- **WHEN** a theme's movies have official genres
- **THEN** an expected high-rated percentage is derived from the global high-rates of those genres and the observed-minus-expected residual is reported for the theme

#### Scenario: Oddball flagged
- **WHEN** a theme's residual exceeds the configured magnitude
- **THEN** the theme is flagged as an oddball with its direction (above or below the genre expectation)

### Requirement: Validate against MovieLens ratings
The system SHALL compute a per-theme mean user rating from the MovieLens ratings, joined through the link files, and SHALL report it alongside the TMDB-based findings.

#### Scenario: Theme user-rating mean
- **WHEN** a theme's movies have MovieLens ratings available at the time of validation
- **THEN** the theme's mean user rating and number of rated movies are computed and reported next to the TMDB findings

### Requirement: Produce a reproducible report
The system SHALL produce a report containing the working-set summary, the theme table, the HIGH/LOW/oddball lists, charts, the validation paragraph, and the stated caveats.

#### Scenario: Report written on completion
- **WHEN** the analysis completes
- **THEN** a report is written containing working-set statistics, the theme table with labels and rating outcomes, the oddball list, the MovieLens validation paragraph, and caveats covering data quality and parameter sensitivity