"""Stage 01: load raw files and build the clean working set."""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.common import ensure_output_dir, save_json

#: TMDB ids are comfortably below this many movies; larger ids indicate
#: shifted columns from the file's dirty tail.
MAX_VALID_ID: int = 10_000_000

WORKING_COLUMNS: list[str] = [
    "id", "title", "genres", "keywords", "vote_average", "vote_count", "high_rating",
]


def parse_dict_list(value: object) -> list[dict[str, object]]:
    """Parse a Python-literal list of dicts; return [] for unparsable input."""
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [item for item in parsed if isinstance(item, dict)]


def parse_genre_names(value: object) -> list[str]:
    """Extract genre names from a raw genres cell."""
    return [
        str(item.get("name", "")).strip()
        for item in parse_dict_list(value)
        if str(item.get("name", "")).strip()
    ]


def parse_keyword_names(value: object) -> list[str]:
    """Extract keyword names from a raw keywords cell."""
    return [
        str(item.get("name", "")).strip()
        for item in parse_dict_list(value)
        if str(item.get("name", "")).strip()
    ]


def _valid_id(value: object) -> int | None:
    """Return a positive, plausibly-real TMDB id or None if invalid."""
    try:
        parsed = int(str(value).strip())
    except (ValueError, TypeError):
        return None
    if parsed <= 0 or parsed > MAX_VALID_ID:
        return None
    return parsed


def _float_in_unit_range(value: object, upper: float) -> float | None:
    """Return the parsed float in [0, upper] or None when invalid."""
    try:
        parsed = float(str(value).strip())
    except (ValueError, TypeError):
        return None
    if not 0.0 <= parsed <= upper:
        return None
    return parsed


def load_metadata(path: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load and clean movies_metadata.csv; return (clean frame, drop counts)."""
    ensure_output_dir(output_dir)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, engine="python", on_bad_lines="warn")
    stats: dict[str, int] = {"total_rows": len(raw)}

    raw["_id"] = raw["id"].map(_valid_id)
    stats["dropped_bad_id"] = int(raw["_id"].isna().sum())
    raw = raw[raw["_id"].notna()]

    stats["dropped_duplicate"] = int(raw.duplicated(subset="_id", keep="first").sum())
    raw = raw[~raw.duplicated(subset="_id", keep="first")]

    raw["_va"] = raw["vote_average"].map(lambda v: _float_in_unit_range(v, 10.0))
    stats["dropped_bad_va"] = int(raw["_va"].isna().sum())
    raw = raw[raw["_va"].notna()]

    raw["_vc"] = raw["vote_count"].map(lambda v: _float_in_unit_range(v, float("inf")))
    stats["dropped_bad_vc"] = int(raw["_vc"].isna().sum())
    raw = raw[raw["_vc"].notna()]

    clean = pd.DataFrame({
        "id": raw["_id"].astype(int),
        "title": raw["title"],
        "genres": raw["genres"].map(parse_genre_names),
        "release_date": raw["release_date"],
        "vote_average": raw["_va"].astype(float),
        "vote_count": raw["_vc"].astype(float),
    }).reset_index(drop=True)

    stats["kept"] = len(clean)
    return clean, stats


def load_keywords(path: Path, output_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load and parse keywords.csv; return (id->keyword-list frame, counts)."""
    ensure_output_dir(output_dir)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, engine="python", on_bad_lines="warn")
    stats: dict[str, int] = {"total_rows": len(raw)}

    raw["id"] = raw["id"].map(_valid_id)
    stats["dropped_bad_id"] = int(raw["id"].isna().sum())
    raw = raw[raw["id"].notna()]

    stats["dropped_duplicate"] = int(raw.duplicated(subset="id", keep="first").sum())
    raw = raw[~raw.duplicated(subset="id", keep="first")]

    raw["keywords"] = raw["keywords"].map(parse_keyword_names)
    stats["unparsable"] = int((raw["keywords"].map(len) == 0).sum())

    frame = pd.DataFrame({
        "id": raw["id"].astype(int),
        "keywords": raw["keywords"],
    }).reset_index(drop=True)
    stats["kept"] = len(frame)
    return frame, stats


def build_working_set(
    metadata: pd.DataFrame,
    keywords: pd.DataFrame,
    min_votes: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join metadata to keywords, apply the vote floor, label high ratings."""
    meta_ids = set(int(x) for x in metadata["id"])
    kw_ids = set(int(x) for x in keywords["id"])
    missing_keywords = sorted(meta_ids - kw_ids)
    missing_metadata = sorted(kw_ids - meta_ids)

    joined = metadata.merge(keywords, on="id", how="inner")
    joined = joined.drop_duplicates(subset="id", keep="first")
    before_floor = len(joined)
    joined = joined[joined["vote_count"] >= float(min_votes)]
    below_floor = before_floor - len(joined)

    working = pd.DataFrame({
        "id": joined["id"].astype(int),
        "title": joined["title"],
        "genres": joined["genres"],
        "keywords": joined["keywords"],
        "vote_average": joined["vote_average"].astype(float),
        "vote_count": joined["vote_count"].astype(float),
        "high_rating": joined["vote_average"] >= 7.5,
    }).reset_index(drop=True)

    stats: dict[str, Any] = {
        "metadata_count": len(meta_ids),
        "keyword_count": len(kw_ids),
        "missing_keywords": len(missing_keywords),
        "missing_keywords_ids": missing_keywords,
        "missing_metadata": len(missing_metadata),
        "missing_metadata_ids": missing_metadata,
        "min_votes": min_votes,
        "below_floor": below_floor,
        "kept": len(working),
        "high_count": int(working["high_rating"].sum()),
    }
    return working[WORKING_COLUMNS], stats


def _json_col(series: pd.Series) -> pd.Series:
    """Serialize object columns (lists) to JSON strings for CSV persistence."""
    return series.map(json.dumps)


def main() -> None:
    """Run stage 01 end-to-end over data/raw/ and write outputs."""
    data_dir = PROJECT_ROOT / "data" / "raw"
    out = ensure_output_dir()

    metadata, meta_stats = load_metadata(data_dir / "movies_metadata.csv", out)
    keywords, kw_stats = load_keywords(data_dir / "keywords.csv", out)
    working, work_stats = build_working_set(metadata, keywords, min_votes=50)

    working_df = working.copy()
    working_df["genres"] = _json_col(working_df["genres"])
    working_df["keywords"] = _json_col(working_df["keywords"])
    working_df.to_csv(out / "working_set.csv", index=False)

    metadata_df = metadata[["id", "title", "genres", "release_date", "vote_average", "vote_count"]].copy()
    metadata_df["genres"] = _json_col(metadata_df["genres"])
    metadata_df.to_csv(out / "metadata_clean.csv", index=False)
    save_json(
        {str(row.id): row.keywords for row in keywords.itertuples()},
        out / "keywords.json",
    )
    save_json({
        "metadata": meta_stats,
        "keywords": kw_stats,
        "working_set": work_stats,
    }, out / "load_stats.json")

    print(f"metadata rows read: {meta_stats['total_rows']}, kept: {meta_stats['kept']}")
    print(f"  dropped: bad_id={meta_stats['dropped_bad_id']} "
          f"duplicate={meta_stats['dropped_duplicate']} "
          f"bad_va={meta_stats['dropped_bad_va']} bad_vc={meta_stats['dropped_bad_vc']}")
    print(f"keywords rows read: {kw_stats['total_rows']}, kept: {kw_stats['kept']}")
    print(f"working set: {work_stats['kept']} movies "
          f"(high-rated: {work_stats['high_count']}, floor={work_stats['min_votes']})")


if __name__ == "__main__":
    main()