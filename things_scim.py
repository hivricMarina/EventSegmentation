"""
Filters THINGS/THINGSplus object concepts by:
  - Category    (53-category system, 03_category-level/category53_wide-format.tsv)
  - Memorability (01_image-level/_images-metadata_things.tsv  → memorability_cr)
  - Recognizability (01_object-level/_images-metadata_things.tsv → recognizability)
  - Holdability  (02_object-level/_property-ratings.tsv → property_hold_mean)
  - Pleasantness (02_object-level/_property-ratings.tsv → property_pleasant_mean)

Expected layout after `osf -p jum2f clone THINGS-database`:

  THINGS-database/
  ├── 01_image-level/
  │   └── _images-metadata_things.tsv
  ├── 02_object-level/
  │   ├── _concepts-metadata_things.tsv
  │   └── _property-ratings.tsv
  └── 03_category-level/
      └── category53_wide-format.tsv

Property scales (THINGSplus):
  memorability_cr        0–1   (image hit rate, averaged to concept level)
  recognizability       0–1   (0 = not recognizable, 1 = fully recognizable)
  property_hold_mean     1–7   (7 = very easy to hold in one hand)
  property_pleasant_mean 1–7   (7 = very pleasant)

Usage
-----
  python things_filter.py                      # interactive wizard
  python things_filter.py --help
  python things_filter.py \\
      --db_root ./THINGS-database \\
      --categories "food,tool" \\
      --mem_min 0.60 --rec_min 0.60 --rec_max 1.0 \\
      --hold_min 4.0 --pleas_min 4.0 --arous_min 4.0 --arous_max 7.0 \\
      --n_per_category 10 --output filtered_objects.csv
"""

import argparse
import logging
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(format="  %(levelname)s  %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

# ── Display options ───────────────────────────────────────────────────────────

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_colwidth", 32)
pd.set_option("display.max_rows", 60)

# ── Raw column names (as they appear in the TSV files) ───────────────────────

class Col:
    MEM        = "memorability_cr"
    REC        = "recognizability"
    HOLD       = "property_hold_mean"
    HOLD_SD    = "property_hold_SD"
    PLEAS      = "property_pleasant_mean"
    PLEAS_SD   = "property_pleasant_SD"
    SIZE       = "size_mean"
    MANMADE    = "property_manmade_mean"
    MANMADE_SD = "property_manmade_SD"
    GRASP      = "property_grasp_mean"
    GRASP_SD   = "property_grasp_SD"
    NATURAL    = "property_natural_mean"
    NATURAL_SD = "property_natural_SD"
    AROUSAL    = "property_arousal_mean"
    AROUSAL_SD = "property_arousal_SD"

PROPERTY_COLS = (
    Col.HOLD, Col.HOLD_SD,
    Col.PLEAS, Col.PLEAS_SD,
    Col.SIZE,
    Col.MANMADE, Col.MANMADE_SD,
    Col.GRASP, Col.GRASP_SD,
    Col.NATURAL, Col.NATURAL_SD,
    Col.AROUSAL, Col.AROUSAL_SD,
)

IMAGE_COLS = [Col.MEM, Col.REC]

DISPLAY_COLS = [
    "Word", "categories_53", Col.SIZE,
    Col.MEM, Col.REC, Col.HOLD, Col.HOLD_SD, Col.PLEAS, Col.PLEAS_SD, Col.AROUSAL, Col.AROUSAL_SD, 
    Col.MANMADE, Col.MANMADE_SD, Col.GRASP, Col.GRASP_SD, Col.NATURAL, Col.NATURAL_SD,
]

SORTABLE_COLS = [Col.MEM, Col.REC, Col.HOLD, Col.PLEAS, Col.AROUSAL, "category"]

# All columns included in descriptive statistics (means paired with their SDs)
STAT_COLS = [
    Col.MEM,
    Col.REC,
    Col.HOLD,    Col.HOLD_SD,
    Col.PLEAS,   Col.PLEAS_SD,
    Col.AROUSAL, Col.AROUSAL_SD,
    Col.MANMADE, Col.MANMADE_SD,
    Col.GRASP,   Col.GRASP_SD,
    Col.NATURAL, Col.NATURAL_SD,
    Col.SIZE,
]

# ── Filter parameters ─────────────────────────────────────────────────────────

Range = tuple[Optional[float], Optional[float]]

@dataclass
class FilterParams:
    categories:     Optional[list[str]] = None
    mem_range:      Range               = (None, None)
    rec_range:      Range               = (None, None)
    hold_range:     Range               = (None, None)
    pleas_range:    Range               = (None, None)
    arous_range:    Range               = (None, None)
    n_per_category: Optional[int]       = None
    sort_by:        str                 = field(default=Col.MEM)
    output:         Optional[str]       = None

# ─────────────────────────────────────────────────────────────────────────────
# Database discovery
# ─────────────────────────────────────────────────────────────────────────────

_SEARCH_ROOTS = [
    Path("."),
    Path("./things_database"),
    Path("/home/hivrim8h/projects/things_database"),
    Path.home() / "Desktop" / "things-database",
]

_UNIQUE_ID = "uniqueID"
 
 
def find_db_root(hint: Optional[str] = None) -> Path:
    """Return the THINGS-database root that contains 01_image-level/."""
    candidates = ([Path(hint)] if hint else []) + _SEARCH_ROOTS
    for p in candidates:
        if (p / "01_image-level").exists():
            return p.resolve()
    raise FileNotFoundError(
        "Cannot find THINGS-database root.\n"
        "Pass --db_root /path/to/THINGS-database explicitly."
    )

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_tsv(path: Path) -> pd.DataFrame:
    """Read a TSV (falls back to CSV if only one column is parsed)."""
    df = pd.read_csv(path, sep="\t", low_memory=False)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def _merge_key(left: pd.DataFrame, right: pd.DataFrame) -> str:
    """Prefer uniqueID as join key, fall back to Word."""
    if _UNIQUE_ID in left.columns and _UNIQUE_ID in right.columns:
        return _UNIQUE_ID
    return "Word"

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_concepts(root: Path) -> pd.DataFrame:
    path = root / "02_object-level" / "_concepts-metadata_things.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

    df = _read_tsv(path)
    df.rename(columns={"word": "Word", "Concept": "Word"}, inplace=True)

    if "Word" not in df.columns:
        first_str_col = df.select_dtypes("object").columns[0]
        df.rename(columns={first_str_col: "Word"}, inplace=True)

    log.info("Concepts: %d rows | columns: %s …", len(df), list(df.columns[:8]))
    return df


def load_property_ratings(root: Path, concepts: pd.DataFrame) -> pd.DataFrame:
    path = root / "02_object-level" / "_property-ratings.tsv"
    
    df  = _read_tsv(path)
    key = _merge_key(concepts, df)

    log.info("Property TSV raw columns: %s", list(df.columns))
    # Match TSV headers to PROPERTY_COLS case-insensitively, then rename to the
    # canonical Col name so the rest of the code can rely on exact spelling.
    tsv_lower = {c.lower(): c for c in df.columns if c != key}
    rename    = {tsv_lower[p.lower()]: p for p in PROPERTY_COLS if p.lower() in tsv_lower}
    df.rename(columns=rename, inplace=True)

    present = [c for c in PROPERTY_COLS if c in df.columns]

    return concepts.merge(df[[key] + present], on=key, how="left")


def load_image_level(root: Path, concepts: pd.DataFrame) -> pd.DataFrame:
    path = root / "01_image-level" / "_images-metadata_things.tsv"
    
    df = _read_tsv(path)
    key = _merge_key(concepts, df)

    mem_mean = df.groupby(key)[Col.MEM].mean().reset_index(name='memorability_cr')
    mem_rec_mean = df.groupby(key)[Col.REC].mean().reset_index(name='recognizability')

    concepts = concepts.merge(mem_mean, on=key, how="left")
    concepts = concepts.merge(mem_rec_mean, on=key, how="left")

    return concepts

def load_categories(root: Path, concepts: pd.DataFrame) -> pd.DataFrame:
    """
    Load the 53-category membership matrix and attach two columns:
      categories_53  – list of category names the concept belongs to
      cat53_string   – semicolon-joined string for easy filtering/display
    """
    path = root / "03_category-level" / "category53_wide-format.tsv"
    
    df = _read_tsv(path)

    id_col_names = {_UNIQUE_ID.lower(), "word", "concept", "object", "index"}
    id_cols  = [c for c in df.columns if c.lower() in id_col_names]
    cat_cols = [c for c in df.columns if c not in id_cols]

    key = _merge_key(concepts, df) if id_cols else None
    if key:
        concepts = concepts.merge(df[[key] + cat_cols], on=key, how="left")
    else:
        n_rows = min(len(df), len(concepts))
        for col in cat_cols:
            concepts.loc[:n_rows - 1, col] = df[col].values[:n_rows]

    concepts["categories_53"] = concepts.apply(
        lambda row: [c for c in cat_cols if row.get(c, 0) == 1], axis=1
    )
    concepts["cat53_string"] = concepts["categories_53"].apply("; ".join)
    log.info("53-category memberships attached | %d categories.", len(cat_cols))
    return concepts


def load_all(root: Path) -> pd.DataFrame:
    """Load and join all THINGS data sources into one DataFrame."""
    concepts = load_concepts(root)
    concepts = load_property_ratings(root, concepts)
    concepts = load_image_level(root, concepts)
    concepts = load_categories(root, concepts)
    return concepts

# ─────────────────────────────────────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────────────────────────────────────

def _range_mask(series: pd.Series, lo: Optional[float], hi: Optional[float]) -> pd.Series:
    mask = pd.Series(True, index=series.index)
    if lo is not None:
        mask &= series.fillna(-np.inf) >= lo
    if hi is not None:
        mask &= series.fillna(np.inf)  <= hi
    return mask


def filter_concepts(df: pd.DataFrame, params: FilterParams) -> pd.DataFrame:
    result = df.copy()

    # ── Category ─────────────────────────────────────────────────────────────
    if params.categories:
        if "cat53_string" not in result.columns:
            log.warning("cat53_string column not found — skipping category filter.")
        else:
            cats_lower = [c.lower().strip() for c in params.categories]
            mask       = result["cat53_string"].str.lower().apply(
                lambda s: any(c in s for c in cats_lower)
            )
            before, result = len(result), result[mask]
            log.info("Category filter → %d/%d kept", len(result), before)

    # ── Numeric ranges ────────────────────────────────────────────────────────
    for col, rng, label in [
        (Col.MEM,   params.mem_range,   "Memorability"),
        (Col.REC,   params.rec_range,   "Recognizability"),
        (Col.HOLD,  params.hold_range,  "Holdability"),
        (Col.PLEAS, params.pleas_range, "Pleasantness"),
        (Col.AROUSAL, params.arous_range, "Arousal"),
    ]:
        if rng == (None, None):
            continue
        if col not in result.columns:
            log.warning("'%s' missing — skipping %s filter.", col, label)
            continue
        before, result = len(result), result[_range_mask(result[col], *rng)]
        log.info("%s (%s) %s → %d/%d kept", label, col, rng, len(result), before)

    # ── Stratified sampling ───────────────────────────────────────────────────
    if params.n_per_category and "categories_53" in result.columns:
        uid_col = _UNIQUE_ID if _UNIQUE_ID in result.columns else result.columns[0]
        sampled = (result.explode("categories_53")
                         .groupby("categories_53", group_keys=False)
                         .head(params.n_per_category))
        result  = sampled.drop_duplicates(subset=[uid_col])
        log.info("Stratified sample (≤%d/category) → %d unique concepts",
                 params.n_per_category, len(result))

    # ── Sort ──────────────────────────────────────────────────────────────────
    if params.sort_by == "category":
        # Explode so each concept appears once per category, sort alphabetically
        # on category name, then keep only the first (alphabetically earliest)
        # category row for each concept to avoid duplicates.
        uid_col = _UNIQUE_ID if _UNIQUE_ID in result.columns else result.columns[0]
        result = (result.explode("categories_53")
                        .sort_values(["categories_53", "Word"])
                        .drop_duplicates(subset=[uid_col])
                        .dropna(subset=["categories_53"])
                        .reset_index(drop=True))
    elif params.sort_by in result.columns:
        result = result.sort_values(params.sort_by, ascending=False, na_position="last")

    return result.reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def report(df: pd.DataFrame) -> None:
    sep = "═" * 72
    print(f"\n{sep}\n  ✔  {len(df)} concepts match all filters\n{sep}")

    present = [c for c in DISPLAY_COLS if c in df.columns]
    
    if len(df):
        print(df[present].to_string(index=True))

    stat_cols = [c for c in STAT_COLS if c in df.columns]
    
    if stat_cols:
        df[stat_cols].describe().round(3).to_csv("descriptive-stats.csv")

    if "categories_53" in df.columns and len(df):
        df.explode("categories_53")["categories_53"].value_counts().to_csv("concepts-per-category.csv")
        

# ─────────────────────────────────────────────────────────────────────────────
# Interactive wizard
# ─────────────────────────────────────────────────────────────────────────────

def _list_categories(df: pd.DataFrame) -> None:
    if "categories_53" not in df.columns:
        return
    all_cats = sorted({c for row in df["categories_53"] for c in row})
    print(f"\nAvailable categories ({len(all_cats)} total):")
    for i, cat in enumerate(all_cats, 1):
        count = df["categories_53"].apply(lambda x: cat in x).sum()
        print(f"  {i:>3}. {cat:<40} ({count} concepts)")


def _ask_range(name: str, scale: str) -> Range:
    print(f"\n  {name}  scale: {scale}  (leave blank = no limit)")
    lo = input("    min: ").strip()
    hi = input("    max: ").strip()
    return (float(lo) if lo else None, float(hi) if hi else None)


def interactive_wizard(df: pd.DataFrame) -> FilterParams:
    print("\n" + "═" * 72)
    print("  THINGS Database — Interactive Filter Wizard")
    print("═" * 72)

    _list_categories(df)

    cats_raw   = input("\nCategories to include (comma-separated, partial OK) [all]: ").strip()
    categories = None if cats_raw.lower() in ("", "all") else \
                 [c.strip() for c in cats_raw.split(",") if c.strip()]

    mem_range   = _ask_range(f"Memorability  [{Col.MEM}]",  "0–1")
    rec_range   = _ask_range(f"Recognizability [{Col.REC}]",  "0–1")
    hold_range  = _ask_range(f"Holdability   [{Col.HOLD}]", "1–7")
    pleas_range = _ask_range(f"Pleasantness  [{Col.PLEAS}]","1–7")
    arous_range = _ask_range(f"Arousal  [{Col.AROUSAL}]","1–7")

    n_str    = input("\nMax concepts per category (blank = unlimited): ").strip()
    sort_raw = input(f"\nSort by [{' / '.join(SORTABLE_COLS)}]: ").strip() or Col.MEM
    out      = input("\nSave results to CSV (blank to skip): ").strip()

    return FilterParams(
        categories     = categories,
        mem_range      = mem_range,
        rec_range      = rec_range,
        hold_range     = hold_range,
        pleas_range    = pleas_range,
        arous_range    = arous_range,
        n_per_category = int(n_str) if n_str.isdigit() else None,
        sort_by        = sort_raw,
        output         = out or None,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(f"""\
            Filter THINGS/THINGSplus object concepts by category and norms.

            Property columns (THINGSplus):
              {Col.MEM:<26} 0–1   (image hit rate)
              {Col.REC:<26} 0-1   (1 = very recognizable)
              {Col.HOLD:<26} 1–7   (7 = easy to hold)
              {Col.PLEAS:<26} 1–7   (7 = very pleasant)
              {Col.AROUSAL:<26} 1–7   (7 = very arousing)
        """),
    )
    p.add_argument("--db_root",        default=None)
    p.add_argument("--categories",     default=None,
                   help="Comma-separated substrings, e.g. 'food,tool,animal'")
    p.add_argument("--mem_min",        type=float)
    p.add_argument("--mem_max",        type=float)
    p.add_argument("--rec_min",        type=float)
    p.add_argument("--rec_max",        type=float)
    p.add_argument("--hold_min",       type=float)
    p.add_argument("--hold_max",       type=float)
    p.add_argument("--arous_min",      type=float)
    p.add_argument("--arous_max",      type=float)
    p.add_argument("--pleas_min",      type=float)
    p.add_argument("--pleas_max",      type=float)
    p.add_argument("--n_per_category", type=int, default=None)
    p.add_argument("--sort_by",        default=Col.MEM, choices=SORTABLE_COLS)
    p.add_argument("--output",         default=None)
    p.add_argument("--interactive",    action="store_true")
    return p


def _params_from_args(args: argparse.Namespace) -> FilterParams:
    cats = [c.strip() for c in args.categories.split(",") if c.strip()] \
           if args.categories else None
    return FilterParams(
        categories     = cats,
        mem_range      = (args.mem_min,   args.mem_max),
        rec_range      = (args.rec_min,   args.rec_max),
        hold_range     = (args.hold_min,  args.hold_max),
        arous_range    = (args.arous_min, args.arous_max),
        pleas_range    = (args.pleas_min, args.pleas_max),
        n_per_category = args.n_per_category,
        sort_by        = args.sort_by,
        output         = args.output,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _build_parser().parse_args()

    try:
        root = find_db_root(args.db_root)
    except FileNotFoundError as e:
        sys.exit(f"\nERROR: {e}\n")
    log.info("THINGS-database root: %s", root)

    concepts = load_all(root)

    params = interactive_wizard(concepts) \
             if (args.interactive or len(sys.argv) == 1) \
             else _params_from_args(args)

    log.info("Filtering …")
    result = filter_concepts(concepts, params)
    report(result)

    if params.output:
        # Save every DISPLAY_COLS column plus uniqueID.
        # categories_53 is a Python list — swap it for the CSV-safe semicolon string.
        save_cols = [_UNIQUE_ID] + [
            "cat53_string" if c == "categories_53" else c
            for c in DISPLAY_COLS
        ]
        
        missing   = [c for c in save_cols if c not in result.columns]
        if missing:
            log.warning("Columns missing from result, skipped in output: %s", missing)

        save_cols = [c for c in save_cols if c in result.columns]
        result[save_cols].to_csv(params.output, index=False)
        log.info("Saved -> %s  (%d rows x %d cols)", params.output, len(result), len(save_cols))


if __name__ == "__main__":
    main()