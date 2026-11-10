#!/usr/bin/env python3
"""
harmonize.py: whatsthepoint Data Harmonization Pipeline
-------------------------------------------------------
Data harmonization for survey and panel research.

FEATURES:
• Multi-format support (.dta, .csv, .xlsx, .sav, .sas7bdat)
• Flexible folder structure (unprocessed/ + processed/ + metaset.xlsx)
• Complete metadata synchronization and enrichment
• Automatic variable recoding and constructed variables
• Response bias indicators (acquiescence, extreme response, etc.)
• Split-ballot combination
• Missing value handling
• LaTeX codebook generation (PDF)
• Master codebook and variable mapping
• Longitudinal dataset merging
• Topic modeling with BERTopic (optional)
• Stata export with value labels
• NO automatic binary/ternary recoding

DIRECTORY STRUCTURE:
    /project_root/
        unprocessed/          ← Raw data files
        processed/            ← Harmonized datasets
        metaset.xlsx          ← Metadata workbook (3 sheets: metadata, constructed, identifiers)
"""

from __future__ import annotations
import ast, re, subprocess, unicodedata, shutil, logging, warnings, contextlib, hashlib, json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from ..utils.optional import require

# Optional dependencies
try:
    from pandas.io.stata import StataWriter117
    STATA_AVAILABLE = True
except ImportError:
    STATA_AVAILABLE = False

try:
    import pingouin as pg
    PINGOUIN = True
except ImportError:
    pg = None
    PINGOUIN = False

try:
    from bertopic import BERTopic
    from umap import UMAP
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path.cwd()
DIR_UNPROCESSED = PROJECT_ROOT / "unprocessed"
DIR_PROCESSED = PROJECT_ROOT / "processed"
# DIR_CODEBOOKS = PROJECT_ROOT / "codebooks"  # Reserved for future PDF codebook generation
# DIR_METADATA = PROJECT_ROOT / "metadata"    # Not used - metadata stored in metaset.xlsx
META_FILE = PROJECT_ROOT / "metaset.xlsx"
CACHE_FILE = PROJECT_ROOT / ".metadata_cache.json"

# Directories will be created on-demand when harmonize() is called
# Removed automatic directory creation to avoid cluttering user workspaces
# for p in (DIR_UNPROCESSED, DIR_PROCESSED):
#     p.mkdir(parents=True, exist_ok=True)

# Stata limits
_MAX_STATA_CAT = 32_000
_MAX_STATA_STR = 244

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# Unicode normalization for LaTeX/Stata compatibility
_BAD_CHARS = {
    "'": "'", "'": "'", "'": "'", "`": "'",
    """: '"', """: '"',
    "\u2013": "-", "\u2014": "-", "\u2026": "...",
    "\u00a0": " ", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "-", "\u2122": "TM",
}


def _latin1(s):
    """Clean text for LaTeX/Stata export."""
    if s is None or pd.isna(s):
        return ""
    s_str = str(s)
    for k, v in _BAD_CHARS.items():
        s_str = s_str.replace(k, v)
    return unicodedata.normalize("NFKD", s_str).encode("latin-1", "ignore").decode("latin-1")


def _safe_str_lower(x) -> str:
    """Convert to lowercase string, handling NaN/float values."""
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def _norm(x) -> str:
    """
    Normalize metadata field for comparison.
    Lists/arrays → sorted, de-duplicated comma-separated string.
    """
    if pd.isna(x) or x is None:
        return ""
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        items = [str(i).strip() for i in x if str(i).strip() and str(i).strip().lower() != "nan"]
        return ", ".join(sorted(set(items)))
    txt = str(x).strip()
    if "," in txt:
        items = [s.strip() for s in txt.split(",") if s.strip()]
        return ", ".join(sorted(set(items)))
    return txt.lower()


def _sec_key(section: str):
    """Sort sections: PROFILE → S/S1/S2 → A/A1/A2/B... → other → CONSTRUCTED."""
    if pd.isna(section) or not isinstance(section, str):
        return (4, 0, "")
    up = section.upper().strip()
    if up == "PROFILE":
        return (0, 0, "")
    if up == "CONSTRUCTED":
        return (4, 0, "")
    m = re.match(r"^(?:SECTION\s+)?([A-Z])(\d*)", up)
    if m:
        letter, digits = m.group(1), m.group(2)
        num = int(digits) if digits else 0
        if letter == "S":
            return (1, num, "")
        return (2, letter, num)
    return (3, 0, up)


# LaTeX escaping
_LATEX_ESC = str.maketrans({
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"
})
texesc = lambda x: _latin1(x).translate(_LATEX_ESC)
TRUNC = lambda s, n=50: s if pd.isna(s) or len(str(s)) <= n else str(s)[:n].rsplit(" ", 1)[0] + "…"


def _as_stata_str(s: pd.Series) -> pd.Series:
    """Convert series to safe Stata string."""
    return s.astype(str).str.slice(0, _MAX_STATA_STR).fillna("").map(_latin1)


def _run_pdflatex(tex: Path) -> bool:
    """Compile LaTeX to PDF."""
    pdf = tex.with_suffix(".pdf")
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex.name],
            cwd=tex.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    return pdf.exists()


def _clean_non_pdfs(path: Path) -> None:
    """Remove LaTeX auxiliary files."""
    for f in path.glob("*"):
        if f.suffix.lower() in {".tex", ".log", ".aux", ".toc", ".out", ".fls", ".fdb_latexmk", ".synctex.gz"}:
            with contextlib.suppress(Exception):
                f.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(filepath: Path) -> pd.DataFrame:
    """
    Load dataset from multiple formats.
    Supports: .dta, .csv, .xlsx, .xls, .sav (SPSS), .sas7bdat
    """
    suffix = filepath.suffix.lower()
    try:
        if suffix == ".dta":
            try:
                return pd.read_stata(filepath, convert_categoricals=True, order_categoricals=True)
            except ValueError as e:
                if "Value labels" in str(e) and "are not unique" in str(e):
                    logger.warning(f"{filepath.name}: Non-unique value labels. Loading without categoricals.")
                    return pd.read_stata(filepath, convert_categoricals=False)
                raise
        elif suffix == ".csv":
            return pd.read_csv(filepath)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(filepath)
        elif suffix == ".sav":
            return pd.read_spss(filepath)
        elif suffix == ".sas7bdat":
            return pd.read_sas(filepath)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    except Exception as e:
        logger.error(f"Failed to load {filepath}: {e}")
        raise


def discover_datasets(folder: Path) -> Dict[str, Path]:
    """Discover all supported datasets in folder."""
    formats = [".dta", ".csv", ".xlsx", ".xls", ".sav", ".sas7bdat"]
    datasets = {}
    for fmt in formats:
        for file in folder.glob(f"*{fmt}"):
            dataset_id = file.stem
            datasets[dataset_id] = file
    return datasets


# ══════════════════════════════════════════════════════════════════════════════
# METADATA MANAGER
# ══════════════════════════════════════════════════════════════════════════════

class MetadataManager:
    """Manages metaset.xlsx with topic modeling and caching."""

    def __init__(self, path: Path):
        self.path = path
        self.cache_path = CACHE_FILE
        self.metadata = pd.DataFrame()
        self.constructed = pd.DataFrame()
        self.identifiers = pd.DataFrame()

        if path.exists():
            self._load()
        else:
            logger.warning(f"Metadata file not found: {path}. Creating template.")
            self._create_template()

    def _load(self):
        """Load all sheets from metaset.xlsx."""
        try:
            xl = pd.ExcelFile(self.path)
            required = {"metadata", "constructed", "identifiers"}
            if not required.issubset(xl.sheet_names):
                raise ValueError(f"Missing required sheets: {required - set(xl.sheet_names)}")

            self.metadata = xl.parse("metadata", dtype=str)
            self.constructed = xl.parse("constructed", dtype=str)
            self.identifiers = xl.parse("identifiers", dtype=str)
            logger.info(f"Loaded metadata: {len(self.metadata)} rows")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            raise

    def _create_template(self):
        """Create template metaset.xlsx."""
        self.metadata = pd.DataFrame(columns=[
            "datasetid", "variable", "varname", "description",
            "question", "responses", "values", "section", "notes"
        ])
        self.constructed = pd.DataFrame(columns=[
            "datasetid", "varname", "vartype", "varnames", "calculation", "description"
        ])
        self.identifiers = pd.DataFrame(columns=[
            "datasetid", "title", "date", "n", "description"
        ])
        self.save()

    def calculate_topic_scores(self):
        """Calculate BERTopic similarity scores with caching."""
        if not BERTOPIC_AVAILABLE:
            logger.warning("BERTopic not available. Skipping topic scores.")
            self.metadata['topic_score'] = 0
            return

        current_ids = set(self.metadata['datasetid'].unique())
        cached_data = {}

        # Load cache
        if self.cache_path.exists():
            with open(self.cache_path, 'r') as f:
                cached_data = json.load(f)

        cached_ids = set(cached_data.get('dataset_ids', []))

        # Check if cache is valid
        if current_ids == cached_ids and 'topic_scores' in cached_data:
            logger.info("Loading topic scores from cache.")
            scores_df = pd.DataFrame(cached_data['topic_scores'])
            self.metadata = self.metadata.merge(scores_df, on=['datasetid', 'variable'], how='left')
            self.metadata['topic_score'] = self.metadata.get('topic_score', 0).fillna(0).astype(int)
            return

        # Calculate topic scores
        logger.info("Calculating topic scores with BERTopic...")
        try:
            df = self.metadata.copy()
            key_cols = ["variable", "section", "question", "responses", "values"]
            df['_doc'] = df[key_cols].fillna('').agg(' '.join, axis=1)

            valid_mask = (df['_doc'].str.strip() != '') & \
                        (df['question'].fillna('').str.strip() != '') & \
                        (df['responses'].fillna('').str.strip() != '')

            docs = df.loc[valid_mask, '_doc'].tolist()

            if not docs:
                self.metadata['topic_score'] = 0
                return

            umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
            topic_model = BERTopic(
                embedding_model="all-MiniLM-L6-v2",
                umap_model=umap_model,
                verbose=False,
                min_topic_size=5
            )
            topics, _ = topic_model.fit_transform(docs)

            # Map topics to 1D positions
            topic_embeddings = topic_model.topic_embeddings_
            umap_1d = UMAP(n_components=1, min_dist=0.0, random_state=42)
            positions = umap_1d.fit_transform(topic_embeddings)

            topic_info = topic_model.get_topic_info()
            topic_info['Position'] = positions
            sorted_topics = topic_info[topic_info.Topic != -1].sort_values('Position')
            topic_map = {old: new + 1 for new, old in enumerate(sorted_topics.Topic)}
            topic_map[-1] = 0

            # Assign scores
            df['topic_score'] = 0
            scores = pd.Series(topics, index=df[valid_mask].index).map(topic_map)
            df.loc[valid_mask, 'topic_score'] = scores
            self.metadata['topic_score'] = df['topic_score'].fillna(0).astype(int)

            logger.info("Topic scores calculated successfully.")
        except Exception as e:
            logger.error(f"Topic score calculation failed: {e}")
            self.metadata['topic_score'] = 0

    def save(self):
        """Save metadata with backup and topic score caching."""
        # Isolate topic scores
        scores_to_keep = None
        if 'topic_score' in self.metadata.columns:
            scores_to_keep = self.metadata[['datasetid', 'variable', 'topic_score']].copy().dropna(
                subset=['datasetid', 'variable']
            )
            self.metadata = self.metadata.drop(columns=['topic_score'])

        # Sort metadata by date and dataset
        if 'datasetid' in self.metadata.columns and not self.identifiers.empty:
            try:
                md = self.metadata.merge(
                    self.identifiers[['datasetid', 'date']], on='datasetid', how='left'
                )
                md['date'] = pd.to_datetime(md['date'], errors='coerce')

                def get_sort_keys(row):
                    did, date = row['datasetid'], row['date']
                    try:
                        group, year_str = did.rsplit('_', 1)
                        year = int(year_str)
                    except (ValueError, IndexError):
                        group, year = did, 0
                    return group, date.year if pd.notna(date) else year

                md[['sort_group', 'sort_year']] = md.apply(get_sort_keys, axis=1, result_type='expand')
                self.metadata = md.sort_values(
                    ['sort_year', 'sort_group'],
                    ascending=[False, True]
                ).drop(columns=['date', 'sort_group', 'sort_year'])
                logger.info("Metadata sorted by date.")
            except Exception as e:
                logger.warning(f"Could not sort metadata: {e}")

        # Re-attach topic scores
        if scores_to_keep is not None:
            self.metadata = self.metadata.merge(scores_to_keep, on=['datasetid', 'variable'], how='left')
            self.metadata['topic_score'] = self.metadata['topic_score'].fillna(0).astype(int)

        # Save topic scores to cache
        if 'topic_score' in self.metadata.columns and self.metadata['topic_score'].notna().any():
            cache_data = {
                'dataset_ids': sorted(list(self.metadata['datasetid'].unique())),
                'topic_scores': self.metadata[['datasetid', 'variable', 'topic_score']].dropna().to_dict('records')
            }
            with open(self.cache_path, 'w') as f:
                json.dump(cache_data, f)
            logger.info(f"Topic score cache saved to {self.cache_path}")

        # Backup existing file
        if self.path.exists():
            backup = self.path.with_suffix(f".{datetime.now():%Y%m%d_%H%M%S}.bak")
            shutil.copy2(self.path, backup)
            logger.info(f"Metadata backed up to {backup}")

        # Write Excel with tables
        try:
            require("openpyxl", "excel")
            from openpyxl.worksheet.table import Table, TableStyleInfo
            from openpyxl.utils import get_column_letter
            with pd.ExcelWriter(self.path, engine='openpyxl', mode='w') as writer:
                sheets = {
                    'metadata': self.metadata,
                    'constructed': self.constructed,
                    'identifiers': self.identifiers
                }

                for sheet_name, df in sheets.items():
                    # Flatten list columns
                    for col in df.select_dtypes(include=['object']).columns:
                        if df[col].map(lambda x: isinstance(x, (list, tuple))).any():
                            df[col] = df[col].map(
                                lambda x: ', '.join(map(str, x)) if isinstance(x, (list, tuple)) else x
                            )

                    df.to_excel(writer, sheet_name=sheet_name, index=False)

                    if not df.empty:
                        ws = writer.sheets[sheet_name]
                        table_range = f"A1:{get_column_letter(df.shape[1])}{df.shape[0] + 1}"
                        table = Table(displayName=sheet_name.capitalize(), ref=table_range)
                        style = TableStyleInfo(
                            name="TableStyleMedium2",
                            showFirstColumn=False,
                            showLastColumn=False,
                            showRowStripes=True
                        )
                        table.tableStyleInfo = style
                        ws.add_table(table)

            logger.info("Metadata workbook saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise


# ══════════════════════════════════════════════════════════════════════════════
# DATASET HARMONIZER
# ══════════════════════════════════════════════════════════════════════════════

class DatasetHarmonizer:
    """
    Harmonizes individual datasets with full feature support.

    NO automatic binary/ternary recoding - only explicit transforms defined in metadata.
    """

    def __init__(self, dataset_id: str, filepath: Path, metadata: MetadataManager):
        self.dataset_id = dataset_id
        self.filepath = filepath
        self.metadata = metadata
        self.df = None
        self.variable_labels = {}
        self.original_var_labels = {}
        self.final_to_original_map = {}
        self.missing_meta_vars = set()
        self.reliability_info = {}

        logger.info(f"→ Loading {dataset_id}")
        self._load_data()
        self._initialize_dataset()

    def _load_data(self):
        """Load dataset and extract variable labels."""
        self.df = load_dataset(self.filepath)
        self.df.columns = [c.strip().lower() for c in self.df.columns]

        # Extract variable labels from Stata files
        if self.filepath.suffix.lower() == ".dta":
            try:
                with pd.read_stata(self.filepath, iterator=True) as reader:
                    stata_labels = reader.variable_labels()
                    self.original_var_labels = {k.lower(): _latin1(v) for k, v in stata_labels.items()}
                    self.variable_labels = {k.lower(): v for k, v in stata_labels.items()}
            except Exception as e:
                logger.warning(f"{self.dataset_id}: Could not extract variable labels: {e}")

        logger.info(f"{self.dataset_id}: Loaded {len(self.df)} rows, {len(self.df.columns)} columns")

    def _initialize_dataset(self):
        """Initialize dataset with identifiers and special variables."""
        # Add identifier columns from metadata
        id_row = self.metadata.identifiers[self.metadata.identifiers['datasetid'] == self.dataset_id]

        if not id_row.empty:
            id_data = id_row.iloc[0]
            try:
                self.df['yos'] = pd.to_datetime(id_data.get('date')).year
            except:
                self.df['yos'] = np.nan

            for col_name in self.metadata.identifiers.columns:
                if col_name not in ['datasetid', 'date']:
                    value = id_data.get(col_name)
                    self.df[col_name] = value if pd.notna(value) else np.nan
        else:
            logger.warning(f"{self.dataset_id}: No identifiers found. Adding placeholder.")
            new_row = {col: "NEEDS_ATTENTION" for col in self.metadata.identifiers.columns}
            new_row['datasetid'] = self.dataset_id
            self.metadata.identifiers = pd.concat(
                [self.metadata.identifiers, pd.DataFrame([new_row])],
                ignore_index=True
            )
            self.df['yos'] = np.nan

        # Clean and prepare columns
        self._clean_columns()
        self._apply_renames()
        self._combine_split_ballots()
        self._clean_categories()

    def _clean_columns(self):
        """Remove blank/NaN columns and handle duplicates."""
        raw = list(self.df.columns)
        keep_idx = []
        new_cols = []
        counts = {}

        for i, col in enumerate(raw):
            name = "" if pd.isna(col) else str(col).strip()

            # Filter placeholders
            if name == "" or re.fullmatch(r"nan(_\d+)?", name.lower()):
                logger.warning(f"{self.dataset_id}: dropped placeholder column #{i + 1}")
                continue

            # Handle duplicates
            if name in counts:
                counts[name] += 1
                name = f"{name}_{counts[name]}"
            else:
                counts[name] = 1

            keep_idx.append(i)
            new_cols.append(name)

        self.df = self.df.iloc[:, keep_idx]
        self.df.columns = new_cols

    def _apply_renames(self):
        """Apply variable renames from metadata."""
        meta_for_dataset = self.metadata.metadata.query("datasetid == @self.dataset_id")
        renames = {}

        for _, row in meta_for_dataset.iterrows():
            old_var = _safe_str_lower(row.get("variable", ""))
            new_var = _safe_str_lower(row.get("varname", "")) or old_var

            if new_var and old_var and new_var != old_var and old_var in self.df.columns:
                renames[old_var] = new_var
                self.final_to_original_map[new_var] = old_var

        if renames:
            self.df = self.df.rename(columns=renames)
            for old, new in renames.items():
                if old in self.variable_labels:
                    self.variable_labels[new] = self.variable_labels.pop(old)
            logger.info(f"{self.dataset_id}: Applied {len(renames)} renames")

    def _combine_split_ballots(self):
        """Combine split-ballot variables (e.g., var_a, var_b → var)."""
        excluded = ('_dk', '_open', '_other')
        pattern = re.compile(r'^(.+)_([a-z])$')
        groups = {}

        for col in self.df.columns:
            if col.endswith(excluded):
                continue
            match = pattern.match(col)
            if match:
                base = match.group(1)
                groups.setdefault(base, []).append(col)

        for base, group_cols in groups.items():
            if len(group_cols) < 2 or base in self.df.columns:
                continue

            subset = self.df[group_cols]
            if all(isinstance(self.df[c].dtype, CategoricalDtype) for c in group_cols):
                # Combine categories
                all_cats = []
                for c in group_cols:
                    all_cats.extend(self.df[c].cat.categories)
                unique_cats = list(dict.fromkeys(all_cats))
                new_dtype = CategoricalDtype(categories=unique_cats, ordered=True)

                combined = subset.bfill(axis=1).iloc[:, 0]
                self.df[base] = pd.Categorical(combined, dtype=new_dtype)
                self.variable_labels[base] = f"Combined from {', '.join(group_cols)}"
                logger.info(f"{self.dataset_id}: Combined split-ballot {base}")

    def _clean_categories(self):
        """Normalize and clean categorical response options."""
        # Convert suitable object columns to categoricals
        for c in self.df.select_dtypes("object"):
            if c.endswith(('_open', 'other')):
                continue
            if 0 < self.df[c].nunique() < 100:
                self.df[c] = pd.Categorical(self.df[c], ordered=True)

        # Clean existing categoricals
        for col in self.df.select_dtypes("category").columns:
            s = self.df[col]

            def clean_text(x):
                text = str(x)
                if '?' in text:
                    text = re.sub(r'can[\?]t', "can't", text, flags=re.I)
                    text = re.sub(r'don[\?]t', "don't", text, flags=re.I)
                return _latin1(re.sub(r"\s+", " ", text).strip().lower())

            original_cats = s.cat.categories
            mapping = {cat: clean_text(cat) for cat in original_cats if pd.notna(cat)}

            if all(k == v for k, v in mapping.items()):
                continue

            mapped = s.map(mapping)
            cleaned_ordered = [mapping.get(cat) for cat in original_cats if pd.notna(cat)]
            final_cats = list(dict.fromkeys(cleaned_ordered))

            self.df[col] = pd.Categorical(mapped, categories=final_cats, ordered=True)

    def apply_recodes(self):
        """Apply variable recodes from metadata (reverse scales, missing indicators, etc.)."""
        meta_for_dataset = self.metadata.metadata.query("datasetid == @self.dataset_id")

        # Identify variables to reverse
        reverse_vars = set()
        for _, row in meta_for_dataset.iterrows():
            if _safe_str_lower(row.get("notes", "")).startswith("reverse"):
                var = _safe_str_lower(row.get("varname", "")) or _safe_str_lower(row.get("variable", ""))
                if var:
                    reverse_vars.add(var)

        # Reverse ordered categoricals
        for col in reverse_vars:
            if col in self.df.columns and isinstance(self.df[col].dtype, CategoricalDtype):
                old_cats = list(self.df[col].cat.categories)
                if len(old_cats) > 1:
                    new_cats = old_cats[::-1]
                    mapping = {old: new for old, new in zip(old_cats, new_cats)}
                    self.df[col] = self.df[col].map(mapping)
                    self.df[col] = pd.Categorical(self.df[col], categories=new_cats, ordered=True)
                    logger.info(f"{self.dataset_id}: Reversed scale for {col}")

        # Create _dk (don't know) versions for missing value tracking
        for col in self.df.select_dtypes("category"):
            if col.endswith("_dk"):
                continue

            s = self.df[col]
            if s.isna().any():
                missing_substrs = {"don't know", "dk", "refused", "not applicable", "prefer not", "n/a"}

                def is_missing(cat):
                    c = str(cat or "").strip().lower()
                    return any(ms in c for ms in missing_substrs)

                cats = list(s.cat.categories)
                for cat in cats:
                    if is_missing(cat):
                        # Create DK version
                        dk_col = f"{col}_dk"
                        self.df[dk_col] = s.copy()
                        self.variable_labels[dk_col] = self.variable_labels.get(col, "") + " (with missing categories)"

                        # Remove missing from original
                        non_missing = [c for c in cats if not is_missing(c)]
                        clean_s = s.copy()
                        clean_s = clean_s.map(lambda x: x if not is_missing(x) else np.nan)
                        self.df[col] = pd.Categorical(clean_s, categories=non_missing, ordered=True)
                        break

        logger.info(f"{self.dataset_id}: Applied recodes")

    def apply_constructs(self):
        """
        Apply constructed variables from metadata.

        Supported types:
        - index: Mean of items (with Cronbach's alpha)
        - recode: Category remapping
        - ordinal: Numeric binning
        - calculate: Formula evaluation
        - reverse_scale: Reverse Likert scales
        - standardize: Z-scores
        - sum: Sum of columns
        - rank: Identify top-ranked column
        """
        cons = self.metadata.constructed.query("datasetid == @self.dataset_id")
        if cons.empty:
            return

        def num_converter(s: pd.Series) -> pd.Series:
            """Convert categorical to numeric."""
            if isinstance(s.dtype, CategoricalDtype):
                cats = list(s.cat.categories)
                mp = {cat: i + 1 for i, cat in enumerate(cats)}
                return s.map(mp).astype(float)
            return pd.to_numeric(s, errors="coerce")

        all_new_cols = {}
        all_new_labels = {}

        # Multi-pass for dependencies
        max_passes = 5
        for pass_num in range(max_passes):
            pass_new_cols = {}
            pass_new_labels = {}
            constructed_count = 0

            for _, r in cons.iterrows():
                vt = _safe_str_lower(r.get("vartype", ""))
                vn = _safe_str_lower(r.get("varname", ""))
                desc = str(r.get("description", "")) if pd.notna(r.get("description")) else ""
                cols = [c.strip().lower() for c in str(r.get("varnames", "")).split(",") if c.strip()]
                calc = str(r.get("calculation", "")).strip() if pd.notna(r.get("calculation")) else ""

                if vn in self.df.columns or vn in all_new_cols:
                    continue

                new_col_data = None
                special_desc = None

                try:
                    if vt == "index":
                        present = [c for c in cols if c in self.df.columns]
                        if not present:
                            continue

                        dnum = pd.DataFrame({c: num_converter(self.df[c]) for c in present})
                        new_col_data = dnum.mean(axis=1, skipna=True)

                        if PINGOUIN and len(present) > 1 and not dnum.dropna().empty:
                            alpha, ci = pg.cronbach_alpha(data=dnum.dropna())
                            self.reliability_info[vn] = {
                                'alpha': alpha, 'ci': ci, 'n_items': len(present), 'items': present
                            }
                            special_desc = f"{desc} (α = {alpha:.2f})" if desc else f"Mean of {present} (α = {alpha:.2f})"

                    elif vt == "recode":
                        if len(cols) == 1 and cols[0] in self.df.columns and calc:
                            recode_map = ast.literal_eval(calc)
                            if isinstance(recode_map, dict):
                                mapped = self.df[cols[0]].map(recode_map)
                                categories = list(dict.fromkeys(v for v in recode_map.values() if v is not None))
                                new_col_data = pd.Categorical(mapped, categories=categories, ordered=True)

                    elif vt == "ordinal":
                        if len(cols) == 1 and cols[0] in self.df.columns and calc:
                            source = pd.to_numeric(self.df[cols[0]], errors='coerce')
                            bin_map = ast.literal_eval(calc)
                            sorted_bins = sorted(bin_map.items(), key=lambda x: int(x[0]))
                            bins = [int(b[0]) for b in sorted_bins] + [np.inf]
                            labels = [b[1] for b in sorted_bins]
                            new_col_data = pd.cut(source, bins=bins, labels=labels, right=False, ordered=True)

                    elif vt == "calculate":
                        present = [c for c in cols if c in self.df.columns]
                        if len(present) == len(cols) and calc:
                            temp_df = self.df[present]
                            new_col_data = temp_df.eval(calc)

                    elif vt in {"reverse_scale", "inverseind"}:
                        if len(cols) == 1 and cols[0] in self.df.columns and calc:
                            calc_dict = dict(item.split(":") for item in calc.replace(" ", "").split(","))
                            scale_min = float(calc_dict['min'])
                            scale_max = float(calc_dict['max'])
                            new_col_data = (scale_min + scale_max) - pd.to_numeric(self.df[cols[0]], errors='coerce')

                    elif vt in {"standardize", "stdindex"}:
                        if len(cols) == 1 and cols[0] in self.df.columns:
                            s = pd.to_numeric(self.df[cols[0]], errors="coerce")
                            new_col_data = (s - s.mean()) / s.std(ddof=0)

                    elif vt == "sum":
                        present = [c for c in cols if c in self.df.columns]
                        if present:
                            dnum = self.df[present].apply(num_converter)
                            new_col_data = dnum.sum(axis=1, skipna=True)

                    elif vt == "rank":
                        present = [c for c in cols if c in self.df.columns]
                        if present:
                            dnum = self.df[present].apply(num_converter)
                            new_col_data = dnum.apply(
                                lambda r: np.nan if r.isna().all() else r.rank(method="min").idxmin(), axis=1
                            )

                    if new_col_data is not None:
                        pass_new_cols[vn] = new_col_data
                        pass_new_labels[vn] = special_desc or desc
                        constructed_count += 1

                except Exception as e:
                    logger.warning(f"{self.dataset_id} construct '{vn}' ({vt}) failed: {e}")

            if pass_new_cols:
                self.df = pd.concat([self.df, pd.DataFrame(pass_new_cols, index=self.df.index)], axis=1)
                all_new_cols.update(pass_new_cols)
                all_new_labels.update(pass_new_labels)
            else:
                break

        self.variable_labels.update(all_new_labels)
        logger.info(f"{self.dataset_id}: Applied {len(all_new_cols)} constructs")

    def add_response_bias_indicators(self):
        """Add response bias variables (acquiescence, extreme response, etc.)."""
        # Find Likert-type variables (3-7 categories, ordered, not _dk)
        likert = [
            c for c in self.df.columns
            if isinstance(self.df[c].dtype, CategoricalDtype)
            and not c.endswith('_dk')
            and 3 <= len(self.df[c].cat.categories) <= 7
        ]

        if not likert:
            return

        # Convert to numeric
        def to_numeric(s):
            if isinstance(s.dtype, CategoricalDtype):
                cats = list(s.cat.categories)
                mp = {cat: i + 1 for i, cat in enumerate(cats)}
                return s.map(mp).astype(float)
            return pd.to_numeric(s, errors="coerce")

        lnum = pd.DataFrame({v: to_numeric(self.df[v]) for v in likert})

        def acquiescence(r):
            v = r.dropna()
            if len(v) < 5:
                return np.nan
            midpoint = (v.min() + v.max()) / 2
            return (v > midpoint).mean()

        def extreme(r):
            v = r.dropna()
            if len(v) < 5:
                return np.nan
            return ((v == v.min()) | (v == v.max())).mean()

        def streak(r):
            v = r.dropna()
            if len(v) < 5:
                return np.nan
            max_streak = cur_streak = 1
            for i in range(1, len(v)):
                if v.iloc[i] == v.iloc[i-1]:
                    cur_streak += 1
                    max_streak = max(max_streak, cur_streak)
                else:
                    cur_streak = 1
            return max_streak

        def middle(r):
            v = r.dropna()
            uni = sorted(v.unique())
            if len(v) < 5 or len(uni) < 3:
                return np.nan
            if len(uni) % 2:
                return (v == uni[len(uni) // 2]).mean()
            mid1, mid2 = uni[len(uni) // 2 - 1], uni[len(uni) // 2]
            return ((v == mid1) | (v == mid2)).mean()

        self.df['acquiescence_score'] = lnum.apply(acquiescence, axis=1)
        self.df['extreme_response_score'] = lnum.apply(extreme, axis=1)
        self.df['max_response_streak'] = lnum.apply(streak, axis=1)
        self.df['response_variance'] = lnum.var(axis=1, skipna=True)
        self.df['middle_response_score'] = lnum.apply(middle, axis=1)

        logger.info(f"{self.dataset_id}: Added response bias indicators")

    def sync_metadata(self):
        """Synchronize metadata with current dataset variables."""
        md = self.metadata.metadata
        md_others = md[md['datasetid'] != self.dataset_id]
        md_current = md[md['datasetid'] == self.dataset_id].copy()

        # Update existing metadata rows
        valid_rows = []
        for _, row in md_current.iterrows():
            original_var = _safe_str_lower(row['variable'])
            final_name = _safe_str_lower(row.get('varname', original_var)) or original_var

            # Check if variable exists and has data
            if final_name in self.df.columns and self.df[final_name].notna().sum() > 0:
                # Enrich metadata
                if isinstance(self.df[final_name].dtype, CategoricalDtype):
                    cats = list(self.df[final_name].cat.categories)
                    row['responses'] = _norm(cats)
                    row['values'] = _norm(list(range(1, len(cats) + 1)))
                valid_rows.append(row)

        # Add new variables not in metadata
        known_vars = set(_safe_str_lower(r.get('varname') or r.get('variable')) for _, r in md_current.iterrows())
        excluded = {'acquiescence_score', 'extreme_response_score', 'max_response_streak', 'response_variance', 'middle_response_score', 'yos'}
        excluded_suffixes = ('_dk',)

        for var in self.df.columns:
            if var not in known_vars and var not in excluded and not var.endswith(excluded_suffixes):
                if self.df[var].notna().sum() > 0:
                    new_row = {
                        'datasetid': self.dataset_id,
                        'variable': var,
                        'varname': var,
                        'description': self.variable_labels.get(var, ""),
                        'question': "",
                        'responses': "",
                        'values': "",
                        'section': "CONSTRUCTED",
                        'notes': ""
                    }
                    valid_rows.append(new_row)
                    self.missing_meta_vars.add(var)

        # Combine and update
        md_updated = pd.DataFrame(valid_rows) if valid_rows else md_current
        self.metadata.metadata = pd.concat([md_others, md_updated], ignore_index=True)

        logger.info(f"{self.dataset_id}: Metadata synchronized")

    def export_stata(self, outdir: Path):
        """Export to Stata format with value labels."""
        if not STATA_AVAILABLE:
            logger.warning("Stata export unavailable. Skipping.")
            return

        outdir.mkdir(parents=True, exist_ok=True)
        target = outdir / f"{self.dataset_id}.dta"

        try:
            df_export = self.df.copy()

            # Safe column names (≤31 chars, unique)
            safe_map = {}
            taken = set()
            for c in df_export.columns:
                base = _latin1(c)[:30]
                while base in taken:
                    base = base[:28] + hashlib.md5((c + base).encode()).hexdigest()[:2]
                taken.add(base)
                safe_map[c] = base

            inv_safe = {v: k for k, v in safe_map.items()}
            df_export = df_export.rename(columns=safe_map)

            # Value labels
            value_labels = {}
            for col in df_export.columns:
                orig_col = inv_safe.get(col, col)
                if orig_col not in self.df.columns:
                    continue

                ser = self.df[orig_col]

                if isinstance(ser.dtype, CategoricalDtype) and len(ser.cat.categories) <= _MAX_STATA_CAT:
                    cats = [cat for cat in ser.cat.categories if pd.notna(cat)]
                    if cats:
                        code = {cat: i + 1 for i, cat in enumerate(cats)}
                        value_labels[col] = {i: _latin1(str(cat))[:80] for cat, i in code.items()}
                        df_export[col] = ser.map(code).astype("float64")

            # Variable labels
            var_labels = {}
            for col in df_export.columns:
                final_name = inv_safe.get(col, col)
                original_name = self.final_to_original_map.get(final_name, final_name)
                label = self.variable_labels.get(final_name, original_name)
                var_labels[col] = _latin1(label)[-80:]  # Truncate from end

            # Write
            StataWriter117(
                target, df_export,
                variable_labels=var_labels,
                value_labels=value_labels,
                write_index=False
            ).write_file()

            logger.info(f"{self.dataset_id}: Exported to {target}")
        except Exception as e:
            logger.error(f"{self.dataset_id}: Stata export failed: {e}")

    def export_csv(self, outdir: Path):
        """Export to CSV format."""
        outdir.mkdir(parents=True, exist_ok=True)
        target = outdir / f"{self.dataset_id}.csv"
        self.df.to_csv(target, index=False)
        logger.info(f"{self.dataset_id}: Exported to {target}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def harmonize_all():
    """Main harmonization pipeline."""
    logger.info("═══ Harmonization Pipeline Start ═══")

    # Create directories on-demand (only when actually running harmonization)
    for p in (DIR_UNPROCESSED, DIR_PROCESSED):
        p.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta = MetadataManager(META_FILE)

    # Discover datasets
    datasets = discover_datasets(DIR_UNPROCESSED)
    if not datasets:
        logger.warning(f"No datasets found in {DIR_UNPROCESSED}")
        return

    logger.info(f"Found {len(datasets)} dataset(s) to process")

    # Process each dataset
    harmonized = {}
    errors = {}

    for dataset_id, filepath in datasets.items():
        try:
            harmonizer = DatasetHarmonizer(dataset_id, filepath, meta)

            # Apply transformations
            harmonizer.apply_recodes()
            harmonizer.apply_constructs()
            harmonizer.add_response_bias_indicators()
            harmonizer.sync_metadata()

            # Export
            harmonizer.export_stata(DIR_PROCESSED)
            harmonizer.export_csv(DIR_PROCESSED)

            harmonized[dataset_id] = harmonizer
            logger.info(f"{dataset_id}: Processing complete")

        except Exception as e:
            logger.error(f"{dataset_id}: Processing failed: {e}")
            errors[dataset_id] = str(e)

    # Calculate topic scores (optional, expensive)
    if BERTOPIC_AVAILABLE:
        meta.calculate_topic_scores()

    # Save updated metadata
    meta.save()

    # Report errors
    if errors:
        logger.warning(f"Errors occurred for {len(errors)} dataset(s): {list(errors.keys())}")
    else:
        logger.info("No errors")

    logger.info("═══ Harmonization Pipeline Complete ═══")
    return harmonized


if __name__ == "__main__":
    harmonize_all()
