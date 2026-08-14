"""
ds004504_common.py
==================

Shared implementation module for the reproducible comparison of EEG preprocessing
pipelines on OpenNeuro ds004504 (the "AHEPA" dataset).

This module is imported by all three notebooks:

    01_author_pipeline_ds004504.ipynb      -> Pipeline A (reproduction of authors' method)
    02_alternative_pipeline_ds004504.ipynb -> Pipeline B (alternative preprocessing)
    03_pipeline_comparison_ds004504.ipynb  -> final statistical / computational comparison

WHY A SHARED MODULE?
--------------------
Requirement #8 of the project brief ("FAIR COMPARISON RULE") demands that everything
downstream of preprocessing be *identical* between Pipeline A and Pipeline B. The most
reliable way to guarantee that is to have literally one implementation of epoching,
feature extraction, cross-validation, metrics and statistics, called by both notebooks.
If the downstream code lived twice inside two notebooks, the pipelines could silently
drift apart.

SCIENTIFIC INTEGRITY
--------------------
This module computes results; it never invents them. Every function either returns a
measured value or raises / records an error. No default or placeholder metric values
are provided anywhere in this file.

Author: (project author)
License: MIT
"""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Third-party scientific stack. These are imported lazily where they are heavy,
# but MNE / sklearn / scipy are needed almost everywhere.
import mne
import scipy.signal as sps
import scipy.stats as sstats

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "Config",
    "MODULE_VERSION",
    "DATASET_ACCESSION",
    "download_metadata",
    "ensure_dataset",
    "get_environment_info",
    "setup_logging",
    "download_subjects",
    "build_inventory",
    "load_raw_subject",
    "load_derivative_subject",
    "pipeline_a",
    "pipeline_b",
    "PIPELINES",
    "make_epochs",
    "extract_features",
    "signal_quality_metrics",
    "compare_signals",
    "process_subject",
    "process_many_subjects",
    "assemble_feature_table",
    "run_cross_validation",
    "paired_bootstrap_subject_level",
    "wilcoxon_fold_level",
    "corrected_resampled_ttest",
    "mcnemar_subject_level",
    "ResourceMonitor",
]

MODULE_VERSION = "1.1.0"

# OpenNeuro accession. Kept as a constant so the download functions and the
# provenance records can never drift apart.
DATASET_ACCESSION = "ds004504"

logger = logging.getLogger("ds004504")


# =====================================================================================
# 1. CONFIGURATION
# =====================================================================================

# The 19 scalp channels documented in the dataset README, in the order listed there.
EXPECTED_CHANNELS: Tuple[str, ...] = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T3", "C3", "Cz", "C4", "T4",
    "T5", "P3", "Pz", "P4", "T6",
    "O1", "O2",
)

# Old (10-20) -> modern (10-10) names. MNE's standard_1020 montage uses T7/T8/P7/P8.
LEGACY_CHANNEL_RENAME: Dict[str, str] = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Canonical frequency bands. Upper edge stops at 45 Hz because the authors' band-pass
# is 0.5-45 Hz; going higher would compare a band that Pipeline A has filtered away.
FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 25.0),
    "gamma_low": (25.0, 45.0),
}

GROUP_LABELS: Dict[str, str] = {"A": "AD", "F": "FTD", "C": "CN"}


@dataclass
class Config:
    """Central experiment configuration.

    Every notebook exposes an instance of this at the top so that a reader can see all
    tunable parameters in one place, and so that the exact configuration can be
    serialised to JSON next to the results (requirement #19).
    """

    # ---- paths -------------------------------------------------------------------
    dataset_path: str = "/content/ds004504"
    output_path: str = "/content/drive/MyDrive/ds004504_experiment"

    # ---- subject selection -------------------------------------------------------
    # None  -> all subjects found in the dataset
    # int   -> first N subjects (after sorting by ID), used for quick/dev modes
    max_subjects: Optional[int] = 5
    subject_whitelist: Optional[List[str]] = None

    # ---- reproducibility ---------------------------------------------------------
    random_seed: int = 42
    n_jobs: int = 1  # Colab free tier typically exposes 2 vCPUs; 1 keeps RAM low.

    # ---- preprocessing (shared between A and B; these are CONTROLLED variables) ---
    l_freq: float = 0.5
    h_freq: float = 45.0
    butter_order: int = 4          # 4th-order Butterworth, applied zero-phase (filtfilt)
    resample_to: Optional[float] = 100.0  # Hz. None keeps native 500 Hz.
    reference: str = "average"     # see NOTE_ON_REFERENCE below

    # ---- Pipeline A specific -----------------------------------------------------
    asr_cutoff: float = 17.0       # authors: "0.5 s window standard deviation of 17"
    asr_win_len: float = 0.5       # authors: 0.5 second window
    ica_method: str = "infomax"
    ica_extended: bool = True
    ica_n_components: Optional[int] = None  # None -> as many as good channels
    iclabel_reject_labels: Tuple[str, ...] = ("eye blink", "muscle artifact")
    iclabel_threshold: float = 0.50
    ica_fit_l_freq: float = 1.0    # ICLabel was trained on 1-100 Hz data

    # ---- Pipeline B specific -----------------------------------------------------
    bad_channel_corr_threshold: float = 0.40   # PREP-style low-correlation criterion
    bad_channel_dev_threshold: float = 5.0     # robust z of channel amplitude
    max_bad_channel_fraction: float = 0.25     # safety valve -> flag subject
    epoch_reject_z: float = 4.0                # robust z on epoch peak-to-peak

    # ---- epoching / features -----------------------------------------------------
    epoch_length_s: float = 4.0
    epoch_overlap_s: float = 0.0   # 0 = non-overlapping (avoids inflating epoch count)
    psd_n_fft_s: float = 4.0       # Welch window in seconds
    drop_first_s: float = 10.0     # discard recording onset (settling / instructions)

    # ---- machine learning --------------------------------------------------------
    primary_classifier: str = "logreg"
    secondary_classifier: str = "rf"
    cv_n_splits: int = 10
    cv_n_repeats: int = 5
    run_loso: bool = True

    # ---- statistics --------------------------------------------------------------
    n_bootstrap: int = 10000
    alpha: float = 0.05

    # ---- runtime -----------------------------------------------------------------
    cache_enabled: bool = True
    overwrite_cache: bool = False
    verbose_mne: str = "ERROR"

    def mode(self) -> str:
        """Human-readable description of the current run size."""
        if self.max_subjects is None:
            return "FULL"
        if self.max_subjects <= 5:
            return "QUICK_TEST"
        if self.max_subjects <= 20:
            return "DEVELOPMENT"
        return "PARTIAL"

    # -- derived paths -------------------------------------------------------------
    @property
    def out(self) -> Path:
        return Path(self.output_path)

    @property
    def cache_dir(self) -> Path:
        return self.out / "cache"

    @property
    def results_dir(self) -> Path:
        return self.out / "results"

    @property
    def figures_dir(self) -> Path:
        return self.out / "figures"

    @property
    def logs_dir(self) -> Path:
        return self.out / "logs"

    def make_dirs(self) -> None:
        for d in (self.out, self.cache_dir, self.results_dir, self.figures_dir, self.logs_dir):
            d.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str = "config.json") -> Path:
        self.make_dirs()
        path = self.results_dir / filename
        payload = asdict(self)
        payload["_module_version"] = MODULE_VERSION
        payload["_mode"] = self.mode()
        payload["_saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path


NOTE_ON_REFERENCE = """
IMPORTANT DOCUMENTED UNCERTAINTY (do not silently resolve this):

The dataset README states two things that cannot both be reproduced from the shared
raw files:

  (a) "The recording montages were anterior-posterior bipolar and referential montage
       using Cz as the common reference. The referential montage was included in this
       dataset."
  (b) "a Butterworth band-pass filter 0.5-45 Hz was applied and the signals were
       re-referenced to A1-A2."

The shared raw .set files contain only the 19 scalp electrodes. A1 and A2 (mastoids)
are NOT present as data channels -- the README describes them as reference electrodes
used for impedance checking. Therefore a literal A1-A2 (linked-mastoid) re-reference
CANNOT be recomputed from the distributed raw data.

We therefore treat `Config.reference` as an explicit, documented substitution rather
than a faithful reproduction, and we:
  1. record this uncertainty in results/pipeline_a_uncertainties.json,
  2. use the SAME reference setting in Pipeline A and Pipeline B so that referencing is
     a controlled variable and not a confound of the A-vs-B comparison,
  3. default to 'average' because ICLabel (used in Pipeline A) was designed for
     average-referenced data.
"""


# =====================================================================================
# 2. ENVIRONMENT, LOGGING, RESOURCE MONITORING
# =====================================================================================


def get_environment_info() -> Dict[str, Any]:
    """Capture the execution environment for the reproducibility record.

    Returns a plain dict (JSON-serialisable). Anything that cannot be determined is
    reported as the string "unknown" rather than guessed.
    """
    info: Dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count_logical": os.cpu_count() or "unknown",
    }

    # Physical RAM
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 1024**3, 2)
        info["ram_available_gb"] = round(vm.available / 1024**3, 2)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False) or "unknown"
    except Exception:
        info["ram_total_gb"] = "unknown"
        info["ram_available_gb"] = "unknown"
        info["cpu_count_physical"] = "unknown"

    # CPU model (Linux/Colab)
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        info.setdefault("cpu_model", "unknown")

    # GPU (recorded for completeness; this experiment is CPU-only by design)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        info["gpu"] = out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else "none"
    except Exception:
        info["gpu"] = "none"

    # Package versions
    packages = [
        "numpy", "scipy", "pandas", "mne", "sklearn", "matplotlib",
        "asrpy", "mne_icalabel", "torch", "onnxruntime", "joblib", "psutil",
    ]
    versions: Dict[str, str] = {}
    for pkg in packages:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not installed"
    info["packages"] = versions
    info["ds004504_common_version"] = MODULE_VERSION

    # In Colab? (affects how we interpret timings)
    info["in_colab"] = "google.colab" in sys.modules

    return info


def setup_logging(cfg: Config, name: str = "run") -> Path:
    """Attach a file handler so that every warning/error is persisted, not just printed."""
    cfg.make_dirs()
    log_path = cfg.logs_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logger.setLevel(logging.INFO)
    logger.handlers = []  # avoid duplicate handlers on notebook re-run

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)-8s | %(message)s"))
    logger.addHandler(sh)

    mne.set_log_level(cfg.verbose_mne)
    return log_path


class ResourceMonitor:
    """Context manager measuring wall-clock time, CPU time and peak RSS of a block.

    Peak RSS is sampled via `resource.getrusage`, which reports the peak for the whole
    process since start. We therefore report both the absolute peak and the *increase*
    during the block; the increase is the more meaningful per-step number, but it can be
    0 if an earlier step already pushed the high-water mark up. Both are recorded so the
    limitation is visible rather than hidden.
    """

    def __init__(self, label: str = "block"):
        self.label = label
        self.result: Dict[str, Any] = {}

    @staticmethod
    def _peak_rss_mb() -> float:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux reports KiB, macOS reports bytes.
            return peak / 1024.0 if sys.platform != "darwin" else peak / 1024.0**2
        except Exception:
            return float("nan")

    @staticmethod
    def _current_rss_mb() -> float:
        try:
            import psutil  # type: ignore

            return psutil.Process().memory_info().rss / 1024**2
        except Exception:
            return float("nan")

    def __enter__(self) -> "ResourceMonitor":
        gc.collect()
        self._t0 = time.perf_counter()
        self._c0 = time.process_time()
        self._peak0 = self._peak_rss_mb()
        self._rss0 = self._current_rss_mb()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.result = {
            "label": self.label,
            "wall_time_s": round(time.perf_counter() - self._t0, 4),
            "cpu_time_s": round(time.process_time() - self._c0, 4),
            "rss_start_mb": round(self._rss0, 2),
            "rss_end_mb": round(self._current_rss_mb(), 2),
            "peak_rss_mb": round(self._peak_rss_mb(), 2),
            "peak_rss_increase_mb": round(self._peak_rss_mb() - self._peak0, 2),
            "failed": exc_type is not None,
        }
        return False  # never swallow exceptions


# =====================================================================================
# 3. DATASET ACQUISITION AND INVENTORY
# =====================================================================================


REQUIRED_METADATA = ["participants.tsv", "dataset_description.json"]


def _openneuro_module():
    """Import openneuro-py, with an actionable message if it is missing."""
    try:
        import openneuro  # type: ignore
        return openneuro
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "openneuro-py is required to download the dataset at runtime.\n"
            "Install with:  pip install openneuro-py"
        ) from exc


def download_metadata(cfg: Config, retries: int = 3) -> Dict[str, Any]:
    """Fetch only the small top-level metadata files (a few kB).

    This must happen before subject selection, because `participants.tsv` is the
    authoritative source of the group labels that stratified selection depends on.
    Downloading it separately means we never need the multi-GB dataset just to find
    out who the subjects are.
    """
    openneuro = _openneuro_module()
    target = Path(cfg.dataset_path)
    target.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {"target": str(target), "errors": [], "attempts": 0}
    include = ["participants.tsv", "participants.json",
               "dataset_description.json", "README"]

    for attempt in range(1, retries + 1):
        report["attempts"] = attempt
        try:
            openneuro.download(dataset=DATASET_ACCESSION, target_dir=target,
                               include=include)
            break
        except Exception as exc:
            report["errors"].append({"attempt": attempt, "error": repr(exc)})
            logger.warning("Metadata download attempt %d/%d failed: %r",
                           attempt, retries, exc)
            if attempt < retries:
                time.sleep(2 ** attempt)

    # Verify rather than trust: a "successful" call that produced no file is a
    # failure we want to surface here, not three cells later as a confusing
    # FileNotFoundError from participants.tsv.
    missing = [f for f in REQUIRED_METADATA if not (target / f).exists()]
    report["missing"] = missing
    report["ok"] = not missing
    if missing:
        logger.error("Metadata download incomplete; missing: %s", missing)
    return report


def download_subjects(
    cfg: Config,
    subjects: Optional[Sequence[str]] = None,
    include_derivatives: bool = True,
    retries: int = 3,
    allow_full_dataset: bool = False,
) -> Dict[str, Any]:
    """Download only the subjects this run needs, using `openneuro-py`.

    The full dataset is several GB. Downloading per-subject keeps Colab disk usage
    low and makes acquisition restartable: openneuro-py skips files already present,
    so re-running after a session timeout is cheap.

    `subjects=None` would fetch the ENTIRE dataset, which is almost never what anyone
    wants on a Colab disk. That now requires `allow_full_dataset=True` so it cannot
    happen by accident.

    Returns a report dict. Per-subject failures are recorded, not raised, so that one
    bad subject does not abort a long batch -- but the report is verified against the
    filesystem so silent failures cannot pass as success.
    """
    # Validate arguments BEFORE touching imports, so a misuse is reported as a misuse
    # rather than being masked by a missing-package error.
    if not subjects:
        if not allow_full_dataset:
            raise ValueError(
                "download_subjects() called with no subject list. That would download "
                "the entire multi-GB dataset.\n"
                "Pass an explicit `subjects=[...]` list, or set "
                "allow_full_dataset=True if you really mean it."
            )
        subjects = []

    openneuro = _openneuro_module()
    target = Path(cfg.dataset_path)
    target.mkdir(parents=True, exist_ok=True)

    include: List[str] = []
    for sub in subjects:
        include.append(f"{sub}/")
        if include_derivatives:
            include.append(f"derivatives/{sub}/")

    report: Dict[str, Any] = {
        "target": str(target),
        "requested": list(subjects),
        "errors": [],
        "attempts": 0,
    }

    with ResourceMonitor("download") as rm:
        for attempt in range(1, retries + 1):
            report["attempts"] = attempt
            try:
                openneuro.download(dataset=DATASET_ACCESSION, target_dir=target,
                                   include=include if include else None)
                break
            except Exception as exc:
                report["errors"].append({"stage": "download", "attempt": attempt,
                                         "error": repr(exc)})
                logger.warning("Download attempt %d/%d failed: %r",
                               attempt, retries, exc)
                if attempt < retries:
                    time.sleep(2 ** attempt)
    report["timing"] = rm.result

    # Verify what actually landed on disk, per subject.
    present, absent = [], []
    for sub in subjects:
        if list((target / sub).rglob("*.set")) or list((target / sub).rglob("*.edf")):
            present.append(sub)
        else:
            absent.append(sub)
    report["downloaded"] = present
    report["missing"] = absent
    report["ok"] = not absent
    if absent:
        logger.error("No EEG file found after download for: %s", absent)
    return report


def ensure_dataset(cfg: Config, subjects: Sequence[str],
                   include_derivatives: bool = True) -> Dict[str, Any]:
    """Download any of `subjects` not already on disk, then verify all are present.

    Idempotent: safe to re-run. This is the function to call at the top of a batch.
    """
    target = Path(cfg.dataset_path)
    needed = [s for s in subjects
              if not (list((target / s).rglob("*.set")) or list((target / s).rglob("*.edf")))]
    if not needed:
        logger.info("All %d subjects already present on disk.", len(subjects))
        return {"target": str(target), "requested": list(subjects),
                "downloaded": [], "missing": [], "ok": True,
                "already_present": list(subjects), "errors": []}

    logger.info("%d/%d subjects missing; downloading.", len(needed), len(subjects))
    report = download_subjects(cfg, needed, include_derivatives=include_derivatives)
    report["already_present"] = [s for s in subjects if s not in needed]
    return report


def _subject_ids_from_disk(bids_root: Path) -> List[str]:
    return sorted(p.name for p in bids_root.glob("sub-*") if p.is_dir())


def _read_participants(bids_root: Path) -> pd.DataFrame:
    """Read participants.tsv. This is the authoritative source for group labels."""
    ptsv = bids_root / "participants.tsv"
    if not ptsv.exists():
        raise FileNotFoundError(
            f"participants.tsv not found at {ptsv}. Download the dataset metadata first."
        )
    df = pd.read_csv(ptsv, sep="\t")
    if "participant_id" not in df.columns:
        raise ValueError(f"participants.tsv has unexpected columns: {list(df.columns)}")
    return df


def select_subjects(cfg: Config, bids_root: Optional[Path] = None) -> List[str]:
    """Resolve the subject list for this run (whitelist > max_subjects > all)."""
    root = Path(bids_root or cfg.dataset_path)
    if cfg.subject_whitelist:
        return list(cfg.subject_whitelist)

    try:
        df = _read_participants(root)
        all_subs = sorted(df["participant_id"].astype(str).tolist())
    except FileNotFoundError:
        all_subs = _subject_ids_from_disk(root)

    if cfg.max_subjects is None:
        return all_subs
    return all_subs[: cfg.max_subjects]


def select_subjects_balanced(cfg: Config, bids_root: Optional[Path] = None) -> List[str]:
    """Like `select_subjects` but keeps group proportions when truncating.

    Taking "the first N subject IDs" is dangerous here: subject IDs are ordered by
    group in this dataset, so `max_subjects=5` would return five AD subjects and no
    controls, making classification impossible. For quick/dev modes we therefore
    stratify by diagnosis.
    """
    root = Path(bids_root or cfg.dataset_path)
    if cfg.subject_whitelist:
        return list(cfg.subject_whitelist)

    df = _read_participants(root)
    group_col = _find_group_column(df)
    df = df.sort_values("participant_id")

    if cfg.max_subjects is None:
        return df["participant_id"].astype(str).tolist()

    rng = np.random.default_rng(cfg.random_seed)
    groups = df[group_col].astype(str).unique()
    n_per = max(1, cfg.max_subjects // max(1, len(groups)))

    picked: List[str] = []
    for g in sorted(groups):
        sub_ids = df.loc[df[group_col].astype(str) == g, "participant_id"].astype(str).tolist()
        take = min(n_per, len(sub_ids))
        idx = rng.choice(len(sub_ids), size=take, replace=False)
        picked.extend(sorted(sub_ids[i] for i in idx))

    # Top up deterministically if integer division left us short.
    remaining = [s for s in df["participant_id"].astype(str) if s not in picked]
    while len(picked) < cfg.max_subjects and remaining:
        picked.append(remaining.pop(0))

    return sorted(picked)


def _find_group_column(df: pd.DataFrame) -> str:
    """Locate the diagnosis column without assuming its exact name."""
    for candidate in ("Group", "group", "diagnosis", "Diagnosis", "condition"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find a group/diagnosis column in participants.tsv. "
        f"Available columns: {list(df.columns)}"
    )


def build_inventory(cfg: Config, subjects: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Generate the dataset inventory required by brief section #22.

    One row per subject with channel count, sampling rate, duration, file size and any
    integrity flags. Never modifies the raw dataset.
    """
    root = Path(cfg.dataset_path)
    participants = _read_participants(root)
    group_col = _find_group_column(participants)
    pmap = dict(zip(participants["participant_id"].astype(str), participants[group_col].astype(str)))

    subs = list(subjects) if subjects is not None else sorted(pmap.keys())
    rows: List[Dict[str, Any]] = []

    for sub in subs:
        row: Dict[str, Any] = {
            "participant_id": sub,
            "group": pmap.get(sub, "MISSING_IN_PARTICIPANTS_TSV"),
        }
        # carry through any extra metadata columns (age, gender, MMSE, ...)
        meta = participants.loc[participants["participant_id"].astype(str) == sub]
        for col in participants.columns:
            if col not in ("participant_id", group_col):
                row[f"meta_{col}"] = meta[col].iloc[0] if len(meta) else np.nan

        raw_path = root / sub / "eeg" / f"{sub}_task-eyesclosed_eeg.set"
        der_path = root / "derivatives" / sub / "eeg" / f"{sub}_task-eyesclosed_eeg.set"

        row["raw_exists"] = raw_path.exists()
        row["derivative_exists"] = der_path.exists()
        row["raw_path"] = str(raw_path)
        row["derivative_path"] = str(der_path)
        row["raw_size_mb"] = round(raw_path.stat().st_size / 1024**2, 3) if raw_path.exists() else np.nan
        row["derivative_size_mb"] = (
            round(der_path.stat().st_size / 1024**2, 3) if der_path.exists() else np.nan
        )

        if raw_path.exists():
            try:
                raw = mne.io.read_raw_eeglab(raw_path, preload=False, verbose="ERROR")
                row["n_channels"] = len(raw.ch_names)
                row["sfreq_hz"] = float(raw.info["sfreq"])
                row["duration_s"] = round(raw.n_times / raw.info["sfreq"], 2)
                row["channel_names"] = "|".join(raw.ch_names)
                missing = set(EXPECTED_CHANNELS) - set(raw.ch_names)
                extra = set(raw.ch_names) - set(EXPECTED_CHANNELS)
                row["unexpected_channels"] = "|".join(sorted(missing | extra)) or ""
                row["load_error"] = ""
                del raw
            except Exception as exc:
                row.update(
                    n_channels=np.nan, sfreq_hz=np.nan, duration_s=np.nan,
                    channel_names="", unexpected_channels="", load_error=repr(exc),
                )
                logger.error("Inventory: failed to read %s -> %r", sub, exc)
        else:
            row.update(
                n_channels=np.nan, sfreq_hz=np.nan, duration_s=np.nan,
                channel_names="", unexpected_channels="", load_error="raw file missing",
            )

        if der_path.exists():
            try:
                der = mne.io.read_raw_eeglab(der_path, preload=False, verbose="ERROR")
                row["derivative_duration_s"] = round(der.n_times / der.info["sfreq"], 2)
                row["derivative_sfreq_hz"] = float(der.info["sfreq"])
                row["derivative_n_channels"] = len(der.ch_names)
                del der
            except Exception as exc:
                row.update(derivative_duration_s=np.nan, derivative_sfreq_hz=np.nan,
                           derivative_n_channels=np.nan)
                logger.error("Inventory: failed to read derivative %s -> %r", sub, exc)
        else:
            row.update(derivative_duration_s=np.nan, derivative_sfreq_hz=np.nan,
                       derivative_n_channels=np.nan)

        rows.append(row)

    inv = pd.DataFrame(rows)
    return inv


def check_inventory_integrity(inv: pd.DataFrame) -> Dict[str, Any]:
    """Flag the integrity problems listed in brief #22. Returns a report, never raises."""
    report: Dict[str, Any] = {}
    report["n_subjects"] = int(len(inv))
    report["n_missing_raw"] = int((~inv["raw_exists"]).sum())
    report["n_missing_derivative"] = int((~inv["derivative_exists"]).sum())
    report["duplicate_subject_ids"] = (
        inv["participant_id"][inv["participant_id"].duplicated()].tolist()
    )

    sfreqs = inv["sfreq_hz"].dropna().unique().tolist()
    report["distinct_sampling_rates"] = sfreqs
    report["inconsistent_sampling_rate"] = len(sfreqs) > 1

    nch = inv["n_channels"].dropna().unique().tolist()
    report["distinct_channel_counts"] = nch
    report["inconsistent_channel_count"] = len(nch) > 1

    report["subjects_with_unexpected_channels"] = inv.loc[
        inv["unexpected_channels"].astype(str) != "", "participant_id"
    ].tolist()
    report["subjects_with_load_errors"] = inv.loc[
        inv["load_error"].astype(str) != "", "participant_id"
    ].tolist()
    report["group_counts"] = inv["group"].value_counts().to_dict()

    if "duration_s" in inv and inv["duration_s"].notna().any():
        report["duration_s_min"] = float(inv["duration_s"].min())
        report["duration_s_max"] = float(inv["duration_s"].max())
        report["duration_s_mean"] = float(inv["duration_s"].mean())
    return report


# =====================================================================================
# 4. LOADING AND MONTAGE
# =====================================================================================


def _standardise_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Rename legacy electrode labels and attach the standard 10-20 montage.

    A montage is mandatory: ICLabel needs topographies, and Pipeline B needs electrode
    positions for spherical-spline interpolation.
    """
    rename = {k: v for k, v in LEGACY_CHANNEL_RENAME.items() if k in raw.ch_names}
    if rename:
        raw.rename_channels(rename)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="warn")
    return raw


def load_raw_subject(cfg: Config, sub: str, preload: bool = True) -> mne.io.BaseRaw:
    """Load one subject's RAW (unprocessed) recording."""
    path = Path(cfg.dataset_path) / sub / "eeg" / f"{sub}_task-eyesclosed_eeg.set"
    if not path.exists():
        raise FileNotFoundError(f"Raw file not found for {sub}: {path}")
    raw = mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
    return _standardise_raw(raw)


def load_derivative_subject(cfg: Config, sub: str, preload: bool = True) -> mne.io.BaseRaw:
    """Load the authors' own preprocessed (derivative) recording, for validation only."""
    path = Path(cfg.dataset_path) / "derivatives" / sub / "eeg" / f"{sub}_task-eyesclosed_eeg.set"
    if not path.exists():
        raise FileNotFoundError(f"Derivative file not found for {sub}: {path}")
    raw = mne.io.read_raw_eeglab(path, preload=preload, verbose="ERROR")
    return _standardise_raw(raw)


# =====================================================================================
# 5. SHARED PREPROCESSING PRIMITIVES (identical in both pipelines -> controlled)
# =====================================================================================


def apply_butterworth_bandpass(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    """Zero-phase Butterworth band-pass, matching the authors' stated filter family.

    MNE's default filter is FIR; the authors specify a Butterworth (IIR). We therefore
    request an IIR design explicitly. `phase='zero'` makes MNE apply it forwards and
    backwards (filtfilt), which doubles the effective order but removes phase
    distortion -- the standard choice for offline EEG analysis.
    """
    iir_params = dict(order=cfg.butter_order, ftype="butter", output="sos")
    raw.filter(
        l_freq=cfg.l_freq,
        h_freq=cfg.h_freq,
        method="iir",
        iir_params=iir_params,
        phase="zero",
        verbose="ERROR",
    )
    return raw


def apply_reference(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    """Apply the referencing scheme. See NOTE_ON_REFERENCE for the A1-A2 caveat."""
    if cfg.reference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif cfg.reference in ("none", "keep", None):
        pass
    elif cfg.reference == "Cz":
        if "Cz" not in raw.ch_names:
            raise ValueError("Cz not present; cannot apply Cz reference.")
        raw.set_eeg_reference(["Cz"], projection=False, verbose="ERROR")
    else:
        # Explicit channel list, e.g. ['A1','A2'] if a future dataset version ships them
        chans = [c.strip() for c in cfg.reference.split(",")]
        missing = [c for c in chans if c not in raw.ch_names]
        if missing:
            raise ValueError(
                f"Reference channels {missing} not present in data. "
                f"Available: {raw.ch_names}. See NOTE_ON_REFERENCE."
            )
        raw.set_eeg_reference(chans, projection=False, verbose="ERROR")
    return raw


def trim_and_resample(raw: mne.io.BaseRaw, cfg: Config) -> mne.io.BaseRaw:
    """Drop the recording onset and (optionally) downsample.

    Downsampling to 100 Hz is safe here because the signal is already low-passed at
    45 Hz (Nyquist 50 Hz), and it cuts memory and ICA cost by 5x -- the single biggest
    lever for making this experiment feasible in Colab. Applied IDENTICALLY to both
    pipelines so it cannot confound the comparison.
    """
    if cfg.drop_first_s and raw.times[-1] > cfg.drop_first_s + 60:
        raw.crop(tmin=cfg.drop_first_s)
    if cfg.resample_to and raw.info["sfreq"] > cfg.resample_to:
        raw.resample(cfg.resample_to, npad="auto", verbose="ERROR")
    return raw


# =====================================================================================
# 6. PIPELINE A -- reproduction of the authors' published method
# =====================================================================================
#
# Documented source (dataset README, OpenNeuro ds004504 / GitHub OpenNeuroDatasets):
#
#   "First, a Butterworth band-pass filter 0.5-45 Hz was applied and the signals were
#    re-referenced to A1-A2. Then, the Artifact Subspace Reconstruction routine (ASR)
#    ... was applied to the signals, removing bad data periods which exceeded the max
#    acceptable 0.5 second window standard deviation of 17 ... Next, the Independent
#    Component Analysis (ICA) method (RunICA algorithm) was performed, transforming the
#    19 EEG signals to 19 ICA components. ICA components that were classified as 'eye
#    artifacts' or 'jaw artifacts' by the automatic classification routine 'ICLabel' in
#    the EEGLAB platform were automatically rejected."
#
# Parameters NOT specified by the source, recorded as uncertainties:
#   - Butterworth filter order
#   - whether the filter was zero-phase or causal
#   - ICLabel probability threshold for rejection
#   - RunICA random seed / initialisation
#   - whether ASR removed or reconstructed bad windows
#   - the ICLabel class corresponding to "jaw artifacts" (ICLabel has no such class;
#     the closest is "muscle artifact")
# =====================================================================================

PIPELINE_A_UNCERTAINTIES: List[Dict[str, str]] = [
    {
        "parameter": "reference (A1-A2)",
        "issue": "A1/A2 are not present as data channels in the shared raw files, so a "
                 "linked-mastoid reference cannot be recomputed.",
        "resolution": "Substituted Config.reference (default 'average'); applied "
                      "identically in Pipeline B so referencing is controlled.",
    },
    {
        "parameter": "Butterworth filter order",
        "issue": "Not stated in the README or data descriptor.",
        "resolution": "Set to Config.butter_order (default 4), applied zero-phase. "
                      "Recorded, not silently invented.",
    },
    {
        "parameter": "ICLabel class 'jaw artifacts'",
        "issue": "ICLabel's seven classes are brain, muscle artifact, eye blink, heart "
                 "beat, line noise, channel noise, other. There is no 'jaw' class.",
        "resolution": "Mapped 'eye artifacts'->'eye blink' and 'jaw artifacts'->"
                      "'muscle artifact' (the closest available class).",
    },
    {
        "parameter": "ICLabel rejection threshold",
        "issue": "The source says components 'classified as' those types were rejected "
                 "but gives no probability threshold.",
        "resolution": "Config.iclabel_threshold (default 0.50, i.e. argmax-equivalent). "
                      "Sensitivity to this value is not assumed -- it can be varied.",
    },
    {
        "parameter": "RunICA seed / convergence settings",
        "issue": "EEGLAB runica defaults and random initialisation are not reported.",
        "resolution": "MNE extended-Infomax with Config.random_seed. Bit-identical "
                      "reproduction is therefore impossible by construction.",
    },
    {
        "parameter": "ASR mode (removal vs reconstruction)",
        "issue": "README says ASR removed 'bad data periods'; standard clean_rawdata ASR "
                 "reconstructs rather than deletes them.",
        "resolution": "ASRpy reconstruction is used (the standard behaviour). Duration "
                      "differences vs the authors' derivative are reported, not hidden.",
    },
    {
        "parameter": "ICA input filtering",
        "issue": "ICLabel was trained on 1-100 Hz average-referenced data; the authors "
                 "ran it on 0.5-45 Hz data.",
        "resolution": "We fit ICA on a 1 Hz high-passed copy (standard MNE practice) but "
                      "cannot supply 100 Hz content, because Pipeline A's own low-pass "
                      "is 45 Hz. This is an inherent limitation of the original design "
                      "and is reported as such.",
    },
]


def pipeline_a(raw: mne.io.BaseRaw, cfg: Config) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Pipeline A: our reproduction of the dataset authors' preprocessing.

    Steps (in the published order):
        1. Butterworth band-pass 0.5-45 Hz
        2. Re-reference (see NOTE_ON_REFERENCE)
        3. Artifact Subspace Reconstruction, 0.5 s window, cutoff SD = 17
        4. Extended-Infomax ICA (RunICA equivalent), n_components = n_channels
        5. ICLabel classification; reject eye-blink and muscle ("jaw") components
        6. Back-projection of the retained components

    Returns
    -------
    raw_clean : mne.io.BaseRaw
    meta : dict  -- per-step timings and what was actually removed
    """
    meta: Dict[str, Any] = {"pipeline": "A", "steps": {}, "warnings": []}
    raw = raw.copy()

    # -- step 1: band-pass ---------------------------------------------------------
    with ResourceMonitor("A1_filter") as rm:
        apply_butterworth_bandpass(raw, cfg)
    meta["steps"]["filter"] = rm.result

    # -- step 2: reference ---------------------------------------------------------
    with ResourceMonitor("A2_reference") as rm:
        apply_reference(raw, cfg)
    meta["steps"]["reference"] = rm.result
    meta["reference_used"] = cfg.reference

    # -- (shared) trim + resample --------------------------------------------------
    with ResourceMonitor("A3_trim_resample") as rm:
        trim_and_resample(raw, cfg)
    meta["steps"]["trim_resample"] = rm.result

    # -- step 3: ASR ---------------------------------------------------------------
    with ResourceMonitor("A4_asr") as rm:
        try:
            import asrpy  # type: ignore

            asr = asrpy.ASR(sfreq=raw.info["sfreq"], cutoff=cfg.asr_cutoff,
                            win_len=cfg.asr_win_len)
            asr.fit(raw)
            raw = asr.transform(raw)
            meta["asr_applied"] = True
        except ImportError:
            meta["asr_applied"] = False
            meta["warnings"].append("asrpy not installed -- ASR step SKIPPED. "
                                    "Pipeline A is incomplete; do not report as a "
                                    "faithful reproduction.")
            logger.error("asrpy missing: Pipeline A ASR step skipped for this subject.")
        except Exception as exc:
            meta["asr_applied"] = False
            meta["warnings"].append(f"ASR failed: {exc!r}")
            logger.error("ASR failed: %r", exc)
    meta["steps"]["asr"] = rm.result

    # -- steps 4-5: ICA + ICLabel --------------------------------------------------
    with ResourceMonitor("A5_ica_iclabel") as rm:
        try:
            from mne_icalabel import label_components  # type: ignore

            # ICA is fitted on a 1 Hz high-passed copy: high-pass below ~1 Hz leaves
            # slow drifts that dominate the decomposition. The unmixing matrix is then
            # applied to the 0.5 Hz data. This is standard MNE practice.
            raw_for_ica = raw.copy().filter(
                l_freq=cfg.ica_fit_l_freq, h_freq=None, method="iir",
                iir_params=dict(order=cfg.butter_order, ftype="butter", output="sos"),
                phase="zero", verbose="ERROR",
            )

            n_comp = cfg.ica_n_components or len(
                [c for c in raw.ch_names if c not in raw.info["bads"]]
            )
            ica = mne.preprocessing.ICA(
                n_components=n_comp,
                method=cfg.ica_method,
                fit_params=dict(extended=cfg.ica_extended) if cfg.ica_method == "infomax" else None,
                random_state=cfg.random_seed,
                max_iter="auto",
                verbose="ERROR",
            )
            ica.fit(raw_for_ica)

            labels = label_components(raw_for_ica, ica, method="iclabel")
            probs = np.asarray(labels["y_pred_proba"]).ravel()
            names = list(labels["labels"])

            exclude = [
                i for i, (lab, p) in enumerate(zip(names, probs))
                if lab in cfg.iclabel_reject_labels and p >= cfg.iclabel_threshold
            ]
            ica.exclude = exclude
            ica.apply(raw, verbose="ERROR")

            meta["n_ica_components"] = int(n_comp)
            meta["ica_labels"] = names
            meta["ica_label_probs"] = [float(p) for p in probs]
            meta["n_components_rejected"] = len(exclude)
            meta["rejected_component_indices"] = exclude
            del raw_for_ica, ica
        except ImportError:
            meta["warnings"].append("mne-icalabel not installed -- ICA/ICLabel SKIPPED. "
                                    "Pipeline A is incomplete.")
            meta["n_components_rejected"] = None
            logger.error("mne-icalabel missing: Pipeline A ICA step skipped.")
        except Exception as exc:
            meta["warnings"].append(f"ICA/ICLabel failed: {exc!r}")
            meta["n_components_rejected"] = None
            logger.error("ICA/ICLabel failed: %r", exc)
    meta["steps"]["ica_iclabel"] = rm.result

    meta["total_wall_time_s"] = round(
        sum(s["wall_time_s"] for s in meta["steps"].values()), 4
    )
    meta["peak_rss_mb"] = max(s["peak_rss_mb"] for s in meta["steps"].values())
    meta["output_duration_s"] = round(raw.n_times / raw.info["sfreq"], 2)
    gc.collect()
    return raw, meta


# =====================================================================================
# 7. PIPELINE B -- alternative preprocessing
# =====================================================================================
#
# SELECTION RATIONALE (see notebook 02 for the full literature argument):
#
# Pipeline A's cost and its non-determinism both come from one stage: ASR + Infomax ICA
# + ICLabel. Delorme (2023, Sci Rep 13:2372) found across three public collections that,
# apart from high-pass filtering and bad-channel interpolation, automated corrections --
# including automated ICA rejection of eye and muscle components -- did not reliably
# improve data quality. de Cheveigne (2023) rebuts the generality of that metric, so the
# question is genuinely open, and it has never been tested on ds004504 for a *diagnostic
# classification* endpoint.
#
# Pipeline B therefore keeps filtering, referencing, resampling and every downstream
# step IDENTICAL to Pipeline A, and replaces ONLY the artifact-handling stage with a
# deterministic, ICA-free alternative:
#
#     bad-channel detection -> spherical-spline interpolation -> robust epoch rejection
#
# This isolates the artifact-removal strategy as the single independent variable, and
# yields a pipeline that is (i) far cheaper, (ii) fully deterministic (no random seed,
# no iterative convergence), which directly addresses objectives 3 and 4 of the brief.
# =====================================================================================


def detect_bad_channels(raw: mne.io.BaseRaw, cfg: Config) -> Tuple[List[str], Dict[str, Any]]:
    """PREP-inspired deterministic bad-channel detection.

    Two criteria, both computed on the filtered data:
      1. Low correlation: a channel whose maximum absolute correlation with any other
         channel is below `bad_channel_corr_threshold` is not sharing the volume-conducted
         signal that all scalp electrodes should share -> likely disconnected/bridged.
      2. Deviation: a channel whose robust z-scored amplitude (median absolute deviation
         based) exceeds `bad_channel_dev_threshold` is flat or noise-dominated.

    Uses median/MAD throughout so the criteria are not themselves driven by outliers.
    """
    data = raw.get_data()
    n_ch = data.shape[0]
    info: Dict[str, Any] = {}

    # criterion 1 -- maximum off-diagonal correlation
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(data)
    np.fill_diagonal(corr, np.nan)
    max_corr = np.nanmax(np.abs(corr), axis=1)
    max_corr = np.nan_to_num(max_corr, nan=0.0)
    low_corr_idx = np.where(max_corr < cfg.bad_channel_corr_threshold)[0]

    # criterion 2 -- robust z of channel standard deviation
    ch_sd = np.std(data, axis=1)
    med = np.median(ch_sd)
    mad = np.median(np.abs(ch_sd - med))
    scale = 1.4826 * mad if mad > 0 else np.nan
    robust_z = (ch_sd - med) / scale if np.isfinite(scale) and scale > 0 else np.zeros(n_ch)
    dev_idx = np.where(np.abs(robust_z) > cfg.bad_channel_dev_threshold)[0]

    bad_idx = sorted(set(low_corr_idx.tolist()) | set(dev_idx.tolist()))
    bads = [raw.ch_names[i] for i in bad_idx]

    info["max_abs_correlation"] = {raw.ch_names[i]: float(max_corr[i]) for i in range(n_ch)}
    info["robust_z_amplitude"] = {raw.ch_names[i]: float(robust_z[i]) for i in range(n_ch)}
    info["bad_by_correlation"] = [raw.ch_names[i] for i in low_corr_idx]
    info["bad_by_deviation"] = [raw.ch_names[i] for i in dev_idx]

    # Safety valve: if "too many" channels look bad, the criteria are probably wrong for
    # this recording. Flag it rather than interpolating most of the montage.
    if len(bads) > cfg.max_bad_channel_fraction * n_ch:
        info["excessive_bad_channels"] = True
        info["bad_channels_before_capping"] = list(bads)
        # keep only the most extreme ones, up to the cap
        order = np.argsort(-np.abs(robust_z))
        cap = int(cfg.max_bad_channel_fraction * n_ch)
        bads = [raw.ch_names[i] for i in order[:cap]]
    else:
        info["excessive_bad_channels"] = False

    info["bad_channels"] = list(bads)
    info["n_bad_channels"] = len(bads)
    return bads, info


def reject_bad_epochs_robust(epochs: mne.Epochs, cfg: Config) -> Tuple[mne.Epochs, Dict[str, Any]]:
    """Drop epochs whose peak-to-peak amplitude is a robust-z outlier.

    A fixed microvolt threshold does not transfer across subjects with different scalp
    impedances and different disease-related amplitude levels. Deriving the threshold
    per subject from the median and MAD of the epoch peak-to-peak distribution keeps the
    criterion adaptive but still fully deterministic.
    """
    data = epochs.get_data(copy=True)                 # (n_epochs, n_ch, n_times)
    ptp = data.max(axis=2) - data.min(axis=2)         # (n_epochs, n_ch)
    epoch_ptp = ptp.max(axis=1)                       # worst channel per epoch

    med = np.median(epoch_ptp)
    mad = np.median(np.abs(epoch_ptp - med))
    scale = 1.4826 * mad if mad > 0 else np.nan

    if not np.isfinite(scale) or scale <= 0:
        info = {"threshold_uv": None, "n_dropped": 0,
                "note": "MAD was zero or undefined; no epochs dropped."}
        return epochs, info

    threshold = med + cfg.epoch_reject_z * scale
    keep = np.where(epoch_ptp <= threshold)[0]

    # Never drop everything: if the criterion would remove >50% of epochs, something is
    # wrong with the recording; keep the best half and record the anomaly.
    info: Dict[str, Any] = {
        "threshold_v": float(threshold),
        "median_ptp_v": float(med),
        "n_epochs_before": int(len(epoch_ptp)),
    }
    if len(keep) < 0.5 * len(epoch_ptp):
        keep = np.argsort(epoch_ptp)[: max(1, len(epoch_ptp) // 2)]
        info["fallback_kept_best_half"] = True
    else:
        info["fallback_kept_best_half"] = False

    info["n_dropped"] = int(len(epoch_ptp) - len(keep))
    info["n_epochs_after"] = int(len(keep))
    return epochs[np.sort(keep)], info


def pipeline_b(raw: mne.io.BaseRaw, cfg: Config) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Pipeline B: deterministic, ICA-free alternative preprocessing.

    Steps:
        1. Butterworth band-pass 0.5-45 Hz   (IDENTICAL to Pipeline A)
        2. Re-reference                       (IDENTICAL to Pipeline A)
        3. Trim + resample                    (IDENTICAL to Pipeline A)
        4. Deterministic bad-channel detection
        5. Spherical-spline interpolation of bad channels
        6. Re-reference after interpolation (only if using average reference, so the
           average is not biased by the channels we just replaced)

    Note: robust epoch rejection is applied at the epoching stage (see `make_epochs`),
    because it operates on epochs rather than continuous data.
    """
    meta: Dict[str, Any] = {"pipeline": "B", "steps": {}, "warnings": []}
    raw = raw.copy()

    with ResourceMonitor("B1_filter") as rm:
        apply_butterworth_bandpass(raw, cfg)
    meta["steps"]["filter"] = rm.result

    with ResourceMonitor("B2_reference") as rm:
        apply_reference(raw, cfg)
    meta["steps"]["reference"] = rm.result
    meta["reference_used"] = cfg.reference

    with ResourceMonitor("B3_trim_resample") as rm:
        trim_and_resample(raw, cfg)
    meta["steps"]["trim_resample"] = rm.result

    with ResourceMonitor("B4_bad_channels") as rm:
        try:
            bads, bad_info = detect_bad_channels(raw, cfg)
            raw.info["bads"] = bads
            meta["bad_channel_info"] = bad_info
        except Exception as exc:
            meta["warnings"].append(f"Bad-channel detection failed: {exc!r}")
            meta["bad_channel_info"] = {"bad_channels": [], "n_bad_channels": 0}
            logger.error("Bad-channel detection failed: %r", exc)
    meta["steps"]["bad_channels"] = rm.result

    with ResourceMonitor("B5_interpolate") as rm:
        try:
            if raw.info["bads"]:
                raw.interpolate_bads(reset_bads=True, mode="accurate", verbose="ERROR")
                meta["interpolated"] = True
                # Re-apply average reference so it is not biased by the old bad channels
                if cfg.reference == "average":
                    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
            else:
                meta["interpolated"] = False
        except Exception as exc:
            meta["warnings"].append(f"Interpolation failed: {exc!r}")
            meta["interpolated"] = False
            logger.error("Interpolation failed: %r", exc)
    meta["steps"]["interpolate"] = rm.result

    meta["total_wall_time_s"] = round(
        sum(s["wall_time_s"] for s in meta["steps"].values()), 4
    )
    meta["peak_rss_mb"] = max(s["peak_rss_mb"] for s in meta["steps"].values())
    meta["output_duration_s"] = round(raw.n_times / raw.info["sfreq"], 2)
    gc.collect()
    return raw, meta


PIPELINES: Dict[str, Callable[[mne.io.BaseRaw, Config], Tuple[mne.io.BaseRaw, Dict[str, Any]]]] = {
    "A": pipeline_a,
    "B": pipeline_b,
}


# =====================================================================================
# 8. EPOCHING AND FEATURE EXTRACTION (identical for both pipelines)
# =====================================================================================


def make_epochs(raw: mne.io.BaseRaw, cfg: Config, pipeline: str) -> Tuple[mne.Epochs, Dict[str, Any]]:
    """Segment continuous data into fixed-length epochs.

    Non-overlapping by default. Overlapping epochs would multiply the apparent sample
    size while adding almost no independent information, and would make epoch-level
    statistics badly over-optimistic.

    Robust epoch rejection is applied here for BOTH pipelines. That is deliberate: it is
    a downstream step, so under the fair-comparison rule it must be identical. Pipeline
    B's distinctive contribution is the *channel-level* handling, not this.
    """
    info: Dict[str, Any] = {}
    step = cfg.epoch_length_s - cfg.epoch_overlap_s
    epochs = mne.make_fixed_length_epochs(
        raw, duration=cfg.epoch_length_s, overlap=cfg.epoch_overlap_s,
        preload=True, verbose="ERROR",
    )
    info["n_epochs_raw"] = len(epochs)
    info["epoch_length_s"] = cfg.epoch_length_s
    info["epoch_step_s"] = step

    epochs, rej = reject_bad_epochs_robust(epochs, cfg)
    info["epoch_rejection"] = rej
    info["n_epochs_final"] = len(epochs)
    return epochs, info


# NumPy 2.0 renamed `trapz` to `trapezoid`. Colab's NumPy version varies, so bind once
# here rather than risking an AttributeError at run time.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # type: ignore[attr-defined]


def _band_powers_from_psd(psd: np.ndarray, freqs: np.ndarray) -> Dict[str, np.ndarray]:
    """Integrate PSD over each band (trapezoid rule). psd shape: (..., n_freqs)."""
    out: Dict[str, np.ndarray] = {}
    for name, (lo, hi) in FREQ_BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            out[name] = np.full(psd.shape[:-1], np.nan)
        else:
            out[name] = _trapezoid(psd[..., mask], freqs[mask], axis=-1)
    return out


def _spectral_entropy(psd: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy of the power spectrum. psd shape: (..., n_freqs)."""
    total = psd.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(total > 0, psd / total, 0.0)
        logp = np.where(p > 0, np.log(p), 0.0)
    ent = -(p * logp).sum(axis=-1)
    return ent / np.log(psd.shape[-1])


def _spectral_edge_frequency(psd: np.ndarray, freqs: np.ndarray, pct: float = 0.95) -> np.ndarray:
    """Frequency below which `pct` of total power lies."""
    cum = np.cumsum(psd, axis=-1)
    total = cum[..., -1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.where(total > 0, cum / total, 0.0)
    idx = np.argmax(norm >= pct, axis=-1)
    return freqs[idx]


def _hjorth(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hjorth mobility and complexity. data shape: (n_epochs, n_ch, n_times)."""
    d1 = np.diff(data, axis=-1)
    d2 = np.diff(d1, axis=-1)
    v0 = np.var(data, axis=-1)
    v1 = np.var(d1, axis=-1)
    v2 = np.var(d2, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mobility = np.sqrt(np.where(v0 > 0, v1 / v0, np.nan))
        mob1 = np.sqrt(np.where(v1 > 0, v2 / v1, np.nan))
        complexity = np.where(mobility > 0, mob1 / mobility, np.nan)
    return mobility, complexity


def extract_features(epochs: mne.Epochs, cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    """Compute the per-epoch, per-channel feature set.

    Feature set (justified in the report's Feature Extraction section):
      - relative band power in delta / theta / alpha / beta / low-gamma  (5 x n_ch)
      - spectral entropy                                                (1 x n_ch)
      - 95% spectral edge frequency                                     (1 x n_ch)
      - Hjorth mobility and complexity                                  (2 x n_ch)

    RELATIVE (not absolute) band power is used because absolute amplitude is strongly
    affected by preprocessing itself -- ASR and ICA remove variance, so absolute power
    would partly measure "how much did the pipeline subtract" rather than a property of
    the brain. Relative power normalises that out and is the representation the AHEPA
    benchmark identifies as the consistent baseline across studies.
    """
    sfreq = float(epochs.info["sfreq"])
    n_fft = int(cfg.psd_n_fft_s * sfreq)
    n_fft = min(n_fft, epochs.get_data(copy=False).shape[-1])

    spectrum = epochs.compute_psd(
        method="welch", fmin=cfg.l_freq, fmax=cfg.h_freq,
        n_fft=n_fft, n_overlap=n_fft // 2, average="mean", verbose="ERROR",
    )
    psd = spectrum.get_data()          # (n_epochs, n_ch, n_freqs)
    freqs = spectrum.freqs
    ch_names = epochs.ch_names

    bp = _band_powers_from_psd(psd, freqs)
    total_power = np.sum(list(bp.values()), axis=0)   # (n_epochs, n_ch)

    cols: Dict[str, np.ndarray] = {}
    for band, vals in bp.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(total_power > 0, vals / total_power, np.nan)
        for ci, ch in enumerate(ch_names):
            cols[f"relpow_{band}_{ch}"] = rel[:, ci]

    ent = _spectral_entropy(psd)
    sef = _spectral_edge_frequency(psd, freqs, pct=0.95)
    mob, cmplx = _hjorth(epochs.get_data(copy=False))

    for ci, ch in enumerate(ch_names):
        cols[f"specent_{ch}"] = ent[:, ci]
        cols[f"sef95_{ch}"] = sef[:, ci]
        cols[f"hjmob_{ch}"] = mob[:, ci]
        cols[f"hjcomp_{ch}"] = cmplx[:, ci]

    df = pd.DataFrame(cols)
    feature_names = list(df.columns)
    df.insert(0, "epoch_index", np.arange(len(df)))
    return df, feature_names


# =====================================================================================
# 9. SIGNAL-QUALITY METRICS
# =====================================================================================


def signal_quality_metrics(raw: mne.io.BaseRaw, cfg: Config, tag: str = "") -> Dict[str, Any]:
    """Scientifically-defined signal descriptors. No opaque composite "quality score".

    Each metric is a standard, individually interpretable quantity:
      time domain    : RMS, variance, kurtosis, mean absolute amplitude
      frequency      : absolute and relative band power (channel-averaged)
      artifact proxy : fraction of samples exceeding a robust amplitude threshold,
                       and the 50/60 Hz-adjacent line-noise ratio where measurable
    """
    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    m: Dict[str, Any] = {"tag": tag, "n_channels": data.shape[0],
                         "duration_s": round(data.shape[1] / sfreq, 2), "sfreq": sfreq}

    # -- time domain ---------------------------------------------------------------
    m["rms_v"] = float(np.sqrt(np.mean(data**2)))
    m["variance_v2"] = float(np.mean(np.var(data, axis=1)))
    m["mean_abs_amplitude_v"] = float(np.mean(np.abs(data)))
    m["kurtosis_mean"] = float(np.mean(sstats.kurtosis(data, axis=1, fisher=True)))
    m["channel_rms_sd_v"] = float(np.std(np.sqrt(np.mean(data**2, axis=1))))

    # -- artifact proxy: high-amplitude sample rate --------------------------------
    # Threshold is robust and derived per channel, so it adapts to the pipeline's scale.
    med = np.median(np.abs(data), axis=1, keepdims=True)
    mad = np.median(np.abs(np.abs(data) - med), axis=1, keepdims=True)
    scale = 1.4826 * mad
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(scale > 0, (np.abs(data) - med) / scale, 0.0)
    m["frac_samples_robust_z_gt5"] = float(np.mean(z > 5.0))
    m["frac_samples_robust_z_gt10"] = float(np.mean(z > 10.0))

    # -- frequency domain ----------------------------------------------------------
    n_fft = int(min(4 * sfreq, data.shape[1]))
    freqs, psd = sps.welch(data, fs=sfreq, nperseg=n_fft, noverlap=n_fft // 2, axis=-1)
    band = _band_powers_from_psd(psd, freqs)
    total = np.sum(list(band.values()), axis=0)
    for name, vals in band.items():
        m[f"abspow_{name}"] = float(np.mean(vals))
        with np.errstate(divide="ignore", invalid="ignore"):
            m[f"relpow_{name}"] = float(np.mean(np.where(total > 0, vals / total, np.nan)))
    m["total_band_power"] = float(np.mean(total))
    m["spectral_entropy_mean"] = float(np.mean(_spectral_entropy(psd)))
    m["sef95_mean_hz"] = float(np.mean(_spectral_edge_frequency(psd, freqs, 0.95)))
    return m


def compare_signals(
    raw_ours: mne.io.BaseRaw,
    raw_reference: mne.io.BaseRaw,
    cfg: Config,
    label: str = "ours_vs_reference",
) -> Dict[str, Any]:
    """Quantitative comparison of two versions of the same recording.

    IMPORTANT: ASR may remove data segments, so the two recordings can differ in length
    and be time-shifted relative to each other. Sample-wise correlation is therefore
    reported ONLY on the overlapping prefix and must be interpreted as a lower bound --
    a low value may reflect misalignment rather than a genuine processing difference.
    The length-robust spectral and amplitude comparisons are the primary evidence.
    """
    out: Dict[str, Any] = {"label": label}

    common = [c for c in raw_ours.ch_names if c in raw_reference.ch_names]
    out["n_common_channels"] = len(common)
    out["channels_only_in_ours"] = [c for c in raw_ours.ch_names if c not in common]
    out["channels_only_in_reference"] = [c for c in raw_reference.ch_names if c not in common]
    if not common:
        out["error"] = "No channels in common; comparison not possible."
        return out

    a = raw_ours.copy().pick(common)
    b = raw_reference.copy().pick(common)

    out["duration_ours_s"] = round(a.n_times / a.info["sfreq"], 2)
    out["duration_reference_s"] = round(b.n_times / b.info["sfreq"], 2)
    out["duration_ratio"] = round(out["duration_ours_s"] / max(out["duration_reference_s"], 1e-9), 4)

    # Align sampling rates before any sample-wise comparison
    if abs(a.info["sfreq"] - b.info["sfreq"]) > 1e-6:
        target = min(a.info["sfreq"], b.info["sfreq"])
        a.resample(target, verbose="ERROR")
        b.resample(target, verbose="ERROR")
        out["resampled_to_hz"] = float(target)

    da, db = a.get_data(), b.get_data()
    n = min(da.shape[1], db.shape[1])
    da_c, db_c = da[:, :n], db[:, :n]

    # -- sample-wise agreement on the overlapping prefix (lower bound) -------------
    per_ch_r: Dict[str, float] = {}
    for i, ch in enumerate(common):
        x, y = da_c[i], db_c[i]
        if np.std(x) > 0 and np.std(y) > 0:
            per_ch_r[ch] = float(np.corrcoef(x, y)[0, 1])
        else:
            per_ch_r[ch] = float("nan")
    vals = np.array(list(per_ch_r.values()), dtype=float)
    out["pearson_r_per_channel"] = per_ch_r
    out["pearson_r_mean"] = float(np.nanmean(vals))
    out["pearson_r_median"] = float(np.nanmedian(vals))
    out["pearson_r_min"] = float(np.nanmin(vals))
    out["pearson_r_note"] = (
        "Computed on the overlapping prefix only; ASR segment removal can misalign the "
        "two recordings, so this is a LOWER BOUND on true agreement."
    )

    # -- length-robust comparisons -------------------------------------------------
    qa = signal_quality_metrics(a, cfg, tag="ours")
    qb = signal_quality_metrics(b, cfg, tag="reference")
    out["quality_ours"] = qa
    out["quality_reference"] = qb

    diffs: Dict[str, Any] = {}
    for key in qa:
        if isinstance(qa[key], (int, float)) and isinstance(qb.get(key), (int, float)):
            va, vb = float(qa[key]), float(qb[key])
            diffs[key] = {
                "ours": va,
                "reference": vb,
                "abs_diff": va - vb,
                "rel_diff_pct": ((va - vb) / vb * 100.0) if vb not in (0,) else float("nan"),
            }
    out["metric_differences"] = diffs

    # -- PSD shape agreement (correlation of log-PSD across frequencies) ----------
    sf = a.info["sfreq"]
    nfft = int(min(4 * sf, n))
    fa, pa = sps.welch(da_c, fs=sf, nperseg=nfft, noverlap=nfft // 2, axis=-1)
    fb, pb = sps.welch(db_c, fs=sf, nperseg=nfft, noverlap=nfft // 2, axis=-1)
    with np.errstate(divide="ignore"):
        la, lb = np.log10(pa + 1e-30), np.log10(pb + 1e-30)
    psd_r = [
        float(np.corrcoef(la[i], lb[i])[0, 1])
        for i in range(la.shape[0])
        if np.std(la[i]) > 0 and np.std(lb[i]) > 0
    ]
    out["log_psd_correlation_mean"] = float(np.mean(psd_r)) if psd_r else float("nan")
    out["log_psd_correlation_min"] = float(np.min(psd_r)) if psd_r else float("nan")
    return out


# =====================================================================================
# 10. PER-SUBJECT DRIVER WITH CACHING AND ERROR HANDLING
# =====================================================================================


def _cache_paths(cfg: Config, sub: str, pipeline: str) -> Tuple[Path, Path]:
    d = cfg.cache_dir / f"pipeline_{pipeline}"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{sub}_features.parquet", d / f"{sub}_meta.json"


def process_subject(
    cfg: Config,
    sub: str,
    pipeline: str,
    group: str,
    compute_quality: bool = True,
) -> Dict[str, Any]:
    """Run one subject through one pipeline, end to end, with caching.

    Returns a status dict. NEVER raises on a per-subject failure: the error is captured,
    logged, and reported so the subject can be listed in the exclusion table rather than
    silently disappearing (brief #21).
    """
    feat_path, meta_path = _cache_paths(cfg, sub, pipeline)

    # -- cache hit -----------------------------------------------------------------
    if cfg.cache_enabled and not cfg.overwrite_cache and feat_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as fh:
                meta = json.load(fh)
            meta["status"] = "cached"
            meta["features_path"] = str(feat_path)
            return meta
        except Exception as exc:
            logger.warning("Cache for %s/%s unreadable (%r); reprocessing.", sub, pipeline, exc)

    record: Dict[str, Any] = {
        "participant_id": sub, "pipeline": pipeline, "group": group,
        "status": "failed", "error": "", "traceback": "",
    }

    try:
        with ResourceMonitor(f"total_{sub}_{pipeline}") as total_rm:
            # load
            with ResourceMonitor("load") as rm_load:
                raw = load_raw_subject(cfg, sub, preload=True)
            record["timing_load"] = rm_load.result
            record["input_duration_s"] = round(raw.n_times / raw.info["sfreq"], 2)
            record["input_n_channels"] = len(raw.ch_names)
            record["input_sfreq"] = float(raw.info["sfreq"])

            if compute_quality:
                record["quality_before"] = signal_quality_metrics(raw, cfg, tag="raw")

            # preprocess
            fn = PIPELINES[pipeline]
            with ResourceMonitor("preprocess") as rm_pre:
                clean, pmeta = fn(raw, cfg)
            record["timing_preprocess"] = rm_pre.result
            record["preprocess_meta"] = pmeta
            del raw

            if compute_quality:
                record["quality_after"] = signal_quality_metrics(clean, cfg, tag=f"pipeline_{pipeline}")

            # epoch
            with ResourceMonitor("epoch") as rm_ep:
                epochs, einfo = make_epochs(clean, cfg, pipeline)
            record["timing_epoch"] = rm_ep.result
            record["epoch_info"] = einfo

            if len(epochs) == 0:
                raise RuntimeError("No epochs survived rejection for this subject.")

            # features
            with ResourceMonitor("features") as rm_ft:
                feats, feature_names = extract_features(epochs, cfg)
            record["timing_features"] = rm_ft.result
            record["n_features"] = len(feature_names)
            record["n_epochs"] = int(len(feats))

            feats.insert(0, "group", group)
            feats.insert(0, "participant_id", sub)

            # persist
            try:
                feats.to_parquet(feat_path, index=False)
            except Exception:
                feat_path = feat_path.with_suffix(".csv")
                feats.to_csv(feat_path, index=False)
            record["features_path"] = str(feat_path)

            del epochs, clean, feats
            gc.collect()

        record["timing_total"] = total_rm.result
        record["status"] = "ok"

    except Exception as exc:
        record["error"] = repr(exc)
        record["traceback"] = traceback.format_exc()
        logger.error("Subject %s pipeline %s FAILED: %r", sub, pipeline, exc)

    # Always write the meta record, success or failure -- this is the audit trail.
    try:
        with open(meta_path, "w") as fh:
            json.dump(record, fh, indent=2, default=str)
    except Exception as exc:
        logger.error("Could not write meta for %s/%s: %r", sub, pipeline, exc)

    return record


def process_many_subjects(
    cfg: Config,
    subjects: Sequence[str],
    pipeline: str,
    groups: Dict[str, str],
    progress: bool = True,
) -> pd.DataFrame:
    """Sequentially process subjects. Restartable: cached subjects are skipped."""
    records: List[Dict[str, Any]] = []
    for i, sub in enumerate(subjects, 1):
        if progress:
            print(f"[{pipeline}] {i}/{len(subjects)}  {sub} ...", flush=True)
        rec = process_subject(cfg, sub, pipeline, groups.get(sub, "UNKNOWN"))
        status = rec.get("status")
        if progress:
            t = rec.get("timing_total", {}).get("wall_time_s", "n/a")
            print(f"    -> {status} (wall {t}s)", flush=True)
        records.append(rec)
        gc.collect()

    df = pd.json_normalize(records, max_level=1)
    return df


def assemble_feature_table(cfg: Config, pipeline: str, subjects: Sequence[str]) -> pd.DataFrame:
    """Concatenate the per-subject cached feature tables into one DataFrame."""
    frames: List[pd.DataFrame] = []
    missing: List[str] = []
    for sub in subjects:
        fp, _ = _cache_paths(cfg, sub, pipeline)
        if not fp.exists():
            fp = fp.with_suffix(".csv")
        if not fp.exists():
            missing.append(sub)
            continue
        try:
            frames.append(pd.read_parquet(fp) if fp.suffix == ".parquet" else pd.read_csv(fp))
        except Exception as exc:
            logger.error("Could not read features for %s: %r", sub, exc)
            missing.append(sub)

    if missing:
        logger.warning("Feature table missing %d subject(s): %s", len(missing), missing)
    if not frames:
        raise RuntimeError(f"No feature files found for pipeline {pipeline}.")
    return pd.concat(frames, ignore_index=True)


# =====================================================================================
# 11. CROSS-VALIDATION -- SUBJECT-LEVEL, LEAKAGE-FREE
# =====================================================================================


def _make_classifier(name: str, cfg: Config):
    """Build the sklearn estimator. Scaling is INSIDE the pipeline (see below)."""
    if name == "logreg":
        # 'lbfgs' (not 'liblinear'): liblinear cannot fit >2 classes, which would break
        # the three-class AD/FTD/CN task. lbfgs handles binary and multinomial alike,
        # so the SAME estimator definition serves every task -- important for the fair
        # comparison rule.
        clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs",
            class_weight="balanced", max_iter=5000, random_state=cfg.random_seed,
        )
    elif name == "rf":
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=cfg.n_jobs,
            random_state=cfg.random_seed,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")

    # StandardScaler is wrapped in the sklearn Pipeline so that `fit` sees ONLY the
    # training fold. Scaling the whole dataset before CV is a classic leakage route.
    return SkPipeline([("scaler", StandardScaler()), ("clf", clf)])


def _aggregate_to_subject(
    subject_ids: np.ndarray, y_epoch: np.ndarray, proba_epoch: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average epoch-level probabilities within each subject.

    The clinical unit of interest is the *subject*, not the 4-second epoch, so all
    headline metrics are computed after this aggregation. Mean probability is used
    rather than majority vote because it retains confidence information and yields a
    continuous score for ROC/PR analysis.
    """
    subs = np.unique(subject_ids)
    y_sub, p_sub = [], []
    for s in subs:
        mask = subject_ids == s
        labels = np.unique(y_epoch[mask])
        if len(labels) != 1:
            raise RuntimeError(f"Subject {s} has inconsistent labels: {labels}")
        y_sub.append(labels[0])
        p_sub.append(proba_epoch[mask].mean(axis=0))
    return subs, np.asarray(y_sub), np.asarray(p_sub)


def _classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, classes: np.ndarray
) -> Dict[str, Any]:
    """Compute the full metric set required by brief #13. Binary and multiclass."""
    n_classes = len(classes)
    m: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "n_samples": int(len(y_true)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    m["confusion_matrix"] = cm.tolist()
    m["confusion_matrix_labels"] = [str(c) for c in classes]

    if n_classes == 2:
        # Positive class = classes[1] by convention (classes are sorted).
        tn, fp, fn, tp = cm.ravel()
        m["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) else float("nan")
        m["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
        m["precision"] = float(tp / (tp + fp)) if (tp + fp) else float("nan")
        m["recall"] = m["sensitivity"]
        m["f1"] = float(f1_score(y_true, y_pred, pos_label=classes[1], zero_division=0))
        try:
            m["roc_auc"] = float(roc_auc_score((y_true == classes[1]).astype(int), proba[:, 1]))
            m["pr_auc"] = float(
                average_precision_score((y_true == classes[1]).astype(int), proba[:, 1])
            )
        except Exception:
            m["roc_auc"] = float("nan")
            m["pr_auc"] = float("nan")
    else:
        try:
            m["roc_auc_ovr_macro"] = float(
                roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
            )
        except Exception:
            m["roc_auc_ovr_macro"] = float("nan")
        per_p = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        per_r = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        per_f = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
        for i, c in enumerate(classes):
            m[f"precision_{c}"] = float(per_p[i])
            m[f"recall_{c}"] = float(per_r[i])
            m[f"f1_{c}"] = float(per_f[i])
    return m


def run_cross_validation(
    features: pd.DataFrame,
    feature_names: Sequence[str],
    task_classes: Sequence[str],
    cfg: Config,
    classifier: str = "logreg",
    scheme: str = "sgkf",
    pipeline_label: str = "",
) -> Dict[str, Any]:
    """Subject-level cross-validation.

    LEAKAGE PREVENTION (brief #11) -- three guarantees, all enforced here:
      1. Splitting uses `groups = participant_id`, so every epoch of a subject lands
         wholly in train or wholly in test. `StratifiedGroupKFold` additionally keeps
         the class balance similar across folds.
      2. StandardScaler lives inside the sklearn Pipeline, so it is fitted on training
         epochs only.
      3. No feature selection or hyper-parameter search is performed on the full data.
         (If you add tuning later, it must go in an INNER CV on the training fold.)

    `scheme` is 'sgkf' (repeated stratified group k-fold, primary) or 'loso'
    (leave-one-subject-out, secondary robustness analysis).
    """
    df = features[features["group"].isin(task_classes)].copy()
    if df.empty:
        raise ValueError(f"No data for classes {task_classes}.")

    classes = np.array(sorted(task_classes))
    X = df[list(feature_names)].to_numpy(dtype=float)
    y = df["group"].to_numpy()
    groups = df["participant_id"].to_numpy()

    # Guard against NaNs, which some features can produce on degenerate epochs.
    finite = np.isfinite(X).all(axis=1)
    n_dropped = int((~finite).sum())
    X, y, groups = X[finite], y[finite], groups[finite]

    n_subjects = len(np.unique(groups))
    subj_class = {s: y[groups == s][0] for s in np.unique(groups)}
    class_counts = pd.Series(list(subj_class.values())).value_counts().to_dict()

    result: Dict[str, Any] = {
        "pipeline": pipeline_label,
        "task": "_vs_".join(classes),
        "classifier": classifier,
        "scheme": scheme,
        "n_subjects": n_subjects,
        "n_epochs": int(len(y)),
        "n_epochs_dropped_nonfinite": n_dropped,
        "subjects_per_class": class_counts,
        "feature_count": len(feature_names),
        "random_seed": cfg.random_seed,
    }

    # Refuse to run a k-fold that cannot be populated.
    min_class = min(class_counts.values()) if class_counts else 0
    if min_class < 2:
        result["error"] = (
            f"Not enough subjects per class ({class_counts}) to cross-validate. "
            "Increase MAX_SUBJECTS."
        )
        return result

    # -- build the list of (train_idx, test_idx, repeat) splits --------------------
    splits: List[Tuple[np.ndarray, np.ndarray, int]] = []
    if scheme == "loso":
        logo = LeaveOneGroupOut()
        splits = [(tr, te, 0) for tr, te in logo.split(X, y, groups)]
    elif scheme == "sgkf":
        n_splits = min(cfg.cv_n_splits, min_class)
        result["effective_n_splits"] = n_splits
        if n_splits < 2:
            result["error"] = f"Only {n_splits} usable fold(s); cannot cross-validate."
            return result
        for rep in range(cfg.cv_n_repeats):
            sgkf = StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=cfg.random_seed + rep
            )
            for tr, te in sgkf.split(X, y, groups):
                splits.append((tr, te, rep))
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    result["n_splits_total"] = len(splits)

    # -- run ------------------------------------------------------------------------
    fold_rows: List[Dict[str, Any]] = []
    # subject -> list of predicted probability vectors (one per time the subject was
    # in a test fold). Used for the paired statistical comparison in notebook 03.
    subject_proba: Dict[str, List[np.ndarray]] = {}
    subject_truth: Dict[str, str] = {}

    fit_times: List[float] = []

    for k, (tr, te, rep) in enumerate(splits):
        # Explicit leakage assertion -- cheap, and it makes the guarantee testable.
        overlap = set(groups[tr]) & set(groups[te])
        if overlap:
            raise RuntimeError(f"LEAKAGE: subjects in both train and test: {overlap}")

        model = _make_classifier(classifier, cfg)
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        fit_times.append(time.perf_counter() - t0)

        proba = model.predict_proba(X[te])
        # Align probability columns to our canonical class order
        order = [list(model.classes_).index(c) for c in classes]
        proba = proba[:, order]

        subs_te, y_sub, p_sub = _aggregate_to_subject(groups[te], y[te], proba)
        y_sub_pred = classes[np.argmax(p_sub, axis=1)]

        fold_metrics = _classification_metrics(y_sub, y_sub_pred, p_sub, classes)
        fold_metrics.update(fold=k, repeat=rep, n_test_subjects=len(subs_te),
                            fit_time_s=round(fit_times[-1], 4))
        fold_rows.append(fold_metrics)

        for s, ps, ys in zip(subs_te, p_sub, y_sub):
            subject_proba.setdefault(str(s), []).append(ps)
            subject_truth[str(s)] = str(ys)

    result["fold_results"] = fold_rows
    result["mean_fit_time_s"] = float(np.mean(fit_times))
    result["total_fit_time_s"] = float(np.sum(fit_times))

    # -- aggregate over folds -------------------------------------------------------
    fold_df = pd.DataFrame(fold_rows)
    numeric = fold_df.select_dtypes(include=[np.number])
    summary: Dict[str, Any] = {}
    for col in numeric.columns:
        if col in ("fold", "repeat"):
            continue
        vals = numeric[col].dropna().to_numpy()
        if len(vals) == 0:
            continue
        summary[f"{col}_mean"] = float(np.mean(vals))
        summary[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        # Percentile CI across folds. NOTE: CV folds are not independent, so this
        # interval is descriptive of fold spread, not a valid frequentist CI.
        summary[f"{col}_p2.5"] = float(np.percentile(vals, 2.5))
        summary[f"{col}_p97.5"] = float(np.percentile(vals, 97.5))
    result["summary"] = summary
    result["fold_ci_caveat"] = (
        "Percentiles across CV folds describe fold-to-fold spread. CV folds share "
        "training data and are therefore NOT independent; do not read these as valid "
        "confidence intervals. Use the subject-level paired bootstrap instead."
    )

    # -- subject-level out-of-fold predictions (for paired tests) -------------------
    oof_rows: List[Dict[str, Any]] = []
    for s, plist in subject_proba.items():
        mean_p = np.mean(np.vstack(plist), axis=0)
        oof_rows.append({
            "participant_id": s,
            "true_class": subject_truth[s],
            "pred_class": classes[int(np.argmax(mean_p))],
            "n_times_tested": len(plist),
            **{f"proba_{c}": float(mean_p[i]) for i, c in enumerate(classes)},
        })
    oof = pd.DataFrame(oof_rows).sort_values("participant_id").reset_index(drop=True)
    result["oof_predictions"] = oof.to_dict(orient="records")

    y_true_all = oof["true_class"].to_numpy()
    y_pred_all = oof["pred_class"].to_numpy()
    proba_all = oof[[f"proba_{c}" for c in classes]].to_numpy()
    result["pooled_subject_level"] = _classification_metrics(
        y_true_all, y_pred_all, proba_all, classes
    )
    return result


# =====================================================================================
# 12. STATISTICAL COMPARISON OF THE TWO PIPELINES
# =====================================================================================


def paired_bootstrap_subject_level(
    oof_a: pd.DataFrame,
    oof_b: pd.DataFrame,
    metric: str = "balanced_accuracy",
    classes: Optional[Sequence[str]] = None,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Paired bootstrap over SUBJECTS -- the primary statistical test.

    Why this and not a t-test over CV folds? Because CV folds overlap in training data,
    fold-level metrics are positively correlated and their variance is badly
    underestimated (Dietterich 1998; Nadeau & Bengio 2003), which inflates Type-I error.
    Subjects, in contrast, are the genuine independent sampling units here.

    Procedure: resample subjects with replacement; for each resample recompute the metric
    for pipeline A and pipeline B on the SAME subjects (that is what makes it paired) and
    record the difference. The reported p-value is a two-sided bootstrap p-value for
    H0: delta = 0.
    """
    a = oof_a.set_index("participant_id").sort_index()
    b = oof_b.set_index("participant_id").sort_index()
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"error": "No subjects in common between the two pipelines."}
    a, b = a.loc[common], b.loc[common]

    if not (a["true_class"].to_numpy() == b["true_class"].to_numpy()).all():
        return {"error": "True labels disagree between pipelines; cannot pair."}

    cls = np.array(sorted(classes)) if classes is not None else np.array(
        sorted(a["true_class"].unique())
    )
    proba_cols = [f"proba_{c}" for c in cls]

    y = a["true_class"].to_numpy()
    pa, pb = a["pred_class"].to_numpy(), b["pred_class"].to_numpy()
    qa = a[proba_cols].to_numpy() if all(c in a.columns for c in proba_cols) else None
    qb = b[proba_cols].to_numpy() if all(c in b.columns for c in proba_cols) else None

    def score(idx: np.ndarray, pred: np.ndarray, proba: Optional[np.ndarray]) -> float:
        yy, pp = y[idx], pred[idx]
        if metric == "balanced_accuracy":
            if len(np.unique(yy)) < 2:
                return float("nan")
            return float(balanced_accuracy_score(yy, pp))
        if metric == "accuracy":
            return float(accuracy_score(yy, pp))
        if metric == "f1_macro":
            return float(f1_score(yy, pp, average="macro", zero_division=0))
        if metric == "roc_auc":
            if proba is None or len(cls) != 2 or len(np.unique(yy)) < 2:
                return float("nan")
            return float(roc_auc_score((yy == cls[1]).astype(int), proba[idx][:, 1]))
        raise ValueError(f"Unsupported metric: {metric}")

    n = len(common)
    all_idx = np.arange(n)
    obs_a, obs_b = score(all_idx, pa, qa), score(all_idx, pb, qb)

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = score(idx, pb, qb) - score(idx, pa, qa)

    valid = deltas[np.isfinite(deltas)]
    if len(valid) == 0:
        return {"error": "All bootstrap replicates were undefined (degenerate resamples)."}

    obs_delta = obs_b - obs_a
    # Two-sided bootstrap p-value: proportion of replicates at least as extreme as 0,
    # centred on the observed difference. +1 correction avoids p = 0 exactly.
    centred = valid - np.mean(valid)
    p = (np.sum(np.abs(centred) >= abs(obs_delta)) + 1) / (len(valid) + 1)

    # Effect size: Cohen's g for paired proportions is not well defined for balanced
    # accuracy, so we report the standardised bootstrap effect instead, plus the raw
    # difference which is the directly interpretable quantity.
    sd = float(np.std(valid, ddof=1))
    return {
        "metric": metric,
        "n_subjects": int(n),
        "n_bootstrap": int(len(valid)),
        "pipeline_A": obs_a,
        "pipeline_B": obs_b,
        "observed_delta_B_minus_A": float(obs_delta),
        "ci95_low": float(np.percentile(valid, 2.5)),
        "ci95_high": float(np.percentile(valid, 97.5)),
        "bootstrap_sd": sd,
        "standardised_effect": float(obs_delta / sd) if sd > 0 else float("nan"),
        "p_value_two_sided": float(p),
        "interpretation_note": (
            "The 95% CI is the primary result. If it contains 0, the data do not "
            "support a difference between pipelines on this metric."
        ),
    }


def wilcoxon_fold_level(
    folds_a: pd.DataFrame, folds_b: pd.DataFrame, metric: str = "balanced_accuracy"
) -> Dict[str, Any]:
    """Secondary, exploratory test: Wilcoxon signed-rank on matched fold metrics.

    Reported for completeness because it is common in the literature, but flagged as
    ANTI-CONSERVATIVE: CV folds are not independent, so the nominal p-value understates
    the true Type-I error rate. The subject-level paired bootstrap is authoritative.
    """
    if metric not in folds_a.columns or metric not in folds_b.columns:
        return {"error": f"Metric '{metric}' not present in fold results."}

    a = folds_a.sort_values(["repeat", "fold"])[metric].to_numpy(dtype=float)
    b = folds_b.sort_values(["repeat", "fold"])[metric].to_numpy(dtype=float)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]

    if len(a) < 5:
        return {"error": f"Only {len(a)} paired folds; too few for a meaningful test."}
    if np.allclose(a, b):
        return {"note": "Fold metrics are identical; test not applicable.",
                "n_pairs": int(len(a))}

    stat, p = sstats.wilcoxon(a, b)
    d = b - a
    return {
        "metric": metric,
        "n_pairs": int(len(a)),
        "mean_A": float(np.mean(a)),
        "mean_B": float(np.mean(b)),
        "mean_delta_B_minus_A": float(np.mean(d)),
        "statistic": float(stat),
        "p_value": float(p),
        "caveat": (
            "ANTI-CONSERVATIVE. Cross-validation folds share training data and are not "
            "independent, so this p-value is optimistic. Treat as exploratory only."
        ),
    }


def corrected_resampled_ttest(
    folds_a: pd.DataFrame,
    folds_b: pd.DataFrame,
    metric: str = "balanced_accuracy",
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
) -> Dict[str, Any]:
    """Nadeau & Bengio (2003) corrected resampled t-test.

    Inflates the variance estimate by (1/k + n_test/n_train) to compensate for the
    dependence between overlapping training sets. This is the statistically defensible
    way to test repeated k-fold results when a subject-level bootstrap is not available.
    """
    if metric not in folds_a.columns or metric not in folds_b.columns:
        return {"error": f"Metric '{metric}' not present in fold results."}

    a = folds_a.sort_values(["repeat", "fold"])[metric].to_numpy(dtype=float)
    b = folds_b.sort_values(["repeat", "fold"])[metric].to_numpy(dtype=float)
    n = min(len(a), len(b))
    d = (b[:n] - a[:n])
    d = d[np.isfinite(d)]
    k = len(d)
    if k < 3:
        return {"error": f"Only {k} paired folds; too few."}
    if np.allclose(d, 0):
        return {"note": "No difference between folds.", "n_pairs": k}

    if n_test is None or n_train is None:
        # Fall back to the uncorrected ratio implied by k-fold: test = 1/k of data.
        ratio = 1.0 / (k - 1) if k > 1 else 1.0
    else:
        ratio = n_test / n_train

    mean_d = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    corrected_var = var_d * (1.0 / k + ratio)
    if corrected_var <= 0:
        return {"error": "Corrected variance is non-positive."}
    t = mean_d / np.sqrt(corrected_var)
    p = 2 * (1 - sstats.t.cdf(abs(t), df=k - 1))
    return {
        "metric": metric,
        "n_pairs": k,
        "mean_delta_B_minus_A": mean_d,
        "sd_of_fold_differences": float(np.sqrt(var_d)),
        # The naive standard error of the mean difference, sd/sqrt(k), assumes the folds
        # are independent. The corrected standard error below is deliberately LARGER; it
        # is the quantity the t-statistic is actually divided by.
        "naive_standard_error": float(np.sqrt(var_d / k)),
        "corrected_standard_error": float(np.sqrt(corrected_var)),
        "inflation_factor": float(np.sqrt(corrected_var) / np.sqrt(var_d / k)),
        "t_statistic": float(t),
        "df": k - 1,
        "p_value": float(p),
        "correction_ratio_ntest_over_ntrain": float(ratio),
        "reference": "Nadeau & Bengio (2003), Machine Learning 52:239-281",
    }


def mcnemar_subject_level(oof_a: pd.DataFrame, oof_b: pd.DataFrame) -> Dict[str, Any]:
    """Exact McNemar test on subject-level correctness -- discordant pairs only.

    Directly answers: "of the subjects the two pipelines classify differently, is the
    split lopsided?" Uses the exact binomial version, which is appropriate for the small
    discordant counts expected with ~88 subjects.
    """
    a = oof_a.set_index("participant_id").sort_index()
    b = oof_b.set_index("participant_id").sort_index()
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return {"error": "No subjects in common."}
    a, b = a.loc[common], b.loc[common]

    ca = (a["pred_class"].to_numpy() == a["true_class"].to_numpy())
    cb = (b["pred_class"].to_numpy() == b["true_class"].to_numpy())

    n01 = int(np.sum(~ca & cb))   # A wrong, B right
    n10 = int(np.sum(ca & ~cb))   # A right, B wrong
    n11 = int(np.sum(ca & cb))
    n00 = int(np.sum(~ca & ~cb))
    disc = n01 + n10

    out = {
        "n_subjects": int(len(common)),
        "both_correct": n11,
        "both_wrong": n00,
        "only_B_correct": n01,
        "only_A_correct": n10,
        "n_discordant": disc,
    }
    if disc == 0:
        out["note"] = "No discordant subjects; the two pipelines agree on every subject."
        out["p_value"] = 1.0
        return out

    res = sstats.binomtest(n01, n=disc, p=0.5, alternative="two-sided")
    out["p_value"] = float(res.pvalue)
    out["test"] = "Exact McNemar (binomial on discordant pairs)"
    return out


# =====================================================================================
# 13. RESULT PERSISTENCE
# =====================================================================================


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands NumPy scalars and arrays.

    Without this, a stray np.float64 falls through to `default=str` and is silently
    written as the STRING "0.82" instead of the number 0.82. Notebook 03 would then load
    it back as text and every downstream arithmetic operation would fail or, worse,
    silently produce nonsense. Numeric fidelity across the save/load boundary matters
    here because the whole comparison is quantitative.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return v if np.isfinite(v) else None
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (Path,)):
            return str(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return str(obj)


def save_results(cfg: Config, name: str, obj: Any) -> Path:
    """Write a result object to results/ as JSON (dict) or CSV (DataFrame)."""
    cfg.make_dirs()
    if isinstance(obj, pd.DataFrame):
        path = cfg.results_dir / f"{name}.csv"
        obj.to_csv(path, index=False)
    else:
        path = cfg.results_dir / f"{name}.json"
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=2, cls=_NumpyJSONEncoder)
    return path


def load_results(cfg: Config, name: str) -> Any:
    """Load a previously saved result; raises a clear error if the producer never ran."""
    jp = cfg.results_dir / f"{name}.json"
    cp = cfg.results_dir / f"{name}.csv"
    if jp.exists():
        with open(jp) as fh:
            return json.load(fh)
    if cp.exists():
        return pd.read_csv(cp)
    raise FileNotFoundError(
        f"Result '{name}' not found in {cfg.results_dir}. "
        "Run the notebook that produces it first (notebook 01 for Pipeline A, "
        "02 for Pipeline B)."
    )


def directory_size_mb(path: Path) -> float:
    """Total size of a directory tree, for the storage-cost benchmark."""
    total = 0
    for p in Path(path).rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return round(total / 1024**2, 3)
