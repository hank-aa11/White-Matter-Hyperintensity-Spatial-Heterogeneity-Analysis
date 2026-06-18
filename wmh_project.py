"""
WMH spatial heterogeneity pipeline.

Input files expected:
  images.zip               ADNI subject folders, each with a nifti/ subdirectory
  MNI152_T1_1mm.nii.gz     MNI T1 template
  all30m.xlsx              phenotype table

Steps:
  1) unzip images.zip, scan for T1 / T2-FLAIR / WMH mask paths
  2) rigid T2/FLAIR -> T1 registration (Mattes MI)
  3) T1 -> MNI affine + B-spline nonrigid registration
  4) warp WMH masks to MNI (nearest-neighbor)
  5) smooth, downsample, threshold at >=1% prevalence, cluster
  6) compute regional WMH proportions per subject
  7) correlate with phenotype variables, save figures

Raw images are never copied to the submission package. All outputs go to --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


# ---------------------------------------------------------------------------
# Hard-coded default paths for all input/output locations.
# These exist purely for development convenience -- sensible defaults mean
# you can run the script on your own machine without typing out every path
# every time. Every single one can be overridden via the command line, so
# they do not affect reproducibility on other machines.
# ---------------------------------------------------------------------------
DEFAULT_IMAGES_ZIP = r"C:\Users\黄济川\OneDrive\Desktop\wmh_project_solution\images.zip"
DEFAULT_MNI_TEMPLATE = r"C:\Users\黄济川\OneDrive\Desktop\wmh_project_solution\MNI152_T1_1mm.nii.gz"
DEFAULT_PHENOTYPE_XLSX = r"C:\Users\黄济川\OneDrive\Desktop\wmh_project_solution\all30m.xlsx"
DEFAULT_OUTPUT_DIR = r"C:\Users\黄济川\OneDrive\Desktop\wmh_project_solution\results"

DEFAULT_X_VARS = ["AGE", "PTGENDER", "APOE4", "AV45", "ADAS11"]


# ---------------------------------------------------------------------------
# Lazy import wrappers for all heavy optional dependencies.
# Each package is imported only when its wrapper is first called, not at
# module load time. The practical payoff: if a package is missing you get a
# clear, actionable error message telling you exactly which library is absent
# and that "pip install -r requirements.txt" will fix it -- rather than a
# bare ModuleNotFoundError before any useful work has started.
# ---------------------------------------------------------------------------

def import_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except Exception as exc:
        raise RuntimeError("Missing dependency: numpy. Run: pip install -r requirements.txt") from exc


def import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception as exc:
        raise RuntimeError("Missing dependency: pandas/openpyxl. Run: pip install -r requirements.txt") from exc


def import_sitk():
    try:
        import SimpleITK as sitk  # type: ignore
        return sitk
    except Exception as exc:
        raise RuntimeError("Missing dependency: SimpleITK. Run: pip install -r requirements.txt") from exc


def import_scipy():
    try:
        from scipy import ndimage, stats  # type: ignore
        return ndimage, stats
    except Exception as exc:
        raise RuntimeError("Missing dependency: scipy. Run: pip install -r requirements.txt") from exc


def import_sklearn():
    try:
        from sklearn.cluster import MiniBatchKMeans, SpectralClustering  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore
        return MiniBatchKMeans, SpectralClustering, StandardScaler
    except Exception as exc:
        raise RuntimeError("Missing dependency: scikit-learn. Run: pip install -r requirements.txt") from exc


def import_matplotlib():
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception as exc:
        raise RuntimeError("Missing dependency: matplotlib. Run: pip install -r requirements.txt") from exc


# ---------------------------------------------------------------------------
# Small utility functions used throughout the rest of the script.
# Nothing algorithmic here -- just timestamp formatting, directory creation,
# path normalization, and JSON writing. Grouping them near the top avoids
# forward references and keeps the domain-specific code further down clean.
# ---------------------------------------------------------------------------

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def natural_key(text: str) -> tuple[Any, ...]:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(text)))


def path_abs(pathlike: str | Path) -> Path:
    return Path(pathlike).expanduser().resolve()


def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)


def write_json(data: dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Scanning the extracted ZIP for subject image files, and reading/writing
# the subject manifest CSV.
# The manifest is the central ledger of which NIfTI files belong to which
# subject. Writing it early means every downstream stage can skip the
# filesystem scan and just read the CSV -- which is especially useful when
# iterating on later steps without re-running extraction.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubjectRecord:
    subject_id: str
    t1_path: Path
    t2_path: Path
    wmh_mask_path: Path

    def as_row(self) -> dict[str, str]:
        return {
            "subject_id": self.subject_id,
            "t1_path": str(self.t1_path),
            "t2_path": str(self.t2_path),
            "wmh_mask_path": str(self.wmh_mask_path),
        }


T1_NAME_HINTS = ["t1_brain", "t1", "mprage", "spgr"]
T2_NAME_HINTS = ["flair", "t2"]
WMH_NAME_HINTS = ["wmh", "lesion", "seg"]


def unzip_images(images_zip: Path, extract_dir: Path, overwrite: bool = False) -> Path:
    if not images_zip.exists():
        raise FileNotFoundError(f"images.zip not found: {images_zip}")
    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)
    if extract_dir.exists() and any(extract_dir.iterdir()):
        log(f"Using already-extracted images folder: {extract_dir}")
        return extract_dir
    ensure_dir(extract_dir)
    log(f"Unzipping {images_zip} -> {extract_dir}")
    with zipfile.ZipFile(images_zip, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def list_nii_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.nii")) + list(root.rglob("*.nii.gz"))
    # On some systems (notably macOS and Windows) rglob("*.nii.gz") can
    # return the same file twice because it also matches the "*.nii" pattern.
    # Deduplicate by resolved absolute path before returning.
    seen: set[str] = set()
    out: list[Path] = []
    for p in sorted(files, key=lambda x: natural_key(str(x))):
        if "__MACOSX" in p.parts:
            continue
        key = str(p.resolve())
        if key not in seen and p.is_file():
            seen.add(key)
            out.append(p)
    return out


def score_t1(path: Path) -> int:
    name = path.name.lower()
    if any(bad in name for bad in ["mask", "wmh", "lesion", "seg"]):
        return -10_000
    score = 0
    if "t1_brain" in name:
        score += 100
    if re.search(r"(^|[_-])t1([_.-]|$)", name):
        score += 80
    if "mprage" in name or "spgr" in name:
        score += 50
    if "brain" in name:
        score += 10
    return score


def score_t2(path: Path) -> int:
    name = path.name.lower()
    if any(bad in name for bad in ["mask", "wmh", "lesion", "seg"]):
        return -10_000
    score = 0
    if "flair" in name:
        score += 100
    if re.search(r"(^|[_-])t2([_.-]|$)", name):
        score += 70
    if "brain" in name:
        score += 5
    return score


def score_wmh(path: Path) -> int:
    name = path.name.lower()
    # Brain extraction masks (e.g. "t1_brain_mask.nii.gz") have naming
    # patterns that accidentally score well on WMH heuristics -- they contain
    # "mask" and sometimes "brain". Assign a large negative score so they
    # can never be mistakenly selected as the WMH lesion mask.
    if "t1_brain_mask" in name or "brain_mask" == name.replace(".nii.gz", "").replace(".nii", ""):
        return -10_000
    if "t1" in name and "wmh" not in name and "lesion" not in name and "seg" not in name:
        return -10_000
    score = 0
    if "wmh" in name:
        score += 120
    if "lesion" in name:
        score += 100
    if "seg" in name:
        score += 80
    if "mask" in name:
        score += 20
    if "flair" in name or "t2" in name:
        score += 10
    return score


def choose_best(files: Sequence[Path], scorer) -> Path | None:
    scored = [(scorer(p), p) for p in files]
    scored = [(s, p) for s, p in scored if s > 0]
    if not scored:
        return None
    scored.sort(key=lambda sp: (-sp[0], natural_key(sp[1].name)))
    return scored[0][1]


def find_subject_dirs(images_root: Path) -> list[Path]:
    # ADNI images.zip typically follows this layout:
    #   <SubjectID>/nifti/<modality files>
    # where SubjectID matches the pattern "###_S_####".
    # Look for that canonical structure first because it is the most reliable.
    candidates: list[Path] = []
    for p in images_root.rglob("nifti"):
        if p.is_dir() and "__MACOSX" not in p.parts:
            subject_dir = p.parent
            if re.search(r"\d{3}_S_\d+", subject_dir.name):
                candidates.append(subject_dir)
    if not candidates:
        # Nothing matched the standard ADNI layout. Fall back to any
        # directory whose name looks like an ADNI subject ID and that
        # contains at least one NIfTI file. This is less reliable but
        # covers non-standard or manually repacked ZIP archives.
        for p in images_root.rglob("*"):
            if p.is_dir() and "__MACOSX" not in p.parts and re.search(r"\d{3}_S_\d+", p.name):
                if list_nii_files(p):
                    candidates.append(p)
    # Deduplicate before returning -- the two search passes above can
    # theoretically yield the same directory twice if the ZIP was packed
    # with nested subject folders.
    seen: set[str] = set()
    out: list[Path] = []
    for p in sorted(candidates, key=lambda x: natural_key(x.name)):
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def scan_images_root(images_root: Path) -> list[SubjectRecord]:
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    records: list[SubjectRecord] = []
    missing_report: list[dict[str, str]] = []
    subject_dirs = find_subject_dirs(images_root)
    log(f"Found {len(subject_dirs)} possible subject folders")
    for sdir in subject_dirs:
        subject_id = sdir.name
        files = list_nii_files(sdir)
        t1 = choose_best(files, score_t1)
        t2 = choose_best(files, score_t2)
        wmh = choose_best(files, score_wmh)
        if t1 and t2 and wmh:
            records.append(SubjectRecord(subject_id, t1.resolve(), t2.resolve(), wmh.resolve()))
        else:
            missing_report.append({
                "subject_id": subject_id,
                "has_t1": str(bool(t1)),
                "has_t2_flair": str(bool(t2)),
                "has_wmh_mask": str(bool(wmh)),
                "folder": str(sdir),
            })
    records.sort(key=lambda r: natural_key(r.subject_id))
    if missing_report:
        log(f"Skipped {len(missing_report)} folders because T1/T2/WMH mask was not confidently detected")
    return records


def write_manifest(records: Sequence[SubjectRecord], out_csv: Path) -> Path:
    ensure_dir(out_csv.parent)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "t1_path", "t2_path", "wmh_mask_path"])
        writer.writeheader()
        for r in records:
            writer.writerow(r.as_row())
    return out_csv


def read_manifest(path: Path) -> list[SubjectRecord]:
    records: list[SubjectRecord] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {"subject_id", "t1_path", "t2_path"}
        if not required <= fields:
            raise ValueError(f"Manifest missing columns: {sorted(required - fields)}")
        mask_col = "wmh_mask_path" if "wmh_mask_path" in fields else "mask_path"
        if mask_col not in fields:
            raise ValueError("Manifest must contain wmh_mask_path or mask_path column")
        for row in reader:
            records.append(SubjectRecord(
                subject_id=str(row["subject_id"]).strip(),
                t1_path=Path(str(row["t1_path"]).strip()),
                t2_path=Path(str(row["t2_path"]).strip()),
                wmh_mask_path=Path(str(row[mask_col]).strip()),
            ))
    records.sort(key=lambda r: natural_key(r.subject_id))
    return records


def validate_records(records: Sequence[SubjectRecord], limit: int = 0) -> list[SubjectRecord]:
    valid: list[SubjectRecord] = []
    for r in records:
        missing = [label for label, p in [("T1", r.t1_path), ("T2/FLAIR", r.t2_path), ("WMH mask", r.wmh_mask_path)] if not p.exists()]
        if missing:
            log(f"Skipping {r.subject_id}: missing {', '.join(missing)}")
            continue
        valid.append(r)
    valid.sort(key=lambda x: natural_key(x.subject_id))
    if limit and limit > 0:
        valid = valid[:limit]
    if not valid:
        raise RuntimeError("No valid subjects found. Check images.zip folder names and NIfTI filenames.")
    return valid


# ---------------------------------------------------------------------------
# SimpleITK registration wrappers.
# Most of the code in this section is parameter configuration for the
# optimizer and image sampler. The design is intentionally explicit --
# every tunable knob is visible at the call site rather than buried inside
# a helper object -- so that registration behavior is easy to audit
# and adjust without hunting through class internals.
# ---------------------------------------------------------------------------

def read_image(path: Path, pixel_type: str = "float32"):
    sitk = import_sitk()
    if pixel_type == "float32":
        return sitk.ReadImage(str(path), sitk.sitkFloat32)
    return sitk.ReadImage(str(path))


def normalize_for_registration(img):
    sitk = import_sitk()
    # Intensity normalization plus a light Gaussian blur before MI-based
    # registration. The key motivation: T1 and FLAIR have very different
    # absolute intensity scales and tissue contrasts. Normalizing both to
    # zero mean / unit variance puts them in a comparable range and
    # stabilizes the Mattes MI metric. The small blur (variance=0.5 voxels)
    # suppresses high-frequency noise that would add variance to the MI
    # gradient without carrying any useful alignment signal.
    img = sitk.Cast(img, sitk.sitkFloat32)
    img = sitk.Normalize(img)
    img = sitk.DiscreteGaussian(img, variance=0.5)
    return img


def center_transform(fixed, moving, transform_kind: str = "rigid"):
    sitk = import_sitk()
    if transform_kind == "rigid":
        tx = sitk.Euler3DTransform()
    elif transform_kind == "affine":
        tx = sitk.AffineTransform(3)
    else:
        raise ValueError(transform_kind)
    return sitk.CenteredTransformInitializer(
        fixed,
        moving,
        tx,
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )


def run_mi_registration(
    fixed,
    moving,
    initial_transform,
    *,
    learning_rate: float,
    iterations: int,
    shrink_factors: Sequence[int],
    smoothing_sigmas: Sequence[float],
    sampling_percentage: float,
    log_file: Path | None = None,
):
    sitk = import_sitk()
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(sampling_percentage), seed=2026)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=float(learning_rate),
        numberOfIterations=int(iterations),
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel(list(shrink_factors))
    reg.SetSmoothingSigmasPerLevel(list(smoothing_sigmas))
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial_transform, inPlace=False)

    if log_file:
        ensure_dir(log_file.parent)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"\n[{now()}] Starting registration\n")

    final_tx = reg.Execute(fixed, moving)

    if log_file:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"Metric value: {reg.GetMetricValue()}\n")
            f.write(f"Optimizer stop: {reg.GetOptimizerStopConditionDescription()}\n")
    return final_tx


def run_bspline_registration(
    fixed,
    moving,
    *,
    mesh_size: int,
    iterations: int,
    sampling_percentage: float,
    log_file: Path | None = None,
):
    sitk = import_sitk()
    mesh = [int(mesh_size)] * 3
    initial = sitk.BSplineTransformInitializer(fixed, mesh)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=40)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(float(sampling_percentage), seed=2027)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=int(iterations),
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=max(200, int(iterations) * 10),
        costFunctionConvergenceFactor=1e7,
    )
    reg.SetShrinkFactorsPerLevel([4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial, inPlace=False)
    final_tx = reg.Execute(fixed, moving)
    if log_file:
        ensure_dir(log_file.parent)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"BSpline metric value: {reg.GetMetricValue()}\n")
            f.write(f"BSpline stop: {reg.GetOptimizerStopConditionDescription()}\n")
    return final_tx


def resample_to_reference(moving, reference, transform, interpolator_name: str, default_value: float = 0.0, pixel_id=None):
    sitk = import_sitk()
    interp = {
        "linear": sitk.sitkLinear,
        "nearest": sitk.sitkNearestNeighbor,
        "bspline": sitk.sitkBSpline,
    }[interpolator_name]
    if pixel_id is None:
        pixel_id = moving.GetPixelID()
    return sitk.Resample(moving, reference, transform, interp, float(default_value), pixel_id)


def subject_output_paths(output_dir: Path, subject_id: str) -> dict[str, Path]:
    base = ensure_dir(output_dir / "registration" / subject_id)
    return {
        "base": base,
        "log": base / "registration_log.txt",
        "t2_to_t1_tx": base / "t2_to_t1_rigid.tfm",
        "t2_in_t1": base / "t2_flair_in_t1.nii.gz",
        "mask_in_t1": base / "wmh_mask_in_t1.nii.gz",
        "t1_to_mni_affine_tx": base / "t1_to_mni_affine.tfm",
        "t1_affine_mni": base / "t1_affine_mni.nii.gz",
        "mask_affine_mni": base / "wmh_mask_affine_mni.nii.gz",
        "t1_to_mni_bspline_tx": base / "t1_to_mni_bspline.tfm",
        "t1_mni": base / "t1_mni.nii.gz",
        "wmh_mask_mni": base / "wmh_mask_mni.nii.gz",
        "t2_mni": base / "t2_flair_mni.nii.gz",
    }


def register_one_subject(
    record: SubjectRecord,
    mni_template: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    fast: bool = False,
) -> dict[str, str]:
    sitk = import_sitk()
    paths = subject_output_paths(output_dir, record.subject_id)
    final_mask = paths["wmh_mask_mni"]
    final_t1 = paths["t1_mni"]
    if final_mask.exists() and final_t1.exists() and not overwrite:
        return {"subject_id": record.subject_id, "status": "SKIP_EXISTS", "wmh_mask_mni": str(final_mask), "message": ""}

    log(f"Registering {record.subject_id}")
    try:
        t1_raw = read_image(record.t1_path, "float32")
        t2_raw = read_image(record.t2_path, "float32")
        mask_raw = read_image(record.wmh_mask_path, "raw")
        mni_raw = read_image(mni_template, "float32")

        t1 = normalize_for_registration(t1_raw)
        t2 = normalize_for_registration(t2_raw)
        mni = normalize_for_registration(mni_raw)

        # Step 1: rigid FLAIR -> T1 intra-subject registration.
        # Only translation and rotation are allowed -- no scaling or shearing.
        # The two scans come from the same session, so the brains are already
        # the same size; we just need to correct for patient movement between
        # acquisitions. Mattes MI is the right choice here because T1 and
        # FLAIR have fundamentally different intensity relationships and a
        # sum-of-squared-differences metric would fail completely.
        rigid_init = center_transform(t1, t2, "rigid")
        rigid_tx = run_mi_registration(
            t1,
            t2,
            rigid_init,
            learning_rate=1.0,
            iterations=120 if fast else 200,
            shrink_factors=[4, 2, 1],
            smoothing_sigmas=[2, 1, 0],
            sampling_percentage=0.12 if fast else 0.20,
            log_file=paths["log"],
        )
        sitk.WriteTransform(rigid_tx, str(paths["t2_to_t1_tx"]))
        t2_in_t1 = resample_to_reference(t2_raw, t1_raw, rigid_tx, "linear", 0.0, sitk.sitkFloat32)
        mask_in_t1 = resample_to_reference(mask_raw, t1_raw, rigid_tx, "nearest", 0.0, sitk.sitkUInt8)
        sitk.WriteImage(t2_in_t1, str(paths["t2_in_t1"]))
        sitk.WriteImage(mask_in_t1, str(paths["mask_in_t1"]))

        # Step 2a: affine T1 -> MNI inter-subject registration.
        # This corrects for global differences in position, orientation, and
        # overall scale between the individual brain and the MNI template.
        # More pyramid levels and more iterations than step 1 because the
        # inter-subject search space is much larger -- we can't assume the
        # brains start close to aligned the way same-session scans do.
        affine_init = center_transform(mni, t1, "affine")
        affine_tx = run_mi_registration(
            mni,
            t1,
            affine_init,
            learning_rate=0.8,
            iterations=150 if fast else 250,
            shrink_factors=[6, 3, 1],
            smoothing_sigmas=[3, 1.5, 0],
            sampling_percentage=0.10 if fast else 0.18,
            log_file=paths["log"],
        )
        sitk.WriteTransform(affine_tx, str(paths["t1_to_mni_affine_tx"]))
        t1_affine_mni = resample_to_reference(t1_raw, mni_raw, affine_tx, "linear", 0.0, sitk.sitkFloat32)
        mask_affine_mni = resample_to_reference(mask_in_t1, mni_raw, affine_tx, "nearest", 0.0, sitk.sitkUInt8)
        t2_affine_mni = resample_to_reference(t2_in_t1, mni_raw, affine_tx, "linear", 0.0, sitk.sitkFloat32)
        sitk.WriteImage(t1_affine_mni, str(paths["t1_affine_mni"]))
        sitk.WriteImage(mask_affine_mni, str(paths["mask_affine_mni"]))

        # Step 2b: B-spline nonrigid refinement on top of the affine result.
        # The affine transform handles the large global shape differences;
        # B-spline handles the residual local variation -- sulcal depth,
        # ventricle size, cortical folding patterns -- that an affine model
        # cannot capture. We initialize from the already-warped affine image,
        # so the optimizer only needs to model small, local deformations
        # rather than the full registration problem from scratch.
        t1_affine_norm = normalize_for_registration(t1_affine_mni)
        bspline_tx = run_bspline_registration(
            mni,
            t1_affine_norm,
            mesh_size=6 if fast else 8,
            iterations=35 if fast else 60,
            sampling_percentage=0.08 if fast else 0.12,
            log_file=paths["log"],
        )
        sitk.WriteTransform(bspline_tx, str(paths["t1_to_mni_bspline_tx"]))
        t1_mni = resample_to_reference(t1_affine_mni, mni_raw, bspline_tx, "linear", 0.0, sitk.sitkFloat32)
        mask_mni = resample_to_reference(mask_affine_mni, mni_raw, bspline_tx, "nearest", 0.0, sitk.sitkUInt8)
        t2_mni = resample_to_reference(t2_affine_mni, mni_raw, bspline_tx, "linear", 0.0, sitk.sitkFloat32)
        sitk.WriteImage(t1_mni, str(paths["t1_mni"]))
        sitk.WriteImage(mask_mni, str(paths["wmh_mask_mni"]))
        sitk.WriteImage(t2_mni, str(paths["t2_mni"]))

        return {"subject_id": record.subject_id, "status": "OK", "wmh_mask_mni": str(final_mask), "message": ""}
    except Exception as exc:
        tb = traceback.format_exc()
        ensure_dir(paths["base"])
        paths["log"].write_text(tb, encoding="utf-8")
        return {"subject_id": record.subject_id, "status": "FAIL", "wmh_mask_mni": str(final_mask), "message": str(exc)}


def run_registration(
    records: Sequence[SubjectRecord],
    mni_template: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
    fast: bool = False,
) -> Path:
    if not mni_template.exists():
        raise FileNotFoundError(f"MNI template not found: {mni_template}")
    rows: list[dict[str, str]] = []
    for i, rec in enumerate(records, 1):
        log(f"[{i}/{len(records)}] subject {rec.subject_id}")
        row = register_one_subject(rec, mni_template, output_dir, overwrite=overwrite, fast=fast)
        rows.append(row)
        log(f"{rec.subject_id}: {row['status']}")
    log_path = ensure_dir(output_dir / "logs") / "registration_summary.csv"
    with log_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id", "status", "wmh_mask_mni", "message"])
        writer.writeheader()
        writer.writerows(rows)
    n_ok = sum(1 for r in rows if r["status"] in {"OK", "SKIP_EXISTS"})
    if n_ok == 0:
        raise RuntimeError(f"Registration failed for all subjects. See logs in {output_dir / 'registration'}")
    return log_path


# ---------------------------------------------------------------------------
# 2D slice extraction utilities and registration QC figure generation.
# Pulling a representative slice from a 3D volume sounds trivial, but the
# naive choice (anatomical midpoint) is often a blank background slice for
# small WMH masks. The middle_lesion_or_brain_slice helper picks the slice
# that sits at the median lesion voxel coordinate instead, which almost
# always lands on something informative.
# ---------------------------------------------------------------------------

def image_to_array(path: Path):
    sitk = import_sitk()
    img = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    return arr, img


def normalize_arr(arr):
    np = import_numpy()
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=float)
    lo, hi = np.nanpercentile(arr[finite], [1, 99])
    if hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def middle_lesion_or_brain_slice(arr, axis: int = 0, threshold: float = 0.5) -> int:
    np = import_numpy()
    coords = np.argwhere(np.asarray(arr) > threshold)
    if coords.size:
        return int(np.median(coords[:, axis]))
    coords = np.argwhere(np.asarray(arr) != 0)
    if coords.size:
        return int(np.median(coords[:, axis]))
    return arr.shape[axis] // 2


def take_slice(arr, axis: int, idx: int):
    np = import_numpy()
    if axis == 0:  # axial z
        return np.rot90(arr[idx, :, :])
    if axis == 1:  # coronal y
        return np.rot90(arr[:, idx, :])
    return np.rot90(arr[:, :, idx])  # sagittal x


def make_registration_qc(records: Sequence[SubjectRecord], mni_template: Path, output_dir: Path, max_subjects: int = 3) -> Path:
    np = import_numpy()
    plt = import_matplotlib()
    out = ensure_dir(output_dir / "figures") / "figure1_registration_qc.png"
    valid: list[SubjectRecord] = []
    for r in records:
        p = subject_output_paths(output_dir, r.subject_id)
        if p["t2_in_t1"].exists() and p["t1_mni"].exists() and p["wmh_mask_mni"].exists():
            valid.append(r)
    if not valid:
        raise RuntimeError("No registered subjects available for QC figure.")
    chosen = valid[:max_subjects]
    mni_arr, _ = image_to_array(mni_template)
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11, 3.5 * len(chosen)), dpi=180, squeeze=False)
    for row, r in enumerate(chosen):
        p = subject_output_paths(output_dir, r.subject_id)
        t1_arr, _ = image_to_array(r.t1_path)
        t2_t1_arr, _ = image_to_array(p["t2_in_t1"])
        t1_mni_arr, _ = image_to_array(p["t1_mni"])
        mask_mni_arr, _ = image_to_array(p["wmh_mask_mni"])

        z_native = t1_arr.shape[0] // 2
        ax = axes[row, 0]
        ax.imshow(take_slice(normalize_arr(t1_arr), 0, z_native), cmap="gray")
        ax.imshow(take_slice(normalize_arr(t2_t1_arr), 0, z_native), cmap="magma", alpha=0.35)
        ax.set_title(f"{r.subject_id}: FLAIR→T1")
        ax.axis("off")

        z_mni = mni_arr.shape[0] // 2
        ax = axes[row, 1]
        ax.imshow(take_slice(normalize_arr(mni_arr), 0, z_mni), cmap="gray")
        ax.imshow(take_slice(normalize_arr(t1_mni_arr), 0, z_mni), cmap="viridis", alpha=0.35)
        ax.set_title("T1→MNI")
        ax.axis("off")

        z_mask = middle_lesion_or_brain_slice(mask_mni_arr, 0, 0.5)
        ax = axes[row, 2]
        ax.imshow(take_slice(normalize_arr(mni_arr), 0, min(z_mask, mni_arr.shape[0] - 1)), cmap="gray")
        mask_slice = take_slice(mask_mni_arr > 0.5, 0, z_mask)
        overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
        ax.imshow(overlay, cmap="autumn", alpha=0.8)
        ax.set_title("WMH mask in MNI")
        ax.axis("off")
    fig.suptitle("Figure 1. Representative registration results", y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Population-level WMH mask stacking, spatial clustering, and regional
# proportion computation.
#
# The core idea: instead of analyzing tens of thousands of voxels
# independently, we find a small number of dominant spatial WMH patterns
# by treating each valid voxel as an observation with one feature value
# per subject (its smoothed lesion probability). Clustering those voxels
# groups together locations that tend to be involved in the same subjects,
# which often corresponds to anatomically meaningful WMH territories.
# The per-subject regional proportion is then a compact, interpretable
# summary that is far easier to correlate with clinical variables than a
# raw voxelwise map.
# ---------------------------------------------------------------------------

def registered_mask_path(output_dir: Path, subject_id: str) -> Path:
    return subject_output_paths(output_dir, subject_id)["wmh_mask_mni"]


def load_registered_mask_array(path: Path):
    arr, img = image_to_array(path)
    np = import_numpy()
    return (np.asarray(arr) > 0.5).astype(np.float32), img


def downsample_array(arr, factor: int, order: int):
    ndimage, _ = import_scipy()
    if factor <= 1:
        return arr
    zoom = [1.0 / factor] * arr.ndim
    return ndimage.zoom(arr, zoom=zoom, order=order)


def make_downsampled_reference(mni_template: Path, factor: int):
    # Downsample the MNI template to match the resolution of the cluster label
    # maps. This copy is only ever used as a background image in figures;
    # we never register to it or derive statistics from it, so a simple
    # linear zoom is perfectly adequate.
    arr, img = image_to_array(mni_template)
    if factor > 1:
        arr_ds = downsample_array(arr.astype("float32"), factor, order=1)
    else:
        arr_ds = arr
    return arr_ds, img


def save_label_image_from_array(label_arr, reference_img, downsample: int, out_path: Path) -> None:
    sitk = import_sitk()
    np = import_numpy()
    img = sitk.GetImageFromArray(np.asarray(label_arr, dtype=np.int16))
    spacing = list(reference_img.GetSpacing())
    if downsample > 1:
        spacing = [s * downsample for s in spacing]
    img.SetSpacing(tuple(spacing))
    img.SetOrigin(reference_img.GetOrigin())
    img.SetDirection(reference_img.GetDirection())
    ensure_dir(out_path.parent)
    sitk.WriteImage(img, str(out_path))


def save_float_image_from_array(arr, reference_img, downsample: int, out_path: Path) -> None:
    sitk = import_sitk()
    np = import_numpy()
    img = sitk.GetImageFromArray(np.asarray(arr, dtype=np.float32))
    spacing = list(reference_img.GetSpacing())
    if downsample > 1:
        spacing = [s * downsample for s in spacing]
    img.SetSpacing(tuple(spacing))
    img.SetOrigin(reference_img.GetOrigin())
    img.SetDirection(reference_img.GetDirection())
    ensure_dir(out_path.parent)
    sitk.WriteImage(img, str(out_path))


def plot_cluster_regions(label_map, mni_background, out_path: Path) -> Path:
    np = import_numpy()
    plt = import_matplotlib()
    ensure_dir(out_path.parent)
    labels = np.asarray(label_map)
    bg = np.asarray(mni_background)
    if labels.shape != bg.shape:
        # Integer rounding during downsampling can cause a one-voxel
        # discrepancy in one or more dimensions between the label array
        # and the background array. Rather than raising an error over a
        # single-voxel size mismatch, just crop both arrays to the
        # common (smaller) shape -- the visual difference is invisible.
        target = labels.shape
        cropped = np.zeros(target, dtype=bg.dtype)
        common = tuple(slice(0, min(target[i], bg.shape[i])) for i in range(3))
        cropped[common] = bg[common]
        bg = cropped
    coords = np.argwhere(labels > 0)
    if not coords.size:
        raise RuntimeError("Cluster label map is empty; cannot draw cluster figure.")
    center = np.median(coords, axis=0).astype(int)
    views = [
        ("Axial", 0, int(center[0])),
        ("Coronal", 1, int(center[1])),
        ("Sagittal", 2, int(center[2])),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=220)
    for ax, (name, axis, idx) in zip(axes, views):
        ax.imshow(take_slice(normalize_arr(bg), axis, idx), cmap="gray")
        lab_slice = take_slice(labels, axis, idx)
        masked = np.ma.masked_where(lab_slice == 0, lab_slice)
        ax.imshow(masked, cmap="tab20", alpha=0.78, interpolation="nearest", vmin=1, vmax=max(1, int(labels.max())))
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Figure 2. Clustered dominant WMH regions in MNI space")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def cluster_wmh_regions(
    records: Sequence[SubjectRecord],
    mni_template: Path,
    output_dir: Path,
    *,
    regions: int = 5,
    sigma: float = 1.0,
    downsample: int = 4,
    prevalence_threshold: float = 0.01,
    cluster_method: str = "kmeans",
    random_state: int = 2026,
    max_spectral_voxels: int = 15000,
) -> dict[str, Path]:
    np = import_numpy()
    ndimage, _ = import_scipy()
    MiniBatchKMeans, SpectralClustering, StandardScaler = import_sklearn()

    cluster_dir = ensure_dir(output_dir / "clusters")
    subject_ids: list[str] = []
    smoothed_features: list[Any] = []
    binary_ds_list: list[Any] = []
    reference_img = None

    for r in records:
        mask_path = registered_mask_path(output_dir, r.subject_id)
        if not mask_path.exists():
            log(f"Skipping {r.subject_id} in clustering: registered mask missing")
            continue
        binary, img = load_registered_mask_array(mask_path)
        if reference_img is None:
            reference_img = img
        smoothed = ndimage.gaussian_filter(binary.astype(np.float32), sigma=float(sigma)) if sigma > 0 else binary
        smoothed_ds = downsample_array(smoothed, downsample, order=1).astype(np.float32)
        binary_ds = downsample_array(binary, downsample, order=0).astype(np.float32)
        subject_ids.append(r.subject_id)
        smoothed_features.append(smoothed_ds)
        binary_ds_list.append((binary_ds > 0.5).astype(np.float32))

    if len(subject_ids) < 2:
        raise RuntimeError("Need at least 2 registered subjects for population clustering.")
    shapes = {tuple(a.shape) for a in smoothed_features}
    if len(shapes) != 1:
        raise RuntimeError(f"Registered masks do not have the same downsampled shape: {sorted(shapes)}")

    feature_stack = np.stack(smoothed_features, axis=0)  # (n_subjects, z, y, x)
    binary_stack = np.stack(binary_ds_list, axis=0)
    prevalence = feature_stack.mean(axis=0)
    valid = prevalence >= float(prevalence_threshold)
    n_voxels = int(valid.sum())
    if n_voxels < regions:
        raise RuntimeError(
            f"Only {n_voxels} voxels remain after prevalence threshold. "
            f"Try --prevalence-threshold 0.001 or fewer --regions."
        )

    L = feature_stack[:, valid].T  # (V, n_subjects)
    L = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
    L_scaled = StandardScaler(with_mean=True, with_std=True).fit_transform(L)

    if cluster_method == "spectral" and n_voxels <= max_spectral_voxels:
        model = SpectralClustering(
            n_clusters=int(regions),
            affinity="nearest_neighbors",
            n_neighbors=min(15, max(2, n_voxels - 1)),
            assign_labels="kmeans",
            random_state=random_state,
        )
        labels = model.fit_predict(L_scaled)
    else:
        if cluster_method == "spectral" and n_voxels > max_spectral_voxels:
            log(f"{n_voxels} valid voxels is too many for spectral; switching to MiniBatchKMeans")
        model = MiniBatchKMeans(
            n_clusters=int(regions),
            random_state=random_state,
            n_init=20,
            batch_size=min(8192, max(512, n_voxels)),
        )
        labels = model.fit_predict(L_scaled)

    label_map = np.zeros(valid.shape, dtype=np.int16)
    label_map[valid] = labels.astype(np.int16) + 1

    label_path = cluster_dir / f"cluster_labels_R{regions}_ds{downsample}.nii.gz"
    prevalence_path = cluster_dir / f"wmh_prevalence_ds{downsample}.nii.gz"
    if reference_img is None:
        raise RuntimeError("No reference image available")
    save_label_image_from_array(label_map, reference_img, downsample, label_path)
    save_float_image_from_array(prevalence, reference_img, downsample, prevalence_path)

    # Regional WMH proportion for a given subject and region:
    #   = mean of the binary mask over all voxels in that region
    #   = (lesion voxels in region) / (total voxels in region)
    # This is stored as a float in [0, 1] and serves as the feature
    # vector for downstream correlation with clinical variables.
    prop_rows: list[dict[str, Any]] = []
    for si, sid in enumerate(subject_ids):
        row: dict[str, Any] = {"subject_id": sid}
        for c in range(1, regions + 1):
            region_mask = label_map == c
            row[f"R{c}"] = float(binary_stack[si][region_mask].mean()) if region_mask.any() else float("nan")
        prop_rows.append(row)
    prop_csv = cluster_dir / "regional_wmh_proportions.csv"
    with prop_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id"] + [f"R{i}" for i in range(1, regions + 1)])
        writer.writeheader()
        writer.writerows(prop_rows)

    meta_csv = cluster_dir / "cluster_metadata.csv"
    with meta_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["region", "voxel_count", "mean_prevalence"])
        writer.writeheader()
        for c in range(1, regions + 1):
            rm = label_map == c
            writer.writerow({
                "region": f"R{c}",
                "voxel_count": int(rm.sum()),
                "mean_prevalence": float(prevalence[rm].mean()) if rm.any() else "",
            })

    mni_ds, _ = make_downsampled_reference(mni_template, downsample)
    figure_path = plot_cluster_regions(label_map, mni_ds, cluster_dir / "figure2_clustered_regions_mni.png")
    return {
        "regional_proportions": prop_csv,
        "cluster_labels": label_path,
        "prevalence_map": prevalence_path,
        "cluster_metadata": meta_csv,
        "cluster_figure": figure_path,
    }


# ---------------------------------------------------------------------------
# Phenotype table merging and correlation analysis.
# The trickiest part here is joining the regional WMH table with the
# phenotype spreadsheet, because ADNI subject identifiers appear in several
# formats: "PTID" strings like "002_S_4262" and bare numeric "RID" values
# like "4262". The infer_merge helper tries both formats automatically and
# reports a clear error if neither produces any matching rows.
# ---------------------------------------------------------------------------

def extract_rid(subject_id: str) -> str | None:
    m = re.search(r"\d{3}_S_(\d+)", str(subject_id))
    if m:
        return str(int(m.group(1)))
    nums = re.findall(r"\d+", str(subject_id))
    return str(int(nums[-1])) if nums else None


def infer_merge(region_df, pheno_df, merge_key: str | None = None):
    pd = import_pandas()
    r = region_df.copy()
    p = pheno_df.copy()
    if merge_key:
        if merge_key not in p.columns:
            raise ValueError(f"--merge-key {merge_key} not found in phenotype columns")
        if merge_key == "RID":
            r["RID"] = r["subject_id"].map(extract_rid).astype(str)
            p["RID"] = p["RID"].astype(str).str.replace(r"\.0$", "", regex=True)
            return r, p, "RID", "RID"
        return r, p, "subject_id", merge_key

    if "PTID" in p.columns:
        if set(r["subject_id"].astype(str)) & set(p["PTID"].astype(str)):
            return r, p, "subject_id", "PTID"
    if "RID" in p.columns:
        r["RID"] = r["subject_id"].map(extract_rid).astype(str)
        p["RID"] = p["RID"].astype(str).str.replace(r"\.0$", "", regex=True)
        if set(r["RID"].astype(str)) & set(p["RID"].astype(str)):
            return r, p, "RID", "RID"
    raise ValueError("Could not merge regional data with phenotype table. Use --merge-key PTID or --merge-key RID.")


def encode_variable(series):
    pd = import_pandas()
    numeric = pd.to_numeric(series, errors="coerce")
    # If at least 75% of non-missing values parse as numeric, treat the
    # entire column as continuous. The 75% threshold is a pragmatic choice:
    # it tolerates a handful of stray text annotations or note fields without
    # accidentally treating a genuinely categorical column (e.g. diagnosis
    # group) as a continuous variable.
    nonmissing = series.notna().sum()
    if nonmissing == 0:
        return numeric, None
    if numeric.notna().sum() >= max(3, int(0.75 * nonmissing)):
        return numeric, None
    text = series.astype(str).str.strip().str.lower()
    unique = set(text.dropna().unique()) - {"", "nan", "none"}
    if unique <= {"male", "female", "m", "f"}:
        encoded = text.map({"female": 0, "f": 0, "male": 1, "m": 1})
        return encoded, {0: "Female", 1: "Male"}
    codes, vals = pd.factorize(series.astype(str).replace({"nan": None}), sort=True)
    encoded = pd.Series(codes, index=series.index).replace(-1, float("nan"))
    return encoded, {i: str(v) for i, v in enumerate(vals)}


def correlation_analysis(
    regional_csv: Path,
    phenotype_xlsx: Path,
    output_dir: Path,
    *,
    x_vars: Sequence[str] = DEFAULT_X_VARS,
    merge_key: str | None = None,
) -> dict[str, Path]:
    pd = import_pandas()
    np = import_numpy()
    _, stats = import_scipy()
    plt = import_matplotlib()

    ensure_dir(output_dir)
    region_df = pd.read_csv(regional_csv)
    pheno_df = pd.read_excel(phenotype_xlsx)
    region_df, pheno_df, left_key, right_key = infer_merge(region_df, pheno_df, merge_key)
    merged = region_df.merge(pheno_df, left_on=left_key, right_on=right_key, how="left", suffixes=("", "_pheno"))
    region_cols = [c for c in region_df.columns if re.fullmatch(r"R\d+", str(c))]
    available_x = [x for x in x_vars if x in merged.columns]
    missing_x = [x for x in x_vars if x not in merged.columns]
    if missing_x:
        log(f"Phenotype variables not found and skipped: {missing_x}")
    if not available_x:
        raise RuntimeError(f"None of the requested phenotype variables were found. Columns include: {list(pheno_df.columns)[:30]}")

    corr_rows: list[dict[str, Any]] = []
    n_rows, n_cols = len(available_x), len(region_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.35 * n_rows), dpi=190, squeeze=False)
    rng = np.random.default_rng(2026)

    for i, xvar in enumerate(available_x):
        x_encoded, labels = encode_variable(merged[xvar])
        for j, reg_col in enumerate(region_cols):
            ax = axes[i, j]
            y = pd.to_numeric(merged[reg_col], errors="coerce")
            valid = x_encoded.notna() & y.notna()
            xv = x_encoded[valid].astype(float)
            yv = y[valid].astype(float)
            if len(xv) >= 3 and xv.nunique(dropna=True) > 1 and yv.nunique(dropna=True) > 1:
                r_value, p_value = stats.pearsonr(xv, yv)
            else:
                r_value, p_value = float("nan"), float("nan")

            corr_rows.append({
                "x_variable": xvar,
                "region": reg_col,
                "n": int(len(xv)),
                "pearson_r": float(r_value) if math.isfinite(float(r_value)) else "",
                "p_value": float(p_value) if math.isfinite(float(p_value)) else "",
            })

            if labels:
                xp = xv.to_numpy() + rng.normal(0, 0.035, len(xv))
                ax.scatter(xp, yv, s=28, alpha=0.82)
                ax.set_xticks(list(labels.keys()))
                ax.set_xticklabels([labels[k] for k in labels], rotation=25, ha="right")
            else:
                ax.scatter(xv, yv, s=28, alpha=0.82)
            if len(xv) >= 3 and xv.nunique(dropna=True) > 1:
                coef = np.polyfit(xv, yv, deg=1)
                xs = np.linspace(float(xv.min()), float(xv.max()), 100)
                ax.plot(xs, coef[0] * xs + coef[1], linewidth=1.1)
            title_r = "NA" if not math.isfinite(float(r_value)) else f"{r_value:.3f}"
            title_p = "NA" if not math.isfinite(float(p_value)) else f"{p_value:.3g}"
            ax.set_title(f"{xvar} vs {reg_col}\nr={title_r}, p={title_p}, n={len(xv)}")
            ax.set_xlabel(xvar)
            ax.set_ylabel(f"WMH proportion {reg_col}")
            ax.grid(True, alpha=0.25)

    fig.suptitle("Figure 3. Correlation between regional WMH lesion percentage and phenotypic data", y=1.01)
    fig.tight_layout()
    fig_path = output_dir / "figure3_correlation_grid.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    corr_csv = output_dir / "correlations.csv"
    with corr_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["x_variable", "region", "n", "pearson_r", "p_value"])
        writer.writeheader()
        writer.writerows(corr_rows)
    merged_csv = output_dir / "merged_regional_phenotype.csv"
    merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")
    return {"correlation_figure": fig_path, "correlations": corr_csv, "merged_data": merged_csv}


# ---------------------------------------------------------------------------
# Report generation and submission ZIP packaging.
# The report is Markdown rather than PDF so it can be version-controlled
# and diffed. The submission ZIP deliberately excludes raw images and
# per-subject registration folders -- those can be several gigabytes and
# are not needed by a reviewer. Only summary statistics, figures, the
# report, and source code go in.
# ---------------------------------------------------------------------------

def make_report_markdown(output_dir: Path, report_path: Path, *, regions: int, x_vars: Sequence[str]) -> Path:
    text = f"""# White Matter Hyperintensity Spatial Heterogeneity Project Report

## 1. Objective
This project analyzes topographic heterogeneity of white matter hyperintensity (WMH) lesions. T2/FLAIR-derived WMH masks are mapped into MNI space, clustered into dominant spatial regions, and summarized as regional lesion proportions for association testing with phenotype/genetic variables.

## 2. Methods

### 2.1 Registration
For each subject, the T2/FLAIR image was rigidly registered to the corresponding T1 image using Mattes mutual information, which is appropriate for multi-modality T1/FLAIR registration. The T1 image was then registered to the MNI152 T1 1 mm template using an affine stage followed by nonrigid B-spline refinement. The WMH mask was transformed to MNI space using nearest-neighbor interpolation to preserve binary lesion labels.

### 2.2 WMH region discovery
Registered MNI-space WMH masks were Gaussian-smoothed, downsampled, and stacked into a voxel-by-subject matrix. Voxels with mean WMH involvement at or above the selected threshold were retained. Retained voxels were clustered into {regions} dominant WMH regions. For each subject, the percentage of lesion within each region was calculated as the mean binary WMH value in that region.

### 2.3 Association analysis
The regional WMH proportions were merged with phenotype data. The variables used for association analysis were: {', '.join(x_vars)}. Pearson correlation coefficients and p-values were calculated for every phenotype-region pair.

## 3. Generated results

- Registration QC figure: `figures/figure1_registration_qc.png`
- Clustered dominant WMH regions: `clusters/figure2_clustered_regions_mni.png`
- Regional WMH proportions: `clusters/regional_wmh_proportions.csv`
- Correlation figure: `correlations/figure3_correlation_grid.png`
- Correlation statistics: `correlations/correlations.csv`

## 4. Discussion template
The regional features reduce voxel-level WMH maps to interpretable lesion-distribution dimensions. Significant associations between regional WMH proportions and variables such as age, APOE4, amyloid PET burden, or ADAS11 suggest that WMH topography may reflect clinically meaningful differences in vascular/neurodegenerative burden.

## 5. Reproducibility
All outputs were generated by `code/wmh_project_run_all.py`. Raw ADNI images and registered images should not be included in the final submission zip.
"""
    ensure_dir(report_path.parent)
    report_path.write_text(text, encoding="utf-8")
    return report_path


def make_submission_zip(output_dir: Path, package_root: Path, out_zip: Path) -> Path:
    """Pack the key output files into a submission ZIP.

    Raw images and per-subject registration folders are intentionally
    excluded: they can easily run to several gigabytes and a reviewer
    does not need them. Only summary CSVs, figures, the report, the
    manifest, and source code are included.
    """
    ensure_dir(out_zip.parent)
    include_paths: list[Path] = []
    for rel in [
        "clusters/regional_wmh_proportions.csv",
        "clusters/cluster_metadata.csv",
        "clusters/figure2_clustered_regions_mni.png",
        "correlations/correlations.csv",
        "correlations/figure3_correlation_grid.png",
        "figures/figure1_registration_qc.png",
        "report/WMH_project_report.md",
        "manifest.csv",
    ]:
        p = output_dir / rel
        if p.exists():
            include_paths.append(p)
    code_files = [package_root / "code" / "wmh_project_run_all.py", package_root / "requirements.txt", package_root / "README_运行说明.md", package_root / "run_windows.bat", package_root / "run_windows.ps1"]
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in include_paths:
            zf.write(p, arcname=f"results/{p.relative_to(output_dir)}")
        for p in code_files:
            if p.exists():
                zf.write(p, arcname=p.name if p.parent == package_root else str(p.relative_to(package_root)))
    return out_zip


# ---------------------------------------------------------------------------
# Top-level pipeline orchestration.
# run_all() calls each stage in sequence and threads the outputs of one
# stage into the inputs of the next. Every stage writes its results to disk
# before the next one starts, so a crash mid-run leaves the completed work
# intact and the pipeline can be resumed with --skip-registration rather
# than starting from scratch.
# ---------------------------------------------------------------------------

def prepare_records(args) -> tuple[list[SubjectRecord], Path]:
    output_dir = path_abs(args.output_dir)
    ensure_dir(output_dir)
    manifest = output_dir / "manifest.csv"
    if args.manifest:
        records = read_manifest(path_abs(args.manifest))
        write_manifest(records, manifest)
    else:
        extracted = unzip_images(path_abs(args.images_zip), output_dir / "images_extracted", overwrite=args.overwrite_extract)
        records = scan_images_root(extracted)
        write_manifest(records, manifest)
    records = validate_records(records, limit=args.limit)
    log(f"Using {len(records)} valid subjects")
    return records, manifest


def run_all(args) -> dict[str, str]:
    output_dir = path_abs(args.output_dir)
    mni_template = path_abs(args.mni_template)
    phenotype_xlsx = path_abs(args.phenotype_xlsx)
    ensure_dir(output_dir)

    records, manifest = prepare_records(args)
    params = {
        "images_zip": str(path_abs(args.images_zip)) if args.images_zip else "",
        "manifest": str(manifest),
        "mni_template": str(mni_template),
        "phenotype_xlsx": str(phenotype_xlsx),
        "output_dir": str(output_dir),
        "subjects": len(records),
        "regions": args.regions,
        "sigma": args.sigma,
        "downsample": args.downsample,
        "prevalence_threshold": args.prevalence_threshold,
        "cluster_method": args.cluster_method,
        "x_vars": args.x_vars,
        "fast": args.fast,
    }
    write_json(params, output_dir / "logs" / "run_parameters.json")

    if not args.skip_registration:
        run_registration(records, mni_template, output_dir, overwrite=args.overwrite_registration, fast=args.fast)
    else:
        log("Skipping registration because --skip-registration was set")

    qc_path = make_registration_qc(records, mni_template, output_dir, max_subjects=min(3, len(records)))
    cluster_outputs = cluster_wmh_regions(
        records,
        mni_template,
        output_dir,
        regions=args.regions,
        sigma=args.sigma,
        downsample=args.downsample,
        prevalence_threshold=args.prevalence_threshold,
        cluster_method=args.cluster_method,
        random_state=args.random_state,
    )
    corr_outputs = correlation_analysis(
        cluster_outputs["regional_proportions"],
        phenotype_xlsx,
        ensure_dir(output_dir / "correlations"),
        x_vars=args.x_vars,
        merge_key=args.merge_key,
    )
    report_path = make_report_markdown(output_dir, output_dir / "report" / "WMH_project_report.md", regions=args.regions, x_vars=args.x_vars)
    summary = {
        "manifest": str(manifest),
        "registration_qc": str(qc_path),
        "cluster_figure": str(cluster_outputs["cluster_figure"]),
        "regional_proportions": str(cluster_outputs["regional_proportions"]),
        "correlation_figure": str(corr_outputs["correlation_figure"]),
        "correlations": str(corr_outputs["correlations"]),
        "report": str(report_path),
    }
    write_json(summary, output_dir / "logs" / "output_summary.json")
    log("Pipeline completed successfully")
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


# ---------------------------------------------------------------------------
# Command-line interface -- three subcommands are exposed:
#
#   scan       unzip images.zip, detect subjects, write manifest.csv only.
#              Useful for a quick sanity check before committing to the full
#              pipeline run.
#   all        run the complete pipeline end to end.
#   correlate  re-run just the correlation step from an existing regional
#              CSV. Separating this out lets you iterate on variable
#              selection or merge-key logic without re-running registration,
#              which can take hours on a large cohort.
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Direct runnable ADNI WMH project pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--images-zip", default=DEFAULT_IMAGES_ZIP, help="Path to images.zip")
        p.add_argument("--manifest", default=None, help="Optional existing manifest CSV. If supplied, images.zip is not scanned.")
        p.add_argument("--mni-template", default=DEFAULT_MNI_TEMPLATE, help="Path to MNI152_T1_1mm.nii.gz")
        p.add_argument("--phenotype-xlsx", default=DEFAULT_PHENOTYPE_XLSX, help="Path to all30m.xlsx")
        p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output folder")
        p.add_argument("--limit", type=int, default=0, help="Use first N subjects; 0 = all")
        p.add_argument("--overwrite-extract", action="store_true", help="Re-extract images.zip")

    p_scan = sub.add_parser("scan", help="Unzip images.zip, scan subjects, and write manifest.csv")
    add_common(p_scan)

    p_all = sub.add_parser("all", help="Run full project pipeline")
    add_common(p_all)
    p_all.add_argument("--regions", type=int, default=5, help="Number of WMH clusters/regions")
    p_all.add_argument("--sigma", type=float, default=1.0, help="Gaussian smoothing sigma before downsample")
    p_all.add_argument("--downsample", type=int, default=4, help="Downsample factor: 2 is detailed, 4 is faster")
    p_all.add_argument("--prevalence-threshold", type=float, default=0.01, help="Voxel inclusion threshold across subjects")
    p_all.add_argument("--cluster-method", choices=["kmeans", "spectral"], default="kmeans")
    p_all.add_argument("--random-state", type=int, default=2026)
    p_all.add_argument("--x-vars", nargs="+", default=DEFAULT_X_VARS)
    p_all.add_argument("--merge-key", choices=["PTID", "RID"], default=None)
    p_all.add_argument("--overwrite-registration", action="store_true", help="Re-run registration even if outputs exist")
    p_all.add_argument("--skip-registration", action="store_true", help="Use existing registered masks in output-dir")
    p_all.add_argument("--fast", action="store_true", help="Faster/lighter registration settings for first test run")

    p_corr = sub.add_parser("correlate", help="Only run phenotype correlation from an existing regional CSV")
    p_corr.add_argument("--regional-csv", required=True)
    p_corr.add_argument("--phenotype-xlsx", default=DEFAULT_PHENOTYPE_XLSX)
    p_corr.add_argument("--output-dir", default=str(Path(DEFAULT_OUTPUT_DIR) / "correlations"))
    p_corr.add_argument("--x-vars", nargs="+", default=DEFAULT_X_VARS)
    p_corr.add_argument("--merge-key", choices=["PTID", "RID"], default=None)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "scan":
        records, manifest = prepare_records(args)
        log(f"Manifest written: {manifest}")
        for r in records[:10]:
            log(f"{r.subject_id}\n  T1={r.t1_path}\n  T2={r.t2_path}\n  WMH={r.wmh_mask_path}")
        if len(records) > 10:
            log(f"... {len(records) - 10} more subjects")
        return 0
    if args.command == "all":
        run_all(args)
        return 0
    if args.command == "correlate":
        outputs = correlation_analysis(
            path_abs(args.regional_csv), path_abs(args.phenotype_xlsx), path_abs(args.output_dir), x_vars=args.x_vars, merge_key=args.merge_key
        )
        log(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2, ensure_ascii=False))
        return 0
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())