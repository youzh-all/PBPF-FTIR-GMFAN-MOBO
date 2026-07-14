from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FTIR_RANGE = (400.0, 4000.0)
DSC_RANGE = (-80.0, 200.0)
ALCOHOL_RATIOS = (73, 82, 91)
CNC_TYPES = ("Pure", "A", "P", "S")
CNC_CONTENTS = (0.0, 0.05, 0.1, 0.2)

_FILENAME_RE = re.compile(r"^\s*(?P<sample_no>\d+)\s*\((?P<composition>PBPF-[^)]+)\)_(?P<suffix>[^.]+)\.csv$")
_COMPOSITION_RE = re.compile(r"^PBPF-(?P<ratio>\d+)(?:(?P<ctype>[APS])-(?P<content>\d+(?:\.\d+)?))?$")


@dataclass(frozen=True)
class SampleMetadata:
    composition_id: str
    alcohol_ratio: int
    cnc_type: str
    cnc_content: float


@dataclass
class CurvePCAProjector:
    n_components: int
    pca: Optional[PCA] = None

    def fit(self, curves: np.ndarray) -> "CurvePCAProjector":
        arr = np.asarray(curves, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("curves must be a 2D array.")
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(arr)
        return self

    def transform(self, curves: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA projector is not fitted.")
        return self.pca.transform(np.asarray(curves, dtype=np.float32)).astype(np.float32)

    def inverse_transform(self, coeffs: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA projector is not fitted.")
        return self.pca.inverse_transform(np.asarray(coeffs, dtype=np.float32)).astype(np.float32)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("PCA projector is not fitted.")
        return self.pca.explained_variance_ratio_

    def save(self, path: str | Path) -> None:
        target = resolve_project_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "CurvePCAProjector":
        source = resolve_project_path(path)
        with source.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}.")
        return obj


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def list_csv_files(directory: str | Path) -> list[Path]:
    root = resolve_project_path(directory)
    return sorted(path for path in root.glob("*.csv") if not path.name.startswith("."))


def parse_filename(path_or_name: str | Path) -> dict[str, object]:
    name = Path(path_or_name).name
    match = _FILENAME_RE.match(name)
    if not match:
        raise ValueError(f"Cannot parse filename: {name}")
    suffix = match.group("suffix")
    repeat_id = int(suffix) if suffix.isdigit() else None
    return {
        "sample_number": int(match.group("sample_no")),
        "composition_id": match.group("composition"),
        "suffix": suffix,
        "repeat_id": repeat_id,
    }


def parse_composition_id(composition_id: str) -> SampleMetadata:
    match = _COMPOSITION_RE.match(composition_id)
    if not match:
        raise ValueError(f"Cannot parse composition_id: {composition_id}")
    alcohol_ratio = int(match.group("ratio"))
    cnc_type = match.group("ctype") or "Pure"
    cnc_content = float(match.group("content")) if match.group("content") is not None else 0.0
    return SampleMetadata(
        composition_id=composition_id,
        alcohol_ratio=alcohol_ratio,
        cnc_type=cnc_type,
        cnc_content=cnc_content,
    )


def encode_metadata(alcohol_ratio: int, cnc_type: str, cnc_content: float) -> np.ndarray:
    vector = np.zeros(len(ALCOHOL_RATIOS) + len(CNC_TYPES) + len(CNC_CONTENTS), dtype=np.float32)
    vector[ALCOHOL_RATIOS.index(alcohol_ratio)] = 1.0
    vector[len(ALCOHOL_RATIOS) + CNC_TYPES.index(cnc_type)] = 1.0
    vector[len(ALCOHOL_RATIOS) + len(CNC_TYPES) + CNC_CONTENTS.index(float(cnc_content))] = 1.0
    return vector


def _read_csv_with_fallback(path: Path, encodings: Sequence[str]) -> pd.DataFrame:
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


def _safe_interp(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, left: Optional[float] = None, right: Optional[float] = None) -> np.ndarray:
    if x.size == 0 or y.size == 0:
        raise ValueError("Interpolation requires non-empty x and y.")
    if x.size == 1:
        fill = float(y[0])
        return np.full_like(x_new, fill, dtype=np.float32)
    f = interp1d(x, y, kind="linear", bounds_error=False, fill_value=(left if left is not None else y[0], right if right is not None else y[-1]), assume_sorted=True)
    return np.asarray(f(x_new), dtype=np.float32)


def _asls_baseline(y: np.ndarray, lam: float, p: float, niter: int) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    n = int(arr.size)
    if n < 3:
        return np.zeros_like(arr)
    diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
    dtd = (diff.T @ diff).tocsc()
    w = np.ones(n, dtype=np.float64)
    baseline = np.zeros(n, dtype=np.float64)
    for _ in range(max(1, int(niter))):
        w_mat = sparse.diags(w, 0, shape=(n, n), format="csc")
        baseline = spsolve(w_mat + float(lam) * dtd, w * arr)
        w = np.where(arr > baseline, float(p), 1.0 - float(p))
    return np.asarray(baseline, dtype=np.float64)


def _apply_msc(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
    x = np.asarray(signal, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if x.shape != ref.shape:
        raise ValueError("MSC requires signal and reference with identical shape.")
    a = np.vstack([np.ones_like(ref), ref]).T
    coeffs, *_ = np.linalg.lstsq(a, x, rcond=None)
    intercept = float(coeffs[0])
    slope = float(coeffs[1])
    if abs(slope) < 1e-12:
        return (x - intercept).astype(np.float32)
    return ((x - intercept) / slope).astype(np.float32)


class FTIRPreprocessor:
    def __init__(
        self,
        points: int = 1800,
        wavenumber_range: tuple[float, float] = FTIR_RANGE,
        smoothing_window: int = 11,
        smoothing_polyorder: int = 2,
        baseline_degree: int = 1,
        asls_lambda: float = 1e6,
        asls_p: float = 0.01,
        asls_niter: int = 10,
        msc_reference_path: str | Path = "data_processed/ftir_msc_reference_1800.npy",
        msc_reference_dir: str | Path = "data_csv/FTIR",
        save_msc_reference: bool = True,
    ) -> None:
        self.points = points
        self.wavenumber_range = wavenumber_range
        self.wavenumber_grid = np.linspace(wavenumber_range[0], wavenumber_range[1], points, dtype=np.float32)
        # Kept for backward compatibility with scripts that still pass these args.
        self.smoothing_window = smoothing_window
        self.smoothing_polyorder = smoothing_polyorder
        self.baseline_degree = baseline_degree
        self.asls_lambda = float(asls_lambda)
        self.asls_p = float(asls_p)
        self.asls_niter = int(asls_niter)
        self.msc_reference_path = resolve_project_path(msc_reference_path)
        self.msc_reference_dir = str(msc_reference_dir)
        self.save_msc_reference = bool(save_msc_reference)
        self._msc_reference: Optional[np.ndarray] = None

    def load_raw(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        source = resolve_project_path(path)
        df = pd.read_csv(source, encoding="latin-1")
        lower_map = {col.lower().strip(): col for col in df.columns}
        w_col = lower_map.get("wavenumber")
        t_col = lower_map.get("transmittance")
        if w_col is None or t_col is None:
            raise ValueError(f"FTIR file missing required columns: {source}")
        w = pd.to_numeric(df[w_col], errors="coerce").to_numpy(dtype=float)
        t = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(w) & np.isfinite(t)
        w, t = w[mask], t[mask]
        order = np.argsort(w)
        w, t = w[order], t[order]
        low, high = self.wavenumber_range
        window = (w >= low) & (w <= high)
        w, t = w[window], t[window]
        if w.size == 0:
            raise ValueError(f"No FTIR points remained after range filter: {source}")
        return w.astype(np.float32), t.astype(np.float32)

    def _to_absorbance(self, w: np.ndarray, t: np.ndarray) -> np.ndarray:
        t_resampled = _safe_interp(w, t, self.wavenumber_grid)
        t_clipped = np.clip(t_resampled, 0.001, 1.0)
        return -np.log10(t_clipped).astype(np.float32)

    def _asls_correct(self, absorbance: np.ndarray) -> np.ndarray:
        baseline = _asls_baseline(absorbance, lam=self.asls_lambda, p=self.asls_p, niter=self.asls_niter)
        corrected = np.asarray(absorbance, dtype=np.float64) - baseline
        return corrected.astype(np.float32)

    def _load_reference_from_disk(self) -> Optional[np.ndarray]:
        path = self.msc_reference_path
        if not path.exists():
            return None
        arr = np.load(path)
        ref = np.asarray(arr, dtype=np.float32).reshape(-1)
        if ref.shape[0] != self.points:
            return None
        return ref

    def _build_reference(self) -> np.ndarray:
        spectra: list[np.ndarray] = []
        for csv_path in list_csv_files(self.msc_reference_dir):
            w, t = self.load_raw(csv_path)
            absorbance = self._to_absorbance(w, t)
            corrected = self._asls_correct(absorbance)
            spectra.append(corrected.astype(np.float32))
        if not spectra:
            raise ValueError(f"No FTIR spectra available for MSC reference: {self.msc_reference_dir}")
        ref = np.mean(np.vstack(spectra), axis=0).astype(np.float32)
        return ref

    def _get_reference(self) -> np.ndarray:
        if self._msc_reference is not None:
            return self._msc_reference
        loaded = self._load_reference_from_disk()
        if loaded is None:
            loaded = self._build_reference()
            if self.save_msc_reference:
                self.msc_reference_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(self.msc_reference_path, loaded.astype(np.float32))
        self._msc_reference = loaded.astype(np.float32)
        return self._msc_reference

    def transform_stages(self, path: str | Path) -> dict[str, np.ndarray]:
        w, t = self.load_raw(path)
        absorbance = self._to_absorbance(w, t)
        asls_corrected = self._asls_correct(absorbance)
        msc_reference = self._get_reference()
        msc_corrected = _apply_msc(asls_corrected, msc_reference)
        return {
            "wavenumber": self.wavenumber_grid.copy(),
            "raw_transmittance": _safe_interp(w, t, self.wavenumber_grid),
            "absorbance": absorbance.astype(np.float32),
            "asls_corrected": asls_corrected.astype(np.float32),
            "msc_corrected": np.asarray(msc_corrected, dtype=np.float32),
        }

    def preprocess(self, path: str | Path) -> np.ndarray:
        stages = self.transform_stages(path)
        return stages["msc_corrected"].astype(np.float32)


class DSCPreprocessor:
    def __init__(self, points: int = 560, temperature_range: tuple[float, float] = DSC_RANGE) -> None:
        self.points = points
        self.temperature_range = temperature_range
        self.temperature_grid = np.linspace(temperature_range[0], temperature_range[1], points, dtype=np.float32)

    def load_raw(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        source = resolve_project_path(path)
        df = _read_csv_with_fallback(source, ("utf-8", "utf-8-sig", "latin-1"))
        cols = {re.sub(r"[^a-z]+", "", col.lower()): col for col in df.columns}
        t_col = next((value for key, value in cols.items() if "temperature" in key), None)
        h_col = next((value for key, value in cols.items() if "heatflow" in key), None)
        if t_col is None or h_col is None:
            raise ValueError(f"DSC file missing required columns: {source}")
        temp = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
        heat = pd.to_numeric(df[h_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(temp) & np.isfinite(heat)
        temp, heat = temp[mask], heat[mask]
        order = np.argsort(temp)
        temp, heat = temp[order], heat[order]
        temp_unique, inverse = np.unique(temp, return_inverse=True)
        if temp_unique.size != temp.size:
            aggregated = np.zeros_like(temp_unique, dtype=float)
            counts = np.zeros_like(temp_unique, dtype=float)
            np.add.at(aggregated, inverse, heat)
            np.add.at(counts, inverse, 1.0)
            heat = aggregated / np.maximum(counts, 1.0)
            temp = temp_unique
        low, high = self.temperature_range
        window = (temp >= low) & (temp <= high)
        temp, heat = temp[window], heat[window]
        if temp.size == 0:
            raise ValueError(f"No DSC points remained after range filter: {source}")
        return temp.astype(np.float32), heat.astype(np.float32)

    def preprocess(self, path: str | Path) -> np.ndarray:
        temp, heat = self.load_raw(path)
        return _safe_interp(temp, heat, self.temperature_grid).astype(np.float32)


@lru_cache(maxsize=1)
def _cached_global_max_strain_percent(utm_dir: str) -> float:
    max_strain = 0.0
    for path in list_csv_files(utm_dir):
        df = _read_csv_with_fallback(path, ("utf-8", "utf-8-sig", "latin-1"))
        cols = {re.sub(r"[^a-z%]+", "", col.lower()): col for col in df.columns}
        strain_col = next((value for key, value in cols.items() if "strain" in key), None)
        if strain_col is None:
            continue
        strain = pd.to_numeric(df[strain_col], errors="coerce").to_numpy(dtype=float)
        strain = strain[np.isfinite(strain)]
        if strain.size:
            max_strain = max(max_strain, float(np.nanmax(strain)))
    if max_strain <= 0:
        raise ValueError("Failed to determine global max strain from UTM_100.")
    return max_strain


class UTMPreprocessor:
    def __init__(self, points: int = 1000, utm_dir: str | Path = "data_csv/UTM_100", global_max_strain_percent: Optional[float] = None) -> None:
        self.points = points
        self.utm_dir = str(resolve_project_path(utm_dir))
        self.global_max_strain_percent = (
            float(global_max_strain_percent)
            if global_max_strain_percent is not None
            else _cached_global_max_strain_percent(self.utm_dir)
        )
        self.strain_grid = np.linspace(0.0, self.global_max_strain_percent, points, dtype=np.float32)

    def load_raw(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        source = resolve_project_path(path)
        df = _read_csv_with_fallback(source, ("utf-8", "utf-8-sig", "latin-1"))
        cols = {re.sub(r"[^a-z%]+", "", col.lower()): col for col in df.columns}
        stress_col = next((value for key, value in cols.items() if "stress" in key), None)
        strain_col = next((value for key, value in cols.items() if "strain" in key), None)
        if stress_col is None or strain_col is None:
            raise ValueError(f"UTM file missing required columns: {source}")
        stress = pd.to_numeric(df[stress_col], errors="coerce").to_numpy(dtype=float)
        strain = pd.to_numeric(df[strain_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(stress) & np.isfinite(strain)
        stress, strain = stress[mask], strain[mask]
        nonneg = strain >= 0.0
        stress, strain = stress[nonneg], strain[nonneg]
        order = np.argsort(strain)
        strain, stress = strain[order], stress[order]
        strain_unique, inverse = np.unique(strain, return_inverse=True)
        if strain_unique.size != strain.size:
            aggregated = np.zeros_like(strain_unique, dtype=float)
            counts = np.zeros_like(strain_unique, dtype=float)
            np.add.at(aggregated, inverse, stress)
            np.add.at(counts, inverse, 1.0)
            stress = aggregated / np.maximum(counts, 1.0)
            strain = strain_unique
        if strain.size == 0:
            raise ValueError(f"No valid UTM points remained after cleanup: {source}")
        return strain.astype(np.float32), stress.astype(np.float32)

    def preprocess(self, path: str | Path) -> np.ndarray:
        strain, stress = self.load_raw(path)
        return _safe_interp(strain, stress, self.strain_grid, left=stress[0], right=stress[-1]).astype(np.float32)


def extract_dsc_scalars(heating_curve: np.ndarray, cooling_curve: np.ndarray, dsc_preprocessor: Optional[DSCPreprocessor] = None) -> np.ndarray:
    prep = dsc_preprocessor or DSCPreprocessor(points=len(heating_curve))
    if len(prep.temperature_grid) != len(heating_curve):
        raise ValueError("DSC curve length must match the preprocessor grid length.")
    tm = float(prep.temperature_grid[int(np.nanargmax(heating_curve))])
    tc = float(prep.temperature_grid[int(np.nanargmin(cooling_curve))])
    return np.asarray([tm, tc], dtype=np.float32)


def _safe_linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2 or np.ptp(x) <= 1e-12:
        return 0.0, float(np.nanmean(y))
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def extract_utm_scalars_from_raw(strain_percent: np.ndarray, stress_mpa: np.ndarray, drop_threshold: float = 0.3) -> np.ndarray:
    if strain_percent.size == 0 or stress_mpa.size == 0:
        return np.asarray([np.nan, np.nan, np.nan], dtype=np.float32)

    uts_idx = int(np.nanargmax(stress_mpa))
    tensile_strength = float(stress_mpa[uts_idx])

    strain_ratio = strain_percent / 100.0
    nonneg_idx = np.where(strain_ratio >= 0.0)[0]
    start = int(nonneg_idx[0]) if nonneg_idx.size else 0
    x0 = strain_ratio[start:]
    y0 = stress_mpa[start:]
    upper_mask = x0 <= 0.02
    x_search = x0[upper_mask] if np.sum(upper_mask) >= 8 else x0[: max(8, min(len(x0), 20))]
    y_search = y0[: len(x_search)]
    slope, intercept = _safe_linear_fit(x_search, y_search)
    offset_line = slope * (strain_ratio - 0.002) + intercept
    diff = stress_mpa - offset_line
    candidates = np.where(strain_ratio >= 0.002)[0]
    yield_strength = np.nan
    if candidates.size >= 2 and np.isfinite(slope) and slope > 0:
        for i in candidates[1:]:
            j = i - 1
            if diff[j] > 0 and diff[i] <= 0:
                x1, x2 = strain_ratio[j], strain_ratio[i]
                d1, d2 = diff[j], diff[i]
                if abs(d2 - d1) < 1e-12:
                    y_strain = x1
                else:
                    y_strain = x1 + (0.0 - d1) * (x2 - x1) / (d2 - d1)
                yield_strength = float(np.interp(y_strain, strain_ratio, stress_mpa))
                break
        if not np.isfinite(yield_strength):
            idx_closest = int(candidates[np.argmin(np.abs(diff[candidates]))])
            yield_strength = float(stress_mpa[idx_closest])

    # Project rule: use the strain at the UTS point as elongation-at-break.
    elongation_at_break = float(strain_percent[uts_idx])

    return np.asarray([tensile_strength, yield_strength, elongation_at_break], dtype=np.float32)


def fit_curve_pca(curves: Iterable[np.ndarray], n_components: int) -> CurvePCAProjector:
    stacked = np.stack([np.asarray(curve, dtype=np.float32) for curve in curves], axis=0)
    return CurvePCAProjector(n_components=n_components).fit(stacked)
