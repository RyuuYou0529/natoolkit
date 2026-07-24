from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MovieInput:
    key: int
    name: str
    data: Any
    source_path: Path | None


@dataclass(frozen=True)
class MotionCorrectionParameters:
    device: str = "auto"
    nonrigid: bool = True
    nimg_init: int = 400
    batch_size: int = 100
    maxregshift: float = 0.1
    smooth_sigma: float = 1.15
    block_size: tuple[int, int] = (128, 128)
    maxregshift_nr: int = 5
    do_bidiphase: bool = False


@dataclass(frozen=True)
class ROIDetectionParameters:
    device: str = "auto"
    fs: float = 10.0
    tau: float = 1.0
    algorithm: str = "sparsery"
    diameter: tuple[float, float] = (12.0, 12.0)
    threshold_scaling: float = 1.0
    max_rois: int = 5000
    spatial_scale: int = 0
    max_overlap: float = 0.75
    nbins: int = 5000
    denoise: bool = False


@dataclass
class Suite2PSession:
    output_root: Path
    suite2p_dir: Path
    plane_dir: Path
    registered_file: Path
    movie_keys: tuple[int, ...]
    frame_ranges: dict[int, tuple[int, int]]
    shape: tuple[int, int, int]
    reg_outputs: dict[str, Any]
    settings: dict[str, Any]
    db: dict[str, Any]

    def registered_movie(self, key: int) -> np.memmap:
        start, stop = self.frame_ranges[key]
        _, height, width = self.shape
        offset = start * height * width * np.dtype(np.int16).itemsize
        return np.memmap(
            self.registered_file,
            dtype=np.int16,
            mode="r",
            offset=offset,
            shape=(stop - start, height, width),
        )


@dataclass(frozen=True)
class ROIDetectionResult:
    labels: np.ndarray
    roi_ids: set[int]
    stat_path: Path


def _load_suite2p():
    try:
        import suite2p
        import torch
    except ImportError as error:
        raise RuntimeError(
            "Suite2p is not installed. Install natoolkit with the suite2p extra."
        ) from error
    return suite2p, torch


def _torch_device(torch, requested: str):
    requested = requested.lower()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        elif mps_available:
            requested = "mps"
        else:
            requested = "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but no CUDA device is available.")
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was selected, but no MPS device is available.")
    return torch.device(requested)


def _validate_movies(movies: list[MovieInput]) -> tuple[int, int, int]:
    if not movies:
        raise ValueError("Import at least one movie before running Suite2p.")
    shapes = [tuple(movie.data.shape) for movie in movies]
    if any(len(shape) != 3 for shape in shapes):
        raise ValueError("Suite2p movies must have shape (time, y, x).")
    if any(shape[0] == 0 for shape in shapes):
        raise ValueError("Suite2p cannot process an empty movie.")
    spatial_shapes = {shape[-2:] for shape in shapes}
    if len(spatial_shapes) != 1:
        raise ValueError("All Suite2p movies must have the same Y/X shape.")
    total_frames = sum(shape[0] for shape in shapes)
    if total_frames < 10:
        raise ValueError("Suite2p requires at least 10 total frames.")
    height, width = shapes[0][-2:]
    return total_frames, height, width


def _int16_frames(frames: Any) -> np.ndarray:
    frames = np.asarray(frames)
    if frames.dtype == np.uint16 or frames.dtype == np.int32:
        frames = frames // 2
    return np.clip(frames, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(
        np.int16,
        copy=False,
    )


def _prepare_output(
    output_root: Path,
    replace_existing: bool,
) -> tuple[Path, Path]:
    output_root = output_root.expanduser().resolve()
    suite2p_dir = output_root / "suite2p"
    if suite2p_dir.exists() and any(suite2p_dir.iterdir()):
        if not replace_existing:
            raise FileExistsError(
                f"{suite2p_dir} is not empty. Replacement was not confirmed."
            )
        shutil.rmtree(suite2p_dir)
    plane_dir = suite2p_dir / "plane0"
    plane_dir.mkdir(parents=True, exist_ok=True)
    return suite2p_dir, plane_dir


def run_motion_correction(
    movies: list[MovieInput],
    output_root: Path,
    parameters: MotionCorrectionParameters,
    replace_existing: bool = False,
) -> Suite2PSession:
    suite2p, torch = _load_suite2p()
    total_frames, height, width = _validate_movies(movies)
    suite2p_dir, plane_dir = _prepare_output(output_root, replace_existing)
    registered_file = plane_dir / "data.bin"
    registered = np.memmap(
        registered_file,
        dtype=np.int16,
        mode="w+",
        shape=(total_frames, height, width),
    )

    frame_ranges: dict[int, tuple[int, int]] = {}
    frame_counts = []
    start = 0
    for movie in movies:
        frame_count = int(movie.data.shape[0])
        stop = start + frame_count
        frame_ranges[movie.key] = (start, stop)
        frame_counts.append(frame_count)
        for batch_start in range(0, frame_count, parameters.batch_size):
            batch_stop = min(batch_start + parameters.batch_size, frame_count)
            registered[start + batch_start : start + batch_stop] = _int16_frames(
                movie.data[batch_start:batch_stop]
            )
        start = stop
    registered.flush()

    settings = suite2p.default_settings()
    device = _torch_device(torch, parameters.device)
    settings["torch_device"] = str(device)
    settings["run"]["do_detection"] = False
    settings["run"]["do_deconvolution"] = False
    settings["run"]["do_regmetrics"] = True
    settings["registration"].update(
        {
            "nonrigid": parameters.nonrigid,
            "nimg_init": parameters.nimg_init,
            "batch_size": parameters.batch_size,
            "maxregshift": parameters.maxregshift,
            "smooth_sigma": parameters.smooth_sigma,
            "block_size": parameters.block_size,
            "maxregshiftNR": parameters.maxregshift_nr,
            "do_bidiphase": parameters.do_bidiphase,
            "reg_tif": False,
            "reg_tif_chan2": False,
            "two_step_registration": False,
        }
    )
    reg_outputs = suite2p.registration_wrapper(
        registered,
        save_path=str(plane_dir),
        settings=settings["registration"],
        device=device,
    )
    registered.flush()
    if settings["run"]["do_regmetrics"] and total_frames >= 1500:
        tpc, regpc, regdx = suite2p.registration.get_pc_metrics(
            registered,
            yrange=reg_outputs["yrange"],
            xrange=reg_outputs["xrange"],
            settings=settings["registration"],
            device=device,
        )
        reg_outputs.update(tPC=tpc, regPC=regpc, regDX=regdx)

    source_paths = [str(movie.source_path) for movie in movies if movie.source_path]
    db = {
        "data_path": sorted({str(Path(path).parent) for path in source_paths}),
        "file_list": source_paths,
        "save_path0": str(output_root.expanduser().resolve()),
        "save_folder": "suite2p",
        "save_path": str(plane_dir),
        "reg_file": str(registered_file),
        "nplanes": 1,
        "nchannels": 1,
        "keep_movie_raw": False,
        "nframes": total_frames,
        "Ly": height,
        "Lx": width,
        "frames_per_file": np.asarray(frame_counts, dtype=int),
        "activity_tracer_movies": [movie.name for movie in movies],
        "activity_tracer_movie_keys": [movie.key for movie in movies],
    }
    np.save(suite2p_dir / "db.npy", db)
    np.save(suite2p_dir / "settings.npy", settings)
    np.save(plane_dir / "db.npy", db)
    np.save(plane_dir / "settings.npy", settings)
    np.save(plane_dir / "reg_outputs.npy", reg_outputs)
    np.save(plane_dir / "ops.npy", {**db, **settings, **reg_outputs})

    return Suite2PSession(
        output_root=output_root.expanduser().resolve(),
        suite2p_dir=suite2p_dir,
        plane_dir=plane_dir,
        registered_file=registered_file,
        movie_keys=tuple(movie.key for movie in movies),
        frame_ranges=frame_ranges,
        shape=(total_frames, height, width),
        reg_outputs=reg_outputs,
        settings=settings,
        db=db,
    )


def labels_from_stats(
    stats: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, set[int]]:
    dtype = np.uint32 if len(stats) >= np.iinfo(np.uint16).max else np.uint16
    labels = np.zeros(shape, dtype=dtype)
    ids: set[int] = set()
    for index, stat in enumerate(stats):
        ypix = np.asarray(stat["ypix"], dtype=int)
        xpix = np.asarray(stat["xpix"], dtype=int)
        overlap = np.asarray(stat.get("overlap", np.zeros(ypix.size, bool)), dtype=bool)
        valid = (
            (ypix >= 0)
            & (ypix < shape[0])
            & (xpix >= 0)
            & (xpix < shape[1])
            & ~overlap
        )
        ypix, xpix = ypix[valid], xpix[valid]
        unassigned = labels[ypix, xpix] == 0
        ypix, xpix = ypix[unassigned], xpix[unassigned]
        if ypix.size:
            roi_id = index + 1
            labels[ypix, xpix] = roi_id
            ids.add(roi_id)
    return labels, ids


def run_roi_detection(
    session: Suite2PSession,
    parameters: ROIDetectionParameters,
    movie_key: int | None = None,
) -> ROIDetectionResult:
    suite2p, torch = _load_suite2p()
    settings = deepcopy(session.settings)
    device = _torch_device(torch, parameters.device)
    settings["torch_device"] = str(device)
    settings["fs"] = parameters.fs
    settings["tau"] = parameters.tau
    settings["diameter"] = list(parameters.diameter)
    settings["run"]["do_detection"] = True
    settings["run"]["do_deconvolution"] = False
    settings["detection"].update(
        {
            "algorithm": parameters.algorithm,
            "denoise": parameters.denoise,
            "nbins": parameters.nbins,
            "threshold_scaling": parameters.threshold_scaling,
            "max_overlap": parameters.max_overlap,
        }
    )
    settings["detection"]["sparsery_settings"].update(
        {
            "max_ROIs": parameters.max_rois,
            "spatial_scale": parameters.spatial_scale,
        }
    )
    if movie_key is None:
        registered = np.memmap(
            session.registered_file,
            dtype=np.int16,
            mode="r",
            shape=session.shape,
        )
        badframes = session.reg_outputs.get("badframes")
    else:
        registered = session.registered_movie(movie_key)
        start, stop = session.frame_ranges[movie_key]
        badframes = session.reg_outputs.get("badframes")
        if badframes is not None:
            badframes = badframes[start:stop]
    detect_outputs, stats, _redcell = suite2p.detection_wrapper(
        registered,
        diameter=list(parameters.diameter),
        tau=parameters.tau,
        fs=parameters.fs,
        yrange=session.reg_outputs.get("yrange"),
        xrange=session.reg_outputs.get("xrange"),
        badframes=badframes,
        preclassify=0.0,
        settings=settings["detection"],
        device=device,
    )
    stat_path = session.plane_dir / "stat.npy"
    np.save(stat_path, stats)
    np.save(session.plane_dir / "detect_outputs.npy", detect_outputs)
    np.save(session.plane_dir / "settings.npy", settings)
    np.save(session.suite2p_dir / "settings.npy", settings)
    np.save(
        session.plane_dir / "ops.npy",
        {**session.db, **settings, **session.reg_outputs, **detect_outputs},
    )
    labels, ids = labels_from_stats(stats, registered.shape[-2:])
    return ROIDetectionResult(labels=labels, roi_ids=ids, stat_path=stat_path)


def run_roi_detection_on_movie(
    movie: MovieInput,
    output_root: Path,
    parameters: ROIDetectionParameters,
    replace_existing: bool = False,
) -> ROIDetectionResult:
    suite2p, _torch = _load_suite2p()
    frame_count, height, width = _validate_movies([movie])
    suite2p_dir, plane_dir = _prepare_output(output_root, replace_existing)
    registered_file = plane_dir / "data.bin"
    registered = np.memmap(
        registered_file,
        dtype=np.int16,
        mode="w+",
        shape=(frame_count, height, width),
    )
    for start in range(0, frame_count, 100):
        stop = min(start + 100, frame_count)
        registered[start:stop] = _int16_frames(movie.data[start:stop])
    registered.flush()
    del registered

    settings = suite2p.default_settings()
    settings["run"]["do_registration"] = False
    settings["run"]["do_detection"] = True
    settings["run"]["do_deconvolution"] = False
    source_path = str(movie.source_path) if movie.source_path is not None else ""
    db = {
        "data_path": [str(movie.source_path.parent)] if movie.source_path else [],
        "file_list": [source_path] if source_path else [],
        "save_path0": str(output_root.expanduser().resolve()),
        "save_folder": "suite2p",
        "save_path": str(plane_dir),
        "reg_file": str(registered_file),
        "nplanes": 1,
        "nchannels": 1,
        "keep_movie_raw": False,
        "nframes": frame_count,
        "Ly": height,
        "Lx": width,
        "frames_per_file": np.asarray([frame_count], dtype=int),
        "activity_tracer_movies": [movie.name],
        "activity_tracer_movie_keys": [movie.key],
    }
    np.save(suite2p_dir / "db.npy", db)
    np.save(suite2p_dir / "settings.npy", settings)
    np.save(plane_dir / "db.npy", db)
    np.save(plane_dir / "settings.npy", settings)
    np.save(plane_dir / "ops.npy", {**db, **settings})

    session = Suite2PSession(
        output_root=output_root.expanduser().resolve(),
        suite2p_dir=suite2p_dir,
        plane_dir=plane_dir,
        registered_file=registered_file,
        movie_keys=(movie.key,),
        frame_ranges={movie.key: (0, frame_count)},
        shape=(frame_count, height, width),
        reg_outputs={},
        settings=settings,
        db=db,
    )
    return run_roi_detection(session, parameters, movie.key)
