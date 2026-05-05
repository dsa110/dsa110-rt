"""tools/viz/common.py — shared utilities for the M2/M3 voltage-fixture viz tools.

Plan §8 line 2172 (M2) + §8 line 2197-2206 (M3). M2 only needs the continuum
slow-corr check; burst / per-chgroup / stage-2-alignment helpers defer to M3.

The viz path operates on either:
  * **UVH5** files produced by `meridian_fringestop` (operator-facing path
    on real continuum fixtures), loaded via h5py to keep the viz tool in
    the `dsa110-rt` conda env (no pyuvdata dep — D13).
  * **Raw bada captures** (`bada_capture.bin`) produced by
    `bench/voltage_fixture_slow_corr.py --skip-meridian` (synthetic /
    smoke-test path; no UVH5 round-trip required).

Both paths land at the same `(NBASE, NCHAN, NPOL)` complex-vis cube +
`(NBASE, 3)` metres-uvw + `(NCHAN,)` Hz freq axis, and feed the same
gridder + iFFT for dirty-image rendering.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dsart.common.constants import (  # noqa: E402
    BADA_NPOL,
    NANTS,
    NBASE,
    NCHAN_PER_CHGROUP,
)

SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


# ---------------------------------------------------------------------------
# Visibility cube container
# ---------------------------------------------------------------------------


@dataclass
class VisCube:
    """Continuum-averaged visibility cube + geometry.

    vis : complex64, shape (NBASE, NCHAN, NPOL)
        Time-averaged visibilities (continuum). One block average ok.
    uvw_m : float64, shape (NBASE, 3)
        Baseline (u, v, w) in metres at the phase centre. East-North-Up.
        Convention: uvw[i] is the displacement (b_a - b_b) for baseline a*(a+1)/2 + b
        (a > b, lower-triangle). Auto-correlation uvw entries are (0, 0, 0).
    freqs_Hz : float64, shape (NCHAN,)
        Channel centre frequencies.
    nant : int
        Number of antennas (96 for DSA-110).
    source_name : str
        Provenance label (e.g. fixture run-id or "SYNTH").
    """
    vis: np.ndarray
    uvw_m: np.ndarray
    freqs_Hz: np.ndarray
    nant: int
    source_name: str = "?"

    def __post_init__(self) -> None:
        if self.vis.ndim != 3:
            raise ValueError(f"vis must be 3D, got {self.vis.ndim}D")
        nbase, nchan, npol = self.vis.shape
        if self.uvw_m.shape != (nbase, 3):
            raise ValueError(f"uvw_m shape {self.uvw_m.shape} != ({nbase}, 3)")
        if self.freqs_Hz.shape != (nchan,):
            raise ValueError(f"freqs_Hz shape {self.freqs_Hz.shape} != ({nchan},)")


# ---------------------------------------------------------------------------
# Antenna positions + uvw helpers
# ---------------------------------------------------------------------------


def antpos_synth_2d_grid(
    nants: int = NANTS, n_x: int = 12, n_y: int = 8,
    spacing_m: float = 0.5,
) -> np.ndarray:
    """Synthetic 2D antenna grid (matches replay_voltage_dump default).

    96 ants in a 12×8 grid with 0.5 m spacing in both x (east) and y
    (north) — gives 2D uv coverage so dirty-image peaks are not
    degenerate along the v=0 axis.
    """
    if n_x * n_y != nants:
        raise ValueError(f"{n_x} × {n_y} = {n_x * n_y} != nants={nants}")
    pos = np.zeros((nants, 3), dtype=np.float64)
    for k in range(nants):
        ix = k % n_x
        iy = k // n_x
        pos[k, 0] = spacing_m * ix
        pos[k, 1] = spacing_m * iy
    return pos


# Backward-compat alias.
antpos_linear_ew = antpos_synth_2d_grid


def upper_tri_indices(nants: int = NANTS) -> tuple[np.ndarray, np.ndarray]:
    """Same convention as `slow_corr_kernel.upper_tri_indices` (xGPU upper-tri)."""
    nbase = nants * (nants + 1) // 2
    a_idx = np.empty(nbase, dtype=np.int64)
    b_idx = np.empty(nbase, dtype=np.int64)
    k = 0
    for a in range(nants):
        for b in range(a + 1):
            a_idx[k] = a
            b_idx[k] = b
            k += 1
    return a_idx, b_idx


def uvw_from_antpos(antpos_m: np.ndarray) -> np.ndarray:
    """Compute baseline displacements (b_a - b_b) for the upper-tri ordering.

    For a phased-up array at zenith on a flat array, uvw at the phase centre
    is just the per-baseline antenna displacement vector (no W-projection).
    """
    nants = antpos_m.shape[0]
    a_idx, b_idx = upper_tri_indices(nants)
    return antpos_m[a_idx] - antpos_m[b_idx]                       # (NBASE, 3)


def channel_freqs_hz(
    nchan: int = NCHAN_PER_CHGROUP,
    nu_top_GHz: float = 1.5,
    nu_bot_GHz: float = 1.45,
) -> np.ndarray:
    """Decreasing per dsa convention; matches replay_voltage_dump default."""
    return np.linspace(nu_top_GHz, nu_bot_GHz, nchan) * 1e9


# ---------------------------------------------------------------------------
# Loaders: bada bin (synth/smoke), UVH5 (operator-facing)
# ---------------------------------------------------------------------------


def load_bada_capture(
    path: Path | str,
    *,
    nbase: int = NBASE,
    nchan: int = NCHAN_PER_CHGROUP,
    npol: int = BADA_NPOL,
    antpos_m: np.ndarray | None = None,
    freqs_Hz: np.ndarray | None = None,
    source_name: str = "bada_capture",
) -> VisCube:
    """Load `bada_capture.bin` (raw complex64 bytes from the slow corr).

    File layout: concatenated `(nbase, nchan, npol)` complex64 blocks.
    Time-averages all blocks; returns a single continuum cube.

    Antenna positions / channel grid default to the synthetic E-W
    linear array used by `bench/replay_voltage_dump.py --synthesize`.
    Pass real antpos / freqs explicitly when loading captures from a
    real fixture.
    """
    path = Path(path)
    arr = np.fromfile(path, dtype=np.complex64)
    block_elems = nbase * nchan * npol
    if arr.size % block_elems != 0:
        raise ValueError(
            f"{path}: byte length {arr.nbytes} not a multiple of "
            f"{block_elems * 8} (one bada block)"
        )
    n_blocks = arr.size // block_elems
    if n_blocks == 0:
        raise ValueError(f"{path}: empty (0 blocks)")
    cube = arr.reshape(n_blocks, nbase, nchan, npol)
    vis_avg = cube.mean(axis=0).astype(np.complex64)               # continuum-average

    if antpos_m is None:
        antpos_m = antpos_synth_2d_grid(NANTS)
    uvw = uvw_from_antpos(antpos_m)
    if freqs_Hz is None:
        freqs_Hz = channel_freqs_hz(nchan)

    return VisCube(
        vis=vis_avg,
        uvw_m=uvw,
        freqs_Hz=freqs_Hz,
        nant=antpos_m.shape[0],
        source_name=source_name,
    )


def load_uvh5_concat(
    paths: Sequence[Path | str], *, source_name: str | None = None,
) -> VisCube:
    """Load N per-subband UVH5 files and stitch them along the freq axis.

    Each input is expected to share the same (Nbls, Ntimes, Npols) layout
    — i.e. the per-sb outputs of `meridian_fringestop` for the same
    fixture run. We continuum-average each input across time first
    (matching `load_uvh5`), then concatenate along the frequency axis
    in ascending-freq order.

    The first input's `uvw_m` (geometry) is reused for the combined cube
    — uvw is wavelength-independent at zenith, so any sb's per-baseline
    metres displacement is correct for all sbs.
    """
    if len(paths) == 0:
        raise ValueError("load_uvh5_concat: no paths supplied")
    cubes: list[VisCube] = []
    for p in paths:
        cubes.append(load_uvh5(p))
    # Sort by ascending f0 so the concatenated freq axis is monotonic
    # (DSA-110 sb files are stored decreasing-within-sb but each sb's
    # window decreases too, so just sort by the *minimum* freq per cube).
    order = sorted(range(len(cubes)), key=lambda k: float(cubes[k].freqs_Hz.min()))
    cubes = [cubes[k] for k in order]
    sorted_paths = [paths[k] for k in order]

    # Concatenate freq + visdata. For each cube, freqs_Hz may be in
    # decreasing order (DSA convention chan_ascending=False), so flip
    # before stacking, then reorder the channel axis of vis to match.
    freqs_parts: list[np.ndarray] = []
    vis_parts: list[np.ndarray] = []
    for c in cubes:
        f = c.freqs_Hz
        v = c.vis  # (Nbls, Nfreqs, Npols)
        if len(f) > 1 and f[0] > f[-1]:
            f = f[::-1].copy()
            v = v[:, ::-1, :].copy()
        freqs_parts.append(f)
        vis_parts.append(v)
    freqs_all = np.concatenate(freqs_parts, axis=0)
    vis_all = np.concatenate(vis_parts, axis=1)

    # uvw is wavelength-independent → first cube's uvw is correct.
    uvw_all = cubes[0].uvw_m
    nant = cubes[0].nant

    src = source_name if source_name is not None else (
        f"concat[{len(cubes)}]: {Path(str(sorted_paths[0])).stem} … "
        f"{Path(str(sorted_paths[-1])).stem}"
    )
    return VisCube(
        vis=vis_all.astype(np.complex64),
        uvw_m=uvw_all,
        freqs_Hz=freqs_all,
        nant=nant,
        source_name=src,
    )


def load_uvh5(path: Path | str, *, source_name: str | None = None) -> VisCube:
    """Load a UVH5 file produced by `meridian_fringestop` via h5py.

    UVH5 schema (pyuvdata convention):
      Header/freq_array              : (Nfreqs,)         float64 Hz
      Header/uvw_array               : (Nblts, 3)        float64 m  (= b_a - b_b)
      Header/Nbls, Nants_data, Npols : ints
      Data/visdata                   : (Nblts, Nfreqs, Npols) complex64
      Header/baseline_array          : (Nblts,)          int (256*ant1+ant2+1)
      Header/time_array              : (Nblts,)          float64 JD
      Header/antenna_positions       : (Nants, 3)        float64 ECEF m

    We continuum-average across time (each time has Nbls visibilities).
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("h5py is required to load UVH5 files") from exc

    path = Path(path)
    with h5py.File(path, "r") as f:
        hdr = f["Header"]
        data = f["Data/visdata"][...]                               # (Nblts, [Nspws,] Nfreqs, Npols)
        # pyuvdata 1.x schema has an extra Nspws=1 axis between Nblts and Nfreqs;
        # pyuvdata 3.x and meridian_fringestop's writer use the flatter 3-D layout.
        if data.ndim == 4:
            if data.shape[1] != 1:
                raise ValueError(
                    f"{path}: visdata shape {data.shape} has Nspws={data.shape[1]} > 1; "
                    f"multi-spw not supported"
                )
            data = data[:, 0, :, :]                                 # (Nblts, Nfreqs, Npols)
        uvw_blts = hdr["uvw_array"][...]                            # (Nblts, 3)
        freqs = np.asarray(hdr["freq_array"][...], dtype=np.float64).reshape(-1)
        nbls = int(hdr["Nbls"][()])
        ntimes = data.shape[0] // nbls
        if data.shape[0] != nbls * ntimes:
            raise ValueError(
                f"{path}: Nblts={data.shape[0]} not divisible by Nbls={nbls}"
            )

    # Reshape (Nblts, Nfreqs, Npols) → (Ntimes, Nbls, Nfreqs, Npols),
    # average across time. Per pyuvdata UVH5 convention, blts are ordered
    # (time, baseline) so the leading dim splits into (Ntimes, Nbls).
    cube = data.reshape(ntimes, nbls, data.shape[1], data.shape[2])
    vis_avg = cube.mean(axis=0).astype(np.complex64)                # (Nbls, Nfreqs, Npols)
    # uvw_blts is (Nblts, 3); take first time (uvw is constant per baseline at zenith).
    uvw_m = uvw_blts.reshape(ntimes, nbls, 3)[0]                    # (Nbls, 3)

    name = source_name if source_name else path.stem
    return VisCube(
        vis=vis_avg, uvw_m=uvw_m, freqs_Hz=freqs,
        nant=int(hdr["Nants_data"][()]) if "Nants_data" in hdr else NANTS,
        source_name=name,
    )


# ---------------------------------------------------------------------------
# Gridder + dirty image
# ---------------------------------------------------------------------------


def grid_uv_natural(
    vis: np.ndarray,           # (Nbls, Nfreqs, Npols) complex
    uvw_m: np.ndarray,         # (Nbls, 3) m
    freqs_Hz: np.ndarray,      # (Nfreqs,) Hz
    *,
    n_grid: int = 256,
    fov_rad: float = 0.5,
    pol: int = 0,
    drop_autos: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Natural-weighted single-side +uv grid, summed across the channel axis.

    Returns
    -------
    grid : complex64, shape (n_grid, n_grid)
        Gridded visibilities; (0, 0) at the centre after np.fft.fftshift.
    weight : float32, shape (n_grid, n_grid)
        Per-cell visibility count (for natural-weighting normalisation).

    The grid coordinates are in λ; cell size is `1 / fov_rad`. Points
    outside the grid are clipped (no toroidal wrap).
    """
    if vis.ndim != 3:
        raise ValueError(f"vis must be 3D, got {vis.ndim}D")

    grid = np.zeros((n_grid, n_grid), dtype=np.complex64)
    weight = np.zeros((n_grid, n_grid), dtype=np.float32)
    half = n_grid // 2
    cell_lambda = 1.0 / fov_rad                                     # cycles per radian per cell

    nbls = vis.shape[0]
    auto_mask = (np.abs(uvw_m).sum(axis=1) < 1e-9)                  # (Nbls,)

    for ich, nu_Hz in enumerate(freqs_Hz):
        wavelength_m = SPEED_OF_LIGHT_M_PER_S / float(nu_Hz)
        # uvw in λ:
        u_lam = uvw_m[:, 0] / wavelength_m                          # (Nbls,)
        v_lam = uvw_m[:, 1] / wavelength_m
        # Pixel coords (centred):
        ix = np.rint(u_lam / cell_lambda).astype(np.int64) + half
        iy = np.rint(v_lam / cell_lambda).astype(np.int64) + half
        in_grid = (ix >= 0) & (ix < n_grid) & (iy >= 0) & (iy < n_grid)
        if drop_autos:
            in_grid &= ~auto_mask
        bls_idx = np.nonzero(in_grid)[0]

        # Accumulate via np.add.at (handles duplicate (ix, iy) pairs).
        v = vis[bls_idx, ich, pol]
        np.add.at(grid, (iy[bls_idx], ix[bls_idx]), v.astype(np.complex64))
        np.add.at(weight, (iy[bls_idx], ix[bls_idx]), 1.0)

    return grid, weight


def dirty_image_from_grid(grid: np.ndarray) -> np.ndarray:
    """Compute the dirty image as Re(iFFT2(grid)), centred via fftshift.

    Per plan §3.6.5 / §3.6.11: single-side +uv grid → Re(iFFT) gives a
    dirty image with `single_side_amplitude_factor ≈ 0.5`. Returns a
    real float32 array of shape `grid.shape`.
    """
    img = np.fft.ifft2(np.fft.ifftshift(grid))
    img = np.fft.fftshift(img)
    return np.real(img).astype(np.float32)


def grid_extent_lm(n_grid: int, fov_rad: float) -> tuple[float, float, float, float]:
    """Image-plane extent in (l, m) (radians) for matplotlib `imshow` / `extent`."""
    half = fov_rad / 2.0
    return (-half, half, -half, half)


# ---------------------------------------------------------------------------
# Peak finding + manifest helpers
# ---------------------------------------------------------------------------


@dataclass
class ImagePeak:
    rank: int
    l_rad: float
    m_rad: float
    flux_image_units: float
    snr_image_plane: float


def find_image_peaks(
    image: np.ndarray, *, fov_rad: float, n_top: int = 5,
    edge_pad: int = 8,
) -> list[ImagePeak]:
    """Top-N local maxima in the dirty image (no deconvolution).

    Returns peaks sorted by descending flux. Edge-padded so peaks within
    `edge_pad` pixels of the border are dropped (image edges have aperture
    cuts that produce false ridges).
    """
    n_grid = image.shape[0]
    interior = image[edge_pad:n_grid - edge_pad, edge_pad:n_grid - edge_pad]
    flat_idx = np.argpartition(interior.flatten(), -n_top)[-n_top:]
    # Sort the top n by descending value.
    flat_vals = interior.flatten()[flat_idx]
    order = np.argsort(-flat_vals)
    flat_idx = flat_idx[order]
    flat_vals = flat_vals[order]

    rms = float(np.std(image))
    cell_rad = fov_rad / n_grid

    peaks: list[ImagePeak] = []
    for rank, (idx, val) in enumerate(zip(flat_idx, flat_vals), start=1):
        iy_int = int(idx) // (n_grid - 2 * edge_pad) + edge_pad
        ix_int = int(idx) % (n_grid - 2 * edge_pad) + edge_pad
        l_rad = (ix_int - n_grid // 2) * cell_rad
        m_rad = (iy_int - n_grid // 2) * cell_rad
        peaks.append(ImagePeak(
            rank=rank, l_rad=float(l_rad), m_rad=float(m_rad),
            flux_image_units=float(val),
            snr_image_plane=float(val / rms) if rms > 0 else float("nan"),
        ))
    return peaks


def expected_sources_from_manifest(
    manifest: dict | None,
) -> list[tuple[str, float, float]]:
    """Extract `[(name, l_rad, m_rad), ...]` from a continuum-fixture manifest.

    Sources without `lm_rad` populated (operator-side TBD) yield (name, nan, nan).
    """
    if not manifest:
        return []
    sources = manifest.get("continuum_sources") or []
    out: list[tuple[str, float, float]] = []
    for src in sources:
        name = str(src.get("name", "?"))
        lm = src.get("lm_rad")
        if lm and len(lm) == 2:
            out.append((name, float(lm[0]), float(lm[1])))
        else:
            out.append((name, float("nan"), float("nan")))
    return out


# ---------------------------------------------------------------------------
# Plot + report rendering
# ---------------------------------------------------------------------------


def render_dirty_image_png(
    image: np.ndarray, fov_rad: float, *,
    title: str, out_path: Path,
    peaks: Sequence[ImagePeak] = (),
    expected_sources: Sequence[tuple[str, float, float]] = (),
    figsize: tuple[float, float] = (8, 7),
) -> None:
    """Render `slow_corr_check.png` with peak + expected-source overlays."""
    try:
        import matplotlib  # noqa: E402
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("matplotlib is required to render PNGs") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    extent = grid_extent_lm(image.shape[0], fov_rad)

    vmax = float(np.percentile(np.abs(image), 99.5))
    vmin = -vmax

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(image, origin="lower", extent=extent, cmap="RdBu_r",
                   vmin=vmin, vmax=vmax, aspect="equal")
    fig.colorbar(im, ax=ax, label="image flux (a.u.)")
    ax.set_xlabel("l (rad, east+)")
    ax.set_ylabel("m (rad, north+)")
    ax.set_title(title)

    # Overlay observed peaks.
    for p in peaks:
        ax.plot(p.l_rad, p.m_rad, "+", color="black", markersize=12, mew=2,
                label=f"peak#{p.rank} SNR={p.snr_image_plane:.1f}" if p.rank == 1 else None)
        ax.text(p.l_rad, p.m_rad + 0.01, f"#{p.rank}", color="black",
                ha="center", fontsize=7)

    # Overlay manifest's expected sources.
    for name, l, m in expected_sources:
        if not (math.isnan(l) or math.isnan(m)):
            ax.plot(l, m, "o", mec="lime", mfc="none", markersize=14, mew=2)
            ax.text(l + 0.01, m, name, color="lime", fontsize=7)

    if peaks:
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_report_html(
    out_dir: Path, *,
    title: str,
    metadata: dict[str, object],
    peaks: Sequence[ImagePeak],
    expected_sources: Sequence[tuple[str, float, float]],
    image_filename: str = "slow_corr_check.png",
) -> Path:
    """Render `report.html` with a peaks-vs-expected table and the dirty-image PNG.

    Per plan §8 line 2174: NO PASS/FAIL banner. The operator inspects
    figures + tables and approves out of band (D11).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_observed = "\n".join(
        f"<tr><td>{p.rank}</td><td>{p.l_rad:+.5f}</td><td>{p.m_rad:+.5f}</td>"
        f"<td>{p.flux_image_units:.4g}</td><td>{p.snr_image_plane:.2f}</td></tr>"
        for p in peaks
    )
    rows_expected = "\n".join(
        f"<tr><td>{name}</td><td>{l:+.5f}</td><td>{m:+.5f}</td></tr>"
        if not (math.isnan(l) or math.isnan(m))
        else f"<tr><td>{name}</td><td colspan='2'>(no lm_rad in manifest)</td></tr>"
        for name, l, m in expected_sources
    )
    meta_rows = "\n".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metadata.items()
    )

    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: sans-serif; max-width: 900px; margin: 1em auto; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 8px; }}
  th {{ background: #eee; }}
  .meta td:first-child {{ font-weight: bold; }}
  img {{ max-width: 100%; }}
  .note {{ background: #fffbe6; padding: 8px; border-left: 4px solid #fc0; }}
</style></head>
<body>
<h1>{title}</h1>
<p class="note"><strong>No PASS/FAIL banner</strong> — per plan §8 line 2174,
this report is for operator inspection only. M2 sub-DoD operator-approval
marker (D11) is recorded out of band in <code>m_operator_approved.yaml</code>.</p>

<h2>Metadata</h2>
<table class="meta">{meta_rows}</table>

<h2>Dirty image (slow-corr continuum check)</h2>
<img src="{image_filename}" alt="slow_corr_check.png">

<h2>Observed peaks (top {len(peaks)})</h2>
<table>
<tr><th>rank</th><th>l (rad)</th><th>m (rad)</th><th>flux (a.u.)</th><th>SNR</th></tr>
{rows_observed}
</table>

<h2>Expected sources (from manifest)</h2>
<table>
<tr><th>name</th><th>l (rad)</th><th>m (rad)</th></tr>
{rows_expected}
</table>
</body></html>
"""

    report_path = out_dir / "report.html"
    report_path.write_text(html)
    return report_path
