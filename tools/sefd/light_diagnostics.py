#!/usr/bin/env python3
"""
Light Diagnostics for SEFD Dashboard

Fast diagnostic computations that run immediately when new data is detected.
Produces PNG plots and summary metrics for each observation.

Optimized: reads MS data in bulk (one open/read per diagnostic group).
"""

import numpy as np
from casatools import table, msmetadata
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MAD_TO_SIGMA = 1.4826
HI_FREQ_MHZ = 1420.405751


# =============================================================================
# Helper functions
# =============================================================================

def get_frequency_info(ms_path):
    """Get frequency information from the MS."""
    msmd = msmetadata()
    msmd.open(ms_path)
    chan_freqs = msmd.chanfreqs(0)
    msmd.close()
    return chan_freqs


def get_baseline_lengths(ms_path):
    """Compute baseline lengths from antenna positions."""
    tb = table()
    tb.open(ms_path + '/ANTENNA')
    positions = tb.getcol('POSITION')
    tb.close()
    n_ant = positions.shape[1]
    baseline_lengths = {}
    for i in range(n_ant):
        for j in range(i + 1, n_ant):
            bl_vec = positions[:, j] - positions[:, i]
            length = np.sqrt(np.sum(bl_vec**2))
            baseline_lengths[(i, j)] = length
    return baseline_lengths


def get_all_baselines(ms_path):
    """Get list of all unique cross-correlation baselines."""
    tb = table()
    tb.open(ms_path)
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')
    tb.close()
    baselines = set()
    for a1, a2 in zip(ant1, ant2):
        if a1 != a2:
            baselines.add((min(a1, a2), max(a1, a2)))
    return sorted(baselines)


def get_antennas_with_autocorr(ms_path):
    """Get list of antennas that have autocorrelation data."""
    tb = table()
    tb.open(ms_path)
    ant1 = tb.getcol('ANTENNA1')
    ant2 = tb.getcol('ANTENNA2')
    tb.close()
    antennas = set()
    for a1, a2 in zip(ant1, ant2):
        if a1 == a2:
            antennas.add(a1)
    return sorted(antennas)


# =============================================================================
# Bulk cross-correlation read: amp + noise for selected baselines
# =============================================================================

def compute_amp_and_noise_bulk(ms_path, baselines):
    """
    Compute mean amplitude AND noise for a set of baselines in one MS read.
    
    Reads the MS once, loops through rows grouped by baseline.
    Returns (mean_amps, noise_estimates) dicts keyed by baseline tuple.
    """
    # Build set of baselines to look for
    bl_set = set()
    for bl in baselines:
        bl_set.add((bl[0], bl[1]))
        bl_set.add((bl[1], bl[0]))
    
    # Read relevant columns once via TAQL to exclude autocorrelations
    tb = table()
    tb.open(ms_path)
    subtb = tb.query('ANTENNA1!=ANTENNA2')
    n_rows = subtb.nrows()
    if n_rows == 0:
        subtb.close()
        tb.close()
        return {}, {}
    
    ant1_col = subtb.getcol('ANTENNA1')
    ant2_col = subtb.getcol('ANTENNA2')
    data_col = subtb.getcol('DATA')      # (pol, chan, row)
    flag_col = subtb.getcol('FLAG')      # (pol, chan, row)
    subtb.close()
    tb.close()
    
    # Group rows by baseline
    from collections import defaultdict
    bl_rows = defaultdict(list)
    for i in range(n_rows):
        a1, a2 = int(ant1_col[i]), int(ant2_col[i])
        key = (min(a1, a2), max(a1, a2))
        if key in set(baselines):
            bl_rows[key].append(i)
    
    mean_amps = {}
    noise_estimates = {}
    
    for bl, rows in bl_rows.items():
        if len(rows) == 0:
            continue
        row_idx = np.array(rows)
        bl_data = data_col[:, :, row_idx]   # (pol, chan, n_time)
        bl_flags = flag_col[:, :, row_idx]
        
        # --- Amplitude ---
        masked = np.ma.array(bl_data, mask=bl_flags)
        amp = np.abs(masked)
        mean_amp = np.ma.mean(amp)
        if mean_amp is not np.ma.masked:
            mean_amps[bl] = float(mean_amp)
        
        # --- Noise (channel differencing) ---
        n_pol = min(2, bl_data.shape[0])
        rms_values = []
        for t in range(bl_data.shape[2]):
            if n_pol >= 2:
                vis0 = bl_data[0, :, t]
                vis1 = bl_data[1, :, t]
                flags0 = bl_flags[0, :, t]
                flags1 = bl_flags[1, :, t]
                combined_flags = flags0 | flags1
                vis_avg = (vis0.real + vis1.real) / 2.0
            else:
                vis0 = bl_data[0, :, t]
                combined_flags = bl_flags[0, :, t]
                vis_avg = vis0.real
            vis_avg = np.where(combined_flags, np.nan, vis_avg)
            diffs = np.diff(vis_avg)
            diffs = diffs[~np.isnan(diffs)]
            if len(diffs) > 10:
                mad = np.median(np.abs(diffs - np.median(diffs)))
                sigma_diff = mad * MAD_TO_SIGMA
                sigma_chan = sigma_diff / np.sqrt(2)
                rms_values.append(sigma_chan)
        if rms_values:
            noise_estimates[bl] = float(np.median(rms_values))
    
    return mean_amps, noise_estimates


def plot_amp_vs_baseline(mean_amps, baseline_lengths, outfile):
    """Plot mean uncalibrated amplitude vs baseline length."""
    bl_lens = []
    amps = []
    for bl, amp in mean_amps.items():
        if bl in baseline_lengths:
            bl_lens.append(baseline_lengths[bl])
            amps.append(amp)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.scatter(bl_lens, amps, s=10, alpha=0.6, color='steelblue')
    median_amp = None
    if amps:
        median_amp = float(np.median(amps))
        ax.axhline(median_amp, color='red', linestyle='--', linewidth=1.5,
                    label=f'Median: {median_amp:.3f}')
        ax.legend(fontsize=10)
    ax.set_xlabel('Baseline Length (m)', fontsize=12)
    ax.set_ylabel('Mean Amplitude', fontsize=12)
    ax.set_title('Uncalibrated Amplitude vs Baseline Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return {'median_amplitude': median_amp}


def plot_noise_vs_baseline(noise_estimates, baseline_lengths, outfile):
    """Plot visibility noise vs baseline length."""
    bl_lens = []
    noise_vals = []
    for bl, noise in noise_estimates.items():
        if bl in baseline_lengths:
            bl_lens.append(baseline_lengths[bl])
            noise_vals.append(noise)

    fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
    ax.scatter(bl_lens, noise_vals, s=10, alpha=0.6, color='#2ca02c')
    median_noise = None
    if noise_vals:
        median_noise = float(np.median(noise_vals))
        ax.axhline(median_noise, color='red', linestyle='--', linewidth=1.5,
                    label=f'Median: {median_noise:.5f}')
        ax.legend(fontsize=10)
    ax.set_xlabel('Baseline Length (m)', fontsize=12)
    ax.set_ylabel('Noise (channel diff)', fontsize=12)
    ax.set_title('Uncalibrated Visibility Noise vs Baseline Length', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return {'median_noise': median_noise}


# =============================================================================
# Diagnostic: Coherence vs Frequency (uses same bulk data)
# =============================================================================

def compute_coherence_bulk(ms_path, baselines):
    """Compute coherence per baseline using one MS read."""
    bl_set = set(baselines)
    
    tb = table()
    tb.open(ms_path)
    subtb = tb.query('ANTENNA1!=ANTENNA2')
    n_rows = subtb.nrows()
    if n_rows == 0:
        subtb.close()
        tb.close()
        return {}
    
    ant1_col = subtb.getcol('ANTENNA1')
    ant2_col = subtb.getcol('ANTENNA2')
    data_col = subtb.getcol('DATA')
    flag_col = subtb.getcol('FLAG')
    subtb.close()
    tb.close()
    
    from collections import defaultdict
    bl_rows = defaultdict(list)
    for i in range(n_rows):
        a1, a2 = int(ant1_col[i]), int(ant2_col[i])
        key = (min(a1, a2), max(a1, a2))
        if key in bl_set:
            bl_rows[key].append(i)
    
    coherence_data = {}
    for bl, rows in bl_rows.items():
        if len(rows) == 0:
            continue
        row_idx = np.array(rows)
        bl_data = np.ma.array(data_col[:, :, row_idx], mask=flag_col[:, :, row_idx])
        
        pol_coherences = {}
        for pol_idx, pol_name in [(0, 'XX'), (1, 'YY')]:
            if pol_idx >= bl_data.shape[0]:
                continue
            pol_data = bl_data[pol_idx]  # (chan, n_time)
            vector_avg = np.abs(np.ma.mean(pol_data, axis=1))  # (chan,)
            scalar_avg = np.ma.mean(np.abs(pol_data), axis=1)  # (chan,)
            coherence = vector_avg / scalar_avg
            pol_coherences[pol_name] = np.ma.filled(coherence, np.nan)
        coherence_data[bl] = pol_coherences
    
    return coherence_data


def plot_coherence_vs_freq(coherence_data, baselines, baseline_lengths, chan_freqs_ghz, outfile):
    """Plot coherence vs frequency for selected baselines."""
    n_bl = len(baselines)
    if n_bl == 0:
        return {'median_coherence': None}
    
    n_cols = min(4, n_bl)
    n_rows = max(1, int(np.ceil(n_bl / n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), facecolor='white')
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle('Coherence vs Frequency\n|<V>| / <|V|>  (1=perfect, <1=decorrelated)',
                 fontsize=14, fontweight='bold', y=1.02)

    coherence_medians = []

    for idx, bl in enumerate(baselines):
        bl_length = baseline_lengths.get(bl, 0)
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        if bl in coherence_data:
            for pol_name, color in [('XX', '#1f77b4'), ('YY', '#ff7f0e')]:
                if pol_name in coherence_data[bl]:
                    coh = coherence_data[bl][pol_name]
                    ax.plot(chan_freqs_ghz, coh, linewidth=0.8, alpha=0.7, color=color,
                            label=pol_name if idx == 0 else None)
                    coherence_medians.append(np.nanmedian(coh))

        ax.set_xlabel('Freq (GHz)', fontsize=9)
        ax.set_ylabel('Coherence', fontsize=9)
        ax.set_title(f'{bl[0]}-{bl[1]} ({bl_length:.0f}m)', fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(n_bl, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return {'median_coherence': float(np.nanmedian(coherence_medians)) if coherence_medians else None}


# =============================================================================
# Diagnostic: Autocorrelation Spectra
# =============================================================================

def plot_autocorr_spectra(ms_path, antennas, outfile_xx, outfile_yy, n_integrations=8):
    """Plot autocorrelation spectra for selected antennas (XX and YY)."""
    tb = table()
    tb.open(ms_path)
    subtb = tb.query('ANTENNA1==ANTENNA2')
    if subtb.nrows() == 0:
        subtb.close()
        tb.close()
        return {}
    ant1 = subtb.getcol('ANTENNA1')
    data = subtb.getcol('DATA')
    subtb.close()
    tb.close()

    spectra = {}
    for ant in antennas:
        ant_mask = ant1 == ant
        row_indices = np.where(ant_mask)[0]
        if len(row_indices) == 0:
            continue
        ant_data = data[:, :, row_indices]
        n_total = ant_data.shape[2]
        if n_total > n_integrations:
            start_idx = (n_total - n_integrations) // 2
            end_idx = start_idx + n_integrations
            ant_data = ant_data[:, :, start_idx:end_idx]
        amp = np.abs(ant_data)
        amp_avg = np.mean(amp, axis=2)
        spectra[ant] = {
            'XX': amp_avg[0, :],
            'YY': amp_avg[1, :] if amp_avg.shape[0] > 1 else amp_avg[0, :],
        }

    if len(spectra) == 0:
        return {}

    n_ants = len(antennas)
    n_cols = min(4, n_ants)
    n_rows = max(1, int(np.ceil(n_ants / n_cols)))

    for pol_name, outfile in [('XX', outfile_xx), ('YY', outfile_yy)]:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), facecolor='white')
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        fig.suptitle(f'Autocorrelation Spectra ({pol_name})',
                     fontsize=14, fontweight='bold', y=1.01)

        for idx, ant in enumerate(antennas):
            if ant not in spectra:
                continue
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            ax.plot(spectra[ant][pol_name], linewidth=0.8, color='steelblue')
            ax.set_xlabel('Channel', fontsize=9)
            ax.set_ylabel('Amplitude', fontsize=9)
            ax.set_title(f'Ant {ant}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        for idx in range(n_ants, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(outfile, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()

    median_xx = np.median([np.median(spectra[a]['XX']) for a in spectra])
    median_yy = np.median([np.median(spectra[a]['YY']) for a in spectra])
    return {'median_autocorr_xx': float(median_xx), 'median_autocorr_yy': float(median_yy)}


# =============================================================================
# Diagnostic: HI Spectra
# =============================================================================

def plot_hi_spectra(ms_path, antennas, outfile, n_integrations=8, bandwidth_mhz=5.0):
    """Plot HI spectra from autocorrelation data."""
    chan_freqs = get_frequency_info(ms_path)
    chan_freqs_mhz = chan_freqs / 1e6

    hi_mask = (chan_freqs_mhz >= HI_FREQ_MHZ - bandwidth_mhz / 2) & \
              (chan_freqs_mhz <= HI_FREQ_MHZ + bandwidth_mhz / 2)
    if not np.any(hi_mask):
        return {}

    hi_indices = np.where(hi_mask)[0]
    freq_hi = chan_freqs_mhz[hi_mask]

    tb = table()
    tb.open(ms_path)
    subtb = tb.query('ANTENNA1==ANTENNA2')
    if subtb.nrows() == 0:
        subtb.close()
        tb.close()
        return {}
    ant1 = subtb.getcol('ANTENNA1')
    data = subtb.getcolslice('DATA', [0, hi_indices[0]], [1, hi_indices[-1]], [1, 1])
    subtb.close()
    tb.close()

    n_ants = len(antennas)
    n_cols = min(4, n_ants)
    n_rows = max(1, int(np.ceil(n_ants / n_cols)))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), facecolor='white')
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle(f'HI Spectrum (Autocorrelations, baseline-normalized, dB)\n{bandwidth_mhz} MHz around {HI_FREQ_MHZ:.1f} MHz',
                 fontsize=14, fontweight='bold', y=1.02)

    hi_peaks = []
    for idx, ant in enumerate(antennas):
        ant_mask = ant1 == ant
        row_indices = np.where(ant_mask)[0]
        if len(row_indices) == 0:
            continue

        ant_data = data[:, :, row_indices]
        n_total = ant_data.shape[2]
        if n_total > n_integrations:
            start_idx = (n_total - n_integrations) // 2
            end_idx = start_idx + n_integrations
            ant_data = ant_data[:, :, start_idx:end_idx]

        amp = np.abs(ant_data)
        amp_avg = np.mean(amp, axis=(0, 2))

        n_edge = max(5, len(amp_avg) // 10)
        edge_freqs = np.concatenate([freq_hi[:n_edge], freq_hi[-n_edge:]])
        edge_amps = np.concatenate([amp_avg[:n_edge], amp_avg[-n_edge:]])
        coeffs = np.polyfit(edge_freqs, edge_amps, 1)
        baseline_fit = np.polyval(coeffs, freq_hi)
        amp_normalized = amp_avg / baseline_fit
        amp_db = 10 * np.log10(amp_normalized)
        hi_peaks.append(np.max(amp_db))

        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        ax.plot(freq_hi, amp_db, linewidth=0.8, color='steelblue')
        ax.axvline(HI_FREQ_MHZ, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.set_xlabel('Freq (MHz)', fontsize=9)
        ax.set_ylabel('Power (dB)', fontsize=9)
        ax.set_title(f'Ant {ant}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    for idx in range(n_ants, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(outfile, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()
    return {'median_hi_peak_db': float(np.median(hi_peaks)) if hi_peaks else None}


# =============================================================================
# Main entry point
# =============================================================================

def run_light_diagnostics(ms_path, date, source, results_base,
                          every_nth_baseline=10, every_nth_antenna=10):
    """
    Run all light diagnostics for a single measurement set.
    
    Optimized: reads the cross-correlation data ONCE for amp+noise,
    and ONCE more for coherence baselines. Autocorr and HI are separate reads.
    
    Returns dict of summary metrics.
    """
    out_dir = os.path.join(results_base, source, date)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[Light] Processing {date} {source}: {ms_path}")
    metrics = {'date': date, 'source': source}

    # Get baseline info
    all_baselines = get_all_baselines(ms_path)
    selected_baselines = all_baselines[::every_nth_baseline]
    baseline_lengths = get_baseline_lengths(ms_path)
    print(f"  Using {len(selected_baselines)} baselines (of {len(all_baselines)})")

    # Get antenna info
    all_antennas = get_antennas_with_autocorr(ms_path)
    selected_antennas = all_antennas[::every_nth_antenna]
    print(f"  Using {len(selected_antennas)} antennas (of {len(all_antennas)})")

    # Select ~10 random baselines for coherence
    random.seed(42)
    coherence_baselines = random.sample(all_baselines, min(10, len(all_baselines)))

    # Get frequencies
    chan_freqs = get_frequency_info(ms_path)
    chan_freqs_ghz = chan_freqs / 1e9

    # 1 & 2. Amplitude + Noise (one bulk read)
    print("  [1-2/5] Amplitude + Noise vs baseline length (bulk read)...")
    mean_amps, noise_estimates = compute_amp_and_noise_bulk(ms_path, selected_baselines)
    
    amp_metrics = plot_amp_vs_baseline(
        mean_amps, baseline_lengths,
        os.path.join(out_dir, 'amp_vs_baseline.png')
    )
    metrics.update(amp_metrics)
    
    noise_metrics = plot_noise_vs_baseline(
        noise_estimates, baseline_lengths,
        os.path.join(out_dir, 'noise_vs_baseline.png')
    )
    metrics.update(noise_metrics)

    # 3. Coherence vs frequency (one bulk read)
    print("  [3/5] Coherence vs frequency (bulk read)...")
    coherence_data = compute_coherence_bulk(ms_path, coherence_baselines)
    coh_metrics = plot_coherence_vs_freq(
        coherence_data, coherence_baselines, baseline_lengths,
        chan_freqs_ghz,
        os.path.join(out_dir, 'coherence_vs_freq.png')
    )
    metrics.update(coh_metrics)

    # 4. Autocorrelation spectra
    print("  [4/5] Autocorrelation spectra...")
    autocorr_metrics = plot_autocorr_spectra(
        ms_path, selected_antennas,
        os.path.join(out_dir, 'autocorr_XX.png'),
        os.path.join(out_dir, 'autocorr_YY.png')
    )
    metrics.update(autocorr_metrics)

    # 5. HI spectra
    print("  [5/5] HI spectra...")
    hi_metrics = plot_hi_spectra(
        ms_path, selected_antennas,
        os.path.join(out_dir, 'hi_spectrum.png')
    )
    metrics.update(hi_metrics)

    print(f"  Done! Metrics: {metrics}")
    return metrics
