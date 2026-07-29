#!/usr/bin/env python3
"""
Full SEFD Pipeline for Dashboard

Wraps estimate_sefd.process_ms() to run the full calibration + SEFD estimation.
"""

import sys
import os
import numpy as np

# Add parent directory to path for importing estimate_sefd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import estimate_sefd


def run_full_pipeline(ms_path, date, source, results_base, cal_flux_jy=10.0,
                      refant='103', avg_time='30s'):
    """
    Run the full SEFD estimation pipeline for a single measurement set.
    
    Parameters
    ----------
    ms_path : str
        Path to input measurement set
    date : str
        Observation date string (e.g., '2026-02-09')
    source : str
        Source name (e.g., '0521+166')
    results_base : str
        Base results directory
    cal_flux_jy : float
        Calibrator flux density in Jy
    refant : str
        Reference antenna
    avg_time : str
        Time averaging interval
    
    Returns dict of summary metrics.
    """
    output_dir = os.path.join(results_base, source, date, 'sefd')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[Full] Running SEFD pipeline for {date} {source} (flux={cal_flux_jy} Jy)")
    print(f"  Input: {ms_path}")
    print(f"  Output: {output_dir}")
    
    try:
        sefd_by_baseline, results = estimate_sefd.process_ms(
            ms_in=ms_path,
            output_dir=output_dir,
            cal_flux_jy=cal_flux_jy,
            refant=refant,
            avg_time=avg_time,
        )
        
        # Compute summary metrics
        all_sefd = np.array(list(sefd_by_baseline.values()))
        metrics = {
            'median_sefd': float(np.median(all_sefd)),
            'mean_sefd': float(np.mean(all_sefd)),
            'std_sefd': float(np.std(all_sefd)),
            'n_baselines': len(all_sefd),
        }
        
        # Add binned results
        for bin_name, stats in results.items():
            if stats['n_baselines'] > 0:
                metrics[f'sefd_{bin_name}'] = float(stats['mean_sefd'])
        
        print(f"  Full pipeline complete. Median SEFD: {metrics['median_sefd']:.0f} Jy")
        return metrics
    
    except Exception as e:
        print(f"  Full pipeline error: {e}")
        raise
