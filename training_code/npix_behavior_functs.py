# import stuff here before definitions
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


def group_puffs_by_trial(data, trial_types, rf_zone_duration=5):
    """Assign puffs to trials defined by RF onsets.

    Returns: list of dicts [{ 'rf_onset': float, 'puffs': np.ndarray, 'count': int }]
    """
    rf_times = np.asarray(data['rf_onset'], dtype=float)
    if len(rf_times) != len(trial_types):
        raise ValueError("Length of rf_onsets does not match length of trial_types.")
    
    puff_times = np.asarray(data['puff_clean'], dtype=float)

    trials = []

    for i, trial_type in enumerate(trial_types):
        if trial_type != 3:
            continue

        end_t = rf_times[i] + rf_zone_duration
        puff_times_in_trial = (puff_times > rf_times[i]) & (puff_times < end_t)

        trials.append({
            'rf_onset': float(rf_times[i]),
            'puffs': puff_times[puff_times_in_trial],
            'count': np.sum(puff_times_in_trial)
        })

    return trials



def plot_puff_from_rf_onset(data, trial_types):
    "Plot puff onsets aligned to RF zone onset."

    trials = group_puffs_by_trial(
        data,
        trial_types
    )


    rel_times = []
    for tr in trials:
        rf_t = tr['rf_onset']
        rel = tr['puffs'] - rf_t
        rel_times.extend(rel.tolist())

    rel_times = np.asarray(rel_times)
    if rel_times.size == 0:
        return

    plt.figure(figsize=(6, 3))
    plt.scatter(rel_times, np.zeros_like(rel_times), s=8)
    plt.axvline(0, color='r', linestyle='--', linewidth=1)
    plt.xlabel('Time since RF onset (s)')
    plt.yticks([])
    plt.title('Puff onsets aligned to RF onset')
    plt.tight_layout()
    plt.show()