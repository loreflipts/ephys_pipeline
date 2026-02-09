#%%
# IMPORT SETUP STUFF

import sys, os
from pathlib import Path
from datetime import datetime
import csv
import numpy as np
import pandas as pd
import json
import scipy.io
import matplotlib.pyplot as plt
import math
from joblib import Parallel, delayed

# import all functions ( *) from npix_analysis_functions.py
from npix_analysis_functs import *
import npix_analysis_functs as naf
from npix_behavior_functs import *
import npix_behavior_functs as nbf

%load_ext autoreload
%autoreload 2

#%% Load TTL pulses 

base_path_ephys = '/Volumes/memoryShare/Leslie_and_Tim/data/ephys/'
#base_path_ephys = '/Volumes/memoryshare/Leslie_and_Tim/data/ephys/NPX2/'
#
sessions = ['P1', 'T1', 'T2', 'T3', 'R1', 'R14', 'R29']
session_class_counts = {}  # session -> {'LR': n, 'HR': n, 'LRHR': n, 'neither': n, 'total': n}
for session in sessions:
    animal = 'NPX2'
    region = 'ACC'

    if session == 'P1':
        file_path = Path(base_path_ephys + animal + '/11_17_25_P1/finalcat/NPX2_11_17_25_training_ACC_TH_g0/event_times.csv')
    elif session == 'T1':
        file_path = Path(base_path_ephys + animal + '/11_18_25_T1/finalcat/NPX2_11_18_25_training_ACC_TH_g0/event_times.csv')
    elif session == 'T2':
        file_path = Path(base_path_ephys + animal + '/11_19_25_T2/finalcat/NPX2_11_19_25_training_ACC_TH_g0/event_times.csv')   
    elif session == 'T3':
        file_path = Path(base_path_ephys + animal + '/11_20_25_T3/finalcat/NPX2_11_20_25_training_CA_TH_g0/event_times.csv')
    elif session == 'R1':
        file_path = Path(base_path_ephys + animal + '/11_21_25_R1_T100/finalcat/NPX2_11_21_25_training_ACC_TH_g0/event_times.csv')
    elif session == 'R14':
        file_path = Path(base_path_ephys + animal + '/12_05_25_R14_T114/finalcat/NPX2_12_05_25_training_ACC_TH_g0/event_times.csv')
    elif session == 'R29':
        file_path = Path(base_path_ephys + animal + '/12_19_25_R29_T129/finalcat/NPX2_12_19_25_training_ACC_TH_g0/event_times.csv')

    behavior_data = pd.read_csv(file_path, header=0)

    # Adjust the last column so that it variables make sense (don't want to use sec_imap0)
    behavior_data['baseline_start']=behavior_data.sec_imap0[0]
    behavior_data['training_start']=behavior_data.sec_imap0[1]
    behavior_data['offline_start']=behavior_data.sec_imap0[2]

    #%% Load behavior matlab data 

    if session == 'P1':
        matfile_path = Path(base_path_ephys +animal + '/11_17_25_P1/finalcat/' + animal + '_11_17_25_training_ACC_TH_g0/behavior/' + animal + '_P1_2025-11-17_13_39.mat')
    elif session == 'T1':
        matfile_path = Path(base_path_ephys +animal + '/11_18_25_T1/finalcat/' + animal + '_11_18_25_training_ACC_TH_g0/behavior/' + animal + '_T1_2025-11-18_13_55.mat')
    elif session == 'T2':
        matfile_path = Path(base_path_ephys +animal + '/11_19_25_T2/finalcat/' + animal + '_11_19_25_training_ACC_TH_g0/behavior/' + animal + '_T2_2025-11-19_13_46.mat')
    elif session == 'T3':
        matfile_path = Path(base_path_ephys +animal + '/11_20_25_T3/finalcat/' + animal + '_11_20_25_training_CA_TH_g0/behavior/' + animal + '_T3_2025-11-20_15_10.mat')   
    elif session == 'R1':
        matfile_path = Path(base_path_ephys +animal + '/11_21_25_R1_T100/finalcat/' + animal + '_11_21_25_training_ACC_TH_g0/behavior/' + animal + '_R1_2025-11-21_10_33.mat')
    elif session == 'R14':
        matfile_path = Path(base_path_ephys +animal + '/12_05_25_R14_T114/finalcat/' + animal + '_12_05_25_training_ACC_TH_g0/behavior/' + animal + '_R14_2025-12-05_13_18.mat')
    elif session == 'R29':
        matfile_path = Path(base_path_ephys +animal + '/12_19_25_R29_T129/finalcat/' + animal + '_12_19_25_training_ACC_TH_g0/behavior/' + animal + '_R29_2025-12-19_12_01.mat')

    mat_contents = scipy.io.loadmat(matfile_path)

    # This 'squeezes' out all the single-dimensional layers (like [1, 1, 36])

    raw_trials=mat_contents['trial_list']
    # Convert to a simple list or array
    flat_trials= raw_trials
    trial_list=flat_trials.flatten().tolist()
    #%%
    if session == 'P1':
        data_path = Path(base_path_ephys + animal + '/11_17_25_P1/finalcat/' + animal + '_11_17_25_training_ACC_TH_g0/')
    elif session == 'T1':
        data_path = Path(base_path_ephys + animal + '/11_18_25_T1/finalcat/' + animal + '_11_18_25_training_ACC_TH_g0/')
    elif session == 'T2':
        data_path = Path(base_path_ephys + animal + '/11_19_25_T2/finalcat/' + animal + '_11_19_25_training_ACC_TH_g0/')   
    elif session == 'T3':
        data_path = Path(base_path_ephys + animal + '/11_20_25_T3/finalcat/' + animal + '_11_20_25_training_CA_TH_g0/')
    elif session == 'R1':
        data_path = Path(base_path_ephys + animal + '/11_21_25_R1_T100/finalcat/' + animal + '_11_21_25_training_ACC_TH_g0/')
    elif session == 'R14':
        data_path = Path(base_path_ephys + animal + '/12_05_25_R14_T114/finalcat/' + animal + '_12_05_25_training_ACC_TH_g0/')
    elif session == 'R29':
        data_path = Path(base_path_ephys + animal + '/12_19_25_R29_T129/finalcat/' + animal + '_12_19_25_training_ACC_TH_g0/')

    data_path = list(data_path.glob('*'+region))[0]
    ops = np.load(data_path / 'ops.npy', allow_pickle=True).item()
    fs = ops['fs']
    time_range = [0, 6200]  # Time window to plot (in seconds)

    spike_times = np.load(data_path / 'spike_times.npy').flatten()
    spike_clusters = np.load(data_path / 'spike_clusters.npy').flatten()
        
    cluster_info = pd.read_csv(data_path / 'cluster_group.tsv', sep='\t')
    good_unit_ids = cluster_info[cluster_info['KSLabel'] == 'good']['cluster_id'].values

    #%%
    # 1) PREPROCESS EVENT TIMESTAMPS in data PD
    cue_clean=behavior_data['cue_onset'].dropna()[:len(trial_list)] # clears out all the nans in empty rows
    puff_clean=behavior_data['puff_onset'].dropna()
    rf_clean=behavior_data['rf_onset'].dropna()[:len(trial_list)]
    of_cam_clean=behavior_data['cam_start_stop'].dropna()
    catgt_change_clean=behavior_data['sec_imap0'].dropna()
    lick_clean=behavior_data['lick_onset'].dropna()

    behavior_data['cue_clean']=cue_clean  # make sure it's the same length as trial_list
    behavior_data['puff_clean']=puff_clean
    behavior_data['rf_clean']=rf_clean
    behavior_data['of_cam_clean']=of_cam_clean
    behavior_data['catgt_change_clean']=catgt_change_clean
    behavior_data['lick_clean']=lick_clean

    # 2) pull out the index of trials
    rew1_indices = [index for index, value in enumerate(trial_list) if value == 1]
    rew2_indices = [index for index, value in enumerate(trial_list) if value == 2]
    fear_indices = [index for index, value in enumerate(trial_list) if value == 3]

    # 3. Take only the good spiking cells from the raw spiking data
    is_good_spike = np.isin(spike_clusters, good_unit_ids)
    good_spike_times = spike_times[is_good_spike] / fs  # Convert samples to seconds
    good_spike_clusters = spike_clusters[is_good_spike]

    # 4) Create trial average reponses per neuron, for all 3 trial types

    cue_zone_duration = 6  # seconds
    rf_zone_duration = 5   # seconds
    iti_duration = 10  # seconds
    entire_trial_s = cue_zone_duration + rf_zone_duration + iti_duration# cue zone 5s, RF zone 6s, ITI 10s
    window = [-2.0, entire_trial_s + 2]  # 2s before cue, 2s after (in seconds)
    bin_size = 0.1       # 100ms bins for the average PSTH

    # 1 = rew1, 2 = rew2, 3 = fear
    type_map = {1: 'rew1', 2: 'rew2', 3: 'fear'}

    # Generate the data
    neural_data = organize_neural_data(good_spike_times, good_spike_clusters, cue_clean, trial_list, window, bin_size)

    # Get all unit names and sort them numerically (so Unit_2 comes before Unit_10)
    all_unit_names = sorted(neural_data['fear'].keys(), key=lambda x: int(x.split('_')[1]))

    #%% find units that change firing rate across concatenation point 
    neural_data = add_fr_over_time(neural_data, good_spike_times, good_spike_clusters, bin_size=10.0, start_time=None, end_time=None)
    good_idx, bad_idx = get_bad_concat_units(neural_data, behavior_data, th=0.9, win=30, plot=True)
    print(f"Good units: {len(good_idx)}, Bad units: {len(bad_idx)}")

    # 3. Take only the good spiking cells from the raw spiking data
    is_good_spike = np.isin(spike_clusters, good_unit_ids[good_idx])
    good_spike_times = spike_times[is_good_spike] / fs  # Convert samples to seconds
    good_spike_clusters = spike_clusters[is_good_spike]

    bin_size = 0.1
    neural_data = organize_neural_data(good_spike_times, good_spike_clusters, cue_clean, trial_list, window, bin_size)

    # Get all unit names and sort them numerically (so Unit_2 comes before Unit_10)
    all_unit_names = sorted(neural_data['fear'].keys(), key=lambda x: int(x.split('_')[1]))

    print(f"Mapped {len(all_unit_names)} units. You can now use index 0 to {len(all_unit_names)-1}")

    #%%
    conditions = ['rew1', 'rew2'] # which conditions to compare
    n_shuffles = 1000

    # First get the real difference
    real_diff = shuffle_and_compute_fr_diff_fast(
            good_spike_times, good_spike_clusters, cue_clean, 
            trial_list, conditions, window, bin_size, shuffle=False)
    real_diff_values = np.array(list(real_diff.values()))

    def run_single_shuffle_fast(iteration):
        """Run a single shuffle iteration and return FR differences."""
        shuffled_diff = shuffle_and_compute_fr_diff_fast(
            good_spike_times, good_spike_clusters, cue_clean, 
            trial_list, conditions, window, bin_size
        )
        return np.array(list(shuffled_diff.values()))

    # Run shuffles in parallel using all available CPU cores
    print(f"Running {n_shuffles} shuffles in parallel (optimized)...")
    shuffled_diff_values = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_shuffle_fast)(i) for i in range(n_shuffles)
    )

    shuffled_diff_values = np.array(shuffled_diff_values)
    #%%
    LRHR_modulated_units = []
    LRHR_modulated_unit_names = []
    for i_unit in range(len(all_unit_names)):
        # over 99th percentile and difference > 2 Hz
        if (real_diff_values[i_unit] > np.percentile(shuffled_diff_values[:,i_unit], 99)) and (real_diff_values[i_unit] > 2):
            LRHR_modulated_units.append(i_unit)
            LRHR_modulated_unit_names.append(all_unit_names[i_unit])

    print(f"Percentage of LR vs HR modulated units: {len(LRHR_modulated_units)/len(all_unit_names)*100:.2f}%")

    #%% Compare offline firing rates: modulated vs non-modulated units
    from npx_analysis_functions_tim import compare_offline_frs, compare_iti_frs, compare_baseline_frs, classify_cue_responsive_cells, compare_lr_hr_cells
    
    offline_start = catgt_change_clean.iloc[-1]
    offline_end = good_spike_times.max()
    offline_frs = compare_offline_frs(all_unit_names, good_spike_times, good_spike_clusters,
                                       offline_start, offline_end, LRHR_modulated_unit_names,
                                       animal, session, region, plot=True)

    #%% Compare ITI firing rates: modulated vs non-modulated units
    iti_offset = 5.0  # seconds after rf onset
    iti_frs = compare_iti_frs(all_unit_names, good_spike_times, good_spike_clusters,
                               rf_clean, iti_duration, iti_offset, LRHR_modulated_unit_names,
                               animal, session, region, plot=True)

    #%% Check if modulated units simply have higher baseline firing rate
    baseline_start = behavior_data['baseline_start'][0]
    training_start = behavior_data['training_start'][0]
    baseline_frs = compare_baseline_frs(all_unit_names, good_spike_times, good_spike_clusters,
                                         baseline_start, training_start, offline_frs, iti_frs,
                                         LRHR_modulated_unit_names, animal, session, region, plot=True)

    #%% Classify cells as LR, HR, LRHR, or non-responsive based on cue vs baseline FR
    task_start = catgt_change_clean.iloc[0]
    task_end = catgt_change_clean.iloc[1]
    alpha = 0.05
    (cell_classes, class_counts, LR_unit_names, HR_unit_names, 
     LRHR_unit_names, neither_unit_names, task_baseline_frs) = classify_cue_responsive_cells(
        all_unit_names, good_spike_times, good_spike_clusters,
        cue_clean, trial_list, task_start, task_end,
        cue_zone_duration, type_map, alpha, animal, session, region, plot=True)

    #%% Compare LR-only vs HR-only cells: baseline, offline, and ITI firing rates
    compare_lr_hr_cells(LR_unit_names, HR_unit_names, baseline_frs, offline_frs, iti_frs,
                        animal, session, region, plot=True)

    # Store classification counts for this session
    session_class_counts[session] = {**class_counts, 'total': len(all_unit_names)}

from npx_analysis_functions_tim import plot_pct_modulated
plot_sessions = ['P1', 'T1', 'T2', 'R1', 'R14', 'R28']
pct_modulated = [21.3, 24.7, 29.1, 25.6, 23.4, 21.4] # rough numbers taken from Lore analysis just for visualisation 
plot_pct_modulated(pct_modulated, plot_sessions, y_lim = [0, 35])
# %%
