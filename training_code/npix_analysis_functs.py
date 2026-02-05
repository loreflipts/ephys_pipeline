# import stuff here before definitions
import matplotlib.pyplot as plt
import numpy as np
import math
import helper_functs as hf
import pandas as pd


# GLOBAL VARIABLES
# 1 = rew1, 2 = rew2, 3 = fear, key to trial types
type_map = {1: 'rew1', 2: 'rew2', 3: 'fear'}
cue_zone_duration = 6  # seconds
rf_zone_duration = 5   # seconds
iti_duration = 10


# FIRST BIG FUNCTION TO ORGANIZE DATA!!! to sort out contexts, cells, and trials
def organize_neural_data(spike_times, spike_clusters, event_times, trial_types, window, bin_size):
    results = {}
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2

    event_times = np.array(event_times)
    trial_types = np.array(trial_types)
    unique_units = np.unique(spike_clusters)
    
    # Loop through contexts (1->rew1, etc.)
    for val, name in type_map.items():
        results[name] = {} # This creates the 'rew1', 'rew2', or 'fear' sub-folders
        
        # Find which trial type we're in
        # The 'val' is the number (1, 2, or 3)
        #event_times = event_times[:len(trial_types)]
        relevant_cues = event_times[trial_types == val]
        
        print(f"Processing {len(relevant_cues)} trials for: {name}")

        for unit in unique_units:
            unit_spikes = spike_times[spike_clusters == unit]
            unit_key = f"Unit_{unit}"
            
            # --- Output 1: Individual Trial Timings ---
            trials_list = []
            all_aligned_spikes = []
            
            for cue in relevant_cues:
                t_start, t_end = cue + window[0], cue + window[1] # window example: [-1, 21] (cue zone 5s, RF zone 6s, ITI 10s)
                mask = (unit_spikes >= t_start) & (unit_spikes <= t_end)
                aligned = unit_spikes[mask] - cue # Center cue at 0
                trials_list.append(aligned)
                all_aligned_spikes.extend(aligned)
            
            # --- 2. Create the Per-Trial Firing Rate Matrix ---
            # Each row is a trial, each column is a time bin
            num_trials = len(trials_list)
            num_bins = len(bin_centers)
            trial_matrix = np.zeros((len(trials_list), len(bin_centers)))

            for i, aligned_spikes in enumerate(trials_list):
                counts, _ = np.histogram(aligned_spikes, bins=bins)
                trial_matrix[i, :] = counts / bin_size # Convert to Hz

            # --- 3. Calculate the Trial-Averaged Trace ---
            # The mean across the trial axis (axis=0)
            avg_trace = np.mean(trial_matrix, axis=0)

            # --- 2) Average FR over all trials ---
            counts, _ = np.histogram(all_aligned_spikes, bins=bins)
            # Firing Rate = (counts / bin_size) / total_number_of_trials
            avg_fr = counts / (bin_size * len(relevant_cues))

            # --- 3) Calculate SEM ---
            # SEM = Standard Deviation / Square Root of N (number of trials)
            sem_trace = np.std(trial_matrix, axis=0) / np.sqrt(len(trials_list))
            
            # Store everything under the NEW descriptive name
            results[name][unit_key] = {
                'trial_spike_matrix': trials_list, #raw spike times per trial
                'trial_fr_matrix': trial_matrix,      # Per-trial traces (2D)
                'avg_psth': avg_trace,            # Trial-averaged response (1D)
                'avg_fr': avg_fr,
                'sem_psth': sem_trace,
                'bin_centers': bin_centers
            }
            
    return results

def add_fr_over_time(data, spike_times_s, spike_clusters, bin_size=60.0, start_time=None, end_time=None):
    
    """Bin spikes for each unit across the full session."""

    # Initialize the new category
    data['recording'] = {}
    
    # Extract sorted unit names from all_data
    unit_names = hf.get_unit_names(data) 

    if start_time is None:
        start_time = float(np.floor(spike_times_s.min()))
    if end_time is None:
        end_time = float(np.ceil(spike_times_s.max()))

    bins = np.arange(start_time, end_time + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size / 2.0

    for unit_name in unit_names:
        # Extract the integer ID from the string 'Unit_X'
        unit_id = int(unit_name.split('_')[1])
        unit_spikes = spike_times_s[spike_clusters == unit_id]

        if unit_spikes.size == 0:
            continue
        counts, _ = np.histogram(unit_spikes, bins=bins)
        unit_fr = counts / bin_size
        data['recording'][unit_name] = {'fr': unit_fr, 
                                      'spike_counts': counts,
                                      'bin_centers': bin_centers}

    return data


# ADD LICK DATA!
def add_lick_data(data, lick_onsets, spike_times, spike_clusters, window_small, bin_size):
    """
    Adds a 'licks' key to your existing all_data structure.
    """
    # Ensure lick_onsets is a numpy array
    lick_onsets = np.array(lick_onsets)
    
    # Initialize the new 'licks' category
    data['licks'] = {}
    
    # Extract sorted unit names from all_data
    unit_names = hf.get_unit_names(data) 

    bins_small = np.arange(window_small[0], window_small[1] + bin_size, bin_size)
    bin_centers_small = bins_small[:-1] + bin_size/2
    
    print(f"Processing licks for {len(unit_names)} units...")

    for unit_name in unit_names:
        # Extract the integer ID from the string 'Unit_X'
        unit_id = int(unit_name.split('_')[1])
        unit_spikes = spike_times[spike_clusters == unit_id]
        
        trials_list = []
        for lick in lick_onsets:
            t_start, t_end = lick + window_small[0], lick + window_small[1]
            mask = (unit_spikes >= t_start) & (unit_spikes <= t_end)
            trials_list.append(unit_spikes[mask] - lick)
        
        # Create the Matrix
        trial_matrix = np.zeros((len(trials_list), len(bin_centers_small)))
        for i, aligned_spikes in enumerate(trials_list):
            counts, _ = np.histogram(aligned_spikes, bins=bins_small)
            trial_matrix[i, :] = counts / bin_size 
        
        # Calculate Mean and SEM
        avg_trace = np.mean(trial_matrix, axis=0)
        sem_trace = np.std(trial_matrix, axis=0) / np.sqrt(len(trials_list))
        
        # Store in the dictionary under the new 'licks' key
        data['licks'][unit_name] = {
            'trial_spike_matrix': trials_list,
            'trial_fr_matrix': trial_matrix,
            'avg_psth': avg_trace,
            'sem_psth': sem_trace,
            'bin_centers': bin_centers_small
        }
    
    print("Lick data successfully integrated!")
    return data


# ADD PUFF DATA 
def add_puff_data(data, puff_onsets, spike_times, spike_clusters, window_small, bin_size):
    """
    Adds a 'puffs' key to your existing all_data structure.
    """
    # Ensure puff_onsets is a numpy array
    puff_onsets = np.array(puff_onsets)
    
    # Initialize the new 'puffs' category
    data['puffs'] = {}
    
    # Extract sorted unit names from all_data
    unit_names = hf.get_unit_names(data) 

    bins_small = np.arange(window_small[0], window_small[1] + bin_size, bin_size)
    bin_centers_small = bins_small[:-1] + bin_size/2
    
    print(f"Processing puffs for {len(unit_names)} units...")

    for unit_name in unit_names:
        # Extract the integer ID from the string 'Unit_X'
        unit_id = int(unit_name.split('_')[1])
        unit_spikes = spike_times[spike_clusters == unit_id]
        
        trials_list = []
        for puff in puff_onsets:
            t_start, t_end = puff + window_small[0], puff + window_small[1]
            mask = (unit_spikes >= t_start) & (unit_spikes <= t_end)
            trials_list.append(unit_spikes[mask] - puff)
        
        # Create the Matrix
        trial_matrix = np.zeros((len(trials_list), len(bin_centers_small)))
        for i, aligned_spikes in enumerate(trials_list):
            counts, _ = np.histogram(aligned_spikes, bins=bins_small)
            trial_matrix[i, :] = counts / bin_size 
        
        # Calculate Mean and SEM
        avg_trace = np.mean(trial_matrix, axis=0)
        sem_trace = np.std(trial_matrix, axis=0) / np.sqrt(len(trials_list))
        
        # Store in the dictionary under the new 'puffs' key
        data['puffs'][unit_name] = {
            'trial_spike_matrix': trials_list,
            'trial_fr_matrix': trial_matrix,
            'avg_psth': avg_trace,
            'sem_psth': sem_trace,
            'bin_centers': bin_centers_small
        }
    
    print("Puff data successfully integrated!")
    return data

## PLOTTING FUNCTIONS BELOW
def plot_unit_trials_heat(data, condition='fear', unit_input=0):
    """
    unit_input: Can be an integer (0, 1, 2...) or a string ('Unit_10')
    """
    # 1. Logic to handle Integer vs String input
    unit_names = hf.get_unit_names(data)
    if isinstance(unit_input, int):
        if unit_input < len(unit_names):
            unit_name = unit_names[unit_input]
            print(f"Plotting index {unit_input}: {unit_name}")
        else:
            print(f"❌ Index {unit_input} out of range. Max index is {len(unit_names)-1}")
            return
    else:
        unit_name = unit_input
    # 1. Extract the data for the specific unit
    unit_data = data[condition][unit_name]
    trials = unit_data['trial_spike_matrix']
    bins = unit_data['bin_centers']
    bin_size = bins[1] - bins[0]
    
    # 2. Convert individual spike times into a firing rate matrix (Trials x Time)
    num_trials = len(trials)
    fr_matrix = np.zeros((num_trials, len(bins)))
    
    for i, spike_times in enumerate(trials):
        # Calculate histogram for this specific trial
        counts, _ = np.histogram(spike_times, bins=np.append(bins, bins[-1] + bin_size))
        fr_matrix[i, :] = counts / bin_size # Convert to Hz

    # 3. Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 2]})

    # Panel A: Raster Plot
    for i, spike_times in enumerate(trials):
        ax1.vlines(spike_times, i + 0.5, i + 1.5, color='black', linewidth=1)
    ax1.set_ylabel('Trial #')
    ax1.set_title(f'Cell: {unit_name} | Condition: {condition}')

    # Panel B: Firing Rate Heatmap
    im = ax2.imshow(fr_matrix, aspect='auto', origin='lower', 
                    extent=[bins[0], bins[-1], 1, num_trials],
                    cmap='viridis') # 'magma' or 'viridis' are great for FR
    
    ax2.set_xlabel('Time from Cue (s)')
    ax2.set_ylabel('Trial #')
    
    # Add a colorbar for the Hz scale
    cbar = plt.colorbar(im, ax=ax2, orientation='vertical', pad=0.02)
    cbar.set_label('Firing Rate (Hz)')
    
    # Add a vertical line at the cue (t=0)
    ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(cue_zone_duration, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(cue_zone_duration + rf_zone_duration, color='red', linestyle='--', alpha=0.5)

    ax2.axvline(0, color='white', linestyle='--', alpha=0.5)
    ax2.axvline(cue_zone_duration, color='white', linestyle='--', alpha=0.5)
    ax2.axvline(cue_zone_duration + rf_zone_duration, color='white', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_all_units_context_grid(data, units_per_row=5):
    # 1. Get the list of units and conditions
    conditions = ['rew1', 'rew2', 'fear']
    # Get all unit names from the first condition
    unit_names = hf.get_unit_names(data)
    num_units = len(unit_names)

    # 2. Calculate grid dimensions
    num_rows = math.ceil(num_units / units_per_row)

    # 3. Create the figure
    fig, axes = plt.subplots(num_rows, units_per_row, figsize=(units_per_row * 3, num_rows * 2), sharex=True)
    axes = axes.flatten() # Flatten to 1D array for easy iteration

    # Define colors for your conditions
    colors = {'rew1': 'green', 'rew2': 'blue', 'fear': 'red'}

    for i, unit in enumerate(unit_names):
        ax = axes[i]
        
        # Plot each condition for this specific unit
        for cond in conditions:
            if unit in data[cond]:
                psth = data[cond][unit]['avg_psth']
                bins = data[cond][unit]['bin_centers']
                ax.plot(bins, psth, color=colors.get(cond, 'black'), label=cond, lw=1.5)
        
        ax.set_title(f"{unit} (i:{i})", fontsize=10, pad=2)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5) # Cue onset
        ax.axvline(cue_zone_duration, color='black', linestyle='--', alpha=0.5) # Cue onset
        ax.axvline(cue_zone_duration + rf_zone_duration, color='black', linestyle='--', alpha=0.5) # Cue onset
        
        # Only show labels on the left-most and bottom-most axes to save space
        if i % units_per_row == 0:
            ax.set_ylabel('Hz')
        if i >= (num_units - units_per_row):
            ax.set_xlabel('Time (s)')

    # 4. Cleanup: Hide empty axes if num_units is not a perfect multiple
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Add a single legend for the whole figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.suptitle("Average PSTH for All Good Units", y=1.02, fontsize=16)
    plt.show()


def plot_single_psth_raster(data, condition='fear', unit_input=0):
    """
    unit_input: Can be an integer (0, 1, 2...) or a string ('Unit_10')
    """
    # 1. Logic to handle Integer vs String input
    unit_names = hf.get_unit_names(data)
    if isinstance(unit_input, int):
        if unit_input < len(unit_names):
            unit_name = unit_names[unit_input]
            print(f"Plotting index {unit_input}: {unit_name}")
        else:
            print(f"❌ Index {unit_input} out of range. Max index is {len(unit_names)-1}")
            return
    else:
        unit_name = unit_input
    # 1. Extract unit data
    unit_data = data[condition][unit_name]
    spikes = unit_data['trial_spike_matrix']        # List of arrays (for Raster)
    avg_psth = unit_data['avg_psth']    # 1D array (for PSTH)
    sem = unit_data['sem_psth']         # 1D array (for Shading)
    bins = unit_data['bin_centers']
    num_trials = len(spikes)

    # 2. Setup Figure (2 rows, 1 column)
    # height_ratios=[1, 2] makes the Raster twice as tall as the PSTH
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True, 
                                    gridspec_kw={'height_ratios': [1, 2]})

    # --- TOP PANEL: PSTH ---
    ax1.fill_between(bins, avg_psth - sem, avg_psth + sem, color='black', alpha=0.2)
    ax1.plot(bins, avg_psth, color='black', lw=2)
    ax1.set_ylabel('Firing Rate (Hz)')
    ax1.set_title(f'{unit_name} - Condition: {condition}', fontweight='bold')
    ax1.axvline(0, color='red', linestyle='--', alpha=0.6) # Cue onset
    ax1.axvline(cue_zone_duration, color='red', linestyle='--', alpha=0.6) # RF onset
    ax1.axvline(cue_zone_duration + rf_zone_duration, color='red', linestyle='--', alpha=0.6) # ITI onset


    # --- BOTTOM PANEL: RASTER ---
    # We loop through each trial and draw the spikes
    for i, trial_spikes in enumerate(spikes):
        # i + 1 ensures the first trial starts at y=1 instead of y=0
        ax2.vlines(trial_spikes, i + 0.6, i + 1.4, color='black', linewidth=1)

    ax2.set_ylabel('Trial Number')
    ax2.set_xlabel('Time from Cue (s)')
    ax2.set_ylim(0.5, num_trials + 0.5)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.6) # Cue onset
    ax2.axvline(cue_zone_duration, color='red', linestyle='--', alpha=0.6) # RF onset
    ax2.axvline(cue_zone_duration + rf_zone_duration, color='red', linestyle='--', alpha=0.6) # ITI onset


    # Optional: Invert Y-axis so Trial 1 is at the top
    ax2.invert_yaxis() 

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05) # Bring panels closer together
    plt.show()


def plot_singleunit_allcontext(data, unit_input='Unit_0'):
    """
    unit_input: Can be an integer (0, 1, 2...) or a string ('Unit_10')
    """
    # 1. Logic to handle Integer vs String input
    unit_names = hf.get_unit_names(data)
    if isinstance(unit_input, int):
        if unit_input < len(unit_names):
            unit_name = unit_names[unit_input]
            print(f"Plotting index {unit_input}: {unit_name}")
        else:
            print(f"❌ Index {unit_input} out of range. Max index is {len(unit_names)-1}")
            return
    else:
        unit_name = unit_input
    # Define our conditions and their specific colors
    conditions = ['rew1', 'rew2', 'fear']
    colors = {'rew1': 'green', 'rew2': 'blue', 'fear': 'red'}
    
    # 1. Setup Figure: 2 rows (PSTH, Raster) x 3 columns (Conditions)
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex=True, 
                             gridspec_kw={'height_ratios': [1, 2]})
    
    for col, cond in enumerate(conditions):
        # Check if condition exists for this unit
        if cond not in data or unit_name not in data[cond]:
            continue
            
        unit_data = data[cond][unit_name]
        spikes = unit_data['trial_spike_matrix']
        avg_psth = unit_data['avg_psth']
        sem = unit_data['sem_psth']
        bins = unit_data['bin_centers']
        num_trials = len(spikes)
        color = colors[cond]

        # --- TOP ROW: PSTH ---
        ax_psth = axes[0, col]
        ax_psth.fill_between(bins, avg_psth - sem, avg_psth + sem, color=color, alpha=0.2)
        ax_psth.plot(bins, avg_psth, color=color, lw=1.5)
        
        ax_psth.set_title(f'{cond.upper()}', fontsize=12, fontweight='bold', color=color)
        if col == 0: ax_psth.set_ylabel('Firing Rate (Hz)')
        ax_psth.axvline(0, color='black', linestyle='--', alpha=0.6)
        ax_psth.axvline(cue_zone_duration, color='black', linestyle='--', alpha=0.6)
        ax_psth.axvline(cue_zone_duration + rf_zone_duration, color='black', linestyle='--', alpha=0.6)

        # --- BOTTOM ROW: RASTER ---
        ax_raster = axes[1, col]
        for i, trial_spikes in enumerate(spikes):
            ax_raster.vlines(trial_spikes, i + 0.6, i + 1.4, color=color, linewidth=0.4)
        
        ax_raster.set_ylim(0.5, num_trials + 0.5)
        ax_raster.invert_yaxis()
        ax_raster.axvline(0, color='black', linestyle='--', alpha=0.6)
        ax_raster.axvline(cue_zone_duration, color='black', linestyle='--', alpha=0.6)
        ax_raster.axvline(cue_zone_duration + rf_zone_duration, color='black', linestyle='--', alpha=0.6)

        if col == 0: ax_raster.set_ylabel('Trial Number')
        ax_raster.set_xlabel('Time from Cue (s)')

    # Overall title for the specific unit
    plt.suptitle(f'Functional Profile: {unit_name}', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08, wspace=0.15)
    plt.show()


def plot_unit_lick_raster(data, unit_input=0, max_licks=100):
    # 1. Handle Index vs Name
    unit_names = hf.get_unit_names(data)
    if isinstance(unit_input, int):
        unit_name = unit_names[unit_input]
    else:
        unit_name = unit_input

    # 2. Extract Lick Data
    if 'licks' not in data or unit_name not in data['licks']:
        print(f"❌ No lick data found for {unit_name}")
        return

    lick_data = data['licks'][unit_name]
    spikes = lick_data['trial_spike_matrix']
    avg_psth = lick_data['avg_psth']
    sem = lick_data['sem_psth']
    bins = lick_data['bin_centers']
    
    # Limit the number of licks shown in the raster to keep it clean
    num_to_plot = min(len(spikes), max_licks)
    
    # 3. Create Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 5), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 2]})

    # --- Top: PSTH ---
    ax1.fill_between(bins, avg_psth - sem, avg_psth + sem, color='forestgreen', alpha=0.3)
    ax1.plot(bins, avg_psth, color='forestgreen', lw=2)
    ax1.set_ylabel('Firing Rate (Hz)')
    ax1.set_title(f'Lick-Locked Activity: {unit_name}\n({len(spikes)} total licks)', fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5)

    # --- Bottom: Raster ---
    # We only plot up to max_licks
    for i in range(num_to_plot):
        ax2.vlines(spikes[i], i + 0.6, i + 1.4, color='black', linewidth=0.8)

    ax2.set_ylabel(f'Lick Number (First {num_to_plot})')
    ax2.set_xlabel('Time from Lick Onset (s)')
    ax2.set_ylim(0.5, num_to_plot + 0.5)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax2.invert_yaxis() # Put first lick at the top

    unit_data = data['licks'][unit_name]
    bins = unit_data['bin_centers'] # This is your -1.0 to 1.0 range
    
    # 1. FIND THE LIMITS FROM THE DATA
    x_min, x_max = bins[0], bins[-1]

    # 2. FORCE THE AXES TO MATCH THE DATA WINDOW
    ax1.set_xlim(x_min, x_max)
    ax2.set_xlim(x_min, x_max)
    
    # Optional: Add a subtle grid to help see the timing
    ax1.grid(axis='x', alpha=0.3)
    ax2.grid(axis='x', alpha=0.3)

    plt.show()

    plt.tight_layout()
    plt.show()

def plot_unit_grid(data, category='licks', color='forestgreen', units_per_row=5):
    """
    category: 'licks', 'puff', 'rew1', etc.
    color: Any matplotlib color string
    """
    # 1. Validation
    if category not in data:
        print(f"❌ Error: Category '{category}' not found in data.")
        return
        
    unit_names = sorted(data[category].keys(), key=lambda x: int(x.split('_')[1]))
    num_units = len(unit_names)
    num_rows = math.ceil(num_units / units_per_row)
    
    # 2. Setup Figure
    fig, axes = plt.subplots(num_rows, units_per_row, 
                             figsize=(units_per_row * 3, num_rows * 2))
    axes = axes.flatten()

    print(f"Plotting {num_units} units for category: {category.upper()}")

    for i, unit in enumerate(unit_names):
        ax = axes[i]
        unit_data = data[category][unit]
        
        # Pull data
        y = unit_data['avg_psth']
        x = unit_data['bin_centers']
        err = unit_data['sem_psth']
        
        # Plotting
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.2)
        ax.plot(x, y, color=color, lw=1.5)

        # --- Labels and Scales ---
        ax.set_title(f"{unit} (i:{i})", fontsize=9, fontweight='bold')
        #ax.set_ylabel('Hz', fontsize=8)
        #ax.set_xlabel('sec', fontsize=8)
        
        # Individual Axis Formatting
        #ax.set_title(f"{unit}", fontsize=9)
        # --- UPDATED TITLE LOGIC ---
        # Displays as "Unit_22 (Idx: 14)"
        ax.set_title(f"{unit} (i : {i})", fontsize=9, pad=2)
        # ---------------------------

        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
        
        # Remove internal labels to keep it clean
        #if i % units_per_row != 0:
        #    ax.set_yticklabels([])
        if i < (num_units - units_per_row):
            ax.set_xticklabels([])

    # 3. Clean up empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Population Profile: {category.upper()}", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_good_units(path, spike_times, spike_clusters, cluster_info, time_range, fs):
    # Filter for only 'good' units
    good_unit_ids = cluster_info[cluster_info['KSLabel'] == 'good']['cluster_id'].values
    
    if len(good_unit_ids) == 0:
        print("No units labeled 'good' found. Check your cluster_info.tsv!")
        return

    # 3. Filter spikes belonging to good units
    is_good_spike = np.isin(spike_clusters, good_unit_ids)
    good_spike_times = spike_times[is_good_spike] / fs  # Convert samples to seconds
    good_spike_clusters = spike_clusters[is_good_spike]

    # 4. Filter for the specific time window
    mask = (good_spike_times >= time_range[0]) & (good_spike_times <= time_range[1])
    plot_times = good_spike_times[mask]
    plot_clusters = good_spike_clusters[mask]

    # 5. Remap cluster IDs to consecutive row numbers
    cluster_to_row = {cid: idx for idx, cid in enumerate(sorted(good_unit_ids))}
    plot_rows = np.array([cluster_to_row[cid] for cid in plot_clusters])

    # 6. Plotting (Raster Plot)
    plt.figure(figsize=(9, 6))
    plt.scatter(plot_times, plot_rows, s=2, c='black', marker='|')
    
    plt.title(f'Raster Plot: {len(good_unit_ids)} "Good" Single Units')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Unit Index')
    plt.xlim(time_range)
    plt.ylim(-0.5, len(good_unit_ids) - 0.5)
    plt.tight_layout()
    plt.show()

def plot_unit_behavior_raster(data, category='puff', unit_input=0, max_trials=150, color='mediumpurple'):
    """
    Plots a PSTH and Raster for a single unit for any behavioral category.
    """
    # 1. Handle Index vs Name
    unit_names = sorted(data[category].keys(), key=lambda x: int(x.split('_')[1]))
    
    if isinstance(unit_input, int):
        unit_name = unit_names[unit_input]
        idx = unit_input
    else:
        unit_name = unit_input
        idx = unit_names.index(unit_name)

    # 2. Extract Data
    unit_data = data[category][unit_name]
    spikes = unit_data['trial_spike_matrix']  # The key we confirmed earlier
    avg_psth = unit_data['avg_psth']
    sem = unit_data['sem_psth']
    bins = unit_data['bin_centers']
    
    num_to_plot = min(len(spikes), max_trials)
    
    # 3. Create Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6), sharex=True, 
                                   gridspec_kw={'height_ratios': [1, 2]})

    # --- Top: PSTH ---
    ax1.fill_between(bins, avg_psth - sem, avg_psth + sem, color=color, alpha=0.3)
    ax1.plot(bins, avg_psth, color=color, lw=2)
    ax1.set_ylabel('Firing Rate (Hz)')
    ax1.set_title(f'{category.upper()} Response: {unit_name} (Idx: {idx})', fontweight='bold')
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5)

    # --- Bottom: Raster ---
    for i in range(num_to_plot):
        ax2.vlines(spikes[i], i + 0.6, i + 1.4, color='black', linewidth=0.8)

    ax2.set_ylabel(f'Trial Number (First {num_to_plot})')
    ax2.set_xlabel('Time from Onset (s)')
    ax2.set_ylim(0.5, num_to_plot + 0.5)
    ax2.invert_yaxis() 
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_single_recording_fr(data, unit_input=0):
    """
    unit_input: Can be an integer (0, 1, 2...) or a string ('Unit_10')
    """
    # 1. Logic to handle Integer vs String input
    unit_names = hf.get_unit_names(data)
    if isinstance(unit_input, int):
        if unit_input < len(unit_names):
            unit_name = unit_names[unit_input]
            print(f"Plotting index {unit_input}: {unit_name}")
        else:
            print(f"❌ Index {unit_input} out of range. Max index is {len(unit_names)-1}")
            return
    else:
        unit_name = unit_input
    # 1. Extract unit data
    unit_data = data['recording'][unit_name]
    fr = unit_data['fr']

    # 2. Setup Figure

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # --- TOP PANEL: PSTH ---
    ax.plot(fr, color='black', lw=2)


def plot_good_units_recording_fr(data, max_units = 10):

    unit_names = sorted(data['recording'].keys(), key=lambda x: int(x.split('_')[1]))
    num_units = len(unit_names)

    # Limit the number of units shown to keep it clean
    num_to_plot = min(num_units, max_units)


    # 3. Create Figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for i in range(num_to_plot):
        unit_name = unit_names[i]
        unit_fr = data['recording'][unit_name]['fr']
        ax.plot(unit_fr, color='black', lw=1, alpha=0.5)


def plot_good_units_fr_per_task_phase(data, behavior_data, max_units = 10):

    unit_names = sorted(data['recording'].keys(), key=lambda x: int(x.split('_')[1]))
    num_units = len(unit_names)

    num_to_plot = min(num_units, max_units)

    bin_size = data['recording'][unit_names[0]]['bin_centers'][1] - data['recording'][unit_names[0]]['bin_centers'][0]

    baseline_start_idx = int(behavior_data['baseline_start'][0] / bin_size)
    training_start_idx = int(behavior_data['training_start'][0] / bin_size)
    offline_start_idx = int(behavior_data['offline_start'][0] / bin_size)

    # 3. Create Figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for i in range(num_to_plot):
        unit_name = unit_names[i]
        unit_fr = data['recording'][unit_name]['fr']
        baseline_fr = np.mean(unit_fr[baseline_start_idx:training_start_idx])
        training_fr = np.mean(unit_fr[training_start_idx:offline_start_idx])
        offline_fr = np.mean(unit_fr[offline_start_idx:])
        ax.plot([baseline_fr, training_fr, offline_fr], color='black', lw=1, alpha=0.5)

def get_good_units_fr_per_task_phase(data, behavior_data, th_fr):

    unit_names = sorted(data['recording'].keys(), key=lambda x: int(x.split('_')[1]))
    num_units = len(unit_names)

    bin_size = data['recording'][unit_names[0]]['bin_centers'][1] - data['recording'][unit_names[0]]['bin_centers'][0]

    baseline_start_idx = int(behavior_data['baseline_start'][0] / bin_size)
    training_start_idx = int(behavior_data['training_start'][0] / bin_size)
    offline_start_idx = int(behavior_data['offline_start'][0] / bin_size)

    good_units = []
    for i in range(num_units):
        unit_name = unit_names[i]
        unit_fr = data['recording'][unit_name]['fr']
        baseline_fr = np.mean(unit_fr[baseline_start_idx:training_start_idx])
        training_fr = np.max(unit_fr[training_start_idx:offline_start_idx])
        offline_fr = np.max(unit_fr[offline_start_idx:])
        if training_fr > th_fr and offline_fr > th_fr:
            good_units.append(unit_name)

    return good_units

def scatter_good_units_fr_per_task_phase(data, behavior_data, th_fr, max_units = 10):

    unit_names = sorted(data['recording'].keys(), key=lambda x: int(x.split('_')[1]))
    num_units = len(unit_names)

    num_to_plot = min(num_units, max_units)

    bin_size = data['recording'][unit_names[0]]['bin_centers'][1] - data['recording'][unit_names[0]]['bin_centers'][0]

    baseline_start_idx = int(behavior_data['baseline_start'][0] / bin_size)
    training_start_idx = int(behavior_data['training_start'][0] / bin_size)
    offline_start_idx = int(behavior_data['offline_start'][0] / bin_size)

    # 3. Create Figure
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    for i in range(num_to_plot):
        unit_name = unit_names[i]
        unit_fr = data['recording'][unit_name]['fr']
        baseline_fr = np.mean(unit_fr[baseline_start_idx:training_start_idx])
        training_fr = np.max(unit_fr[training_start_idx:offline_start_idx])
        offline_fr = np.max(unit_fr[offline_start_idx:])

        ax.scatter(training_fr, offline_fr, color='black', s=3, alpha=0.5)

    ax.axhline(th_fr, color='red', linestyle='--', alpha=0.5)
    ax.axvline(th_fr, color='red', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.set_xlabel('Training FR (Hz)')
    ax.set_ylabel('Offline FR (Hz)')


def hist_good_units_fr_per_task_phase(data, behavior_data):

    unit_names = sorted(data['recording'].keys(), key=lambda x: int(x.split('_')[1]))
    num_units = len(unit_names)

    bin_size = data['recording'][unit_names[0]]['bin_centers'][1] - data['recording'][unit_names[0]]['bin_centers'][0]

    baseline_start_idx = int(behavior_data['baseline_start'][0] / bin_size)
    training_start_idx = int(behavior_data['training_start'][0] / bin_size)
    offline_start_idx = int(behavior_data['offline_start'][0] / bin_size)


    # 3. Create Figure
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    training_frs = []
    offline_frs = []
    for i in range(num_units):
        unit_name = unit_names[i]
        unit_fr = data['recording'][unit_name]['fr']
        baseline_fr = np.mean(unit_fr[baseline_start_idx:training_start_idx])
        training_fr = np.mean(unit_fr[offline_start_idx-1:offline_start_idx])
        offline_fr = np.mean(unit_fr[offline_start_idx:offline_start_idx+1])
        training_frs.append(training_fr)
        offline_frs.append(offline_fr)

    ratio1 = np.array(np.array(offline_frs) / (np.array(training_frs)+0.000001))
    ratio2 = np.array(np.array(training_frs) / (np.array(offline_frs)+0.000001))
    print(np.max(ratio1))
    axs[0].hist(ratio1, color='black', alpha=0.4, bins = np.arange(0, 1000, 1))
    axs[0].set_xlim(0, 50)
    axs[0].set_yscale('log')

    axs[1].hist(ratio2, color='black', alpha=0.4, bins = np.arange(0, 1000, 1))
    axs[1].set_xlim(0, 50)
    axs[1].set_yscale('log')

    good_units_idx = np.where((ratio1 < 20))[0]# & (ratio2 < 10))[0] 
    bad_units_idx = np.where((ratio1 >= 20))[0]# | (ratio2 >= 10))[0]

    #bad_units_idx = np.array(training_frs) < 0.1
    print(f"Number of units with offline/training FR ratio < 10: {len(good_units_idx)} / {num_units}")

    return good_units_idx, bad_units_idx


def plot_units(spike_times, spike_clusters, unit_ids, time_range, fs, behavior_data):

    if len(unit_ids) == 0:
        print("No units labeled 'good' found. Check your cluster_info.tsv!")
        return

    # 3. Filter spikes belonging to good units
    is_good_spike = np.isin(spike_clusters, unit_ids)
    good_spike_times = spike_times[is_good_spike] / fs  # Convert samples to seconds
    good_spike_clusters = spike_clusters[is_good_spike]

    # 4. Filter for the specific time window
    mask = (good_spike_times >= time_range[0]) & (good_spike_times <= time_range[1])
    plot_times = good_spike_times[mask]
    plot_clusters = good_spike_clusters[mask]

    # 5. Organize spikes by unit for eventplot
    sorted_unit_ids = sorted(unit_ids)
    spike_trains = []
    for unit_id in sorted_unit_ids:
        unit_mask = plot_clusters == unit_id
        spike_trains.append(plot_times[unit_mask])

    # 6. Plotting (Raster Plot using eventplot)
    plt.figure(figsize=(9, 6))
    plt.eventplot(spike_trains, colors='black', lineoffsets=1, linelengths=0.8, linewidths=0.8)
    plt.axvline(behavior_data['offline_start'][0], color='red', linestyle='--', lw=1, alpha=0.5)
    plt.title(f'Raster Plot: {len(unit_ids)} Units')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Unit Index')
    plt.xlim(time_range)
    plt.ylim(-0.5, len(unit_ids)-0.5)
    plt.tight_layout()
    plt.show()


def get_bad_concat_units(data, behavior_data, th=0.9, win=30, plot=False):

    unit_names = sorted(data['recording'].keys(), key=lambda x: int(x.split('_')[1]))
    num_units = len(unit_names)

    bin_size = data['recording'][unit_names[0]]['bin_centers'][1] - data['recording'][unit_names[0]]['bin_centers'][0]
    bin_centers = data['recording'][unit_names[0]]['bin_centers']

    baseline_start_idx = int(behavior_data['baseline_start'][0] / bin_size)
    training_start_idx = int(behavior_data['training_start'][0] / bin_size)
    offline_start_idx = int(behavior_data['offline_start'][0] / bin_size)
    win_bin = int(win / bin_size)

    # 3. Create Figure
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    training_frs = []
    offline_frs = []
    
    # Get time window for x-axis, relative to concatenation point (offline_start = 0)
    # +1 is needed because Python slicing is exclusive on the right end
    time_window = bin_centers[offline_start_idx-win_bin:offline_start_idx+win_bin+1]
    time_window_relative = time_window - bin_centers[offline_start_idx]
    
    for i in range(num_units):
        unit_name = unit_names[i]
        unit_fr = data['recording'][unit_name]['fr']
        training_mean_fr = np.mean(unit_fr[offline_start_idx-win_bin:offline_start_idx])
        offline_mean_fr = np.mean(unit_fr[offline_start_idx:offline_start_idx+win_bin])
        training_frs.append(training_mean_fr)
        offline_frs.append(offline_mean_fr)
        
        if plot:
            if np.abs(training_mean_fr - offline_mean_fr)/(np.maximum(training_mean_fr,offline_mean_fr)) > th:
                ax.plot(time_window_relative, unit_fr[offline_start_idx-win_bin:offline_start_idx+win_bin+1], color='red', lw=0.5, alpha=0.6)
            else:
                ax.plot(time_window_relative, unit_fr[offline_start_idx-win_bin:offline_start_idx+win_bin+1], color='black', lw=0.2, alpha=0.2)
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time relative to concatenation (s)')
    ax.set_ylabel('Firing Rate (Hz)')
    
    #ax.set_ylim(0, 10)
    fr_change_across_concat = np.abs(np.array(training_frs) - np.array(offline_frs))/(np.maximum(np.array(training_frs), np.array(offline_frs)))
    bad_idx = np.where(fr_change_across_concat > th)[0]
    good_idx = np.where(fr_change_across_concat <= th)[0]
    return good_idx, bad_idx



def plot_singleunit_allcontext_v2(data, unit_input='Unit_0'):

    unit_names = hf.get_unit_names(data)
    if isinstance(unit_input, int):
        if unit_input < len(unit_names):
            unit_name = unit_names[unit_input]
            print(f"Plotting index {unit_input}: {unit_name}")
        else:
            print(f"❌ Index {unit_input} out of range.")
            return
    else:
        unit_name = unit_input

    conditions = ['rew1', 'rew2', 'fear']
    colors = {'rew1': 'green', 'rew2': 'blue', 'fear': 'red'}

    # --- max trial count ---
    max_trials = 0
    max_psth = 0
    for cond in conditions:
        if cond in data and unit_name in data[cond]:
            max_trials = max(max_trials, len(data[cond][unit_name]['trial_spike_matrix']))
            max_psth = max(max_psth, float(np.max(data[cond][unit_name]['avg_psth'])))

    # --- figure + gridspec ---
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[2, 1, 3],
        hspace=0.15,
        wspace=0.15
    )


    for col, cond in enumerate(conditions):

        if cond not in data or unit_name not in data[cond]:
            continue

        unit_data = data[cond][unit_name]
        spikes = list(unit_data['trial_spike_matrix'])
        avg_psth = unit_data['avg_psth']
        sem = unit_data['sem_psth']
        bins = unit_data['bin_centers']
        color = colors[cond]

        # pad trials
        spikes.extend([[]] * (max_trials - len(spikes)))
        # --- RASTER ---
        ax_raster = fig.add_subplot(gs[0, col])

        for i, trial_spikes in enumerate(spikes):
            if len(trial_spikes) == 0:
                continue
            ax_raster.vlines(
                trial_spikes,
                i + 0.6,
                i + 1.4,
                color=color,
                linewidth=0.4
            )

        ax_raster.set_ylim(0.5, max_trials + 0.5)
        ax_raster.invert_yaxis()
        ax_raster.set_title(cond.upper(), color=color, fontweight='bold')

        if col == 0:
            ax_raster.set_ylabel('Trial Number')
        ax_raster.set_xlabel('Time from Cue (s)')



        # --- PSTH ---
        ax_psth = fig.add_subplot(gs[1, col])
        ax_psth.plot(bins, avg_psth, color=color, lw=1.5)
        ax_psth.fill_between(
            bins,
            avg_psth - sem,
            avg_psth + sem,
            color=color,
            alpha=0.25
        )
        ax_psth.set_ylim(0, max_psth * 1.2)

        
        if col == 0:
            ax_psth.set_ylabel('Firing Rate (Hz)')

        for t in [0, cue_zone_duration, cue_zone_duration + rf_zone_duration, cue_zone_duration + rf_zone_duration + iti_duration]:
            ax_psth.axvline(t, color='black', linestyle='--', alpha=0.6)


        for t in [0, cue_zone_duration, cue_zone_duration + rf_zone_duration, cue_zone_duration + rf_zone_duration + iti_duration]:
            ax_raster.axvline(t, color='black', linestyle='--', alpha=0.6)


        ax_shared = fig.add_subplot(gs[2, :])

        for cond in conditions:
            if cond not in data or unit_name not in data[cond]:
                continue

            unit_data = data[cond][unit_name]
            bins = unit_data['bin_centers']
            avg_psth = unit_data['avg_psth']
            sem = unit_data['sem_psth']
            color = colors[cond]

            ax_shared.plot(bins, avg_psth, color=color, lw=2, alpha=0.5, label=cond.upper())


        for t in [0, cue_zone_duration, cue_zone_duration + rf_zone_duration, cue_zone_duration + rf_zone_duration + iti_duration]:
            ax_shared.axvline(t, color='black', linestyle='--', alpha=0.8)

        ax_shared.set_ylabel('Firing Rate (Hz)')
        ax_shared.legend(frameon=False)
        ax_shared.set_xticklabels([])

    plt.suptitle(f'Functional Profile: {unit_name}', fontsize=18, y=0.92)
    #plt.show()

    return fig




def shuffle_neural_data(spike_times, spike_clusters, event_times, trial_types, window, bin_size, shuffle_types=None):
    """
    shuffle_types: list of trial type values or names to shuffle together
                   Can be numeric: [1, 2] or string: ['rew1', 'rew2']
                   If None, shuffles all trial types.
    """
    results = {}
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2

    event_times = np.array(event_times)
    trial_types = np.array(trial_types)
    
    # Create shuffled trial types - only shuffle specified types
    shuffled_trial_types = trial_types.copy()
    if shuffle_types is None:
        shuffle_types = list(type_map.keys())  # Default: shuffle all types
    else:
        # Convert string names to numeric values if needed
        if len(shuffle_types) > 0 and isinstance(shuffle_types[0], str):
            # Create reverse map: {'rew1': 1, 'rew2': 2, 'fear': 3}
            reverse_map = {v: k for k, v in type_map.items()}
            shuffle_types = [reverse_map[name] for name in shuffle_types]
    
    # Get indices of trials to shuffle
    shuffle_mask = np.isin(trial_types, shuffle_types)
    shuffle_indices = np.where(shuffle_mask)[0]
    
    # Shuffle only the selected trial types
    shuffled_values = trial_types[shuffle_indices]
    np.random.shuffle(shuffled_values)
    shuffled_trial_types[shuffle_indices] = shuffled_values
    
    unique_units = np.unique(spike_clusters)
    
    # Loop through contexts (1->rew1, etc.)
    for val, name in type_map.items():
        results[name] = {} # This creates the 'rew1', 'rew2', or 'fear' sub-folders
        
        # Find which trial type we're in - using SHUFFLED trial types
        # The 'val' is the number (1, 2, or 3)
        relevant_cues = event_times[shuffled_trial_types == val]
        
        for unit in unique_units:
            unit_spikes = spike_times[spike_clusters == unit]
            unit_key = f"Unit_{unit}"
            
            # --- Output 1: Individual Trial Timings ---
            trials_list = []
            all_aligned_spikes = []
            
            for cue in relevant_cues:
                t_start, t_end = cue + window[0], cue + window[1] # window example: [-1, 21] (cue zone 5s, RF zone 6s, ITI 10s)
                mask = (unit_spikes >= t_start) & (unit_spikes <= t_end)
                aligned = unit_spikes[mask] - cue # Center cue at 0
                trials_list.append(aligned)
                all_aligned_spikes.extend(aligned)
            
            # --- 2. Create the Per-Trial Firing Rate Matrix ---
            # Each row is a trial, each column is a time bin
            num_trials = len(trials_list)
            num_bins = len(bin_centers)
            trial_matrix = np.zeros((len(trials_list), len(bin_centers)))

            for i, aligned_spikes in enumerate(trials_list):
                counts, _ = np.histogram(aligned_spikes, bins=bins)
                trial_matrix[i, :] = counts / bin_size # Convert to Hz

            # --- 3. Calculate the Trial-Averaged Trace ---
            # The mean across the trial axis (axis=0)
            avg_trace = np.mean(trial_matrix, axis=0)

            # --- 2) Average FR over all trials ---
            counts, _ = np.histogram(all_aligned_spikes, bins=bins)
            # Firing Rate = (counts / bin_size) / total_number_of_trials
            avg_fr = counts / (bin_size * len(relevant_cues))

            # --- 3) Calculate SEM ---
            # SEM = Standard Deviation / Square Root of N (number of trials)
            sem_trace = np.std(trial_matrix, axis=0) / np.sqrt(len(trials_list))
            
            # Store everything under the NEW descriptive name
            results[name][unit_key] = {
                'trial_spike_matrix': trials_list, #raw spike times per trial
                'trial_fr_matrix': trial_matrix,      # Per-trial traces (2D)
                'avg_psth': avg_trace,            # Trial-averaged response (1D)
                'avg_fr': avg_fr,
                'sem_psth': sem_trace,
                'bin_centers': bin_centers
            }
            
    return results


def get_cue_fr_diff(data, conditions):
    """
    Calculate the absolute difference in average firing rate between two conditions.
    conditions: List of two condition names or numeric values (e.g., ['rew1', 'fear'] or [1, 3])
    """
    # Convert numeric conditions to string names if needed
    condition_names = []
    for cond in conditions:
        if isinstance(cond, int):
            condition_names.append(type_map[cond])
        else:
            condition_names.append(cond)
    
    unit_names = hf.get_unit_names(data)
    fr_diffs = {}
    
    for unit in unit_names:
        # Check if both conditions exist for this unit
        if all(cond_name in data and unit in data[cond_name] for cond_name in condition_names):
            # Get bin centers and find cue zone indices (0 to 6 seconds)
            bins = data[condition_names[0]][unit]['bin_centers']
            cue_mask = (bins >= 0) & (bins <= cue_zone_duration)
            
            # Calculate mean firing rate for each condition (cue zone only)
            fr_cond1 = np.mean(data[condition_names[0]][unit]['avg_psth'][cue_mask])
            fr_cond2 = np.mean(data[condition_names[1]][unit]['avg_psth'][cue_mask])
            
            # Store absolute difference
            fr_diffs[unit] = np.abs(fr_cond1 - fr_cond2)
    
    return fr_diffs


def shuffle_and_compute_fr_diff_fast(spike_times, spike_clusters, event_times, trial_types, 
                                      conditions, window, bin_size, shuffle=True):
    """
    Fast version that only computes what's needed for FR difference calculation.
    Optimized for permutation testing - skips unnecessary calculations.
    
    Parameters:
    -----------
    spike_times : array
        Spike times for all units
    spike_clusters : array
        Cluster IDs for all spikes
    event_times : array
        Event timestamps (e.g., cue onsets)
    trial_types : array
        Trial type for each event
    conditions : list of str
        Two condition names to compare (e.g., ['rew1', 'rew2'])
    window : list
        [start, end] time window around events
    bin_size : float
        Bin size for PSTH
        
    Returns:
    --------
    dict : FR differences for each unit
    """
    # Convert condition names to numeric values
    reverse_map = {v: k for k, v in type_map.items()}
    shuffle_types = [reverse_map[name] for name in conditions]
    
    # Shuffle trial types
    event_times = np.array(event_times)
    trial_types = np.array(trial_types)
    shuffled_trial_types = trial_types.copy()
    shuffle_mask = np.isin(trial_types, shuffle_types)
    shuffle_indices = np.where(shuffle_mask)[0]
    shuffled_values = trial_types[shuffle_indices]
    if shuffle:
        np.random.shuffle(shuffled_values)
    shuffled_trial_types[shuffle_indices] = shuffled_values
    
    # Pre-compute bins for cue zone only (saves memory and computation)
    bins = np.arange(window[0], window[1] + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2
    cue_mask = (bin_centers >= 0) & (bin_centers <= cue_zone_duration)
    
    unique_units = np.unique(spike_clusters)
    fr_diffs = {}
    
    # Process each unit
    for unit in unique_units:
        unit_spikes = spike_times[spike_clusters == unit]
        unit_key = f"Unit_{unit}"
        
        # Compute avg PSTH for both conditions
        avg_psths = []
        avg_spike_rates = []
        
        for cond_name in conditions:
            cond_val = reverse_map[cond_name]
            relevant_cues = event_times[shuffled_trial_types == cond_val]
            
            # Collect all aligned spikes
            all_aligned = []
            cue_spike_rates = []
            for cue in relevant_cues:
                t_start, t_end = cue + window[0], cue + window[1]
                mask = (unit_spikes >= t_start) & (unit_spikes <= t_end)
                aligned = unit_spikes[mask] - cue
                all_aligned.extend(aligned)
                cue_spike_rates.append((np.sum((aligned >= 0) & (aligned <= cue_zone_duration))) / cue_zone_duration)

            avg_spike_rates.append(np.mean(cue_spike_rates) if len(cue_spike_rates) > 0 else 0)
             
            # Compute PSTH
            if len(relevant_cues) > 0:
                counts, _ = np.histogram(all_aligned, bins=bins)
                avg_psth = counts / (bin_size * len(relevant_cues))
            else:
                avg_psth = np.zeros(len(bin_centers))
            
            avg_psths.append(avg_psth)


        # Calculate FR difference in cue zone only
        fr_cond1 = np.mean(avg_psths[0][cue_mask])
        fr_cond2 = np.mean(avg_psths[1][cue_mask])

        fr_cond1 = np.median(avg_spike_rates[0])
        fr_cond2 = np.median(avg_spike_rates[1])

        fr_diffs[unit_key] = np.abs(fr_cond1 - fr_cond2)

    
    return fr_diffs