# import stuff here before definitions
import matplotlib.pyplot as plt
import numpy as np
import math


# GLOBAL VARIABLES
# 1 = rew1, 2 = rew2, 3 = fear, key to trial types
type_map = {1: 'rew1', 2: 'rew2', 3: 'fear'}

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
        relevant_cues = event_times[trial_types == val]
        
        print(f"Processing {len(relevant_cues)} trials for: {name}")

        for unit in unique_units:
            unit_spikes = spike_times[spike_clusters == unit]
            unit_key = f"Unit_{unit}"
            
            # --- Output 1: Individual Trial Timings ---
            trials_list = []
            all_aligned_spikes = []
            
            for cue in relevant_cues:
                t_start, t_end = cue + window[0], cue + window[1]
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

# ADD LICK DATA!
def add_lick_data(data, lick_onsets, spike_times, spike_clusters, window_small, bin_size):
    """
    Adds a 'licks' key to your existing all_data structure.
    """
    # Ensure lick_onsets is a numpy array
    lick_onsets = np.array(lick_onsets)
    
    # Initialize the new 'licks' category
    data['licks'] = {}
    
    # Use your existing sorted unit names
    unit_names = all_unit_names 

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
    
    # Use your existing sorted unit names
    unit_names = all_unit_names 

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
    if isinstance(unit_input, int):
        if unit_input < len(all_unit_names):
            unit_name = all_unit_names[unit_input]
            print(f"Plotting index {unit_input}: {unit_name}")
        else:
            print(f"❌ Index {unit_input} out of range. Max index is {len(all_unit_names)-1}")
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
    ax1.axvline(5, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(11, color='red', linestyle='--', alpha=0.5)

    ax2.axvline(0, color='white', linestyle='--', alpha=0.5)
    ax2.axvline(5, color='white', linestyle='--', alpha=0.5)
    ax2.axvline(11, color='white', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def plot_all_units_context_grid(data, units_per_row=5):
    # 1. Get the list of units and conditions
    conditions = ['rew1', 'rew2', 'fear']
    # Get all unit names from the first condition
    unit_names = sorted(list(data[conditions[0]].keys()), key=lambda x: int(x.split('_')[1]))
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
        ax.axvline(5, color='black', linestyle='--', alpha=0.5) # Cue onset
        ax.axvline(11, color='black', linestyle='--', alpha=0.5) # Cue onset
        
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
    if isinstance(unit_input, int):
        if unit_input < len(all_unit_names):
            unit_name = all_unit_names[unit_input]
            print(f"Plotting index {unit_input}: {unit_name}")
        else:
            print(f"❌ Index {unit_input} out of range. Max index is {len(all_unit_names)-1}")
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
    ax1.axvline(5, color='red', linestyle='--', alpha=0.6) # RF onset
    ax1.axvline(11, color='red', linestyle='--', alpha=0.6) # ITI onset


    # --- BOTTOM PANEL: RASTER ---
    # We loop through each trial and draw the spikes
    for i, trial_spikes in enumerate(spikes):
        # i + 1 ensures the first trial starts at y=1 instead of y=0
        ax2.vlines(trial_spikes, i + 0.6, i + 1.4, color='black', linewidth=1)

    ax2.set_ylabel('Trial Number')
    ax2.set_xlabel('Time from Cue (s)')
    ax2.set_ylim(0.5, num_trials + 0.5)
    ax2.axvline(0, color='red', linestyle='--', alpha=0.6) # Cue onset
    ax2.axvline(5, color='red', linestyle='--', alpha=0.6) # RF onset
    ax2.axvline(11, color='red', linestyle='--', alpha=0.6) # ITI onset


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
    if isinstance(unit_input, int):
        if unit_input < len(all_unit_names):
            unit_name = all_unit_names[unit_input]
            print(f"Plotting index {unit_input}: {unit_name}")
        else:
            print(f"❌ Index {unit_input} out of range. Max index is {len(all_unit_names)-1}")
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
        ax_psth.plot(bins, avg_psth, color=color, lw=2)
        
        ax_psth.set_title(f'{cond.upper()}', fontsize=12, fontweight='bold', color=color)
        if col == 0: ax_psth.set_ylabel('Firing Rate (Hz)')
        ax_psth.axvline(0, color='black', linestyle='--', alpha=0.6)
        ax_psth.axvline(5, color='black', linestyle='--', alpha=0.6)
        ax_psth.axvline(11, color='black', linestyle='--', alpha=0.6)

        # --- BOTTOM ROW: RASTER ---
        ax_raster = axes[1, col]
        for i, trial_spikes in enumerate(spikes):
            ax_raster.vlines(trial_spikes, i + 0.6, i + 1.4, color=color, linewidth=0.4)
        
        ax_raster.set_ylim(0.5, num_trials + 0.5)
        ax_raster.invert_yaxis()
        ax_raster.axvline(0, color='black', linestyle='--', alpha=0.6)
        ax_raster.axvline(5, color='black', linestyle='--', alpha=0.6)
        ax_raster.axvline(11, color='black', linestyle='--', alpha=0.6)

        if col == 0: ax_raster.set_ylabel('Trial Number')
        ax_raster.set_xlabel('Time from Cue (s)')

    # Overall title for the specific unit
    plt.suptitle(f'Functional Profile: {unit_name}', fontsize=18, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08, wspace=0.15)
    plt.show()


def plot_unit_lick_raster(data, unit_input=0, max_licks=100):
    # 1. Handle Index vs Name
    if isinstance(unit_input, int):
        unit_name = all_unit_names[unit_input]
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

    # 5. Plotting (Raster Plot)
    plt.figure(figsize=(9, 6))
    plt.scatter(plot_times, plot_clusters, s=2, c='black', marker='|')
    
    plt.title(f'Raster Plot: {len(good_unit_ids)} "Good" Single Units')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Cluster ID')
    plt.xlim(time_range)
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
