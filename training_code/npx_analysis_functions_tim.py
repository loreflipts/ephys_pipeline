import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from scipy.stats import wilcoxon    
def plot_pct_modulated(pct_modulated, plot_sessions, y_lim = [0, 35]):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(plot_sessions, pct_modulated, 'o-', color='black', linewidth=2, markersize=7)
    ax.set_xlabel('Session')
    ax.set_ylabel('% of cells modulated by LR/HR')
    ax.set_title(f'Cue-responsive cells across sessions', fontweight='bold')
    ax.set_ylim(y_lim)
    plt.tight_layout()
    plt.show()

def compare_offline_frs(all_unit_names, good_spike_times, good_spike_clusters, 
                        offline_start, offline_end, LRHR_modulated_unit_names, 
                        animal, session, region, plot=True):
    """Compare offline firing rates between modulated and non-modulated units."""
    #%% Compare offline firing rates: modulated vs non-modulated units
    
    offline_duration = offline_end - offline_start
    print(f"Offline period: {offline_start:.1f}s to {offline_end:.1f}s ({offline_duration:.1f}s)")
    
    # Compute offline firing rate for each unit
    offline_frs = {}
    for unit_name in all_unit_names:
        cluster_id = int(unit_name.split('_')[1])
        unit_spikes = good_spike_times[good_spike_clusters == cluster_id]
        n_spikes_offline = np.sum((unit_spikes >= offline_start) & (unit_spikes <= offline_end))
        offline_frs[unit_name] = n_spikes_offline / offline_duration
    
    # Split into modulated vs non-modulated
    modulated_offline_frs = [offline_frs[name] for name in LRHR_modulated_unit_names]
    non_modulated_names = [name for name in all_unit_names if name not in LRHR_modulated_unit_names]
    non_modulated_offline_frs = [offline_frs[name] for name in non_modulated_names]
    
    print(f"Modulated units (n={len(modulated_offline_frs)}): mean FR = {np.mean(modulated_offline_frs):.2f} Hz")
    print(f"Non-modulated units (n={len(non_modulated_offline_frs)}): mean FR = {np.mean(non_modulated_offline_frs):.2f} Hz")
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Box/strip plot
        positions = [0, 1]
        axes[0].boxplot([modulated_offline_frs, non_modulated_offline_frs],
                        positions=positions, widths=0.4, patch_artist=True,
                        boxprops=dict(facecolor='white', edgecolor='black'),
                        medianprops=dict(color='red', linewidth=1.5))
        
        for i, (data, pos) in enumerate(zip([modulated_offline_frs, non_modulated_offline_frs], positions)):
            jitter = np.random.uniform(-0.1, 0.1, size=len(data))
            axes[0].scatter(np.full(len(data), pos) + jitter, data, alpha=0.5, s=15,
                            color=['tab:orange', 'tab:blue'][i], zorder=3)
        
        axes[0].set_xticks(positions)
        axes[0].set_xticklabels([f'Modulated\n(n={len(modulated_offline_frs)})',
                                f'Non-modulated\n(n={len(non_modulated_offline_frs)})'])
        axes[0].set_ylabel('Firing rate (Hz)')
        axes[0].set_title(f'Offline FR')
        
        # Cumulative distribution
        sorted_mod = np.sort(modulated_offline_frs)
        sorted_non = np.sort(non_modulated_offline_frs)
        axes[1].step(sorted_mod, np.linspace(0, 1, len(sorted_mod)), color='tab:orange', label='Modulated', linewidth=2)
        axes[1].step(sorted_non, np.linspace(0, 1, len(sorted_non)), color='tab:blue', label='Non-modulated', linewidth=2)
        axes[1].set_xlabel('Firing rate (Hz)')
        axes[1].set_ylabel('Cumulative proportion')
        axes[1].set_title('CDF of offline firing rates')
        axes[1].legend()
        
        fig.suptitle(f'{animal} {session} {region} — Offline period FR comparison', fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return offline_frs


def compare_iti_frs(all_unit_names, good_spike_times, good_spike_clusters,
                    rf_clean, iti_duration, iti_offset, LRHR_modulated_unit_names,
                    animal, session, region,
                    plot=True):
    """Compare ITI firing rates between modulated and non-modulated units."""
    
    # ITI starts iti_offset seconds after each rf_clean entry and lasts iti_duration
    rf_clean_arr = np.array(rf_clean)
    iti_starts = rf_clean_arr + iti_offset
    iti_ends = iti_starts + iti_duration
    
    n_trials_iti = len(iti_starts)
    total_iti_time = n_trials_iti * iti_duration
    print(f"ITI periods: {n_trials_iti} trials × {iti_duration}s = {total_iti_time:.0f}s total ITI time")
    
    # Compute mean ITI firing rate for each unit (averaged across trials)
    iti_frs = {}
    for unit_name in all_unit_names:
        cluster_id = int(unit_name.split('_')[1])
        unit_spikes = good_spike_times[good_spike_clusters == cluster_id]
        
        trial_frs = []
        for iti_start, iti_end in zip(iti_starts, iti_ends):
            n_spikes = np.sum((unit_spikes >= iti_start) & (unit_spikes <= iti_end))
            trial_frs.append(n_spikes / iti_duration)
        iti_frs[unit_name] = np.mean(trial_frs)
    
    # Split into modulated vs non-modulated
    modulated_iti_frs = [iti_frs[name] for name in LRHR_modulated_unit_names]
    non_modulated_names = [name for name in all_unit_names if name not in LRHR_modulated_unit_names]
    non_modulated_iti_frs = [iti_frs[name] for name in non_modulated_names]
    
    print(f"Modulated units (n={len(modulated_iti_frs)}): mean ITI FR = {np.mean(modulated_iti_frs):.2f} Hz")
    print(f"Non-modulated units (n={len(non_modulated_iti_frs)}): mean ITI FR = {np.mean(non_modulated_iti_frs):.2f} Hz")
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Box/strip plot
        positions = [0, 1]
        axes[0].boxplot([modulated_iti_frs, non_modulated_iti_frs],
                        positions=positions, widths=0.4, patch_artist=True,
                        boxprops=dict(facecolor='white', edgecolor='black'),
                        medianprops=dict(color='red', linewidth=1.5))
        
        for i, (data, pos) in enumerate(zip([modulated_iti_frs, non_modulated_iti_frs], positions)):
            jitter = np.random.uniform(-0.1, 0.1, size=len(data))
            axes[0].scatter(np.full(len(data), pos) + jitter, data, alpha=0.5, s=15,
                            color=['tab:orange', 'tab:blue'][i], zorder=3)
        
        axes[0].set_xticks(positions)
        axes[0].set_xticklabels([f'Modulated\n(n={len(modulated_iti_frs)})',
                                f'Non-modulated\n(n={len(non_modulated_iti_frs)})'])
        axes[0].set_ylabel('Firing rate (Hz)')
        axes[0].set_title(f'ITI FR')
        
        # Cumulative distribution
        sorted_mod_iti = np.sort(modulated_iti_frs)
        sorted_non_iti = np.sort(non_modulated_iti_frs)
        axes[1].step(sorted_mod_iti, np.linspace(0, 1, len(sorted_mod_iti)), color='tab:orange', label='Modulated', linewidth=2)
        axes[1].step(sorted_non_iti, np.linspace(0, 1, len(sorted_non_iti)), color='tab:blue', label='Non-modulated', linewidth=2)
        axes[1].set_xlabel('Firing rate (Hz)')
        axes[1].set_ylabel('Cumulative proportion')
        axes[1].set_title('CDF of ITI firing rates')
        axes[1].legend()
        
        fig.suptitle(f'{animal} {session} {region} — ITI period FR comparison', fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return iti_frs


def compare_baseline_frs(all_unit_names, good_spike_times, good_spike_clusters,
                         baseline_start, training_start, offline_frs, iti_frs,
                         LRHR_modulated_unit_names, animal, session, region, plot=True):
    """Check if modulated units have higher baseline firing rate and normalize offline/ITI by baseline."""
    
    baseline_duration = training_start - baseline_start
    print(f"Baseline period: {baseline_start:.1f}s to {training_start:.1f}s ({baseline_duration:.1f}s)")
    
    # Compute baseline firing rate for each unit
    baseline_frs = {}
    for unit_name in all_unit_names:
        cluster_id = int(unit_name.split('_')[1])
        unit_spikes = good_spike_times[good_spike_clusters == cluster_id]
        n_spikes_baseline = np.sum((unit_spikes >= baseline_start) & (unit_spikes <= training_start))
        baseline_frs[unit_name] = n_spikes_baseline / baseline_duration
    
    # Split into modulated vs non-modulated
    modulated_baseline_frs = [baseline_frs[name] for name in LRHR_modulated_unit_names]
    non_modulated_names = [name for name in all_unit_names if name not in LRHR_modulated_unit_names]
    non_modulated_baseline_frs = [baseline_frs[name] for name in non_modulated_names]
    
    print(f"Baseline — Modulated: {np.mean(modulated_baseline_frs):.2f} Hz, Non-mod: {np.mean(non_modulated_baseline_frs):.2f} Hz")
    
    # Normalize offline & ITI rates by each unit's baseline FR
    modulated_offline_norm = [offline_frs[name] / baseline_frs[name] if baseline_frs[name] > 0 else np.nan for name in LRHR_modulated_unit_names]
    non_modulated_offline_norm = [offline_frs[name] / baseline_frs[name] if baseline_frs[name] > 0 else np.nan for name in non_modulated_names]
    modulated_iti_norm = [iti_frs[name] / baseline_frs[name] if baseline_frs[name] > 0 else np.nan for name in LRHR_modulated_unit_names]
    non_modulated_iti_norm = [iti_frs[name] / baseline_frs[name] if baseline_frs[name] > 0 else np.nan for name in non_modulated_names]
    
    # Drop nans
    modulated_offline_norm = [x for x in modulated_offline_norm if not np.isnan(x)]
    non_modulated_offline_norm = [x for x in non_modulated_offline_norm if not np.isnan(x)]
    modulated_iti_norm = [x for x in modulated_iti_norm if not np.isnan(x)]
    non_modulated_iti_norm = [x for x in non_modulated_iti_norm if not np.isnan(x)]
    

    print(f"Normalized offline — Modulated: {np.mean(modulated_offline_norm):.2f}x, Non-mod: {np.mean(non_modulated_offline_norm):.2f}x")
    print(f"Normalized ITI    — Modulated: {np.mean(modulated_iti_norm):.2f}x, Non-mod: {np.mean(non_modulated_iti_norm):.2f}x")
    
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        
        # Panel 1: Raw baseline FR
        for i, (data, pos) in enumerate(zip([modulated_baseline_frs, non_modulated_baseline_frs], [0, 1])):
            axes[0].boxplot([data], positions=[pos], widths=0.4, patch_artist=True,
                            boxprops=dict(facecolor='white', edgecolor='black'),
                            medianprops=dict(color='red', linewidth=1.5))
            jitter = np.random.uniform(-0.1, 0.1, size=len(data))
            axes[0].scatter(np.full(len(data), pos) + jitter, data, alpha=0.5, s=15,
                            color=['tab:orange', 'tab:blue'][i], zorder=3)
        axes[0].set_xticks([0, 1])
        axes[0].set_xticklabels(['Modulated', 'Non-mod'])
        axes[0].set_ylabel('Firing rate (Hz)')
        axes[0].set_title(f'Baseline FR')
        
        # Panel 2: Normalized offline FR
        for i, (data, pos) in enumerate(zip([modulated_offline_norm, non_modulated_offline_norm], [0, 1])):
            axes[1].boxplot([data], positions=[pos], widths=0.4, patch_artist=True,
                            boxprops=dict(facecolor='white', edgecolor='black'),
                            medianprops=dict(color='red', linewidth=1.5))
            jitter = np.random.uniform(-0.1, 0.1, size=len(data))
            axes[1].scatter(np.full(len(data), pos) + jitter, data, alpha=0.5, s=15,
                            color=['tab:orange', 'tab:blue'][i], zorder=3)
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(['Modulated', 'Non-mod'])
        axes[1].set_ylabel('FR / baseline FR')
        axes[1].set_title(f'Offline (normalized)')
        axes[1].axhline(1, color='gray', linestyle='--', alpha=0.5)
        
        # Panel 3: Normalized ITI FR
        for i, (data, pos) in enumerate(zip([modulated_iti_norm, non_modulated_iti_norm], [0, 1])):
            axes[2].boxplot([data], positions=[pos], widths=0.4, patch_artist=True,
                            boxprops=dict(facecolor='white', edgecolor='black'),
                            medianprops=dict(color='red', linewidth=1.5))
            jitter = np.random.uniform(-0.1, 0.1, size=len(data))
            axes[2].scatter(np.full(len(data), pos) + jitter, data, alpha=0.5, s=15,
                            color=['tab:orange', 'tab:blue'][i], zorder=3)
        axes[2].set_xticks([0, 1])
        axes[2].set_xticklabels(['Modulated', 'Non-mod'])
        axes[2].set_ylabel('FR / baseline FR')
        axes[2].set_title(f'ITI (normalized)')
        axes[2].axhline(1, color='gray', linestyle='--', alpha=0.5)
        
        fig.suptitle(f'{animal} {session} {region} — Baseline rate control', fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return baseline_frs


def classify_cue_responsive_cells(all_unit_names, good_spike_times, good_spike_clusters,
                                   cue_clean, trial_list, task_start, task_end,
                                   cue_zone_duration, type_map, alpha, animal, session, region, plot=True):
    """Classify cells as LR, HR, LRHR, or non-responsive based on cue vs baseline FR."""
    
    
    cue_window = [0.0, cue_zone_duration]
    task_duration = task_end - task_start
    print(f"Task baseline period: {task_start:.1f}s to {task_end:.1f}s ({task_duration:.1f}s)")
    
    reverse_map = {v: k for k, v in type_map.items()}
    trial_types_arr = np.array(trial_list)
    cue_clean_arr = np.array(cue_clean)
    
    # Pre-compute task baseline FR for each unit
    task_baseline_frs = {}
    for unit_name in all_unit_names:
        cluster_id = int(unit_name.split('_')[1])
        unit_spikes = good_spike_times[good_spike_clusters == cluster_id]
        n_spikes_task = np.sum((unit_spikes >= task_start) & (unit_spikes <= task_end))
        task_baseline_frs[unit_name] = n_spikes_task / task_duration
    
    # For each unit, compare per-trial cue FR against task baseline
    cell_classes = {}
    cell_pvals = {}
    
    for unit_name in all_unit_names:
        cluster_id = int(unit_name.split('_')[1])
        unit_spikes = good_spike_times[good_spike_clusters == cluster_id]
        baseline_fr = task_baseline_frs[unit_name]
        
        responsive = {}
        pvals = {}
        
        for cond_name in ['rew1', 'rew2']:
            cond_val = reverse_map[cond_name]
            trial_cues = cue_clean_arr[trial_types_arr == cond_val]
            
            cue_frs = []
            for cue_time in trial_cues:
                mask_cue = (unit_spikes >= cue_time + cue_window[0]) & (unit_spikes < cue_time + cue_window[1])
                cue_frs.append(np.sum(mask_cue) / (cue_window[1] - cue_window[0]))
            
            cue_frs = np.array(cue_frs)
            
            # Wilcoxon signed-rank: test if per-trial cue FRs differ from task baseline
            diffs = cue_frs - baseline_fr
            if np.any(diffs != 0) and len(diffs) >= 5:
                _, p = wilcoxon(diffs)
            else:
                p = 1.0
            
            pvals[cond_name] = p
            responsive[cond_name] = p < alpha
        
        cell_pvals[unit_name] = pvals
        
        if responsive['rew1'] and responsive['rew2']:
            cell_classes[unit_name] = 'LRHR'
        elif responsive['rew1']:
            cell_classes[unit_name] = 'HR'
        elif responsive['rew2']:
            cell_classes[unit_name] = 'LR'
        else:
            cell_classes[unit_name] = 'neither'
    
    # Summary
    class_counts = {c: sum(1 for v in cell_classes.values() if v == c) for c in ['LR', 'HR', 'LRHR', 'neither']}
    n_total = len(all_unit_names)
    print(f"\nCell classification (cue-responsive, α={alpha}):")
    print(f"  LR only:   {class_counts['LR']:3d} ({class_counts['LR']/n_total*100:.1f}%)")
    print(f"  HR only:   {class_counts['HR']:3d} ({class_counts['HR']/n_total*100:.1f}%)")
    print(f"  LRHR both: {class_counts['LRHR']:3d} ({class_counts['LRHR']/n_total*100:.1f}%)")
    print(f"  Neither:   {class_counts['neither']:3d} ({class_counts['neither']/n_total*100:.1f}%)")
    
    # Collect unit name lists per class
    LR_unit_names = [name for name, c in cell_classes.items() if c == 'LR']
    HR_unit_names = [name for name, c in cell_classes.items() if c == 'HR']
    LRHR_unit_names = [name for name, c in cell_classes.items() if c == 'LRHR']
    neither_unit_names = [name for name, c in cell_classes.items() if c == 'neither']
    
    if plot:
        fig, ax = plt.subplots(figsize=(5, 5))
        labels = ['LR only', 'HR only', 'LR & HR', 'Neither']
        sizes = [class_counts['LR'], class_counts['HR'], class_counts['LRHR'], class_counts['neither']]
        colors = ['tab:green', 'tab:purple', 'tab:orange', 'lightgray']
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                           startangle=90, textprops={'fontsize': 11})
        ax.set_title(f'{animal} {session} {region} — Cue-responsive classification\n(n={n_total} units)', fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    return cell_classes, class_counts, LR_unit_names, HR_unit_names, LRHR_unit_names, neither_unit_names, task_baseline_frs


def compare_lr_hr_cells(LR_unit_names, HR_unit_names, baseline_frs, offline_frs, iti_frs, animal, session, region, plot=True):
    """Compare LR-only vs HR-only cells: baseline, offline, and ITI firing rates."""
    
    
    # Gather FRs for LR-only and HR-only units
    LR_baseline = [baseline_frs[name] for name in LR_unit_names]
    HR_baseline = [baseline_frs[name] for name in HR_unit_names]
    LR_offline = [offline_frs[name] for name in LR_unit_names]
    HR_offline = [offline_frs[name] for name in HR_unit_names]
    LR_iti = [iti_frs[name] for name in LR_unit_names]
    HR_iti = [iti_frs[name] for name in HR_unit_names]
    
    print(f"LR-only units: n={len(LR_unit_names)}, HR-only units: n={len(HR_unit_names)}")
    

    
    print(f"Baseline — LR: {np.mean(LR_baseline):.2f} Hz, HR: {np.mean(HR_baseline):.2f} Hz")
    print(f"Offline  — LR: {np.mean(LR_offline):.2f} Hz, HR: {np.mean(HR_offline):.2f} Hz")
    print(f"ITI      — LR: {np.mean(LR_iti):.2f} Hz, HR: {np.mean(HR_iti):.2f} Hz")
    
    # Normalize by baseline
    LR_offline_norm = [offline_frs[name] / baseline_frs[name] if baseline_frs[name] > 0 else np.nan for name in LR_unit_names]
    HR_offline_norm = [offline_frs[name] / baseline_frs[name] if baseline_frs[name] > 0 else np.nan for name in HR_unit_names]
    LR_iti_norm = [iti_frs[name] / baseline_frs[name] if baseline_frs[name] > 0 else np.nan for name in LR_unit_names]
    HR_iti_norm = [iti_frs[name] / baseline_frs[name] if baseline_frs[name] > 0 else np.nan for name in HR_unit_names]
    
    LR_offline_norm = [x for x in LR_offline_norm if not np.isnan(x)]
    HR_offline_norm = [x for x in HR_offline_norm if not np.isnan(x)]
    LR_iti_norm = [x for x in LR_iti_norm if not np.isnan(x)]
    HR_iti_norm = [x for x in HR_iti_norm if not np.isnan(x)]
    

    
    print(f"Offline (norm) — LR: {np.mean(LR_offline_norm):.2f}x, HR: {np.mean(HR_offline_norm):.2f}x")
    print(f"ITI (norm)     — LR: {np.mean(LR_iti_norm):.2f}x, HR: {np.mean(HR_iti_norm):.2f}x")
    
    if plot:
        fig, axes = plt.subplots(1, 5, figsize=(22, 4))
        lr_color, hr_color = 'tab:green', 'tab:purple'
        
        panel_data = [
            (LR_baseline, HR_baseline, 'Baseline FR (Hz)', 'Baseline', None),
            (LR_offline, HR_offline, 'Firing rate (Hz)', 'Offline (raw)', None),
            (LR_iti, HR_iti, 'Firing rate (Hz)', 'ITI (raw)', None),
            (LR_offline_norm, HR_offline_norm, 'FR / baseline FR', 'Offline (norm)', 1),
            (LR_iti_norm, HR_iti_norm, 'FR / baseline FR', 'ITI (norm)', 1),
        ]
        
        for ax, (lr_data, hr_data, ylabel, title, hline) in zip(axes, panel_data):
            for i, (data, pos, color) in enumerate(zip([lr_data, hr_data], [0, 1], [lr_color, hr_color])):
                if len(data) > 0:
                    ax.boxplot([data], positions=[pos], widths=0.4, patch_artist=True,
                               boxprops=dict(facecolor='white', edgecolor='black'),
                               medianprops=dict(color='red', linewidth=1.5))
                    jitter = np.random.uniform(-0.1, 0.1, size=len(data))
                    ax.scatter(np.full(len(data), pos) + jitter, data, alpha=0.5, s=15,
                               color=color, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f'LR\n(n={len(lr_data)})', f'HR\n(n={len(hr_data)})'])
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        fig.suptitle(f'{animal} {session} {region} — LR-only vs HR-only cells', fontweight='bold')
        plt.tight_layout()
        plt.show()