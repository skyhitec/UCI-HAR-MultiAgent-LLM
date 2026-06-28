import matplotlib.pyplot as plt
import numpy as np

# Set professional fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']

# 1. Define dataset accuracy values from Table II of the paper
datasets = {
    'MotionSense': {'SVM': 0.8603, 'RF': 0.8796, 'LSTM': 0.8508, 'DNN': 0.8544},
    'Shoaib': {'SVM': 0.9352, 'RF': 0.9894, 'LSTM': 0.9695, 'DNN': 0.9782},
    'UCI HAR': {'SVM': 0.9190, 'RF': 0.9747, 'LSTM': 0.9765, 'DNN': 0.9821},
    'WISDM': {'SVM': 0.9895, 'RF': 0.9965, 'LSTM': 0.9668, 'DNN': 0.9616}
}

models = ['SVM', 'RF', 'LSTM', 'DNN']
x = np.arange(len(models))

# Create 1x4 subplot grid (matching the user's image)
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)

# Soft blue color matching the user's image
bar_color = '#7FB3D5'  # Soft light blue

for i, (ds_name, model_scores) in enumerate(datasets.items()):
    ax = axes[i]
    scores = [model_scores[m] for m in models]
    
    # Draw bars
    bars = ax.bar(x, scores, color=bar_color, alpha=0.9, width=0.55)
    
    # Grid lines (light grey, y-axis only)
    ax.grid(axis='y', linestyle='-', linewidth=0.3, color='#BDC3C7', zorder=0)
    ax.set_axisbelow(True)
    
    # Subplot details
    ax.set_title(f"({chr(97 + i)}) {ds_name}", fontsize=10, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8, rotation=45)
    ax.set_xlabel('Model', fontsize=8, labelpad=5)
    ax.set_ylim(0, 1.1)
    
    if i == 0:
        ax.set_ylabel('Accuracy', fontsize=8)

    # Clean borders (remove top and right spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#BDC3C7')
    ax.spines['bottom'].set_color('#BDC3C7')

plt.suptitle('Fig. 11: Performance analysis for four datasets.', fontsize=12, fontweight='bold', y=1.05)
plt.tight_layout()

# Save figure in high-resolution
plt.savefig('./fig14_performance_analysis_four_datasets.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig14_performance_analysis_four_datasets.png successfully!")
