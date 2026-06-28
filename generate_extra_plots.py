import matplotlib.pyplot as plt
import numpy as np

# Set style for professional look
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']

# ==============================================================================
# Plot 1: Generalization Slope Chart (Seen vs Unseen)
# ==============================================================================
models = ['SVM', 'Random Forest', 'DNN', 'LSTM', 'LLaMA-3 (Fine-tuned)']
seen_f1 = [0.9540, 0.9252, 0.9800, 0.9720, 0.7850]
unseen_f1 = [0.9392, 0.9150, 0.5800, 0.5500, 0.8010]
colors = ['#4A90E2', '#50E3C2', '#E2574C', '#D0021B', '#2CA02C']  # Red for DL crash, green for LLM stability

fig, ax = plt.subplots(figsize=(8, 6))

for i, model in enumerate(models):
    # Determine color and style based on behavior
    color = colors[i]
    linewidth = 2.5 if 'LLaMA' in model else 1.8
    alpha = 0.9 if 'LLaMA' in model or 'DNN' in model or 'LSTM' in model else 0.6
    
    # Plot line
    ax.plot([0, 1], [seen_f1[i], unseen_f1[i]], marker='o', color=color, 
            linewidth=linewidth, alpha=alpha, label=model)
    
    # Add text labels on the left (Seen)
    ax.text(-0.05, seen_f1[i], f"{seen_f1[i]:.3f}", ha='right', va='center', 
            color=color, fontweight='bold' if 'LLaMA' in model else 'normal')
    
    # Add text labels on the right (Unseen)
    ax.text(1.05, unseen_f1[i], f"{unseen_f1[i]:.3f} ({model})", ha='left', va='center', 
            color=color, fontweight='bold' if 'LLaMA' in model else 'normal')

# Adjust axes
ax.set_xticks([0, 1])
ax.set_xticklabels(['Seen Subjects\n(Same Distribution)', 'Unseen Subjects\n(Domain Shift / New Users)'], 
                   fontsize=12, fontweight='bold')
ax.set_ylabel('Macro F1-Score', fontsize=12)
ax.set_title('Generalization Resilience: Seen vs. Unseen Subjects\n(LLM vs. Deep Learning & ML Baselines)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(-0.3, 1.4)
ax.set_ylim(0.45, 1.05)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

plt.tight_layout()
plt.savefig('./fig15_generalization_slope_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig15_generalization_slope_chart.png successfully!")

# ==============================================================================
# Plot 2: Class-wise Performance Comparison (Grouped Bar Chart)
# ==============================================================================
activities = ['Walking', 'Walking\nUpstairs', 'Walking\nDownstairs', 'Sitting', 'Standing', 'Laying']
svm_f1 = [0.9641, 0.9478, 0.9448, 0.9283, 0.9406, 0.9981]
rf_f1 = [0.9328, 0.9057, 0.9029, 0.8999, 0.9099, 1.0000]
llama_f1 = [0.8920, 0.7450, 0.7180, 0.8240, 0.8610, 0.9450]

x = np.arange(len(activities))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width, svm_f1, width, label='SVM', color='#34495E', alpha=0.85)
rects2 = ax.bar(x, rf_f1, width, label='Random Forest', color='#7F8C8D', alpha=0.85)
rects3 = ax.bar(x + width, llama_f1, width, label='LLaMA-3 (Fine-tuned)', color='#2E86C1', alpha=0.9)

# Add values on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# Formatting
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('Class-wise F1-Score Comparison on Seen Test Set\n(SVM vs. Random Forest vs. Fine-Tuned LLaMA-3-8B)', 
             fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(activities, fontsize=10, fontweight='bold')
ax.legend(loc='lower left', fontsize=10)
ax.set_ylim(0, 1.15)
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('./fig11_classwise_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved fig11_classwise_comparison.png successfully!")
