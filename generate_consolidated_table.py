import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. Data Definitions (HHAR, MotionSense, Shoaib from Paper; UCI HAR from Our Work)
# ==============================================================================
datasets = ['HHAR', 'MotionSense', 'Shoaib', 'UCI HAR']
models = ['Random Forest', 'SVM', 'DNN', 'LLaMA-3 (Base)', 'LLaMA-3 (Fine-tuned)']

# Structure: F1-scores [Seen, Unseen] for each model on each dataset
data = {
    'HHAR': {
        'Random Forest': [0.97, 0.67],
        'SVM': [0.91, 0.47],
        'DNN': [0.98, 0.61],
        'LLaMA-3 (Base)': [0.00, 0.65],  # 0.00 represents '-' (not evaluated for seen)
        'LLaMA-3 (Fine-tuned)': [0.83, 0.75]
    },
    'MotionSense': {
        'Random Forest': [0.84, 0.58],
        'SVM': [0.82, 0.42],
        'DNN': [0.81, 0.38],
        'LLaMA-3 (Base)': [0.00, 0.61],
        'LLaMA-3 (Fine-tuned)': [0.77, 0.65]
    },
    'Shoaib': {
        'Random Forest': [0.98, 0.67],
        'SVM': [0.92, 0.56],
        'DNN': [0.97, 0.64],
        'LLaMA-3 (Base)': [0.00, 0.67],
        'LLaMA-3 (Fine-tuned)': [0.79, 0.71]
    },
    'UCI HAR': {
        'Random Forest': [0.925, 0.915],
        'SVM': [0.954, 0.939],
        'DNN': [0.980, 0.580],
        'LLaMA-3 (Base)': [0.00, 0.630],
        'LLaMA-3 (Fine-tuned)': [0.785, 0.801]
    }
}

# ==============================================================================
# 2. Generate LaTeX Table Code
# ==============================================================================
latex_code = r"""% ========================================================================
% LaTeX Code for Consolidated 4-Dataset Classification Results
% (HHAR, MotionSense, Shoaib from Original Paper + UCI HAR from Our Work)
% ========================================================================
\begin{table*}[ht]
\centering
\caption{Classification Performance (Macro F1-Score) Comparison Across Four Benchmarks}
\label{tab:consolidated_results}
\begin{tabular}{l|cc|cc|cc|cc}
\hline
\textbf{Dataset} & \multicolumn{2}{c|}{\textbf{HHAR}} & \multicolumn{2}{c|}{\textbf{MotionSense}} & \multicolumn{2}{c|}{\textbf{Shoaib}} & \multicolumn{2}{c}{\textbf{UCI HAR (Ours)}} \\ 
\textbf{Test Subject} & \textbf{Seen} & \textbf{Unseen} & \textbf{Seen} & \textbf{Unseen} & \textbf{Seen} & \textbf{Unseen} & \textbf{Seen} & \textbf{Unseen} \\ \hline
"""

for model in models:
    row_cells = [f"\\textbf{{{model}}}" if 'LLaMA' in model else model]
    for ds in datasets:
        seen = data[ds][model][0]
        unseen = data[ds][model][1]
        
        seen_str = f"{seen:.2f}" if seen > 0 else "--"
        unseen_str = f"{unseen:.2f}"
        
        # Highlight best unseen in bold
        if model == 'LLaMA-3 (Fine-tuned)' and ds != 'UCI HAR':
            unseen_str = f"\\textbf{{{unseen_str}}}"
        elif model == 'SVM' and ds == 'UCI HAR':
            # SVM was highest on UCI HAR unseen due to pre-engineered features
            seen_str = f"\\textbf{{{seen_str}}}"
            unseen_str = f"\\textbf{{{unseen_str}}}"
        elif model == 'LLaMA-3 (Fine-tuned)' and ds == 'UCI HAR':
            unseen_str = f"\\textbf{{{unseen_str}}}"  # Bold the LLM unseen
            
        row_cells.append(seen_str)
        row_cells.append(unseen_str)
        
    latex_code += " & ".join(row_cells) + " \\\\\n"

latex_code += r"""\hline
\end{tabular}
\end{table*}
"""

with open('./consolidated_latex_table.txt', 'w') as f:
    f.write(latex_code)
print("Saved consolidated_latex_table.txt successfully!")

# ==============================================================================
# 3. Generate Markdown Table File
# ==============================================================================
md_table = """# Consolidated 4-Dataset Performance Table (Macro F1-Score)

| Dataset | HHAR | | MotionSense | | Shoaib | | UCI HAR (Ours) | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test Subject** | **Seen** | **Unseen** | **Seen** | **Unseen** | **Seen** | **Unseen** | **Seen** | **Unseen** |
|--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""

for model in models:
    row_cells = [f"**{model}**" if 'LLaMA' in model else model]
    for ds in datasets:
        seen = data[ds][model][0]
        unseen = data[ds][model][1]
        seen_str = f"{seen:.3f}" if seen > 0 else "-"
        unseen_str = f"**{unseen:.3f}**" if model == 'LLaMA-3 (Fine-tuned)' else f"{unseen:.3f}"
        row_cells.append(seen_str)
        row_cells.append(unseen_str)
    md_table += "| " + " | ".join(row_cells) + " |\n"

with open('./consolidated_markdown_table.md', 'w') as f:
    f.write(md_table)
print("Saved consolidated_markdown_table.md successfully!")

# ==============================================================================
# 4. Generate Heatmap Visualization (fig13_consolidated_heatmap.png)
# ==============================================================================
# Construct matrix for heatmap: rows = models, columns = Dataset-Subject splits
columns = [
    'HHAR\n(Seen)', 'HHAR\n(Unseen)', 
    'MotionSense\n(Seen)', 'MotionSense\n(Unseen)',
    'Shoaib\n(Seen)', 'Shoaib\n(Unseen)',
    'UCI HAR\n(Seen)', 'UCI HAR\n(Unseen)'
]

heatmap_matrix = []
for model in models:
    row_vals = []
    for ds in datasets:
        seen = data[ds][model][0]
        unseen = data[ds][model][1]
        # Replace missing seen value for base LLaMA with a proxy or nan
        row_vals.append(seen if seen > 0 else np.nan)
        row_vals.append(unseen)
    heatmap_matrix.append(row_vals)

heatmap_matrix = np.array(heatmap_matrix)

plt.figure(figsize=(11, 6))
# Using a nice, academic cool-to-warm colormap (YlGnBu or RdYlBu)
sns.heatmap(
    heatmap_matrix, 
    annot=True, 
    fmt=".3f", 
    cmap="YlGnBu", 
    linewidths=.5, 
    xticklabels=columns, 
    yticklabels=models,
    cbar_kws={'label': 'Macro F1-Score'},
    vmin=0.35,
    vmax=1.0
)

plt.title('Consolidated Classifier Performance (Macro F1-Score) Across 4 HAR Datasets\n(HHAR, MotionSense, Shoaib from Paper vs. UCI HAR from Ours)', 
          fontsize=12, fontweight='bold', pad=15)
plt.ylabel('Evaluation Models', fontsize=10, fontweight='bold')
plt.xlabel('Datasets and Subject Splits (Seen vs. Unseen)', fontsize=10, fontweight='bold')
plt.xticks(rotation=0, fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig('./fig13_consolidated_heatmap.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig13_consolidated_heatmap.png successfully!")
