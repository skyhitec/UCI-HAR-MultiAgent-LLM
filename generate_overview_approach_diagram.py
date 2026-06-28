import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set professional fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']

# Create a figure with a vertical layout containing subfigures (a) and (b)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))

# Custom helper function to draw rounded boxes
def draw_box(ax, label, x, y, w, h, bg_color, text_color, fontsize=8.5, border_color=None):
    if border_color is None:
        border_color = bg_color
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.06",
        linewidth=1.2,
        edgecolor=border_color,
        facecolor=bg_color,
        alpha=0.9
    )
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', color=text_color, 
            fontsize=fontsize, fontweight='bold', wrap=True)

# Custom helper to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, color='#34495E', style="->", lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               shrinkA=3, shrinkB=3, patchA=None, patchB=None))

# ==============================================================================
# SUBFIGURE (a): Creating Instruction Tuning Dataset using RAG
# ==============================================================================
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 5.5)
ax1.axis('off')

# Title for (a)
ax1.text(5, 5.2, "(a) Creating Instruction Tuning Dataset using RAG", 
         fontsize=11, fontweight='bold', color='#1B365D', ha='center')

# Draw components
draw_box(ax1, "HAR Sensor Data\n(Acc / Gyro)", 1.2, 3.8, 1.8, 0.6, '#34495E', 'white')
draw_box(ax1, "Prompt Template\n(System & User queries)", 1.2, 1.6, 1.8, 0.6, '#2E4053', 'white')

draw_box(ax1, "Large Language Model\n(Sentence Embedding Encoder)", 4.5, 3.8, 2.8, 0.6, '#2980B9', 'white')
draw_box(ax1, "Retrieval-Augmented Generator\n(FAISS Physical Patterns Index)", 4.5, 1.6, 2.8, 0.6, '#16A085', 'white')

draw_box(ax1, "Classification & Reasoning\nPrompt-Response Assembly", 8.0, 3.8, 2.4, 0.6, '#D35400', 'white')
draw_box(ax1, "Instruction-Tuning Dataset\n(Saved as JSON file)", 8.0, 1.6, 2.4, 0.6, '#8E44AD', 'white')

# Connecting arrows for (a)
draw_arrow(ax1, 2.1, 3.8, 3.1, 3.8) # HAR to Encoder
draw_arrow(ax1, 2.1, 1.6, 3.1, 1.6) # Prompt to FAISS

# RAG loop arrows
draw_arrow(ax1, 4.5, 3.5, 4.5, 1.9, style="<->", color='#E67E22', lw=1.8) # Encoder <-> FAISS loop
ax1.text(4.6, 2.7, "RAG Retrieval\nQuery & Knowledge", fontsize=7.5, color='#E67E22', fontweight='bold', va='center')

# To assembly
draw_arrow(ax1, 5.9, 3.8, 6.8, 3.8) # Encoder to Assembly
draw_arrow(ax1, 5.9, 1.6, 6.8, 3.5) # FAISS context to Assembly

# Assembly to JSON Dataset
draw_arrow(ax1, 8.0, 3.5, 8.0, 1.9, color='#8E44AD') 

# Outer boundary box for (a)
boundary_a = patches.Rectangle((0.1, 0.2), 9.8, 4.9, fill=False, edgecolor='#BDC3C7', linestyle='--', linewidth=1)
ax1.add_patch(boundary_a)


# ==============================================================================
# SUBFIGURE (b): Fine-tuning LLM for various downstream tasks
# ==============================================================================
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 5.5)
ax2.axis('off')

# Title for (b)
ax2.text(5, 5.2, "(b) Fine-tuning LLM for various downstream tasks", 
         fontsize=11, fontweight='bold', color='#1B365D', ha='center')

# Draw components
draw_box(ax2, "Pre-trained Base Model\n(Meta-Llama-3-8B-Instruct)", 1.5, 3.5, 2.2, 0.7, '#2C3E50', 'white')
draw_box(ax2, "Instruction-Tuned Dataset\n(20,264 RAG samples)", 1.5, 1.5, 2.2, 0.7, '#8E44AD', 'white')

draw_box(ax2, "Fine-Tuned HAR LLM Agent\n(Saved LoRA Adapters)", 5.2, 2.5, 2.2, 0.7, '#27AE60', 'white')

# Downstream tasks list on the right
draw_box(ax2, "Downstream Tasks", 8.2, 4.4, 2.0, 0.45, '#D35400', 'white')
draw_box(ax2, "Activity Classification", 8.2, 3.6, 2.0, 0.4, '#EBF5FB', '#2C3E50', border_color='#2980B9')
draw_box(ax2, "Chain-of-Thought Reasoning", 8.2, 2.8, 2.0, 0.4, '#EBF5FB', '#2C3E50', border_color='#2980B9')
draw_box(ax2, "Static / Dynamic Posture QA", 8.2, 2.0, 2.0, 0.4, '#EBF5FB', '#2C3E50', border_color='#2980B9')
draw_box(ax2, "Tailored Recommendations", 8.2, 1.2, 2.0, 0.4, '#EBF5FB', '#2C3E50', border_color='#2980B9')

# Connecting arrows for (b)
draw_arrow(ax2, 2.6, 3.5, 4.1, 2.7) # Base LLM to SFT
draw_arrow(ax2, 2.6, 1.5, 4.1, 2.3) # Dataset to SFT

# Supervised Fine-Tuning label
ax2.text(3.1, 2.6, "Supervised\nFine-Tuning\n(QLoRA SFT)", fontsize=8, color='#27AE60', fontweight='bold', ha='center', va='center')

# Fine-tuned LLM to Downstream Tasks
draw_arrow(ax2, 6.3, 2.5, 7.2, 2.5)

# Branching arrows to each downstream task
draw_arrow(ax2, 7.2, 2.5, 7.2, 4.4, style="-", color='#D35400')
draw_arrow(ax2, 7.2, 2.5, 7.2, 1.2, style="-", color='#D35400')
draw_arrow(ax2, 7.2, 4.4, 7.2, 4.4)
draw_arrow(ax2, 7.2, 3.6, 7.2, 3.6)
draw_arrow(ax2, 7.2, 2.8, 7.2, 2.8)
draw_arrow(ax2, 7.2, 2.0, 7.2, 2.0)
draw_arrow(ax2, 7.2, 1.2, 7.2, 1.2)

# Outer boundary box for (b)
boundary_b = patches.Rectangle((0.1, 0.2), 9.8, 4.9, fill=False, edgecolor='#BDC3C7', linestyle='--', linewidth=1)
ax2.add_patch(boundary_b)

plt.suptitle('Fig. 10: Overview of the proposed approach.', fontsize=13, fontweight='bold', y=1.02, color='#1B365D')
plt.tight_layout()

# Save figure in high-resolution
plt.savefig('./fig10_overview_proposed_approach.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved fig10_overview_proposed_approach.png successfully!")
