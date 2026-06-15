import matplotlib.pyplot as plt
import numpy as np
import os

# Create directory if not exists
os.makedirs('experiments/visualizations', exist_ok=True)

# Data from our analysis
datasets = ['SQuAD\n(Extractive / Exact Match)', 'Thai Wiki\n(Contextual)', 'Wongnai\n(Abstractive / Paraphrased)']
speedups = [3.50, 1.24, 1.59]
avg_accepted = [3.00, 0.24, 0.59]

x = np.arange(len(datasets))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Speedup (Bar Chart)
color1 = '#4a90e2'
bars1 = ax1.bar(x - width/2, speedups, width, label='Max Speedup (x)', color=color1, alpha=0.8)
ax1.set_ylabel('Speedup Ratio (x)', color=color1, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(0, 4.0)

# Add value labels for bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval}x', ha='center', va='bottom', color=color1, fontweight='bold')

# Plot Average Accepted Tokens (Line Chart on secondary Y-axis)
ax2 = ax1.twinx()
color2 = '#e74c3c'
line1 = ax2.plot(x + width/2, avg_accepted, color=color2, marker='o', linewidth=3, markersize=10, label='Avg Accepted Tokens/Step')
ax2.set_ylabel('Avg Accepted Tokens per Step', color=color2, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(0, 3.5)

# Add value labels for line
for i, txt in enumerate(avg_accepted):
    ax2.annotate(f'{txt} tokens', (x[i] + width/2, avg_accepted[i] + 0.15), ha='center', color=color2, fontweight='bold')

# Customize X-axis
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, fontsize=11)
ax1.set_xlabel('Dataset Data Type (Level of N-gram Rigidity)', fontsize=12, fontweight='bold')

# Add title and grid
plt.title('Visualization of N-gram Rigidity:\nHow Paraphrasing (Abstractive text) drops N-gram Drafter Performance', fontsize=14, pad=20, fontweight='bold')
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# Add legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.tight_layout()
output_path = 'experiments/visualizations/ngram_rigidity.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Visualization saved to {output_path}")
