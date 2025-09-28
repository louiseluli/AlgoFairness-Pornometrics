import matplotlib.pyplot as plt
import numpy as np

# Actual data from your mitigation results
strategies = ['Baseline\n(No mitigation)', 'Pre-processing\n(Reweighing)', 
              'In-processing\n(Fairness constraints)', 'Post-processing\n(Threshold opt.)']

accuracy = [74.7, 73.1, 73.9, 73.2]  # Test accuracy percentages
eod_reduction = [0, 40.4, 53.8, 63.5]  # EOD reduction percentages
accuracy_retained = [100, 97.9, 98.9, 98.0]  # Percentage of baseline accuracy retained

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Panel 1: Accuracy vs Fairness Trade-off
x = np.arange(len(strategies))
width = 0.35

bars1 = ax1.bar(x - width/2, accuracy, width, label='Test Accuracy', color='#4E79A7')
bars2 = ax1.bar(x + width/2, [100-e for e in eod_reduction], width, 
                label='Residual Bias', color='#E15759')

ax1.set_ylabel('Percentage (%)', fontsize=12)
ax1.set_title('(a) Accuracy vs. Bias Trade-off', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(strategies, fontsize=10)
ax1.legend(loc='upper right')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Panel 2: Fairness Improvement
colors = ['#808080', '#F28E2B', '#76B7B2', '#59A14F']
bars3 = ax2.bar(strategies, eod_reduction, color=colors)
ax2.set_ylabel('EOD Reduction (%)', fontsize=12)
ax2.set_title('(b) Fairness Improvement', fontsize=14, fontweight='bold')
ax2.set_ylim(0, 70)
ax2.grid(axis='y', alpha=0.3)

for bar, val in zip(bars3, eod_reduction):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Panel 3: Cost-Benefit Analysis
# Create scatter plot
ax3.scatter(accuracy_retained, eod_reduction, s=200, c=colors, alpha=0.7, edgecolors='black', linewidth=2)

for i, strat in enumerate(['Baseline', 'Pre-proc', 'In-proc', 'Post-proc']):
    ax3.annotate(strat, (accuracy_retained[i], eod_reduction[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax3.set_xlabel('Accuracy Retained (%)', fontsize=12)
ax3.set_ylabel('Fairness Improvement (%)', fontsize=12)
ax3.set_title('(c) Pareto-Optimal Region', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim(96, 101)
ax3.set_ylim(-5, 70)

# Add optimal region
ax3.axvspan(97.5, 99, alpha=0.2, color='green', label='Optimal region')
ax3.legend(loc='lower left')

plt.suptitle('Bias Mitigation Strategy Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('outputs/figures/discussion/mitigation_effectiveness.png', dpi=300, bbox_inches='tight')