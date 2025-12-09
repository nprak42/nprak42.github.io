"""
Essay Classification Results Visualization Script
Run this locally after downloading predictions CSV from Kaggle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report
import os

# Configuration
PREDICTIONS_FILE = "roberta_predictions.csv"  # Update this path
OUTPUT_DIR = "analysis_outputs"
TRAIT = "courage"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100

print("="*70)
print("ESSAY CLASSIFICATION ANALYSIS")
print("="*70)

# ============================================================================
# 1. LOAD DATA & BASIC STATS
# ============================================================================
print("\n[1/7] Loading data and computing basic statistics...")

df = pd.read_csv(PREDICTIONS_FILE)

print(f"\nTotal predictions: {len(df)}")
overall_accuracy = (df['true_label'] == df['predicted_label']).mean()
print(f"Overall Accuracy: {overall_accuracy:.2%}")

print(f"\nTrue label distribution:")
print(df['true_label'].value_counts().sort_index().to_string())

# ============================================================================
# 2. CONFUSION MATRIX
# ============================================================================
print("\n[2/7] Generating confusion matrix...")

cm = confusion_matrix(df['true_label'], df['predicted_label'])

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Score {i+1}' for i in range(5)],
            yticklabels=[f'Score {i+1}' for i in range(5)],
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/confusion_matrix.png")
plt.close()

# Most common misclassifications
errors = df[df['true_label'] != df['predicted_label']]
if len(errors) > 0:
    error_pairs = errors.groupby(['true_label', 'predicted_label']).size().sort_values(ascending=False)
    print(f"\nMost common misclassifications:")
    for (true_l, pred_l), count in error_pairs.head(5).items():
        print(f"  True Score {true_l+1} → Predicted Score {pred_l+1}: {count} times")

# ============================================================================
# 3. PREDICTION CONFIDENCE ANALYSIS
# ============================================================================
print("\n[3/7] Analyzing prediction confidence...")

df['max_prob'] = df[[f'prob_class_{i}' for i in range(5)]].max(axis=1)
df['correct'] = (df['true_label'] == df['predicted_label']).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram
correct_conf = df[df['correct'] == 1]['max_prob']
incorrect_conf = df[df['correct'] == 0]['max_prob']

axes[0].hist([correct_conf, incorrect_conf], bins=20, 
             label=['Correct', 'Incorrect'], alpha=0.7, color=['green', 'red'])
axes[0].set_xlabel('Prediction Confidence', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot
bp = axes[1].boxplot([incorrect_conf, correct_conf], labels=['Incorrect', 'Correct'],
                       patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], ['lightcoral', 'lightgreen']):
    patch.set_facecolor(color)
axes[1].set_ylabel('Prediction Confidence', fontsize=11)
axes[1].set_title('Confidence by Correctness', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confidence_analysis.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/confidence_analysis.png")
plt.close()

print(f"\nMean confidence when correct: {correct_conf.mean():.3f}")
print(f"Mean confidence when incorrect: {incorrect_conf.mean():.3f}")

# ============================================================================
# 4. PER-CLASS PERFORMANCE
# ============================================================================
print("\n[4/7] Computing per-class performance metrics...")

precision, recall, f1, support = precision_recall_fscore_support(
    df['true_label'], df['predicted_label'], labels=[0,1,2,3,4]
)

metrics_df = pd.DataFrame({
    'Score': [f'Score {i+1}' for i in range(5)],
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

print("\nPer-class metrics:")
print(metrics_df.to_string(index=False))

# Save to CSV
metrics_df.to_csv(f"{OUTPUT_DIR}/per_class_metrics.csv", index=False)
print(f"✅ Saved: {OUTPUT_DIR}/per_class_metrics.csv")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

x = np.arange(5)
width = 0.6

axes[0].bar(x, precision, width, color='skyblue', edgecolor='black', linewidth=0.5)
axes[0].set_title('Precision by Score', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Precision', fontsize=11)
axes[0].set_xlabel('Score', fontsize=11)
axes[0].set_xticks(x)
axes[0].set_xticklabels([f'{i+1}' for i in range(5)])
axes[0].set_ylim(0, 1)
axes[0].grid(alpha=0.3, axis='y')

axes[1].bar(x, recall, width, color='lightcoral', edgecolor='black', linewidth=0.5)
axes[1].set_title('Recall by Score', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Recall', fontsize=11)
axes[1].set_xlabel('Score', fontsize=11)
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'{i+1}' for i in range(5)])
axes[1].set_ylim(0, 1)
axes[1].grid(alpha=0.3, axis='y')

axes[2].bar(x, f1, width, color='lightgreen', edgecolor='black', linewidth=0.5)
axes[2].set_title('F1-Score by Score', fontsize=12, fontweight='bold')
axes[2].set_ylabel('F1-Score', fontsize=11)
axes[2].set_xlabel('Score', fontsize=11)
axes[2].set_xticks(x)
axes[2].set_xticklabels([f'{i+1}' for i in range(5)])
axes[2].set_ylim(0, 1)
axes[2].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/per_class_performance.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/per_class_performance.png")
plt.close()

# ============================================================================
# 5. PROBABILITY DISTRIBUTION PATTERNS
# ============================================================================
print("\n[5/7] Analyzing probability distribution patterns...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()

for true_label in range(5):
    ax = axes[true_label]
    
    subset = df[df['true_label'] == true_label]
    correct = subset[subset['correct'] == 1]
    incorrect = subset[subset['correct'] == 0]
    
    # Average probability distribution
    correct_probs = correct[[f'prob_class_{i}' for i in range(5)]].mean() if len(correct) > 0 else None
    incorrect_probs = incorrect[[f'prob_class_{i}' for i in range(5)]].mean() if len(incorrect) > 0 else None
    
    x = np.arange(5)
    width = 0.35
    
    if correct_probs is not None:
        ax.bar(x - width/2, correct_probs, width, label='Correct', 
               alpha=0.8, color='green', edgecolor='black', linewidth=0.5)
    if incorrect_probs is not None:
        ax.bar(x + width/2, incorrect_probs, width, label='Incorrect', 
               alpha=0.8, color='red', edgecolor='black', linewidth=0.5)
    
    ax.set_title(f'True Label: Score {true_label+1}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted Class', fontsize=10)
    ax.set_ylabel('Mean Probability', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(5)])
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis='y')

# Remove extra subplot
axes[5].axis('off')

plt.suptitle('Average Probability Distribution by True Label', 
             fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/probability_patterns.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/probability_patterns.png")
plt.close()

# ============================================================================
# 6. LOW CONFIDENCE PREDICTIONS
# ============================================================================
print("\n[6/7] Identifying low confidence predictions...")

low_confidence = df[df['max_prob'] < 0.5].copy()

print(f"\n{len(low_confidence)} predictions with confidence < 0.5 ({100*len(low_confidence)/len(df):.1f}%)")

if len(low_confidence) > 0:
    low_conf_accuracy = (low_confidence['true_label'] == low_confidence['predicted_label']).mean()
    print(f"Accuracy on low-confidence predictions: {low_conf_accuracy:.2%}")
    
    # Save low confidence predictions
    low_confidence_export = low_confidence[['true_label', 'predicted_label', 'max_prob'] + 
                                           [f'prob_class_{i}' for i in range(5)]]
    low_confidence_export.to_csv(f"{OUTPUT_DIR}/low_confidence_predictions.csv", index=False)
    print(f"✅ Saved: {OUTPUT_DIR}/low_confidence_predictions.csv")
else:
    print("No low-confidence predictions found!")

# ============================================================================
# 7. ERROR MAGNITUDE ANALYSIS (ADJACENCY)
# ============================================================================
print("\n[7/7] Analyzing error magnitude (off-by-one errors)...")

df['error_magnitude'] = abs(df['true_label'] - df['predicted_label'])

print("\nError magnitude distribution:")
error_dist = df['error_magnitude'].value_counts().sort_index()
for mag, count in error_dist.items():
    print(f"  Off by {mag}: {count} ({100*count/len(df):.1f}%)")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Bar chart of error magnitudes
error_dist.plot(kind='bar', ax=axes[0], color='coral', edgecolor='black', linewidth=0.5)
axes[0].set_xlabel('Prediction Error (classes off)', fontsize=11)
axes[0].set_ylabel('Count', fontsize=11)
axes[0].set_title('Error Magnitude Distribution', fontsize=12, fontweight='bold')
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].grid(alpha=0.3, axis='y')

# Grouped bar chart
exact = (df['error_magnitude'] == 0).sum()
off_by_one = (df['error_magnitude'] == 1).sum()
off_by_two_plus = (df['error_magnitude'] >= 2).sum()

axes[1].bar(['Exact', 'Off by 1', 'Off by 2+'], 
            [exact, off_by_one, off_by_two_plus],
            color=['green', 'orange', 'red'], 
            edgecolor='black', linewidth=0.5)
axes[1].set_ylabel('Count', fontsize=11)
axes[1].set_title('Prediction Accuracy Grouping', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

# Add percentage labels on bars
for i, (label, val) in enumerate(zip(['Exact', 'Off by 1', 'Off by 2+'], 
                                      [exact, off_by_one, off_by_two_plus])):
    axes[1].text(i, val + len(df)*0.02, f'{100*val/len(df):.1f}%', 
                ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/error_magnitude.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/error_magnitude.png")
plt.close()

print(f"\nExact matches: {exact} ({100*exact/len(df):.1f}%)")
print(f"Off by 1: {off_by_one} ({100*off_by_one/len(df):.1f}%)")
print(f"Off by 2+: {off_by_two_plus} ({100*off_by_two_plus/len(df):.1f}%)")

# ============================================================================
# SUMMARY REPORT
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS COMPLETE - SUMMARY")
print("="*70)

summary = f"""
Dataset: {PREDICTIONS_FILE}
Total Predictions: {len(df)}
Overall Accuracy: {overall_accuracy:.2%}

Confidence Statistics:
  - Mean confidence (correct): {correct_conf.mean():.3f}
  - Mean confidence (incorrect): {incorrect_conf.mean():.3f}
  - Low confidence predictions (<0.5): {len(low_confidence)} ({100*len(low_confidence)/len(df):.1f}%)

Error Analysis:
  - Exact matches: {exact} ({100*exact/len(df):.1f}%)
  - Off by 1 class: {off_by_one} ({100*off_by_one/len(df):.1f}%)
  - Off by 2+ classes: {off_by_two_plus} ({100*off_by_two_plus/len(df):.1f}%)

All outputs saved to: {OUTPUT_DIR}/
"""

print(summary)

# Save summary to text file
with open(f"{OUTPUT_DIR}/analysis_summary.txt", "w") as f:
    f.write("ESSAY CLASSIFICATION ANALYSIS SUMMARY\n")
    f.write("="*70 + "\n")
    f.write(summary)
    f.write("\n" + "="*70 + "\n")
    f.write("\nCLASSIFICATION REPORT:\n")
    f.write(classification_report(df['true_label'], df['predicted_label'], 
                                   target_names=[f'Score {i+1}' for i in range(5)]))

print(f"✅ Saved: {OUTPUT_DIR}/analysis_summary.txt")
print("\n✨ All visualizations and analyses complete!")