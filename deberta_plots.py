import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load your CSV
df = pd.read_csv('deberta_predictions.csv')

# Clean up the data - remove 'tensor(' and ')' from essay_id if present
if df['essay_id'].dtype == 'object':
    df['essay_id'] = df['essay_id'].str.replace('tensor(', '').str.replace(')', '')

print(f"Loaded {len(df)} predictions")
print(f"Accuracy: {df['is_correct'].mean():.2%}")
print(f"Within 1 score: {df['within_1'].mean():.2%}")

# ============================================
# 1. CONFUSION MATRIX
# ============================================
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

cm = confusion_matrix(df['true_score'], df['predicted_score'])

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[1,2,3,4,5], yticklabels=[1,2,3,4,5],
            cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Score', fontsize=12, fontweight='bold')
ax.set_ylabel('True Score', fontsize=12, fontweight='bold')
ax.set_title('DeBERTa Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('deberta_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: deberta_confusion_matrix.png")
plt.close()

# ============================================
# 2. CONFIDENCE vs ACCURACY
# ============================================
# Bin by confidence levels
df['confidence_bin'] = pd.cut(df['confidence'], 
                               bins=[0, 0.4, 0.5, 0.6, 0.7, 1.0],
                               labels=['<40%', '40-50%', '50-60%', '60-70%', '>70%'])

confidence_accuracy = df.groupby('confidence_bin', observed=True).agg({
    'is_correct': 'mean',
    'essay_id': 'count'
}).reset_index()
confidence_accuracy.columns = ['Confidence', 'Accuracy', 'Count']

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(confidence_accuracy['Confidence'], 
               confidence_accuracy['Accuracy'], 
               color='#5eadbd', alpha=0.8, edgecolor='black')

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, confidence_accuracy['Count'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'n={count}',
            ha='center', va='bottom', fontsize=10)

ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% baseline')
ax.set_xlabel('Model Confidence', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax.set_title('Model Accuracy by Confidence Level', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()
plt.savefig('confidence_vs_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: confidence_vs_accuracy.png")
plt.close()

# ============================================
# 3. ENTROPY DISTRIBUTION
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['entropy'], bins=30, color='#5eadbd', alpha=0.7, edgecolor='black')
ax.axvline(df['entropy'].median(), color='red', linestyle='--', 
           label=f'Median: {df["entropy"].median():.2f}')
ax.set_xlabel('Entropy (Uncertainty)', fontsize=12, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax.set_title('Distribution of Model Uncertainty', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('entropy_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: entropy_distribution.png")
plt.close()

# ============================================
# 4. ERROR ANALYSIS - Entropy vs Correctness
# ============================================
fig, ax = plt.subplots(figsize=(10, 6))

correct = df[df['is_correct'] == True]
incorrect = df[df['is_correct'] == False]

ax.scatter(correct['entropy'], correct['confidence'], 
           alpha=0.5, label='Correct', color='green', s=50)
ax.scatter(incorrect['entropy'], incorrect['confidence'], 
           alpha=0.5, label='Incorrect', color='red', s=50, marker='x')

ax.set_xlabel('Entropy (Uncertainty)', fontsize=12, fontweight='bold')
ax.set_ylabel('Confidence', fontsize=12, fontweight='bold')
ax.set_title('Model Confidence vs Uncertainty: Correct vs Incorrect Predictions', 
             fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('error_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: error_analysis.png")
plt.close()

# ============================================
# 5. PROBABILITY DISTRIBUTION EXAMPLES
# ============================================
# Get interesting examples: high confidence correct, high confidence wrong, uncertain

high_conf_correct = df[(df['confidence'] > 0.7) & (df['is_correct'] == True)].iloc[0]
high_conf_wrong = df[(df['confidence'] > 0.6) & (df['is_correct'] == False)].iloc[0] if len(df[(df['confidence'] > 0.6) & (df['is_correct'] == False)]) > 0 else None
uncertain = df[(df['entropy'] > df['entropy'].quantile(0.75))].iloc[0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

examples = [
    (high_conf_correct, 'High Confidence - Correct', 0),
    (high_conf_wrong, 'High Confidence - Incorrect', 1) if high_conf_wrong is not None else (None, None, 1),
    (uncertain, 'High Uncertainty', 2)
]

for example, title, idx in examples:
    if example is None:
        axes[idx].text(0.5, 0.5, 'No examples', ha='center', va='center')
        axes[idx].set_title(title)
        continue
        
    probs = [example['prob_class_1'], example['prob_class_2'], 
             example['prob_class_3'], example['prob_class_4'], example['prob_class_5']]
    
    colors = ['#cccccc'] * 5
    colors[int(example['true_score'])-1] = '#10b981'  # True score in green
    colors[int(example['predicted_score'])-1] = '#5eadbd'  # Predicted in teal
    
    bars = axes[idx].bar([1,2,3,4,5], probs, color=colors, alpha=0.8, edgecolor='black')
    axes[idx].set_xlabel('Score', fontweight='bold')
    axes[idx].set_ylabel('Probability', fontweight='bold')
    axes[idx].set_title(f'{title}\nTrue: {int(example["true_score"])}, Pred: {int(example["predicted_score"])} ({example["confidence"]:.1%} conf)', 
                        fontweight='bold')
    axes[idx].set_ylim(0, 1)
    axes[idx].axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig('probability_distribution_examples.png', dpi=300, bbox_inches='tight')
print("✓ Saved: probability_distribution_examples.png")
plt.close()

# ============================================
# 6. CLASS-WISE PERFORMANCE
# ============================================
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    df['true_score'], df['predicted_score'], labels=[1,2,3,4,5])

metrics_df = pd.DataFrame({
    'Class': [1, 2, 3, 4, 5],
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(metrics_df))
width = 0.25

bars1 = ax.bar(x - width, metrics_df['Precision'], width, label='Precision', color='#5eadbd')
bars2 = ax.bar(x, metrics_df['Recall'], width, label='Recall', color='#10b981')
bars3 = ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score', color='#f59e0b')

ax.set_xlabel('Score Class', fontsize=12, fontweight='bold')
ax.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
ax.set_title('DeBERTa Per-Class Performance', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics_df['Class'])
ax.legend()
ax.set_ylim(0, 1)

# Add support counts as text
for i, support in enumerate(metrics_df['Support']):
    ax.text(i, 0.05, f'n={int(support)}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('class_wise_performance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: class_wise_performance.png")
plt.close()

# ============================================
# SUMMARY STATISTICS
# ============================================
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"Total predictions: {len(df)}")
print(f"Overall accuracy: {df['is_correct'].mean():.2%}")
print(f"Within 1 score accuracy: {df['within_1'].mean():.2%}")
print(f"Mean confidence: {df['confidence'].mean():.2%}")
print(f"Median entropy: {df['entropy'].median():.2f}")
print(f"\nClass distribution (predictions):")
print(df['predicted_score'].value_counts().sort_index())
print(f"\nClass distribution (true):")
print(df['true_score'].value_counts().sort_index())
print("\nPer-class metrics:")
print(metrics_df.to_string(index=False))
print("="*50)

print("\n✓ All visualizations complete!")
print("\nGenerated files:")
print("  1. deberta_confusion_matrix.png")
print("  2. confidence_vs_accuracy.png")
print("  3. entropy_distribution.png")
print("  4. error_analysis.png")
print("  5. probability_distribution_examples.png")
print("  6. class_wise_performance.png")