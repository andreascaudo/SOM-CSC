import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

# Load the classified dataset
input_filename = 'classification results/dataset_classified-5.csv'
df = pd.read_csv(input_filename)

SAVE_PLOTS = True
SAVE_CSV = False
# Prepare output filename
base_name = os.path.splitext(input_filename)[0]
output_filename = f"{base_name}_per_source.csv"

print("=" * 80)
print("CLASSIFICATION RESULTS ANALYSIS - GROUPED BY SOURCE")
print("=" * 80)
print()

# Overall statistics
print("1. OVERALL STATISTICS")
print("-" * 80)
print(f"Total detections: {len(df)}")
print(f"Total unique sources: {df['source name'].nunique()}")
print(f"Detections with classification: {(~df['abstain']).sum()}")
print(f"Abstained detections: {df['abstain'].sum()}")
print(f"Abstention rate: {df['abstain'].sum() / len(df) * 100:.2f}%")
print()

# Classification distribution
print("2. CLASSIFICATION DISTRIBUTION (ALL DETECTIONS)")
print("-" * 80)
class_counts = df[~df['abstain']]['pred'].value_counts()
print(class_counts)
print()
for cls, count in class_counts.items():
    print(f"{cls}: {count / (~df['abstain']).sum() * 100:.2f}%")
print()

# Group by source
print("3. SOURCE-LEVEL ANALYSIS")
print("-" * 80)
grouped = df.groupby('source name')

# Calculate per-source statistics
source_stats = []
for source_name, group in grouped:
    n_detections = len(group)
    n_classified = (~group['abstain']).sum()
    n_abstained = group['abstain'].sum()

    if n_classified > 0:
        # Get classifications (excluding abstained)
        classifications = group[~group['abstain']]['pred'].tolist()
        class_counter = Counter(classifications)

        # Find the maximum count
        max_count = max(class_counter.values())

        # Get all classes with the maximum count (potential tie)
        tied_classes = [cls for cls,
                        cnt in class_counter.items() if cnt == max_count]

        # Check if there's a tie
        has_tie = len(tied_classes) > 1

        # Break ties using average confidence scores
        if has_tie:
            # Calculate average confidence for each tied class
            conf_by_class = {}
            for cls in tied_classes:
                conf_by_class[cls] = group[(~group['abstain']) & (
                    group['pred'] == cls)]['conf'].mean()
            # Pick class with highest average confidence
            most_common_class = max(conf_by_class, key=conf_by_class.get)
            tie_broken_by_conf = True
        else:
            most_common_class = tied_classes[0]
            tie_broken_by_conf = False

        most_common_count = max_count

        # Agreement rate: percentage of detections agreeing with most common class
        agreement_rate = most_common_count / n_classified if n_classified > 0 else 0

        # Check if unanimous (all agree)
        is_unanimous = len(class_counter) == 1

        # Average confidence for the most common class
        avg_conf_most_common = group[(~group['abstain']) & (
            group['pred'] == most_common_class)]['conf'].mean()

        # Overall average confidence
        avg_conf_all = group[~group['abstain']]['conf'].mean()

        # Get BMU coordinates and flags from first detection
        first_row = group.iloc[0]
        bmu_x = first_row['bmu_x'] if 'bmu_x' in group.columns else None
        bmu_y = first_row['bmu_y'] if 'bmu_y' in group.columns else None
        edge_bmu = first_row.get(
            'edge_bmu', None) if 'edge_bmu' in group.columns else None
        truncated_R = first_row.get(
            'truncated_R', None) if 'truncated_R' in group.columns else None

        source_stats.append({
            'source_name': source_name,
            'n_detections': n_detections,
            'n_classified': n_classified,
            'n_abstained': n_abstained,
            'most_common_class': most_common_class,
            'agreement_count': most_common_count,
            'agreement_rate': agreement_rate,
            'is_unanimous': is_unanimous,
            'n_unique_classes': len(class_counter),
            'all_classes': dict(class_counter),
            'avg_conf_most_common': avg_conf_most_common,
            'avg_conf_all': avg_conf_all,
            'has_tie': has_tie,
            'tie_broken_by_conf': tie_broken_by_conf,
            'bmu_x': bmu_x,
            'bmu_y': bmu_y,
            'edge_bmu': edge_bmu,
            'truncated_R': truncated_R
        })
    else:
        # All detections abstained
        first_row = group.iloc[0]
        bmu_x = first_row['bmu_x'] if 'bmu_x' in group.columns else None
        bmu_y = first_row['bmu_y'] if 'bmu_y' in group.columns else None
        edge_bmu = first_row.get(
            'edge_bmu', None) if 'edge_bmu' in group.columns else None
        truncated_R = first_row.get(
            'truncated_R', None) if 'truncated_R' in group.columns else None

        source_stats.append({
            'source_name': source_name,
            'n_detections': n_detections,
            'n_classified': 0,
            'n_abstained': n_abstained,
            'most_common_class': None,
            'agreement_count': 0,
            'agreement_rate': 0,
            'is_unanimous': False,
            'n_unique_classes': 0,
            'all_classes': {},
            'avg_conf_most_common': 0,
            'avg_conf_all': 0,
            'has_tie': False,
            'tie_broken_by_conf': False,
            'bmu_x': bmu_x,
            'bmu_y': bmu_y,
            'edge_bmu': edge_bmu,
            'truncated_R': truncated_R
        })

source_df = pd.DataFrame(source_stats)

# Calculate interestingness score for sorting
# Formula: n_detections * (1 - agreement_rate) * n_unique_classes
# This prioritizes sources with many detections, low agreement, and high class diversity
source_df['interestingness_score'] = (
    source_df['n_detections'] *
    (1 - source_df['agreement_rate']) *
    source_df['n_unique_classes']
)

# Sort by interestingness (descending) and then by n_detections (descending)
source_df_sorted = source_df.sort_values(
    by=['interestingness_score', 'n_detections'],
    ascending=[False, False]
).reset_index(drop=True)

# Create a formatted output dataframe with cleaner column names
output_df = pd.DataFrame({
    'source_name': source_df_sorted['source_name'],
    'n_detections': source_df_sorted['n_detections'],
    'n_classified': source_df_sorted['n_classified'],
    'n_abstained': source_df_sorted['n_abstained'],
    'final_classification': source_df_sorted['most_common_class'],
    'agreement_count': source_df_sorted['agreement_count'],
    'agreement_rate': source_df_sorted['agreement_rate'].round(4),
    'n_unique_classes': source_df_sorted['n_unique_classes'],
    'all_classes': source_df_sorted['all_classes'].apply(str),
    'avg_confidence_final_class': source_df_sorted['avg_conf_most_common'].round(4),
    'avg_confidence_all': source_df_sorted['avg_conf_all'].round(4),
    'interestingness_score': source_df_sorted['interestingness_score'].round(4),
    'bmu_x': source_df_sorted['bmu_x'],
    'bmu_y': source_df_sorted['bmu_y'],
    'edge_bmu': source_df_sorted['edge_bmu'],
    'truncated_R': source_df_sorted['truncated_R']
})

# Save to CSV
if SAVE_CSV:
    output_df.to_csv(output_filename, index=False)
    print(f"✓ Source-level classification saved to: {output_filename}")
print(f"  Total sources in output: {len(output_df)}")
print()
print("  Interestingness Score Formula:")
print("    score = n_detections × (1 - agreement_rate) × n_unique_classes")
print("  → Higher scores = more detections + lower agreement + more class diversity")
print()

# Show preview of most interesting sources
print("PREVIEW: Top 10 Most Interesting Sources (saved to file)")
print("-" * 80)
for idx, row in output_df.head(10).iterrows():
    print(f"{idx+1}. {row['source_name']}")
    print(
        f"   Detections: {row['n_detections']}, Classified: {row['n_classified']}, Abstained: {row['n_abstained']}")
    print(f"   Final: {row['final_classification']}, Agreement: {row['agreement_rate']*100:.1f}% ({row['agreement_count']}/{row['n_classified']})")
    print(f"   All classes: {row['all_classes']}")
    print(f"   Interestingness score: {row['interestingness_score']:.2f}")
print()

# Filter sources with classifications
classified_sources = source_df[source_df['n_classified'] > 0]

print(f"Sources with at least one classification: {len(classified_sources)}")
print(
    f"Sources with all detections abstained: {len(source_df[source_df['n_classified'] == 0])}")
print()

# BMU edge and truncation flag statistics
if len(classified_sources) > 0:
    edge_count = classified_sources['edge_bmu'].sum(
    ) if 'edge_bmu' in classified_sources.columns else 0
    truncated_count = classified_sources['truncated_R'].sum(
    ) if 'truncated_R' in classified_sources.columns else 0
    both_count = ((classified_sources['edge_bmu'] == True) & (classified_sources['truncated_R'] == True)).sum(
    ) if 'edge_bmu' in classified_sources.columns and 'truncated_R' in classified_sources.columns else 0
    either_count = ((classified_sources['edge_bmu'] == True) | (classified_sources['truncated_R'] == True)).sum(
    ) if 'edge_bmu' in classified_sources.columns and 'truncated_R' in classified_sources.columns else 0

    print(
        f"Classified sources with edge_bmu flag = True: {edge_count} ({edge_count/len(classified_sources)*100:.2f}%)")
    print(
        f"Classified sources with truncated_R flag = True: {truncated_count} ({truncated_count/len(classified_sources)*100:.2f}%)")
    print(
        f"Classified sources with both flags = True: {both_count} ({both_count/len(classified_sources)*100:.2f}%)")
    print(
        f"Classified sources with either flag = True: {either_count} ({either_count/len(classified_sources)*100:.2f}%)")
    print()

# Tie statistics
n_ties = classified_sources['has_tie'].sum()
n_ties_broken = classified_sources['tie_broken_by_conf'].sum()
print(
    f"Sources with tied classifications: {n_ties} ({n_ties/len(classified_sources)*100:.2f}%)")
if n_ties > 0:
    print(f"  → Ties broken by confidence scores: {n_ties_broken}")

    # Show examples of tied sources
    tied_sources = classified_sources[classified_sources['has_tie']].sort_values(
        'n_detections', ascending=False)
    print()
    print("  Examples of tied sources (showing top 10 by detection count):")
    for idx, (_, row) in enumerate(tied_sources.head(10).iterrows(), 1):
        print(f"    {idx}. {row['source_name']}")
        print(f"       Classifications: {row['all_classes']}")
        print(
            f"       Winner: {row['most_common_class']} (chosen by confidence)")

        # Calculate and show confidence for each tied class
        source_group = df[df['source name'] == row['source_name']]
        for cls, count in row['all_classes'].items():
            if count == row['agreement_count']:  # This is a tied class
                avg_conf = source_group[(~source_group['abstain']) & (
                    source_group['pred'] == cls)]['conf'].mean()
                winner_marker = " ← WINNER" if cls == row['most_common_class'] else ""
                print(
                    f"       {cls}: count={count}, avg_conf={avg_conf:.4f}{winner_marker}")
print()

# Multi-detection sources
multi_detection = classified_sources[classified_sources['n_classified'] > 1]
print(f"Sources with multiple detections: {len(multi_detection)}")
print()

if len(multi_detection) > 0:
    print("4. AGREEMENT STATISTICS (MULTI-DETECTION SOURCES)")
    print("-" * 80)
    unanimous = multi_detection[multi_detection['is_unanimous']]
    print(
        f"Sources with unanimous classification: {len(unanimous)} ({len(unanimous)/len(multi_detection)*100:.2f}%)")
    print(
        f"Sources with disagreement: {len(multi_detection) - len(unanimous)} ({(len(multi_detection) - len(unanimous))/len(multi_detection)*100:.2f}%)")
    print()

    print(
        f"Average agreement rate (multi-detection sources): {multi_detection['agreement_rate'].mean()*100:.2f}%")
    print(
        f"Median agreement rate (multi-detection sources): {multi_detection['agreement_rate'].median()*100:.2f}%")
    print()

    # Show sources with disagreement
    disagreement_sources = multi_detection[~multi_detection['is_unanimous']].sort_values(
        'n_detections', ascending=False)

    if len(disagreement_sources) > 0:
        print("5. SOURCES WITH CLASSIFICATION DISAGREEMENT")
        print("-" * 80)
        print(f"Total sources with disagreement: {len(disagreement_sources)}")
        print()
        print("Top 20 sources with most detections showing disagreement:")
        print()
        for idx, (_, row) in enumerate(disagreement_sources.head(20).iterrows(), 1):
            print(f"{idx}. {row['source_name']}")
            print(
                f"   Total detections: {row['n_detections']}, Classified: {row['n_classified']}")
            print(
                f"   Most common: {row['most_common_class']} ({row['agreement_count']}/{row['n_classified']} = {row['agreement_rate']*100:.1f}%)")
            print(f"   All classifications: {row['all_classes']}")
            print(f"   Avg confidence: {row['avg_conf_all']:.3f}")
            print()

    print()
    print("6. FINAL CLASSIFICATION PER SOURCE (MAJORITY VOTE)")
    print("-" * 80)
    final_class_counts = classified_sources['most_common_class'].value_counts()
    print(final_class_counts)
    print()
    for cls, count in final_class_counts.items():
        print(f"{cls}: {count / len(classified_sources) * 100:.2f}%")
    print()

# Create visualizations
print("Generating visualizations...")
print()

# Create output directory for plots
plots_dir = os.path.join(os.path.dirname(
    os.path.abspath(input_filename)), base_name)
os.makedirs(plots_dir, exist_ok=True)
print(f"Saving individual plots to: {plots_dir}/")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Classification Results Analysis - Source Level',
             fontsize=16, fontweight='bold')

# 1. Distribution of detections per source
ax1 = axes[0, 0]
detection_counts = source_df['n_detections'].value_counts().sort_index()
ax1.bar(detection_counts.index, detection_counts.values,
        color='steelblue', alpha=0.7)
ax1.set_xlabel('Number of Detections per Source')
ax1.set_ylabel('Number of Sources (log scale)')
ax1.set_yscale('log')
ax1.set_title(
    f'Distribution of Detections per Source\n(All {len(source_df)} sources including fully abstained)')
ax1.grid(True, alpha=0.3, which='both')

# 2. Classification distribution (detection level)
ax2 = axes[0, 1]
class_counts_plot = df[~df['abstain']]['pred'].value_counts()
colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
ax2.pie(class_counts_plot.values, labels=class_counts_plot.index, autopct='%1.1f%%',
        colors=colors[:len(class_counts_plot)], startangle=90)
ax2.set_title('Classification Distribution\n(Detection Level)')

# 3. Classification distribution (source level - majority vote)
ax3 = axes[0, 2]
if len(classified_sources) > 0:
    source_class_counts = classified_sources['most_common_class'].value_counts(
    )
    ax3.pie(source_class_counts.values, labels=source_class_counts.index, autopct='%1.1f%%',
            colors=colors[:len(source_class_counts)], startangle=90)
    n_multi = len(classified_sources[classified_sources['n_detections'] > 1])
    n_single = len(classified_sources) - n_multi
    ax3.set_title(
        f'Classification Distribution\n(Source Level - Majority Vote)\n{len(classified_sources)} classified sources (single: {n_single}, multi: {n_multi})')
else:
    ax3.text(0.5, 0.5, 'No classified sources', ha='center', va='center')
    ax3.set_title('Classification Distribution\n(Source Level)')

# 4. Agreement rate distribution (for multi-detection sources)
ax4 = axes[1, 0]
if len(multi_detection) > 0:
    ax4.hist(multi_detection['agreement_rate'] * 100,
             bins=20, color='green', alpha=0.7, edgecolor='black')
    ax4.axvline(multi_detection['agreement_rate'].mean() * 100, color='red', linestyle='--',
                linewidth=2, label=f'Mean: {multi_detection["agreement_rate"].mean()*100:.1f}%')
    ax4.set_xlabel('Agreement Rate (%)')
    ax4.set_ylabel('Number of Sources')
    ax4.set_title(
        f'Agreement Rate Distribution\n(Multi-Detection Sources, n={len(multi_detection)})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
else:
    ax4.text(0.5, 0.5, 'No multi-detection sources',
             ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Agreement Rate Distribution')

# 5. Confidence distribution
ax5 = axes[1, 1]
classified_df = df[~df['abstain']]
ax5.hist(classified_df['conf'], bins=30,
         color='purple', alpha=0.7, edgecolor='black')
ax5.axvline(classified_df['conf'].mean(), color='red', linestyle='--',
            linewidth=2, label=f'Mean: {classified_df["conf"].mean():.3f}')
ax5.set_xlabel('Confidence Score')
ax5.set_ylabel('Number of Detections')
ax5.set_title('Confidence Score Distribution\n(All Classifications)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Number of unique classes per source (for multi-detection sources)
ax6 = axes[1, 2]
if len(multi_detection) > 0:
    unique_classes_counts = multi_detection['n_unique_classes'].value_counts(
    ).sort_index()
    ax6.bar(unique_classes_counts.index,
            unique_classes_counts.values, color='coral', alpha=0.7)
    ax6.set_xlabel('Number of Unique Classes Assigned')
    ax6.set_ylabel('Number of Sources')
    ax6.set_title(
        f'Classification Diversity\n(Multi-Detection Sources, n={len(multi_detection)})')
    ax6.set_xticks(unique_classes_counts.index)
    ax6.grid(True, alpha=0.3)
else:
    ax6.text(0.5, 0.5, 'No multi-detection sources',
             ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Classification Diversity')

plt.tight_layout()
plt.show()

# Save individual plots
print("\nSaving individual plots...")

# Plot 1: Distribution of detections per source
fig1, ax = plt.subplots(figsize=(10, 8))
detection_counts = source_df['n_detections'].value_counts().sort_index()
ax.bar(detection_counts.index, detection_counts.values,
       color='steelblue', alpha=0.7)
ax.set_xlabel('Number of Detections per Source', fontsize=12)
ax.set_ylabel('Number of Sources (log scale)', fontsize=12)
ax.set_yscale('log')
ax.set_title(
    f'Distribution of Detections per Source\n(All {len(source_df)} sources including fully abstained)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
plt.tight_layout()
if SAVE_PLOTS:
    fig1.savefig(os.path.join(plots_dir, '01_detections_per_source.png'),
                 dpi=300, bbox_inches='tight')
plt.close(fig1)

# Common settings for both plots
class_order = ["neoAGNs", "Binaries", "Stars", "neoYSOs"]
colors = ['#DBDC79', '#B4E3C4', '#F8C761', '#FF7627']  # AGN, Bin, Stars, YSO

# -------------------------
# Plot 2: Detection level
# -------------------------
fig2, ax = plt.subplots(figsize=(6, 6))

kept = df[~df["abstain"]]
# rename kept classes to the new class names
kept['pred'] = kept['pred'].replace('YSO', 'neoYSOs')
kept['pred'] = kept['pred'].replace('AGN', 'neoAGNs')

class_counts = (
    kept["pred"]
    .value_counts()
    .reindex(class_order)   # fixed order
    .dropna()
    .astype(int)
)

wedges, texts, autotexts = ax.pie(
    class_counts.values,
    labels=class_counts.index,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors[:len(class_counts)],
    textprops={"fontsize": 12}
)

ax.axis('equal')
ax.set_title("Class Distribution\n(Detection Level)", fontsize=14)

plt.tight_layout()
if SAVE_PLOTS:
    fig2.savefig(
        os.path.join(plots_dir, "02_classification_detection_level.png"),
        dpi=600,
        bbox_inches="tight"
    )
plt.show()
plt.close(fig2)

# -------------------------
# Plot 3: Source level
# -------------------------
fig3, ax = plt.subplots(figsize=(6, 6))

classified_sources['most_common_class'] = classified_sources['most_common_class'].replace(
    'YSO', 'neoYSOs')
classified_sources['most_common_class'] = classified_sources['most_common_class'].replace(
    'AGN', 'neoAGNs')

if len(classified_sources) > 0:
    source_class_counts = (
        classified_sources["most_common_class"]
        .value_counts()
        .reindex(class_order)   # SAME fixed order
        .dropna()
        .astype(int)
    )

    ax.pie(
        source_class_counts.values,
        labels=source_class_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors[:len(source_class_counts)],
        textprops={"fontsize": 12}
    )

    ax.set_title("Class Distribution\n(Source Level)", fontsize=14)
    ax.axis('equal')
else:
    ax.text(0.5, 0.5, 'No classified sources',
            ha='center', va='center', fontsize=14)
    ax.set_title('Class Distribution\n(Source Level)', fontsize=14)

plt.tight_layout()
if SAVE_PLOTS:
    fig3.savefig(
        os.path.join(plots_dir, "03_classification_source_level.png"),
        dpi=600,
        bbox_inches="tight"
    )
plt.show()
plt.close(fig3)

# Plot 4: Agreement rate distribution
fig4, ax = plt.subplots(figsize=(10, 8))
if len(multi_detection) > 0:
    ax.hist(multi_detection['agreement_rate'] * 100,
            bins=20, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(multi_detection['agreement_rate'].mean() * 100, color='red', linestyle='--',
               linewidth=2, label=f'Mean: {multi_detection["agreement_rate"].mean()*100:.1f}%')
    ax.set_xlabel('Agreement Rate (%)', fontsize=12)
    ax.set_ylabel('Number of Sources', fontsize=12)
    ax.set_title(f'Agreement Rate Distribution\n(Multi-Detection Sources, n={len(multi_detection)})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No multi-detection sources', ha='center',
            va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('Agreement Rate Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
if SAVE_PLOTS:
    fig4.savefig(os.path.join(
        plots_dir, '04_agreement_rate_distribution.png'), dpi=300, bbox_inches='tight')
plt.close(fig4)

# Plot 5: Confidence distribution
fig5, ax = plt.subplots(figsize=(10, 8))
classified_df = df[~df['abstain']]
ax.hist(classified_df['conf'], bins=30,
        color='purple', alpha=0.7, edgecolor='black')
ax.axvline(classified_df['conf'].mean(), color='red', linestyle='--',
           linewidth=2, label=f'Mean: {classified_df["conf"].mean():.3f}')
ax.set_xlabel('Confidence Score', fontsize=12)
ax.set_ylabel('Number of Detections', fontsize=12)
ax.set_title('Confidence Score Distribution\n(All Classifications)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
if SAVE_PLOTS:
    fig5.savefig(os.path.join(plots_dir, '05_confidence_distribution.png'),
                 dpi=300, bbox_inches='tight')
plt.close(fig5)

# Plot 6: Classification diversity
fig6, ax = plt.subplots(figsize=(10, 8))
if len(multi_detection) > 0:
    unique_classes_counts = multi_detection['n_unique_classes'].value_counts(
    ).sort_index()
    ax.bar(unique_classes_counts.index,
           unique_classes_counts.values, color='coral', alpha=0.7)
    ax.set_xlabel('Number of Unique Classes Assigned', fontsize=12)
    ax.set_ylabel('Number of Sources', fontsize=12)
    ax.set_title(f'Classification Diversity\n(Multi-Detection Sources, n={len(multi_detection)})',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(unique_classes_counts.index)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'No multi-detection sources', ha='center',
            va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title('Classification Diversity', fontsize=14, fontweight='bold')
plt.tight_layout()
if SAVE_PLOTS:
    fig6.savefig(os.path.join(
        plots_dir, '06_classification_diversity.png'), dpi=300, bbox_inches='tight')
plt.close(fig6)

print(f"✓ Saved 6 individual plots to {plots_dir}/")
print("  - 01_detections_per_source.png")
print("  - 02_classification_detection_level.png")
print("  - 03_classification_source_level.png")
print("  - 04_agreement_rate_distribution.png")
print("  - 05_confidence_distribution.png")
print("  - 06_classification_diversity.png")

print("\nAnalysis complete!")
