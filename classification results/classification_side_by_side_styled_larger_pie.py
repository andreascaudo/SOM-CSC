import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os
from matplotlib.backends.backend_pdf import PdfPages

# --- Style injections from SOM-CSC script ---


def set_size(width, fraction=1, subplots=(1, 1)):
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width
    fig_width_pt = width_pt * fraction
    inches_per_pt = 1 / 72.27
    golden_ratio = (5**.5 - 1) / 2
    fig_width_in = fig_width_pt * inches_per_pt
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    return (fig_width_in, fig_height_in)


plt.style.use("seaborn-v0_8")
width_pt = 513.11743  # pt

tex_fonts = {
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}
plt.rcParams.update(tex_fonts)
# --- End of style injections ---


# Load the classified dataset
input_filename = 'classification results/dataset_classified-5.csv'
# Fallback to local file if run locally
if not os.path.exists(input_filename) and os.path.exists('dataset_classified.csv'):
    input_filename = 'dataset_classified.csv'

df = pd.read_csv(input_filename)

SAVE_PLOTS = True
# Prepare output directory
plots_dir = os.path.join(os.path.dirname(input_filename), 'plots')
if plots_dir == 'plots' or plots_dir == '/plots' or plots_dir == '':
    plots_dir = 'plots'
os.makedirs(plots_dir, exist_ok=True)

# Group by source
grouped = df.groupby('source name')

# Calculate per-source statistics
source_stats = []
for source_name, group in grouped:
    n_detections = len(group)
    n_classified = (~group['abstain']).sum()

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

        # Break ties using average confidence scores (EXACTLY AS ORIGINAL CODE)
        if has_tie:
            # Calculate average confidence for each tied class
            conf_by_class = {}
            for cls in tied_classes:
                conf_by_class[cls] = group[(~group['abstain']) & (
                    group['pred'] == cls)]['conf'].mean()
            # Pick class with highest average confidence
            most_common_class = max(conf_by_class, key=conf_by_class.get)
        else:
            most_common_class = tied_classes[0]

    else:
        most_common_class = None

    source_stats.append({
        'source name': source_name,
        'n_detections': n_detections,
        'n_classified': n_classified,
        'most_common_class': most_common_class
    })

# Convert to DataFrame
source_df = pd.DataFrame(source_stats)

# Filter sources with at least one classification
classified_sources = source_df[source_df['n_classified'] > 0].copy()

# Common settings for both plots
class_order = ["neoAGNs", "Binaries", "Stars", "neoYSOs"]
colors = ['#DBDC79', '#B4E3C4', '#F8C761', '#FF7627']  # AGN, Bin, Stars, YSO

# Create a single PDF with side-by-side plots
pdf_path = os.path.join(plots_dir, "classification_combined_side_by_side.pdf")

with PdfPages(pdf_path) as pdf:

    # Use subplots(1, 2) and size correctly based on your preferred style
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=set_size(width_pt, subplots=(1, 2)))

    # -------------------------
    # Plot 1: Detection level (ax1)
    # -------------------------
    kept = df[~df["abstain"]].copy()
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

    ax1.pie(
        class_counts.values,
        labels=class_counts.index,
        autopct='%1.1f',
        startangle=90,
        colors=colors[:len(class_counts)],
        textprops={"fontsize": 10},
        pctdistance=0.75,     # Move percentages further out to fit better
        labeldistance=1.15,   # Move labels slightly further out
        radius=1.2          # Make the pie chart itself larger
    )

    ax1.axis('equal')
    ax1.set_title("Class Distribution\n(Detection Level)", fontsize=10, pad=20)

    # -------------------------
    # Plot 2: Source level (ax2)
    # -------------------------
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

        ax2.pie(
            source_class_counts.values,
            labels=source_class_counts.index,
            autopct='%1.1f',
            startangle=90,
            colors=colors[:len(source_class_counts)],
            textprops={"fontsize": 10},
            pctdistance=0.76,
            labeldistance=1.15,
            radius=1.1
        )

        ax2.set_title("Class Distribution\n(Source Level)",
                      fontsize=10, pad=20)
        ax2.axis('equal')
    else:
        ax2.text(0.5, 0.5, 'No classified sources',
                 ha='center', va='center', fontsize=10)
        ax2.set_title('Class Distribution\n(Source Level)', fontsize=10)

    # plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(f"\nSaved side-by-side PDF to: {pdf_path}")
