#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-Optimization Analysis Script
=================================

This module performs a post-optimization analysis that includes:

- Running a multi-objective optimization via :func:`multiobjective_optimization`.
- Merging feedstock usage data with constraints.
- Creating a feedstock-availability bar chart.
- Generating chord and Sankey diagrams for flow visualization.

:author: tghosh
:date: Wed Mar 12 20:42:23 2025
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# Third-party chord diagram library
from mpl_chord_diagram import chord_diagram

# Local module with the optimization function
from optimization import multiobjective_optimization

warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output


def parse_user_inputs() -> tuple[int, bool, str, bool]:
    """
    Define or parse user inputs for year, TRL filter, metric, and fix_2030_results toggle.

    :return: A tuple containing:
        - **yr** (*int*): Year to analyze (e.g., 2030 or 2050).
        - **trl** (*bool*): True if 'High TRL' data should be used.
        - **metric** (*str*): The name of the metric to optimize (e.g., 'ghg_emissions').
        - **fix_2030_results** (*bool*): Whether to fix certain flows/variables to a 2030 reference solution.
    :rtype: tuple[int, bool, str, bool]
    """
    # Hard-coded inputs here; could be replaced with arg parsing
    yr = [2030,2050]  # Note: if multiple years are provided, the first element is used for folder naming.
    trl = True  # High TRL if year == 2030 (this may need adjustment if yr is a list)
    metric_list = ["ghg_emissions", "price_total", "human_toxicity"]  # Similar note as above.
    fix_2030_results = False

    return yr, trl, metric_list, fix_2030_results


def run_optimization(yr: int, trl: bool, fix_2030_results: bool, metric: str):
    """
    Run the multiobjective_optimization function and return the resulting DataFrames.

    :param yr: The target year for constraints (e.g., 2030 or 2050).
    :type yr: int
    :param trl: Whether to filter for high TRL only.
    :type trl: bool
    :param fix_2030_results: Whether to constrain flows based on 2030 data.
    :type fix_2030_results: bool
    :param metric: The metric to optimize.
    :type metric: str
    :return: A tuple containing:
        - **eq_df** (*pd.DataFrame*): Detailed optimization results at equation-level.
        - **feed_df** (*pd.DataFrame*): Aggregated feedstock usage data.
        - **prod_df** (*pd.DataFrame*): Aggregated product flow data (including co-products).
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    eq_df, feed_df, prod_df = multiobjective_optimization(
        constrained_metric=[],
        metrics=[metric],
        yr=yr,
        trl=trl,
        fix_2030_results=fix_2030_results
    )
    return eq_df, feed_df, prod_df


def merge_feedstock_availability(feed_df: pd.DataFrame, yr: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge the feedstock usage with its availability for the given year, compute the remaining
    availability, and sort the results in descending order of usage.

    :param feed_df: Feedstock usage dataframe from the optimization.
    :type feed_df: pd.DataFrame
    :param yr: The year used for selecting the correct availability column.
    :type yr: int
    :return: A tuple containing:
        - **feed_df_merged** (*pd.DataFrame*): The merged dataframe with additional columns:
          ['Availability {yr} (MMT)'] and 'Difference'.
        - **feedstock_avail** (*pd.DataFrame*): The feedstock availability dataframe.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    # Load feedstock availability
    feedstock = pd.read_csv('feedstock_limits.csv')
    feedstock_avail = feedstock[['Feedstock', f'Availability {yr} (MMT)']]

    # Merge and compute difference
    feed_df_merged = feed_df.merge(feedstock_avail, on="Feedstock", how='left').fillna(0)
    feed_df_merged['Difference'] = (
        feed_df_merged[f'Availability {yr} (MMT)'] - feed_df_merged['Feedstock_flow']
    )
    feed_df_merged = feed_df_merged.sort_values('Feedstock_flow', ascending=False)
    return feed_df_merged, feedstock_avail


def plot_feedstock_bars(feed_df: pd.DataFrame, yr: int, output_folder: str):
    """
    Create a stacked horizontal bar chart of feedstock usage versus availability.

    :param feed_df: DataFrame with columns ['Feedstock_flow', f'Availability {yr} (MMT)', 'Feedstock'].
    :type feed_df: pd.DataFrame
    :param yr: The target year (for labeling).
    :type yr: int
    :param output_folder: Path to the folder where the output files will be saved.
    :type output_folder: str
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        x=-feed_df[f'Availability {yr} (MMT)'],
        y=feed_df['Feedstock'],
        color='red',
        label='Available',
        ax=ax
    )
    sns.barplot(
        x=feed_df['Feedstock_flow'],
        y=feed_df['Feedstock'],
        color='blue',
        label='Used',
        ax=ax
    )

    ax.set_title(f"Feedstock Usage vs. Availability ({yr})")
    ax.set_xlabel("Quantity (MMT)")
    ax.set_ylabel("Feedstock")
    ax.legend()
    plt.tight_layout()
    # plt.show()
    pdf_path = os.path.join(output_folder, 'tornado_plot.pdf')
    fig.savefig(pdf_path)


def create_chord_diagram(eq_df: pd.DataFrame, feed_df: pd.DataFrame, yr: int, output_folder: str):
    """
    Build and display a chord diagram illustrating the flow from feedstock to chemicals.

    :param eq_df: DataFrame that must include the columns ['Feedstock', 'Chemical', 'Feedstock_flow'].
    :type eq_df: pd.DataFrame
    :param feed_df: DataFrame used to determine main feedstocks.
    :type feed_df: pd.DataFrame
    :param yr: The target year, used only for naming or in the figure title if desired.
    :type yr: int
    :param output_folder: Path to the folder where the output files will be saved.
    :type output_folder: str
    """
    network_df = eq_df[['Feedstock', 'Chemical', 'Feedstock_flow']].copy()
    network_df = network_df[network_df['Feedstock_flow'] > 1E-10]  # Filter negligible flows

    main_feedstocks = list(pd.unique(feed_df['Feedstock']))
    tertiary_feedstocks = list(pd.unique(eq_df['Feedstock']))

    # Build a combined sorted list of unique feedstocks and chemicals
    feedstocks = network_df['Feedstock'].unique()
    chemicals = network_df['Chemical'].unique()
    combined = sorted(set(feedstocks) | set(chemicals))

    # Create a square adjacency matrix
    flow_df = pd.DataFrame(0, index=combined, columns=combined, dtype=float)
    for _, row in network_df.iterrows():
        flow_df.loc[row['Feedstock'], row['Chemical']] += row['Feedstock_flow']

    flow_array = flow_df.to_numpy()

    # Generate three color lists from different colormaps
    n1 = 43  # 41-color list using viridis
    n2 = 10  # 10-color list using reds
    n3 = 9   # 9-color list using blues

    colors_list1 = [plt.cm.viridis(i / (n1 - 1)) for i in range(n1)]
    colors_list2 = [plt.cm.Reds(0.3 + 0.4 * i / (n2 - 1)) for i in range(n2)]
    colors_list3 = [plt.cm.Blues(0.3 + 0.4 * i / (n3 - 1)) for i in range(n3)]

    def rgba_to_hex(rgba):
        """
        Convert an RGBA tuple (with floats in the range [0,1]) to a hexadecimal color string.

        :param rgba: A tuple representing an RGBA color.
        :type rgba: tuple
        :return: The hexadecimal color string.
        :rtype: str
        """
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgba[0] * 255),
            int(rgba[1] * 255),
            int(rgba[2] * 255)
        )

    hex_colors_list1 = [rgba_to_hex(c) for c in colors_list1]
    hex_colors_list2 = [rgba_to_hex(c) for c in colors_list2]
    hex_colors_list3 = [rgba_to_hex(c) for c in colors_list3]

    colors = []
    c1, c2, c3 = 0, 0, 0
    res = combined

    for r in res:
        if r in main_feedstocks:
            colors.append(hex_colors_list3[c1])
            c1 += 1
        elif (r in tertiary_feedstocks) and (r not in main_feedstocks):
            colors.append(hex_colors_list2[c2])
            c2 += 1
        else:
            colors.append(hex_colors_list1[c3])
            c3 += 1

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.rcParams["font.family"] = "Arial"

    chord_diagram(
        flow_array,
        combined,
        ax=ax,
        fontsize=9,
        sort="distance",
        directed=True,
        rotate_names=True,
        colors=colors,
        use_gradient=True
    )
    # plt.show()
    chord_pdf_path = os.path.join(output_folder, 'chord_diagram.pdf')
    fig.savefig(chord_pdf_path)


def create_sankey_feedstocks(eq_df: pd.DataFrame, html_filename: str, output_folder: str):
    """
    Generate a Sankey diagram showing multi-use feedstocks flowing to chemicals,
    excluding flows that are 100% dedicated to one chemical.

    :param eq_df: DataFrame that must contain columns:
                  ['Feedstock_flow', 'Chemical', 'Feedstock', 'Long name'] for flow details.
    :type eq_df: pd.DataFrame
    :param html_filename: File path (including folder) to save the resulting Sankey diagram as an HTML file.
    :type html_filename: str
    :param output_folder: Path to the folder where CSV outputs will be saved.
    :type output_folder: str
    """
    temp_df = eq_df[['Feedstock_flow', 'Chemical', 'Feedstock', 'Long name']].copy()
    total_flow = temp_df.groupby('Chemical')['Feedstock_flow'].sum().reset_index()
    merged = temp_df.merge(total_flow, on='Chemical', suffixes=('_x', '_y'))
    merged['percent'] = merged['Feedstock_flow_x'] / merged['Feedstock_flow_y'] * 100
    merged = merged[merged['percent'] != 100]

    csv_path = os.path.join(output_folder, 'feedstock_multiuse.csv')
    merged.to_csv(csv_path, index=False)

    feedstock_list = list(pd.unique(merged['Feedstock']))
    chemical_list = list(pd.unique(merged['Chemical']))
    all_nodes = sorted(set(feedstock_list + chemical_list))

    source = [all_nodes.index(f) for f in merged['Feedstock']]
    target = [all_nodes.index(c) for c in merged['Chemical']]
    value = merged['percent'].tolist()
    link_labels = merged['Long name'].tolist()

    cmap = plt.get_cmap("coolwarm", len(merged))
    rgba_colors = [
        "rgba({}, {}, {}, 0.4)".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in (cmap(i) for i in range(len(merged)))
    ]

    link_data = dict(source=source, target=target, value=value, label=link_labels, color=rgba_colors)
    node_data = dict(pad=15, thickness=15, label=all_nodes, color=rgba_colors)

    sankey_data = go.Sankey(node=node_data, link=link_data)
    fig = go.Figure(sankey_data)

    fig.update_layout(
        title_text="Sankey Diagram: Feedstock → Chemical (Multi-Use Feedstocks Only)",
        font=dict(size=20)
    )
    # fig.show()

    fig.write_html(html_filename)


def create_sankey_pathways(eq_df: pd.DataFrame, html_filename: str, output_folder: str):
    """
    Generate a Sankey diagram showing multi-use pathways flowing to chemicals,
    excluding flows that are 100% dedicated to one chemical.

    :param eq_df: DataFrame that must contain columns:
                  ['product_flow', 'Chemical', 'Long name'] for flow details.
    :type eq_df: pd.DataFrame
    :param html_filename: File path (including folder) to save the resulting Sankey diagram as an HTML file.
    :type html_filename: str
    :param output_folder: Path to the folder where CSV outputs will be saved.
    :type output_folder: str
    """
    temp_df = eq_df[['Chemical', 'Long name', 'product_flow']].copy()
    total_flow = temp_df.groupby('Chemical')['product_flow'].sum().reset_index()
    merged = temp_df.merge(total_flow, on='Chemical', suffixes=('_x', '_y'))
    merged['percent'] = merged['product_flow_x'] / merged['product_flow_y'] * 100
    merged = merged[merged['percent'] != 100]
    merged['Feedstock'] = merged['Long name']

    csv_path = os.path.join(output_folder, 'pathways_multiuse.csv')
    merged.to_csv(csv_path, index=False)

    source_list = list(pd.unique(merged['Feedstock']))
    target_list = list(pd.unique(merged['Chemical']))
    all_nodes = sorted(set(source_list + target_list))

    source = [all_nodes.index(f) for f in merged['Feedstock']]
    target = [all_nodes.index(c) for c in merged['Chemical']]
    value = merged['percent'].tolist()
    link_labels = merged['Long name'].tolist()

    cmap = plt.get_cmap("coolwarm", len(merged))
    rgba_colors = [
        "rgba({}, {}, {}, 0.4)".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in (cmap(i) for i in range(len(merged)))
    ]

    link_data = dict(source=source, target=target, value=value, label=link_labels, color=rgba_colors)
    node_data = dict(pad=15, thickness=15, label=all_nodes, color=rgba_colors)

    sankey_data = go.Sankey(node=node_data, link=link_data)
    fig = go.Figure(sankey_data)

    fig.update_layout(
        title_text="Sankey Diagram: Pathways → Chemical (Multi-Use Pathways Only)",
        font=dict(size=20)
    )
    # fig.show()

    fig.write_html(html_filename)


def main():
    """
    Main function tying the whole workflow together.

    The workflow includes:
    
    1. Parsing user inputs.
    2. Running the optimization.
    3. Analyzing feedstock usage by merging with availability.
    4. Plotting the feedstock usage bar chart.
    5. Creating the chord diagram.
    6. Generating Sankey diagrams for feedstock and pathway flows.
    """
    # 1. User inputs
    yr, trl, metric, fix_2030_results = parse_user_inputs()

    # If the inputs are lists, use the first element for folder naming.
    # Loop through every year and metric combination
    for yr_val in yr:
        for metric_val in metric:

            if (yr_val == 2030):
                trl = True
            else:
                trl = False

            # Create output folder with a name that includes metric, year, and fix_2030_results flag.
            output_folder = f"results_{metric_val}_{yr_val}_fix2030_{fix_2030_results}"
            os.makedirs(output_folder, exist_ok=True)

            # 2. Optimization
            if fix_2030_results == True and yr_val == 2050 and trl == False:
                # Saving the pickle file
                run_optimization(2030, True, False, metric_val)

            eq_df, feed_df, prod_df = run_optimization(yr_val, trl, fix_2030_results, metric_val)

            # 3. Merge feedstock usage with availability
            feed_df_merged, feedstock_availability = merge_feedstock_availability(feed_df, yr_val)

            # 4. Plot feedstock usage
            plot_feedstock_bars(feed_df_merged, yr_val, output_folder)

            # 5. Chord diagram: feedstock → chemical
            create_chord_diagram(eq_df, feedstock_availability, yr_val, output_folder)

            # 6. Sankey diagrams:
            #    (a) Feedstock → Chemical
            sankey_feedstocks_html = os.path.join(output_folder, "sankey-diagram-plotly-real_feedstock.html")
            create_sankey_feedstocks(eq_df, sankey_feedstocks_html, output_folder)

            #    (b) Pathways → Chemical
            sankey_pathways_html = os.path.join(output_folder, "sankey-diagram-plotly-real_pathways.html")
            create_sankey_pathways(eq_df, sankey_pathways_html, output_folder)

    print("All steps completed successfully.")


if __name__ == "__main__":
    main()
