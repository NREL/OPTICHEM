"""
Pareto Optimization Plotting Module 2030
=====================================

This module generates Pareto front plots by calling the
:func:`pareto.multiobjective_optimization` function. Two main routines are provided:
one to explore the trade-off between GHG emissions (as a constraint) and price, and
another to explore the trade-off between human toxicity and GHG emissions.

:author: tghosh
"""

from pareto import multiobjective_optimization
import pandas as pd
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import os


years = [2030]
for yr in years:
    high_trl = (yr == 2030)
    fix_2030_results = False

    def main():
        """
        Perform multi-objective optimization with a constraint on GHG emissions and plot
        the Pareto front for GHG emissions vs. price.

        The routine initializes optimization parameters, repeatedly calls
        :func:`pareto.multiobjective_optimization` while decreasing the GHG constraint,
        and stores the resulting objective values. Finally, it creates and saves a scatterplot
        that visualizes the trade-off between total GHG emissions and total price.

        :return: None
        """
        # Parameters and settings
        constrained_ghg_metric = "ghg_emissions"
        optimized_metric = "price_total"
        
        ghg_upper_bounds = 311.1746
        ghg_lower_bounds = -26.701
        constrained_ghg_metric_val = ghg_upper_bounds
        count = 3.36

        # Initialize lists to store results
        ghg_value = []
        price_value = []
        human_tox = []

        output_folder = f"./pareto_results_{yr}_fix2030_{fix_2030_results}"
        os.makedirs(output_folder, exist_ok=True)
        
        # Single iteration (adjust the loop range if more iterations are needed)
        for i in range(100):
            print(f"Optimizing {constrained_ghg_metric} with constraint value: {constrained_ghg_metric_val}")
            # Multiobjective optimization call
            a, b, c = multiobjective_optimization(
                [constrained_ghg_metric, constrained_ghg_metric_val],
                [optimized_metric],
                yr,
                high_trl,
                fix_2030_results
            )
            # Adjust the constraint for subsequent iterations if applicable
            constrained_ghg_metric_val -= count

            # Sum the values and store in lists
            ghg_value.append(sum(a['ghg_emissions']))
            price_value.append(sum(a['price_total']))
            human_tox.append(sum(a['human_toxicity']))

        # Create a DataFrame with results
        multi_objective_result_df = pd.DataFrame({
            'GHG Emissions': ghg_value,
            'Price Total': price_value,
            'Human Toxicity': human_tox
        })

        # Plotting improvements for publication
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.2)
        plt.figure(figsize=(8, 6))
        plt.rcParams["font.family"] = "Arial"
        # Create a scatterplot with enhanced aesthetics
        scatter_plot = sns.scatterplot(
            data=multi_objective_result_df,
            x='GHG Emissions',
            y='Price Total',
            hue='Human Toxicity',
            palette="viridis",
            s=100,           # Marker size
            edgecolor="black",
            alpha=0.8
        )

        # Set axis labels and title with improved formatting
        plt.xlabel("Total GHG Emissions ", fontsize=14)
        plt.ylabel("Total Price ", fontsize=14)
        plt.title(f"Pareto Front: GHG Emissions vs Price {yr}", fontsize=16)

        # Customize legend for clarity
        plt.legend(title="Human Toxicity", title_fontsize='13', fontsize='11', loc='best')

        # Adjust layout and save the figure with publication-quality resolution
        plt.tight_layout()
        plt.savefig(output_folder + f"/pareto_front GHG Emissions vs Price {yr}.png", dpi=300)
        # plt.show()

    if __name__ == "__main__":
        main()

    #%%

    def main2():
        """
        Perform multi-objective optimization with a constraint on human toxicity and plot
        the Pareto front for human toxicity vs. GHG emissions.

        This routine sets optimization parameters for human toxicity, calls the
        :func:`pareto.multiobjective_optimization` function repeatedly while reducing the
        human toxicity constraint, and collects the optimization results. It then creates
        a scatterplot that visualizes the trade-off between total human toxicity and total GHG emissions.

        :return: None
        """
        # Parameters and settings
        constrained_ht_metric = "human_toxicity"
        optimized_metric = "ghg_emissions"
        
        ht_upper_bounds = 419.348
        ht_lower_bounds = 177.17763
        constrained_ht_metric_val = ht_upper_bounds
        count = 2.3977

        # Initialize lists to store results
        ghg_value = []
        price_value = []
        human_tox = []
        
        output_folder = f"./pareto_results_{yr}_fix2030_{fix_2030_results}"
        os.makedirs(output_folder, exist_ok=True)

        # Single iteration (adjust the loop range if more iterations are needed)
        for i in range(100):
            print(f"Optimizing {constrained_ht_metric} with constraint value: {constrained_ht_metric_val}")
            # Multiobjective optimization call
            a, b, c = multiobjective_optimization(
                [constrained_ht_metric, constrained_ht_metric_val],
                [optimized_metric],
                yr,
                high_trl,
                fix_2030_results
            )
            # Adjust the constraint for subsequent iterations if applicable
            constrained_ht_metric_val -= count

            # Sum the values and store in lists
            ghg_value.append(sum(a['ghg_emissions']))
            price_value.append(sum(a['price_total']))
            human_tox.append(sum(a['human_toxicity']))

        # Create a DataFrame with results
        multi_objective_result_df = pd.DataFrame({
            'GHG Emissions': ghg_value,
            'Price Total': price_value,
            'Human Toxicity': human_tox
        })

        # Plotting improvements for publication
        sns.set_theme(style="whitegrid")
        sns.set_context("paper", font_scale=1.2)
        plt.figure(figsize=(8, 6))
        plt.rcParams["font.family"] = "Arial"
        # Create a scatterplot with enhanced aesthetics
        scatter_plot = sns.scatterplot(
            data=multi_objective_result_df,
            x='Human Toxicity',
            y='GHG Emissions',
            hue='Price Total',
            palette="viridis",
            s=100,           # Marker size
            edgecolor="black",
            alpha=0.8
        )

        # Set axis labels and title with improved formatting
        plt.xlabel("Total Human Toxicity ", fontsize=14)
        plt.ylabel("Total GHG Emissions ", fontsize=14)
        plt.title(f"Pareto Front: Human Toxicity vs GHG Emissions {yr}", fontsize=16)

        # Customize legend for clarity
        plt.legend(title="Price Total", title_fontsize='13', fontsize='11', loc='best')

        # Adjust layout and save the figure with publication-quality resolution
        plt.tight_layout()
        plt.savefig(output_folder + f"/pareto_Human Toxicity vs GHG_Totals {yr}.png", dpi=300)
        # plt.show()

    if __name__ == "__main__":
        main2()
