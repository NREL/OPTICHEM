
---

# Chemicals production pathway optimizer

This project contains two main Python modules that work together to perform multi-objective optimization of feedstock usage and post-optimization analysis. The outputs from both modules (e.g., CSV files, PDF diagrams, HTML reports, and pickle files) are automatically saved in dedicated output folders. The folder names include key parameters such as the optimization metric, target year, and whether 2030 results are fixed.

---

## Documentation

[Documentation](https://pages.github.nrel.gov/tghosh/chemicals_optimization/)

---

## Table of Contents

- [Overview](#overview)
- [Modules Description](#modules-description)
  - [Post-Optimization Analysis Script](#post-optimization-analysis-script)
  - [Multi-Objective Optimization (Refactored)](#multi-objective-optimization-refactored)
- [Pathways Data & Optimization Scenarios](#pathways-data--optimization-scenarios)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Customization](#customization)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project is designed to:

1. **Perform multi-objective optimization:**  
   The optimization module uses Pyomo to model feedstock constraints, product demand, and co-product relationships. Multiple metrics (e.g., greenhouse gas emissions, price, human toxicity) can be optimized individually.
   
2. **Conduct post-optimization analysis:**  
   After running the optimization, the analysis module merges the optimization results with feedstock availability, generates visualizations (bar charts, chord diagrams, Sankey diagrams), and saves the outputs.

3. **Automate output management:**  
   Both modules automatically create output folders whose names include the metric, target year, and a flag indicating if 2030 results are fixed. This makes it easier to track and organize the generated results.

---

## Modules Description

### Post-Optimization Analysis Script

- **Functionality:**
  - Parses user inputs (year, TRL filter, metric list, fix_2030_results flag).
  - Calls the multi-objective optimization function.
  - Merges feedstock usage with availability data.
  - Generates a stacked bar chart of feedstock usage vs. availability.
  - Produces a chord diagram to visualize feedstock-to-chemical flows.
  - Creates two types of Sankey diagrams:
    - One for multi-use feedstocks.
    - One for pathways (products) flows.
  - Saves all outputs (charts, CSVs, HTML files) in an output folder whose name reflects the selected parameters.

- **Key Functions:**
  - `parse_user_inputs()`
  - `run_optimization()`
  - `merge_feedstock_availability()`
  - `plot_feedstock_bars()`
  - `create_chord_diagram()`
  - `create_sankey_feedstocks()`
  - `create_sankey_pathways()`

---

### Multi-Objective Optimization (Refactored)

- **Functionality:**
  - Loads and filters data from `full_data.csv` (with optional filtering based on TRL).
  - Adjusts feedstock usage to account for co-product fractions.
  - Loads constraint data from `feedstock_limits.csv` and `product_limits.csv`.
  - Constructs a co-product dictionary to link processes.
  - Builds an equation dataframe linking optimization variables to physical flows.
  - Defines constraints (feedstock availability, product demand, co-product linking).
  - Supports extra constraint handling (fixing results for 2030 if needed).
  - Iterates through a list of metrics and solves the optimization problem for each.
  - Post-processes the results by merging optimization outputs with input data, recomputing metrics, and saving CSVs and pickle files.
  - Saves all output files in an output folder named according to the metric, target year, and fix flag.

- **Key Functions:**
  - `load_and_filter_data()`
  - `sum_co_products()`
  - `load_constraint_data()`
  - `build_co_product_dictionary()`
  - `build_equation_dataframe()`
  - `product_dataframe_editor()`
  - `reduce_extra_production()`
  - `multiobjective_optimization()`

- **Looping Through Parameter Lists:**  
  The code includes a snippet to loop through lists of years and metrics to process every combination. For example:

  ```python
  # Ensure yr and metric are lists
  yr_list = yr if isinstance(yr, list) else [yr]
  metric_list = metric if isinstance(metric, list) else [metric]

  # Loop through every year and metric combination
  for y in yr_list:
      for m in metric_list:
          output_folder = f"results_{m}_{y}_fix2030_{fix_2030_results}"
          os.makedirs(output_folder, exist_ok=True)
          print(f"Processing for Year: {y}, Metric: {m}. Output folder: {output_folder}")
          # Place additional processing code here using y and m as needed.
  ```

---

## Pathways Data & Optimization Scenarios

The **'pathways'** tab contains detailed information on all the alternative chemical production pathways. The columns provide the following data:

- **Column A:** Chemical being produced.
- **Column B:** Indicates if the chemical is a Tier 1 (platform) chemical:
  - `TRUE` if Tier 1.
  - `FALSE` otherwise.
  
  **Note:**  
  A three-step optimization approach is proposed:
  1. **First Step:** Optimize for Tier 1 chemicals only (TRUE entries).
  2. **Second Step:** For non-Tier 1 chemicals that use a Tier 1 chemical as feedstock, calculate the costs and impacts by multiplying the quantity of the required Tier 1 chemical with the optimized costs and impacts from the Tier 1 optimization. Then add these values to the baseline impacts of the non-Tier 1 production.
  3. **Third Step:** Run a second optimization for non-Tier 1 chemicals (FALSE entries).

- **Column C:** Feedstock being used to make the chemical.
- **Column D:** Long name of the production pathway.
- **Column E:** Indicates if the pathway is a high TRL technology:
  - `TRUE` means the pathway is available for the 2030 scenario.
  - `FALSE` means it can be used for the 2050 scenario.
- **Column F:** Price of the pathway in billion USD per million metric ton (MMT) of chemical produced.
- **Column G:** GHG emissions of the pathway in MMT CO2e per MMT of chemical produced.
- **Column H:** Fossil fuel depletion of the pathway in MMT oil eq per MMT of chemical produced.
- **Column I:** Human toxicity of the pathway in MMT 1,4-DCB eq per MMT of chemical produced.
- **Column J:** Land use of the pathway in km² per MMT of chemical produced.
- **Column K:** Water depletion of the pathway in km³ per MMT of chemical produced.
- **Column L:** Quantity of the feedstock (from Column C) in MMT required to produce 1 MMT of chemical.
- **Column M:** Quantity of electricity in TWh needed to produce 1 MMT of chemical.
- **Column N:** Quantity of hydrogen in MMT required to produce 1 MMT of chemical.
- **Column O:** Indicates if the pathway generates a co-product:
  - `TRUE` if a co-product is generated.
  - `FALSE` otherwise.
- **Column P:** If Column O is `TRUE`, this column lists the long name of the co-product.  
  **Note:** Some pathways can have up to four co-products. The optimization must select the pathway for all co-products simultaneously (e.g., if pyrolysis produces both ethylene and propylene, both must be selected).

### Overall Optimization Goals

Key objectives for the optimization include:

1. **Scenarios for 2030:**  
   Use only pathways that are labeled `TRUE` in Column E. The DOE's goal for this scenario is a >70% reduction in GHG emissions.
   - **Sub-scenarios:**
     - Optimize solely for GHG emissions reduction.
     - Optimize with a cost constraint.
     - Optimize for the best combination of all other metrics.

2. **Scenarios for 2050:**  
   Use all pathways (regardless of the TRL label). The DOE's goal is net-zero GHG emissions.
   - **Sub-scenarios:**
     - Optimize solely for GHG emissions reduction.
     - Optimize with a cost constraint.
     - Optimize for the best combination of all other metrics.
     - Optionally, compare a scenario where all pathways are optimized versus a scenario where the pathways selected for 2030 are retained and only additional pathways are supplemented.

3. **Resource Calculations:**  
   For each scenario, calculate the total:
   - Electricity use
   - Hydrogen use
   - Feedstock usage

---

## Prerequisites

- **Python Version:** Python 3.x  
- **Libraries/Packages:**
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - plotly
  - pyomo
  - glpk (solver)
  - mpl_chord_diagram

- **Data Files Required:**
  - `full_data.csv`
  - `feedstock_limits.csv`
  - `product_limits.csv`
  - `2030data.pkl` (if using fix_2030_results functionality)

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://your-repository-url.git
   cd your-repository-directory
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

   *(Ensure that your `requirements.txt` lists all necessary packages.)*

3. **Verify that all required CSV and pickle files are in place.**

---

## Usage

- **Run the Post-Optimization Analysis Script:**

  ```bash
  python post_optimization_analysis.py
  ```

- **Run the Multi-Objective Optimization Script:**

  ```bash
  python multiobjective_optimization.py
  ```

The scripts will automatically create output folders based on the selected year, metric, and fix flag. Check these folders for the generated CSV files, PDF diagrams, HTML reports, and pickle files.

---

## File Structure

```
.
├── post_optimization_analysis.py    # Post-Optimization Analysis Script
├── multiobjective_optimization.py   # Multi-Objective Optimization (Refactored)
├── full_data.csv                    # Main dataset
├── feedstock_limits.csv             # Feedstock availability data
├── product_limits.csv               # Product demand data
├── 2030data.pkl                     # Pickle file for fixing 2030 results
├── requirements.txt                 # List of Python package dependencies
└── README.md                        # This README file
```

---

## Customization

- **Parameter Lists:**  
  You can update the lists of years and metrics in the `parse_user_inputs()` function (or in your custom loop) to run the analysis for multiple scenarios.

- **Output Folder Naming:**  
  The output folders are named using a convention that includes the metric, year, and the `fix_2030_results` flag. Feel free to adjust the naming convention in the code if needed.

- **Solver Settings:**  
  The optimization model currently uses GLPK. You can change the solver or adjust its settings within the `multiobjective_optimization()` function.

---

## License

*(Include license details here, for example: MIT License, Apache 2.0, etc.)*

---

## Acknowledgements

- **Pyomo:** For providing a robust optimization modeling framework.
- **Matplotlib, Seaborn, Plotly:** For powerful visualization capabilities.
- **mpl_chord_diagram:** For chord diagram generation.
- **Your Data Providers:** Acknowledge any data sources if applicable.

---
