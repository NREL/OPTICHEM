#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Objective Optimization (Refactored)
===========================================

Created on Thu Mar 13 10:22:35 2025

This module implements a multi-objective optimization framework using Pyomo.
It loads and processes data, builds an optimization model with various constraints
and objective metrics, and performs post-processing to generate output files.

:author: tghosh
"""

import os
import pandas as pd
import pickle
import warnings
from pyomo.environ import (
    ConcreteModel,
    Set,
    Var,
    NonNegativeReals,
    ConstraintList,
    Objective,
    minimize,
    SolverFactory
)

warnings.filterwarnings('ignore')  # Suppress warning messages for clarity


def load_and_filter_data(yr: int, trl: bool) -> pd.DataFrame:
    """
    Load the main 'full_data.csv' dataset and filter it if TRL is True.

    :param yr: The target year (currently not used in filtering).
    :type yr: int
    :param trl: Flag indicating whether to filter for high TRL data.
    :type trl: bool
    :return: A pandas DataFrame containing the loaded (and possibly filtered) data.
    :rtype: pd.DataFrame
    """
    data = pd.read_csv('full_data.csv')
    if trl:
        data = data[data['High TRL']]
    return data


def sum_co_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjust feedstock usage based on co-product fractions.

    This function computes the total product flow by summing the co-product amounts
    and adjusts the main product mass fraction accordingly, then calculates the feedstock
    inflow and corresponding product yield.

    :param df: DataFrame containing feedstock and co-product usage data.
    :type df: pd.DataFrame
    :return: A pandas DataFrame with additional columns for adjusted feedstock usage.
    :rtype: pd.DataFrame
    """
    df['Total product flow'] = 1 + df[
        [
            'Co-product 1 amount (MMT/MMT)',
            'Co-product 2 amount (MMT/MMT)',
            'Co-product 3 amount (MMT/MMT)',
            'Co-product 4 amount (MMT/MMT)'
        ]
    ].sum(axis=1)

    df['mass fraction of main poduct'] = 1 / df['Total product flow']
    df['total feedstock inflow for 1 MMT product'] = (
        df['Feedstock use (MMT/MMT)'] / df['mass fraction of main poduct']
    )
    df['MMT product per 1 MMT feedstock'] = (
        1 / df['total feedstock inflow for 1 MMT product']
    )
    return df


def load_constraint_data(yr: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load additional constraint data for feedstock availability and product demand.

    :param yr: The target year for constraints.
    :type yr: int
    :return: A tuple of two pandas DataFrames:
             - The first DataFrame contains feedstock availability data.
             - The second DataFrame contains product demand data.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    feedstock = pd.read_csv('feedstock_limits.csv')
    product = pd.read_csv('product_limits.csv')
    feedstock2 = feedstock[['Feedstock', f'Availability {yr} (MMT)']]
    product2 = product[['Chemical', f'Demand by society {yr} (MMT)']]
    return feedstock2, product2


def build_co_product_dictionary(data: pd.DataFrame) -> dict:
    """
    Create a dictionary mapping co-product relationships from the data.

    This function extracts rows with co-product information and builds a dictionary
    with unique keys (using the row index) and sorted lists of co-product names.

    :param data: DataFrame containing production process data with co-product columns.
    :type data: pd.DataFrame
    :return: A dictionary where each key corresponds to a process and its value is a sorted list of co-products.
    :rtype: dict
    """
    data_cp = data[data['Co-product']]
    data_cp = data_cp[
        [
            'Long name',
            'Co-product 1 long name',
            'Co-product 2 long name',
            'Co-product 3 long name',
            'Co-product 4 long name'
        ]
    ]

    data_cp_dic = {}
    for index, row in data_cp.iterrows():
        columns_names = [
            'Co-product 1 long name',
            'Co-product 2 long name',
            'Co-product 3 long name',
            'Co-product 4 long name'
        ]
        cp_list = [row['Long name']]
        for cn in columns_names:
            if isinstance(row[cn], str) and row[cn] != 0:
                cp_list.append(row[cn])
        cp_list = sorted(cp_list)
        if cp_list not in data_cp_dic.values():
            data_cp_dic[str(index) + 'Set'] = cp_list

    return data_cp_dic


def build_equation_dataframe(model, data: pd.DataFrame) -> pd.DataFrame:
    """
    Link model variables (x) to the underlying DataFrame representing physical flows.

    This function constructs a DataFrame that maps Pyomo decision variables to their
    corresponding process pathways, merges it with the original data, and computes derived
    metrics such as product flow and environmental impacts.

    :param model: A Pyomo model instance containing decision variables.
    :param data: DataFrame with process data and associated parameters.
    :type data: pd.DataFrame
    :return: A pandas DataFrame containing flow data and calculated metrics.
    :rtype: pd.DataFrame
    """
    variable_list = []
    pathway_list_var = []
    for m in model.M:
        variable_list.append(model.x[m])
        pathway_list_var.append(m)

    # Construct a dataframe that links Pyomo Var objects to each pathway
    model_df = pd.DataFrame({
        'Feedstock_flow': variable_list,
        'pathway': pathway_list_var
    })

    # Merge with original data
    equation_dataframe = model_df.merge(
        data,
        left_on='pathway',
        right_on='Long name'
    )

    # Calculate product flow and various metrics
    equation_dataframe['product_flow'] = (
        equation_dataframe['Feedstock_flow'] *
        equation_dataframe['MMT product per 1 MMT feedstock']
    )

    equation_dataframe['product_name'] = equation_dataframe['Chemical']
    equation_dataframe['price_total'] = (
        equation_dataframe['Price (billion USD / MMT)'] *
        equation_dataframe['product_flow']
    )
    equation_dataframe['ghg_emissions'] = (
        equation_dataframe['GHG emissions (MMT CO2e/MMT)'] *
        equation_dataframe['product_flow']
    )
    equation_dataframe['human_toxicity'] = (
        equation_dataframe['Human toxicity (MMT 1,4-DCB eq/MMT)'] *
        equation_dataframe['product_flow']
    )
    equation_dataframe['electricity_use'] = (
        equation_dataframe['Electricity use (TWh/MMT)'] *
        equation_dataframe['product_flow']
    )
    equation_dataframe['hydrogen_use'] = (
        equation_dataframe['Hydrogen use (MMT/MMT)'] *
        equation_dataframe['product_flow']
    )
    equation_dataframe['fossil_fuel_depletion'] = (
        equation_dataframe['Fossil fuel depletion (MMT oil eq/MMT)'] *
        equation_dataframe['product_flow']
    )
    equation_dataframe['land_use'] = (
        equation_dataframe['Land use (km2*a crop eq/MMT)'] *
        equation_dataframe['product_flow']
    )
    equation_dataframe['water_depletion'] = (
        equation_dataframe['Water depletion (km3/MMT)'] *
        equation_dataframe['product_flow']
    )   

    return equation_dataframe


def product_dataframe_editor(
    product_df: pd.DataFrame,
    equation_df: pd.DataFrame,
    main_product_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Adjust the product DataFrame to account for intermediate feedstocks that are also products
    from a previous step.

    Negative flow values indicate the consumption of intermediate feedstocks.

    :param product_df: DataFrame containing product flow data.
    :type product_df: pd.DataFrame
    :param equation_df: DataFrame linking model variables with process data.
    :type equation_df: pd.DataFrame
    :param main_product_df: DataFrame flagging main-product processes.
    :type main_product_df: pd.DataFrame
    :return: A modified product DataFrame with corrected sign and adjusted flow values.
    :rtype: pd.DataFrame
    """
    product_dataframe_temp = product_df[['product_name']].drop_duplicates()
    intermediate_feedstock_product = equation_df.merge(
        product_dataframe_temp,
        left_on='Feedstock',
        right_on='product_name'
    )

    if intermediate_feedstock_product.empty:
        product_df['sign'] = 'positive'
        return product_df
    else:
        intermediate_feedstock_product['sign'] = 'negative'
        intermediate_feedstock_product['product_name'] = intermediate_feedstock_product['Feedstock']
        intermediate_feedstock_product['product_flow'] = -intermediate_feedstock_product['Feedstock_flow']

        intermediate_cp = intermediate_feedstock_product[
            intermediate_feedstock_product['Co-product'] == True
        ]
        intermediate_wcp = intermediate_feedstock_product[
            intermediate_feedstock_product['Co-product'] == False
        ]

        if not intermediate_cp.empty:
            intermediate_cp = intermediate_cp.merge(main_product_df, on='pathway')
            intermediate_cp = intermediate_cp[['product_name','product_flow','sign']]
        intermediate_wcp = intermediate_wcp[['product_name','product_flow','sign']]

        intermediate_edited = pd.concat([intermediate_cp, intermediate_wcp])

        product_without_co_product = product_df[product_df['Co-product'] == False].copy()
        product_with_coproduct = product_df[product_df['Co-product'] == True].copy()

        product_without_co_product['sign'] = 'positive'
        product_with_coproduct['sign'] = 'positive'

        product_without_co_product = product_without_co_product[['product_name','product_flow','sign']]
        product_with_coproduct = product_with_coproduct[['product_name','product_flow','sign']]

        complete_product_dataframe = pd.concat([
            intermediate_edited,
            product_without_co_product,
            product_with_coproduct
        ])

        return complete_product_dataframe


def reduce_extra_production(
    demand_summed: pd.DataFrame,
    yr: int
) -> float:
    """
    Calculate the total excess production beyond the demanded quantity.

    Excess production is computed as the difference between the actual product flow
    and the demand by society.

    :param demand_summed: DataFrame containing aggregated product flow and demand.
    :type demand_summed: pd.DataFrame
    :param yr: The target year for demand constraints.
    :type yr: int
    :return: The total excess production as a float.
    :rtype: float
    """
    demand_summed['excess_flow'] = (
        demand_summed['product_flow'] -
        demand_summed[f'Demand by society {yr} (MMT)']
    )
    return demand_summed['excess_flow'].sum()


def create_metrics(
    metrics: list,
    equation_dataframe: pd.DataFrame,
) -> list:
    """
    Create the objective function based on a given metric.

    The function aggregates the specified metric from the equation_dataframe
    and constructs an objective expression.

    :param metrics: List of metrics. Only the first metric in the list is used.
    :type metrics: list
    :param equation_dataframe: DataFrame containing process data and calculated metrics.
    :type equation_dataframe: pd.DataFrame
    :return: A tuple containing the total metric value and the metric name.
    :rtype: list
    """
    # metrics should be just one
    met = metrics[0]
    # 6.1 Construct objective expression
    metric_val_total = 0
    metric_df = equation_dataframe[['product_name', met]].groupby('product_name').sum().reset_index()
    for _, row in metric_df.iterrows():
        metric_val_total += row[met]
    return metric_val_total, met


def multiobjective_optimization(
    constrained_metric,
    metrics,
    yr,
    trl,
    fix_2030_results
):
    """
    Perform multi-objective optimization using Pyomo.

    This function sets up and solves an optimization model. It loads and preprocesses data,
    defines decision variables and constraints, constructs the objective function based on the provided metric,
    and performs post-processing to generate output files.

    :param constrained_metric: List of metrics to be constrained (currently not used).
    :param metrics: List of metrics for the objective function. Only the first metric is used.
    :param yr: The target year for constraints.
    :param trl: Boolean flag indicating whether to filter for high TRL data.
    :param fix_2030_results: Boolean flag determining if 2030 results should be fixed.
    :return: A tuple containing:
             - The final equation DataFrame with calculated flows and metrics.
             - The aggregated feedstock DataFrame.
             - The complete product DataFrame.
    :rtype: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
    """
    # ----------------- Create Output Folder -----------------
    # Use the first metric in the list for naming purposes.
    metric_val = metrics[0] if isinstance(metrics, list) else metrics
    output_folder = f"results_{metric_val}_{yr}_fix2030_{fix_2030_results}"
    os.makedirs(output_folder, exist_ok=True)
    
    # --------------------------------------------------------------
    # 1. Initialize and Read Data
    # --------------------------------------------------------------
    model = ConcreteModel()
    data = load_and_filter_data(yr, trl)

    # Split into co-product & no co-product
    data_cp = data[data['Co-product']].fillna(0)
    data_wcp = data[~data['Co-product']].fillna(0)

    # Adjust feedstock usage with co-product logic
    data_cp = sum_co_products(data_cp)
    data_wcp = sum_co_products(data_wcp)
    data = pd.concat([data_cp, data_wcp])

    # Load constraint data
    feedstock2, product2 = load_constraint_data(yr)

    # Merge constraints into main dataset
    data_product_limit = data.merge(product2, on='Chemical')
    data_feedstock_limit = data.merge(feedstock2, on='Feedstock')

    # Create smaller supply & demand DataFrames
    supply_dataframe = (
        data_feedstock_limit[['Feedstock', f'Availability {yr} (MMT)']]
        .drop_duplicates()
        .dropna()
    )
    demand_dataframe = (
        data_product_limit[['Chemical', f'Demand by society {yr} (MMT)']]
        .drop_duplicates()
        .dropna()
    )

    # Build dictionary for co-product processes
    data_cp_dic = build_co_product_dictionary(data)

    # Create a DataFrame to flag main-product processes
    main_product_df = pd.DataFrame({
        'pathway': [v[0] for v in data_cp_dic.values()],
        'indicator': 'copy of one product'
    })

    # --------------------------------------------------------------
    # 2. Define Sets & Variables
    # --------------------------------------------------------------
    pathways = list(pd.unique(data['Long name']))
    model.M = Set(initialize=pathways)
    model.x = Var(model.M, domain=NonNegativeReals, bounds=(0,1000))
    model.y = Var(model.M, domain=NonNegativeReals, bounds=(0,1000))

    # --------------------------------------------------------------
    # 3. Build Equation DataFrame
    # --------------------------------------------------------------
    equation_dataframe = build_equation_dataframe(model, data)

    # --------------------------------------------------------------
    # 4. Define Constraints
    # --------------------------------------------------------------
    model.constraints = ConstraintList()

    # 4.1 Feedstock constraints
    feedstock_dataframe = equation_dataframe[['Feedstock','Feedstock_flow','Co-product','pathway']]
    feedstock_cp = feedstock_dataframe[feedstock_dataframe['Co-product']]
    feedstock_wcp = feedstock_dataframe[~feedstock_dataframe['Co-product']]

    if not feedstock_cp.empty:
        feedstock_cp = feedstock_cp.merge(main_product_df, on='pathway')
        feedstock_cp = feedstock_cp[['Feedstock', 'Feedstock_flow']]

    feedstock_wcp = feedstock_wcp[['Feedstock', 'Feedstock_flow']]
    feedstock_combined = pd.concat([feedstock_cp, feedstock_wcp])
    feedstock_summed = feedstock_combined.groupby('Feedstock')['Feedstock_flow'].sum().reset_index()
    feedstock_summed = feedstock_summed.merge(supply_dataframe, on='Feedstock')

    for _, row in feedstock_summed.iterrows():
        model.constraints.add(row['Feedstock_flow'] <= row[f'Availability {yr} (MMT)'])

    # 4.2 Demand constraints
    product_dataframe_temp = equation_dataframe[['product_flow','product_name','Co-product','price_total']]
    complete_product_dataframe = product_dataframe_editor(product_dataframe_temp, equation_dataframe, main_product_df)
    demand_summed = complete_product_dataframe.groupby('product_name')['product_flow'].sum().reset_index()
    demand_summed = demand_summed.merge(demand_dataframe, left_on='product_name', right_on='Chemical')

    for _, row in demand_summed.iterrows():
        model.constraints.add(
            row['product_flow'] >= row[f'Demand by society {yr} (MMT)']
        )

    # 4.3 Co-product linking constraints
    co_product_df = equation_dataframe[['pathway','Feedstock_flow']].copy()
    co_product_df.set_index('pathway', inplace=True)
    for dic_key in data_cp_dic.keys():
        all_prods = data_cp_dic[dic_key]
        main_prod = all_prods[0]
        for i in range(1, len(all_prods)):
            model.constraints.add(
                co_product_df.loc[main_prod, 'Feedstock_flow'] ==
                co_product_df.loc[all_prods[i], 'Feedstock_flow']
            )

    # --------------------------------------------------------------
    # 5. Extra Constraint Handling (Fix 2030 if needed)
    # --------------------------------------------------------------
    if fix_2030_results:
        with open('./pickle/'+metric_val+'2030data.pkl', 'rb') as file:
            loaded_data = pickle.load(file)
        for key, val in loaded_data.items():
            model.constraints.add(model.x[key] >= val)

        model.constraints.add(
            reduce_extra_production(demand_summed, yr) <= 0.36379692400003269
        )
    else:
        model.constraints.add(
            reduce_extra_production(demand_summed, yr) <= 9.089951014118469e-8
        )


    # --------------------------------------------------------------
    # 6. Derive Constrained Metric
    # --------------------------------------------------------------
    c_metric_val_total, c_met = create_metrics(constrained_metric, equation_dataframe)
    model.constraints.add(c_metric_val_total <= float(constrained_metric[1]))


    # --------------------------------------------------------------
    # 6. Solve Model for Each Metric
    # --------------------------------------------------------------

    metric_val_total, met = create_metrics(metrics, equation_dataframe)
    try:
        model.del_component(model.obj)
    except:
        pass
    model.obj = Objective(expr=metric_val_total, sense=minimize)

    solver = SolverFactory('glpk')
    result = solver.solve(model)
    print()
    print(met+f"{yr}"+"-"+str(trl)+"-"+str(fix_2030_results))
    print(f"Optimal Solution for metric: {met}")
    print(f"Objective value = {model.obj.expr()}")
    
    """
    for m in model.M:
        print(f"{m}: {model.x[m].value}")

    if 'Iterations' in result.solver:
        print(f"Total Iterations: {result.solver.Iterations}")
    if 'Nodes' in result.solver:
        print(f"Total Nodes Explored: {result.solver.Nodes}")
    print(result.solver.termination_condition)
    """

    # ----------------------------------------------------------
    # 7. Post-Processing: Final Solution Details
    # ----------------------------------------------------------
    variable_list_final = []
    pathway_list_final = []
    model_df_dic_result = {}

    for m in model.M:
        val = model.x[m].value
        variable_list_final.append(val)
        pathway_list_final.append(m)
        model_df_dic_result[m] = val

    model_df_result = pd.DataFrame({
        'Feedstock_flow': variable_list_final,
        'pathway': pathway_list_final
    })

    # Save pickle file into output folder
    pickle_path = os.path.join("./pickle", met+f"{yr}data.pkl")
    with open(pickle_path, "wb") as file:
        pickle.dump(model_df_dic_result, file)

    equation_dataframe_result = model_df_result.merge(
        data,
        left_on='pathway',
        right_on='Long name'
    )

    wcp_mask = (equation_dataframe_result['Co-product'] == False)
    cp_mask = (equation_dataframe_result['Co-product'] == True)
    feedstock_wcp_res = equation_dataframe_result[wcp_mask]
    feedstock_cp_res = equation_dataframe_result[cp_mask]

    if feedstock_cp_res.empty:
        feedstock_dataframe_temp_result_joined = pd.DataFrame()
    else:
        feedstock_cp_res = feedstock_cp_res.merge(main_product_df, on='pathway')
        feedstock_dataframe_temp_result_joined = pd.concat([
            feedstock_wcp_res,
            feedstock_cp_res
        ])

    if not feedstock_cp_res.empty:
        feedstock_dataframe_temp_result = feedstock_dataframe_temp_result_joined[
            ['Feedstock', 'Feedstock_flow']
        ]
    else:
        feedstock_dataframe_temp_result = feedstock_wcp_res[
            ['Feedstock', 'Feedstock_flow']
        ]
    feedstock_dataframe_temp_result = feedstock_dataframe_temp_result.groupby('Feedstock')['Feedstock_flow'].sum().reset_index()

    equation_dataframe_result['product_flow'] = (
        equation_dataframe_result['Feedstock_flow'] *
        equation_dataframe_result['MMT product per 1 MMT feedstock']
    )
    equation_dataframe_result['price_total'] = (
        equation_dataframe_result['Price (billion USD / MMT)'] *
        equation_dataframe_result['product_flow']
    )
    equation_dataframe_result['ghg_emissions'] = (
        equation_dataframe_result['GHG emissions (MMT CO2e/MMT)'] *
        equation_dataframe_result['product_flow']
    )
    equation_dataframe_result['human_toxicity'] = (
        equation_dataframe_result['Human toxicity (MMT 1,4-DCB eq/MMT)'] *
        equation_dataframe_result['product_flow']
    )
    equation_dataframe_result['electricity_use'] = (
        equation_dataframe_result['Electricity use (TWh/MMT)'] *
        equation_dataframe_result['product_flow']
    )
    equation_dataframe_result['hydrogen_use'] = (
        equation_dataframe_result['Hydrogen use (MMT/MMT)'] *
        equation_dataframe_result['product_flow']
    )
    equation_dataframe_result['fossil_fuel_depletion'] = (
        equation_dataframe_result['Fossil fuel depletion (MMT oil eq/MMT)'] *
        equation_dataframe_result['product_flow']
    )
    equation_dataframe_result['land_use'] = (
        equation_dataframe_result['Land use (km2*a crop eq/MMT)'] *
        equation_dataframe_result['product_flow']
    )
    equation_dataframe_result['water_depletion'] = (
        equation_dataframe_result['Water depletion (km3/MMT)'] *
        equation_dataframe_result['product_flow']
    )   

    equation_dataframe_result['product_name'] = equation_dataframe_result['Chemical']
    product_dataframe_result = equation_dataframe_result[
        [
            'product_flow', 'product_name', 'Co-product',
            'Co-product 1 long name', 'price_total'
        ]
    ]
    complete_product_dataframe_result = (
        product_dataframe_editor(
            product_dataframe_result,
            equation_dataframe_result,
            main_product_df
        )
        .groupby('product_name')['product_flow']
        .sum()
        .reset_index()
    )

    # Save CSV outputs to output folder
    eq_csv_path = os.path.join(output_folder, f"{met}equation_dataframe.csv")
    prod_csv_path = os.path.join(output_folder, f"{met}product_dataframe.csv")
    feed_csv_path = os.path.join(output_folder, f"{met}feedstock_dataframe.csv")
    demand_csv_path = os.path.join(output_folder, f"{met}demand_validity.csv")
    equation_dataframe_result.to_csv(eq_csv_path, index=False)
    complete_product_dataframe_result.to_csv(prod_csv_path, index=False)
    feedstock_dataframe_temp_result.to_csv(feed_csv_path, index=False)

    product_data = pd.read_csv('product_limits.csv')
    product_data['product_name'] = product_data['Chemical']
    data_product_limit2 = product_data.merge(
        complete_product_dataframe_result,
        on='product_name',
        how='left'
    )
    data_product_limit2['excess'] = abs(
        data_product_limit2['product_flow'] -
        data_product_limit2[f'Demand by society {yr} (MMT)']
    )
    data_product_limit2 = data_product_limit2.fillna("missing")
    data_product_limit3 = data_product_limit2[data_product_limit2['product_flow'] == "missing"]
    data_product_limit4 = data_product_limit2[data_product_limit2['product_flow'] != "missing"]
    data_product_limit4.to_csv(demand_csv_path, index=False)
    print('Total excess flow: ', data_product_limit4['excess'].sum())
    print()

    missing_products = data_product_limit3
    missing_products.to_csv(os.path.join(output_folder, 'missing_products.csv'), index=False)

    indicators = ['price_total','ghg_emissions','human_toxicity','water_depletion','fossil_fuel_depletion','land_use','electricity_use','hydrogen_use']
    print("Values of other metrics:")
    for i in indicators:
        print(f"{i} ", sum(equation_dataframe_result[i]))
    """
    print("\nModel Properties:")
    print(f"Total Variables: {model.nvariables()}")
    print(f"Total Constraints: {model.nconstraints()}")
    print(f"Total Objectives: {model.nobjectives()}")
    """

    # Return results for the last metric processed (or store them differently if needed)
    return (
        equation_dataframe_result,
        feedstock_dataframe_temp_result,
        complete_product_dataframe_result
    )
