.. Chemicals_Pathway_Optimizer documentation master file, created by
   sphinx-quickstart on Thu Mar 13 14:18:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Chemicals_Pathway_Optimizer documentation
=========================================


Welcome to the Multi-Objective Optimization project documentation! This project includes
scripts for performing multi-objective optimization, post-optimization analysis, and Pareto
front visualizations. This project contains two main Python modules that work together to perform multi-objective optimization of feedstock usage and post-optimization analysis. The outputs from both modules (e.g., CSV files, PDF diagrams, HTML reports, and pickle files) are automatically saved in dedicated output folders. The folder names include key parameters such as the optimization metric, target year, and whether 2030 results are fixed.

This project is designed to:

1. **Perform multi-objective optimization:**  
   The optimization module uses Pyomo to model feedstock constraints, product demand, and co-product relationships. Multiple metrics (e.g., greenhouse gas emissions, price, human toxicity) can be optimized individually.
   
2. **Conduct post-optimization analysis:**  
   After running the optimization, the analysis module merges the optimization results with feedstock availability, generates visualizations (bar charts, chord diagrams, Sankey diagrams), and saves the outputs.

3. **Automate output management:**  
   Both modules automatically create output folders whose names include the metric, target year, and a flag indicating if 2030 results are fixed. This makes it easier to track and organize the generated results.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Modules:

   installation
   usage
   main_run
   optimization
   pareto
   run_multiobjective_2030
   run_multiobjective_2050


.. autosummary::
   :toctree: _autosummary  

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`