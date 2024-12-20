This repository contains the code used in the pre-print "Predicting missing links in food webs using stacked models and species traits" -  https://doi.org/10.1101/2024.11.22.624890, Lucy Van Kleunen, Laura E. Dee, Kate L. Wootton, Francois Massol, Aaron Clauset. 2024.

Methods files:

Methods/stacking_model/OLP.py - this is the main file that implements the functions used in the stacked generalization missing link prediction procedure
Methods/stacking_model/KNN_predictors.py - this file provides helper functions for KNN predictors
Methods/stacking_model/directed_triangles.py - this file provides helper functions for directed triangle predictors
Methods/stacking_model/eco_predictors_helper.py - this file provides helper functions for custom ecological predictors
Methods/stacking_model/link_prediction_helper.py - this file provides helper functions for testing the link prediction procedure with different feature subsets on a single network

Synthetic food web tests:

Synthetic_Networks/Network_Generation_Scripts - scripts used to generate synthetic networks
Synthetic_Networks/generate_helper.py - helper functions for generating synthetic networks
Synthetic_Networks/Link_Prediction_Scripts - scripts used to run link prediction on the synthetic networks
Synthetic_Networks/link_prediction_demo_synthetic.ipynb - Jupyter notebook providing a demonstration of running link prediction on a single synthetic network, with details printed
Synthetic_Networks/Results_Demo - folder containing results from running the above demonstration script
Synthetic_Networks/visualize_synthetic_directed_results.ipynb - Jupyter notebook visualizing the results across all synthetic networks
Synthetic_Networks/viz_helper.py - helper functions for visualizing synthetic network results

Empirical food web tests:

Real_Networks/Data_Processing_Code_(Dis)Aggregated_Lifestage - Jupyter notebooks for pre-processing the data files to produce the versions of the empirical network files used in our link prediction tests for aggregated and disaggregated versions by lifestage. Metadata files are produced.
Real_Networks/data_processing_helper.py - helper functions for running data pre-processing scripts
Real_Networks/Original_Data - contains a CSV of the original data file for reproducing the processed data steps. Our study uses no original data, and this database should be cited as Brose, U. GlobAL daTabasE of traits and food Web Architecture (GATEWAy) version 1.0. iDiv https://doi.org/10.25829/idiv.283-3-756 (2018), https://idata.idiv.de/ddm/Data/ShowData/283?version=3
Real_Networks/Processed_Data_(Dis)Aggregated_Lifestage - processed data and metadata files for aggregated and disaggregated by lifestage versions of the empirical food webs used in our tests.
Real_Networks/apply_link_prediction_food_web_demo.ipynb - Jupyter notebook providing a demonstration of running link prediction on a single food web, with details printed
Real_Networks/Results_Demo - folder containing results from running the above demonstration script
Real_Networks/link_prediction_food_webs.py - script for running link prediction tests across all the processed food web data files
Real_Networks/visualize_food_web_database_results.ipynb - Jupyter notebook for visualizing the predictive performance results across the food webs
Real_Networks/visualize_importance_results.ipynb - Jupyter notebook for visualizing the feature importance results across the food webs
Real_Networks/summarize_results_food_webs.py - Helper functions for summarizing and visualizing results
Real_Networks/net_props_display_names.csv - Display names for network properties
Real_Networks/feature_display_names.csv - Display names for features
Real_Networks/Summarized_Results - results files summarizing across all food webs
Real_Networks/food_web_property_regressions.R - R script for running regressions between summarized predictive performance results and food web properties
Real_Networks/Regression_Results - regression results

Additional processed data files and results files across all food webs and syntehtic networks can be regenerated from the above code, or provided upon request.




