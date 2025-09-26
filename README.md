This repository contains the code used in the pre-print "Predicting missing links in food webs using stacked models and species traits" -  https://doi.org/10.1101/2024.11.22.624890

Lucy Van Kleunen, Laura E. Dee, Kate L. Wootton, Francois Massol, Aaron Clauset. 2024.

# Empirical food web tests:

Data and code for the empirical food web tests are provided in the Real_Networks folder. A demonstration of running missing link prediction on one empirical food web for one iteration of 5-fold cross validation each for three different models (attribute-only, structure-only, and full) can be found in the following Jupyter Notebook: https://github.com/lucyvk/food_web_link_prediction/blob/main/Real_Networks/apply_link_prediction_food_web_demo.ipynb

In this file, a number of arguments are set including the location of the food web dataset file, the data types of the node attributes for this food web, and parameters used in missing link prediction. Detailed output of the entire link prediction workflow is printed, including displays of training and test set size and features for each of the folds, the random forest hyperparameters selected for each fold, and performance metrics. The results from running this demo file are stored in Real_Networks/Results_Demo as an example of the intermediate and results files produced for a single food web from this analysis. The results for running missing link prediction for all 290 food web datasets are not included in this repository due to size, but can be reproduced by running the file Real_Networks/link_prediction_food_webs.py. This file runs one iteration of 5-fold cross validation. For the paper, the results were produced by running different versions of this file with Slurm scripts on a high performance computing cluster and it has been provided in simplified form here with comments indicating how to modify the file to reproduce results across 5 iterations of 5-fold cross validation as reported in the paper. This file can be run to produce Real_Networks/Results_Food_Webs_{n} across iterations 0-4 on the food web dataset with nodes disaggregated by lifestage (shown in the main text) and Real_Networks/Results_Food_Webs_Aggregated (supplemental test on the food web dataset with nodes aggregated across lifestage, for one iteration).

The predictive performance results across the entire food web database (Real_Networks/Results_Food_Webs_{n} across iterations 0-4, Real_Networks/Results_Food_Webs_Aggregated) are visualized in the file Real_Networks/visualize_food_web_database_results.ipynb. The feature importance results (for Results_Food_Webs_{n} across iterations 0-4) are visualized in Real_Networks/visualize_importance_results.ipynb. These two Jupyter Notebooks use helper functions defined in Real_Networks/summarize_results_food_webs.py. The feature importance visualization notebook also uses the full display names for the features saved in Real_Networks/feature_display_names.csv.

The Jupyter notebook Real_Networks/visualize_food_web_database_results.ipynb also saves a summary file of the aggreaged predictive performance results to Real_Networks/Summarized_Results. This file, food_web_lp_res.csv, reports the mean performance for each network of the structure-only, attribute-only, and full models for AUC-ROC and PR-AUC.

Finally, the Real_Networks/food_web_property_regressions_Beta.R file provides the code used for the analysis of aggregated predictive performance by food web properties, which also uses the full display names for the food web properties saved in Real_Networks/net_props_display_names.csv. These results are saved in Real_Networks/Regression_Results_Beta.

# Empirical food web processing:

Real_Networks/Original_Data contains the original food web database file. This file is provided for reproducing the data processing steps. Our study uses no original data, and this database should be cited as Brose, U. GlobAL daTabasE of traits and food Web Architecture (GATEWAy) version 3. iDiv https://doi.org/10.25829/idiv.283-3-756 (2018), accessible from: https://idata.idiv.de/ddm/Data/ShowData/283?version=3.

In the folder Real_Networks/Data_Processing_Code_Disaggregated_Lifestage, a Jupyter Notebook is provided for processing the original data file so that the networks are saved in the format we use in our missing link prediction experiments in the main text results. Seeds can be set as indicated to reproduce the 5 randomized edge orders for each food web used in the iterations for which results are reported in the paper. Metadata files are also produced in this notebook and saved in the folder.

The folder Real_Networks/Data_Processing_Code_Aggregated_Lifestage similarly provides the code for processing the original data if species nodes are aggregated across lifestages.

Real_Networks/data_processing_helper.py includes helper functions used in these data processing scripts.

Real_Networks/Processed_Data_Disaggregated_Lifestage includes the edge lists and and node lists with attributes for all 290 of the food webs that were evaluated to produce the main text results. 

The processed data files with lifestage aggregation are similarly provided in Real_Networks/Processed_Data_Aggregated_Lifestage.

# Synthetic food web tests:

A demonstration is also provided for running missing link prediction across the three models and 5-fold cross validation on a single synthetic network in the following file: https://github.com/lucyvk/food_web_link_prediction/blob/main/Synthetic_Networks/link_prediction_demo_synthetic.ipynb, with intermediate and results files saved in Synthetic_Networks/Results_Demo. 

All of the synthetic networks generated for this paper are not provided in this repository due to space, but an example network on which this demonstration script is run (edge list and node list with attributes) is provided in Synthetic_Networks/Synth_Nets_Directed_Assortative_Demo. Synthetic_Networks/Network_Generation_Scripts includes scripts that can be used to reproduce generating the synthetic networks used in the paper across varying rho  for both assortative and disassortative RGG anchor networks, which are saved in Synthetic_Networks/Synth_Nets_Directed_Assortative and Synthetic_Networks/Synth_Nets_Directed_Disassortative. Synthetic_Networks/generate_helper.py provides helper functions.

Synthetic_Networks/Link_Prediction_Scripts includes scripts to run missing link prediction on these synthetic networks and save the results. For the paper, results were produced by running these via a Slurm script on a high performance computing cluster, although these results are not provided in the repository due to space. Running these files produces results that are saved in Synthetic_Networks/Results_Synthetic_Directed_Assortative and Synthetic_Networks/Results_Synthetic_Directed_Disassortative.

The Jupyter Notebook Synthetic_Networks/visualize_synthetic_directed_results.ipynb visualizes the predictive performance results from the synthetic network tests as well as example synthetic networks. Synthetic_Networks/viz_helper.py provides helper functions used in this visualization. 

# Methods:

The main file that implements the functions used in the stacked generalization missing link prediction procedure is Methods/stacking_model/OLP.py.

The file that implements the basic experiment for this paper to compare performance across three versions of the stacking model using 5-fold cross validation and save the results by calling functions in Methods/stacking_model/OLP.py is Methods/stacking_model/link_prediction_helper.py.

Helper functions implementing predictors used in the stacking model are provided in the following files:
* Methods/stacking_model/KNN_predictors.py - this file provides helper functions for KNN predictors
* Methods/stacking_model/directed_triangles.py - this file provides helper functions for directed triangle predictors
* Methods/stacking_model/eco_predictors_helper.py - this file provides helper functions for custom ecological predictors

----------------------------------------------------------------------------------------------------------------------------------------------------------
Additional processed data files and results files across all food webs and synthetic networks can be regenerated from the above code, or provided upon request. The data processing, link prediction scripts, and synthetic network visualizations were run on CentOS Linux 7, and the food web visualizations and demonstration scripts for both empirical and synthetic networks were run on Windows 11. Cloning this repository should typically take less than 1 minute.


