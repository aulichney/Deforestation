
Results
=======

Contents
========

* [Links to Code](#links-to-code)
* [Motivating Questions](#motivating-questions)
* [Test Train Split](#test-train-split)
* [Models & Fit](#models--fit)
* [Feature Importance](#feature-importance)
	* [Feature Importance without the **forest** variables.](#feature-importance-without-the-forest-variables)
* [Predicted Deforestation](#predicted-deforestation)
* [MSE](#mse)
* [Feature Importance Changing in Time](#feature-importance-changing-in-time)
* [Remaining Questions](#remaining-questions)
* [Appendix: Bigger Feature Importance Plots](#appendix-bigger-feature-importance-plots)
	* [2004_2005_2006_PREDICT_2007 feature importances](#2004_2005_2006_predict_2007-feature-importances)
	* [2005_2006_2007_PREDICT_2008 feature importances](#2005_2006_2007_predict_2008-feature-importances)
	* [2006_2007_2008_PREDICT_2009 feature importances](#2006_2007_2008_predict_2009-feature-importances)
	* [2007_2008_2009_PREDICT_2010 feature importances](#2007_2008_2009_predict_2010-feature-importances)
	* [2008_2009_2010_PREDICT_2011 feature importances](#2008_2009_2010_predict_2011-feature-importances)
	* [2009_2010_2011_PREDICT_2012 feature importances](#2009_2010_2011_predict_2012-feature-importances)
	* [2010_2011_2012_PREDICT_2013 feature importances](#2010_2011_2012_predict_2013-feature-importances)
	* [2011_2012_2013_PREDICT_2014 feature importances](#2011_2012_2013_predict_2014-feature-importances)
	* [2012_2013_2014_PREDICT_2015 feature importances](#2012_2013_2014_predict_2015-feature-importances)
	* [2013_2014_2015_PREDICT_2016 feature importances](#2013_2014_2015_predict_2016-feature-importances)
* [Appendix: All Features Input](#appendix-all-features-input)

# Links to Code
  
[Models and Utils: deforestutils.py](deforestutils.py)  
[Notebook for models and analysis: ResultsParallel.ipynb](ResultsParallel.ipynb)  
[Notebook for markdown: ResultsMarkdownGenerator.ipynb](ResultsMarkdownGenerator.ipynb)  

# Motivating Questions

- What are important features in explaining variance in deforestation?
- How do the features change across the time interval 2004-2016?
- How much information leakage occurrs if spatial dependence is not considered?
- Do important feature vary accross the region?

# Test Train Split


Data split 70/30 between Train and Test data sets by municipality where municipalities are selected at random. Splitting by municipality is done to account for the spatial dependence of the data.  
![Test/Train split plot](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/EntirePlot.png)  
Note that I use this same split for each panel of the sliding window and for each method within each panel.
# Models & Fit

- Random Forest
    - Model: RandomForestRegressor in Sklearn
    - Hyperparam: {'model__max_depth': np.arange(3, 11, 8)}
    - coefficients: best_model._final_estimator.feature_importances_
        - unsigned feature importances are calculated based on the Gini importance or mean decrease impurity method
- Lasso
    - Model: Lasso in Sklearn
    - Hyperparam: {'model__alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}
    - signed coefficients: best_model.named_steps['model'].coef_
        - feature importances calculated as coeffs resulting from iterative coordinate descent optimization algo where each updated one at a time while others held fixed
- Gradient Boosting
    - Model: GradientBoostingRegressor(learning_rate = 0.1, min_samples_leaf = 2) in Sklearn
    - Hyperparam: {'model__max_depth': np.arange(3, 11, 8)}
    - coefficients: best_model._final_estimator.feature_importances_
        - unsigned feature importances, normalized by design: total reduction in the loss function achieved by splits on a given feature, summed over all trees in the ensemble of decision trees
- Neural Network
    - Model: MLPRegressor(activation = 'logistic') in Sklearn
    - Hyperparam: {'model__hidden_layer_sizes':[(50,),(100,)], 'model__alpha':np.arange(0.00001, 0.001, 0.001)}
    - coefficients: shap.KernelExplainer(best_model.predict,shap.sample(X_train, 100), nsamples = 100), explainer.shap_values(shap.sample(X_test, 1000), nsamples=100)
        - signed feature importances: shap approximates Shapley values as coefficients of linear regression of perturbed inputs
- Super Learner
    - Model: I implement by combining the above. predictions from best models from each of the above are input to Ridge() meta learner
    - Hyperparam: {'model__max_depth': np.arange(3, 11, 8)}
    - coefficients: best_model._final_estimator.feature_importances_
        - signed feature importances: mean of feature importance of each input model weighted by importance of each base model in meta model
  
I use 3 years of data to predict the  following year's **deforest_diff** variable.  
Parallelized gridsearchCV used to tune hyperparameters of each model.  
All hyperparameters are selected with 5-fold cross validation where folds are selected on municipality level to control for spatial dependence.  
StandardScaler transform applied
# Feature Importance
  
Feature importance scores for each method are normalized to sum to 1 here to understand their relative differences. Absolute vals are taken when necessary since some methods return signed feature importance and some don't.  
![](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2009_2010_2011_PREDICT_2012/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2010_2011_2012_PREDICT_2013/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2011_2012_2013_PREDICT_2014/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2012_2013_2014_PREDICT_2015/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2013_2014_2015_PREDICT_2016/FeatureImportance/features_all.png)
## Feature Importance without the **forest** variables.
  
'''forest_lag''' and '''forest_formation''' are both directly proportional to the response variable forest_diff, so it was expected that they would dominate the important features consistently.  
I repeat the above plots but exclude forest vars so others are visible.  
![](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_all_forest_exclude.png)  
![](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/FeatureImportance/features_all_forest_exclude.png)  
![](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/FeatureImportance/features_all_forest_exclude.png)  
![](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/FeatureImportance/features_all_forest_exclude.png)  
![](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/FeatureImportance/features_all_forest_exclude.png)  
![](FeatureImportanceResults/2009_2010_2011_PREDICT_2012/FeatureImportance/features_all_forest_exclude.png)  
![](FeatureImportanceResults/2010_2011_2012_PREDICT_2013/FeatureImportance/features_all_forest_exclude.png)  
![](FeatureImportanceResults/2011_2012_2013_PREDICT_2014/FeatureImportance/features_all_forest_exclude.png)  
![](FeatureImportanceResults/2012_2013_2014_PREDICT_2015/FeatureImportance/features_all_forest_exclude.png)  
![](FeatureImportanceResults/2013_2014_2015_PREDICT_2016/FeatureImportance/features_all_forest_exclude.png)
# Predicted Deforestation
  
![](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/DeforestPlot_all.png)  
![](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/DeforestPlot_all.png)  
![](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/DeforestPlot_all.png)  
![](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/DeforestPlot_all.png)  
![](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/DeforestPlot_all.png)  
![](FeatureImportanceResults/2009_2010_2011_PREDICT_2012/DeforestPlot_all.png)  
![](FeatureImportanceResults/2010_2011_2012_PREDICT_2013/DeforestPlot_all.png)  
![](FeatureImportanceResults/2011_2012_2013_PREDICT_2014/DeforestPlot_all.png)  
![](FeatureImportanceResults/2012_2013_2014_PREDICT_2015/DeforestPlot_all.png)  
![](FeatureImportanceResults/2013_2014_2015_PREDICT_2016/DeforestPlot_all.png)
# MSE
  
![Average of all methods](FeatureImportanceResults/MSE.png)
# Feature Importance Changing in Time
  
![Average of all methods](FeatureImportanceResults/Evolution/evolution_exclude_forest_avg.png)  
![Random Forest](FeatureImportanceResults/Evolution/evolution_exclude_forest_randomforest.png)  
![Lasso](FeatureImportanceResults/Evolution/evolution_exclude_forest_lasso.png)  
![Gradient Boosting](FeatureImportanceResults/Evolution/evolution_exclude_forest_gradientboosting.png)  
![Neural Network](FeatureImportanceResults/Evolution/evolution_exclude_forest_neuralnetwork.png)  
![Superlearner](FeatureImportanceResults/Evolution/evolution_exclude_forest_superlearner.png)
# Remaining Questions

- Should I keep the deforest variables in? 
- Should I run again with random cross validation and see if the mse changes?
- Should I do the same analysis with a sliding window to see if important features change over the area?
- Should I exclude year?
- Should I use the same test/train split for all fits?

# Appendix: Bigger Feature Importance Plots

## 2004_2005_2006_PREDICT_2007 feature importances
  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_randomforest.png)  
![Lasso](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_lasso.png)  
![GradientBoosting](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_gradientboosting.png)  
![NeuralNetwork](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_neuralnetwork.png)  
![SuperLearner](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_superlearner.png)
## 2005_2006_2007_PREDICT_2008 feature importances
  
![Random Forest](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/FeatureImportance/features_randomforest.png)  
![Lasso](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/FeatureImportance/features_lasso.png)  
![GradientBoosting](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/FeatureImportance/features_gradientboosting.png)  
![NeuralNetwork](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/FeatureImportance/features_neuralnetwork.png)  
![SuperLearner](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/FeatureImportance/features_superlearner.png)
## 2006_2007_2008_PREDICT_2009 feature importances
  
![Random Forest](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/FeatureImportance/features_randomforest.png)  
![Lasso](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/FeatureImportance/features_lasso.png)  
![GradientBoosting](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/FeatureImportance/features_gradientboosting.png)  
![NeuralNetwork](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/FeatureImportance/features_neuralnetwork.png)  
![SuperLearner](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/FeatureImportance/features_superlearner.png)
## 2007_2008_2009_PREDICT_2010 feature importances
  
![Random Forest](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/FeatureImportance/features_randomforest.png)  
![Lasso](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/FeatureImportance/features_lasso.png)  
![GradientBoosting](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/FeatureImportance/features_gradientboosting.png)  
![NeuralNetwork](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/FeatureImportance/features_neuralnetwork.png)  
![SuperLearner](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/FeatureImportance/features_superlearner.png)
## 2008_2009_2010_PREDICT_2011 feature importances
  
![Random Forest](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/FeatureImportance/features_randomforest.png)  
![Lasso](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/FeatureImportance/features_lasso.png)  
![GradientBoosting](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/FeatureImportance/features_gradientboosting.png)  
![NeuralNetwork](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/FeatureImportance/features_neuralnetwork.png)  
![SuperLearner](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/FeatureImportance/features_superlearner.png)
## 2009_2010_2011_PREDICT_2012 feature importances
  
![Random Forest](FeatureImportanceResults/2009_2010_2011_PREDICT_2012/FeatureImportance/features_randomforest.png)  
![Lasso](FeatureImportanceResults/2009_2010_2011_PREDICT_2012/FeatureImportance/features_lasso.png)  
![GradientBoosting](FeatureImportanceResults/2009_2010_2011_PREDICT_2012/FeatureImportance/features_gradientboosting.png)  
![NeuralNetwork](FeatureImportanceResults/2009_2010_2011_PREDICT_2012/FeatureImportance/features_neuralnetwork.png)  
![SuperLearner](FeatureImportanceResults/2009_2010_2011_PREDICT_2012/FeatureImportance/features_superlearner.png)
## 2010_2011_2012_PREDICT_2013 feature importances
  
![Random Forest](FeatureImportanceResults/2010_2011_2012_PREDICT_2013/FeatureImportance/features_randomforest.png)  
![Lasso](FeatureImportanceResults/2010_2011_2012_PREDICT_2013/FeatureImportance/features_lasso.png)  
![GradientBoosting](FeatureImportanceResults/2010_2011_2012_PREDICT_2013/FeatureImportance/features_gradientboosting.png)  
![NeuralNetwork](FeatureImportanceResults/2010_2011_2012_PREDICT_2013/FeatureImportance/features_neuralnetwork.png)  
![SuperLearner](FeatureImportanceResults/2010_2011_2012_PREDICT_2013/FeatureImportance/features_superlearner.png)
## 2011_2012_2013_PREDICT_2014 feature importances
  
![Random Forest](FeatureImportanceResults/2011_2012_2013_PREDICT_2014/FeatureImportance/features_randomforest.png)  
![Lasso](FeatureImportanceResults/2011_2012_2013_PREDICT_2014/FeatureImportance/features_lasso.png)  
![GradientBoosting](FeatureImportanceResults/2011_2012_2013_PREDICT_2014/FeatureImportance/features_gradientboosting.png)  
![NeuralNetwork](FeatureImportanceResults/2011_2012_2013_PREDICT_2014/FeatureImportance/features_neuralnetwork.png)  
![SuperLearner](FeatureImportanceResults/2011_2012_2013_PREDICT_2014/FeatureImportance/features_superlearner.png)
## 2012_2013_2014_PREDICT_2015 feature importances
  
![Random Forest](FeatureImportanceResults/2012_2013_2014_PREDICT_2015/FeatureImportance/features_randomforest.png)  
![Lasso](FeatureImportanceResults/2012_2013_2014_PREDICT_2015/FeatureImportance/features_lasso.png)  
![GradientBoosting](FeatureImportanceResults/2012_2013_2014_PREDICT_2015/FeatureImportance/features_gradientboosting.png)  
![NeuralNetwork](FeatureImportanceResults/2012_2013_2014_PREDICT_2015/FeatureImportance/features_neuralnetwork.png)  
![SuperLearner](FeatureImportanceResults/2012_2013_2014_PREDICT_2015/FeatureImportance/features_superlearner.png)
## 2013_2014_2015_PREDICT_2016 feature importances
  
![Random Forest](FeatureImportanceResults/2013_2014_2015_PREDICT_2016/FeatureImportance/features_randomforest.png)  
![Lasso](FeatureImportanceResults/2013_2014_2015_PREDICT_2016/FeatureImportance/features_lasso.png)  
![GradientBoosting](FeatureImportanceResults/2013_2014_2015_PREDICT_2016/FeatureImportance/features_gradientboosting.png)  
![NeuralNetwork](FeatureImportanceResults/2013_2014_2015_PREDICT_2016/FeatureImportance/features_neuralnetwork.png)  
![SuperLearner](FeatureImportanceResults/2013_2014_2015_PREDICT_2016/FeatureImportance/features_superlearner.png)
# Appendix: All Features Input

- year
- rain1
- elevation
- slope
- aspect
- near_mines
- near_roads
- near_hidrovia
- indigenous_homol
- mun_election_year
- new_forest_code
- lula
- dilma
- temer
- bolsonaro
- fed_election_year
- populacao
- pib_pc
- ironore
- silver
- copper
- gold
- soy_price
- beef_price
- ag_jobs
- mining_jobs
- public_jobs
- construction_jobs
- PIB
- n_companies_PUBLIC ADMIN
- n_companies_AGRICULTURE
- n_companies_FOOD AND DRINKS
- n_companies_ACCOMODATION AND FOOD
- n_companies_EQUIPMENT RENTAL
- n_companies_WHOLESALE
- n_companies_ASSOCIATIVE ACTIVITIES
- n_companies_AUTOMOBILES AND TRANSPORT
- n_companies_FINANCIAL ASSISTANCE
- n_companies_TRADE REP VEHICLES
- n_companies_CONSTRUCTION
- n_companies_MAIL AND TELECOM
- n_companies_CULTURE AND SPORT
- n_companies_EDITING AND PRINTING
- n_companies_EDUCATION
- n_companies_ELECTRICITY AND GAS
- n_companies_FINANCES
- n_companies_CLEANING AND SEWAGE
- n_companies_MACHINERY
- n_companies_BASIC METALLURGY
- n_companies_MINING
- n_companies_WOOD PROD
- n_companies_NON-METALLIC MINERAL PRODUCTS
- n_companies_HEALTH
- n_companies_SERVICES FOR COMPANIES
- n_companies_PERSONAL SERVICES
- n_companies_TRANSPORTATION
- n_companies_GROUND TRANSPORT
- n_companies_WATER TREATMENT AND DISTRIBUTION
- n_companies_RETAIL
- n_companies_COMPUTING
- n_companies_INSURANCE AND SOCIAL SECURITY
- n_companies_METALLIC PRODUCTS
- n_companies_DOMESTIC SERVICES
- n_companies_FORESTRY
- n_companies_CLOTHING
- n_companies_PAPER
- n_companies_INTERNATIONAL BODIES
- n_companies_OIL AND GAS
- n_companies_FISHING AND AQUACULTURE
- n_companies_CHEMICALS
- n_companies_WATER-BASED TRANSPORTATION
- n_companies_REAL ESTATE
- n_companies_RECYCLING
- n_companies_LEATHERS AND FOOTWEAR
- n_companies_RUBBER AND PLASTIC
- n_companies_TEXTILES
- n_companies_RESEARCH AND DEVELOPMENT
- n_companies_AERO TRANSPORT
- n_companies_SMOKE
- n_companies_PETROLEUM REFINING
- n_companies_
- n_jobs_PUBLIC ADMIN
- n_jobs_AGRICULTURE
- n_jobs_FOOD AND DRINKS
- n_jobs_ACCOMODATION AND FOOD
- n_jobs_EQUIPMENT RENTAL
- n_jobs_WHOLESALE
- n_jobs_ASSOCIATIVE ACTIVITIES
- n_jobs_AUTOMOBILES AND TRANSPORT
- n_jobs_FINANCIAL ASSISTANCE
- n_jobs_TRADE REP VEHICLES
- n_jobs_CONSTRUCTION
- n_jobs_MAIL AND TELECOM
- n_jobs_CULTURE AND SPORT
- n_jobs_EDITING AND PRINTING
- n_jobs_EDUCATION
- n_jobs_ELECTRICITY AND GAS
- n_jobs_FINANCES
- n_jobs_CLEANING AND SEWAGE
- n_jobs_MACHINERY
- n_jobs_BASIC METALLURGY
- n_jobs_MINING
- n_jobs_WOOD PROD
- n_jobs_NON-METALLIC MINERAL PRODUCTS
- n_jobs_HEALTH
- n_jobs_SERVICES FOR COMPANIES
- n_jobs_PERSONAL SERVICES
- n_jobs_TRANSPORTATION
- n_jobs_GROUND TRANSPORT
- n_jobs_WATER TREATMENT AND DISTRIBUTION
- n_jobs_RETAIL
- n_jobs_COMPUTING
- n_jobs_INSURANCE AND SOCIAL SECURITY
- n_jobs_METALLIC PRODUCTS
- n_jobs_DOMESTIC SERVICES
- n_jobs_FORESTRY
- n_jobs_CLOTHING
- n_jobs_PAPER
- n_jobs_INTERNATIONAL BODIES
- n_jobs_OIL AND GAS
- n_jobs_FISHING AND AQUACULTURE
- n_jobs_CHEMICALS
- n_jobs_WATER-BASED TRANSPORTATION
- n_jobs_REAL ESTATE
- n_jobs_RECYCLING
- n_jobs_LEATHERS AND FOOTWEAR
- n_jobs_RUBBER AND PLASTIC
- n_jobs_TEXTILES
- n_jobs_RESEARCH AND DEVELOPMENT
- n_jobs_AERO TRANSPORT
- n_jobs_SMOKE
- n_jobs_PETROLEUM REFINING
- n_jobs_
- n_jobs_TOTAL INDUSTRIAL
- n_jobs_TOTAL SERVICE
- n_companies_TOTAL INDUSTRIAL
- n_companies_TOTAL SERVICE
- n_companies_TOTAL
- n_jobs_TOTAL
- murder_threats
- assassination
- assassination_attempt
- f_emitted_count
- expen_agri
- expen_env_man
- expen_agr_org
- expen_mining
- expen_petrol
- expen_prom_ani_pro
- expen_prom_veg_pro
- expen_other_agr
- expen_agr_defense
- expen_min_fuel
- illegal_mining
- illegal_other
- illegal_industry
- audits
- emiss_pec_full
- emiss_agr_full
- emiss_agropec_full
- incumbant
- term_limited_seat
- special
- overall_winner_complete_college
- overall_winner_feminino
- overall_winner_agriculture_job
- overall_winner_public_service_job
- overall_winner_health_job
- overall_winner_corporate_job
- overall_winner_law_job
- overall_winner_technical_job
- overall_winner_professional_job
- overall_winner_mining_job
- overall_winner_partido_PT
- overall_winner_partido_PMDB_MDB
- overall_winner_partido_PSDB
- overall_winner_partido_DEM
- overall_winner_partido_PL
- overall_winner_partido_other
- runnerup_partido_PT
- runnerup_partido_PMDB_MDB
- runnerup_partido_PSDB
- runnerup_partido_DEM
- runnerup_partido_PL
- runnerup_partido_other
- winner_votes_proportion
- vote_participation_proportion
- forest_formation
- savanna
- mangrove
- silvicultura
- pasture
- sugarcane
- mosaic_ag
- urban
- mining
- water
- soybean
- rice
- other_crop
- coffee
- citrus
- other_perennial
- forest_lag
