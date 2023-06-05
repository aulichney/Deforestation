
Results
=======

# Questions we're answering

- What are important features in explaining variance in deforestation?
- How do the features change across the time interval 2004-2016?
- How much information leakage occurrs if spatial dependence is not considered?
- Do important feature vary accross the region?

# Test Train Split


Data split 70/30 between Train and Test data sets by municipality where municipalities are selected at random. Splitting by municipality is done to account for the spatial dependence of the data.  
![Test/Train split plot](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/EntirePlot.png)  
Note that I use this same split for each panel of the sliding window and for each method within each panel.
# Procedure

## Models Fit

# Sliding Window Subsets
  
I use 3 years of data to predict the  following year's **deforest_diff** variable.
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
# Feature Importance Changing in Time
  
![Average of all methods](FeatureImportanceResults/Evolution/evolution_exclude_forest_avg.png)  
![Random Forest](FeatureImportanceResults/Evolution/evolution_exclude_forest_randomforest.png)  
![Lasso](FeatureImportanceResults/Evolution/evolution_exclude_forest_lasso.png)  
![Gradient Boosting](FeatureImportanceResults/Evolution/evolution_exclude_forest_gradientboosting.png)  
![Neural Network](FeatureImportanceResults/Evolution/evolution_exclude_forest_neuralnetwork.png)  
![Superlearner](FeatureImportanceResults/Evolution/evolution_exclude_forest_superlearner.png)
# Feature Importance

## 2004_2005_2006 predict 2007 feature importances
  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_lasso.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_randomforest.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_gradientboosting.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_neuralnetwork.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_superlearner.png)