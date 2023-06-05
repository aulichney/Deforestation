
Results
=======

# Test Train Split
  
Data split 70/30 between Train and Test data sets by municipality where municipalities are selected at random. Splitting by municipality is done to account for the spatial dependence of the data.  
![Test/Train split plot](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/EntirePlot.png)  
Note that I use this same split for each panel of the sliding window and for each method within each panel.
# Sliding Window Subsets
  
I use 3 years of features as the 
# Feature Importance
  
Feature importance scores for each method are   
![](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/FeatureImportance/features_all.png)  
![](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/FeatureImportance/features_all.png)
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
# Feature Importance

## 2004_2005_2006 predict 2007 feature importances
  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_lasso.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_randomforest.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_gradientboosting.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_neuralnetwork.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_superlearner.png)