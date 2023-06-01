
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
# Predicted Deforestation

## 2004_2005_2006 predict 2007 predicted deforestation
  

|year|  randomforest  |     lasso      |gradientboosting|       nn       |  superlearner  |
| :---: | :---: | :---: | :---: | :---: | :---: |
|2004_2005_2006_PREDICT_2007|![Image](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/DeforestPlot_randomforest.png)|![Image](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/DeforestPlot_lasso.png)|![Image](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/DeforestPlot_gradientboosting.png)|![Image](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/DeforestPlot_nn.png)|![Image](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/DeforestPlot_superlearner.png)|
|2005_2006_2007_PREDICT_2008|![Image](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/DeforestPlot_randomforest.png)|![Image](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/DeforestPlot_lasso.png)|![Image](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/DeforestPlot_gradientboosting.png)|![Image](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/DeforestPlot_nn.png)|![Image](FeatureImportanceResults/2005_2006_2007_PREDICT_2008/DeforestPlot_superlearner.png)|
|2006_2007_2008_PREDICT_2009|![Image](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/DeforestPlot_randomforest.png)|![Image](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/DeforestPlot_lasso.png)|![Image](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/DeforestPlot_gradientboosting.png)|![Image](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/DeforestPlot_nn.png)|![Image](FeatureImportanceResults/2006_2007_2008_PREDICT_2009/DeforestPlot_superlearner.png)|
|2007_2008_2009_PREDICT_2010|![Image](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/DeforestPlot_randomforest.png)|![Image](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/DeforestPlot_lasso.png)|![Image](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/DeforestPlot_gradientboosting.png)|![Image](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/DeforestPlot_nn.png)|![Image](FeatureImportanceResults/2007_2008_2009_PREDICT_2010/DeforestPlot_superlearner.png)|
|2008_2009_2010_PREDICT_2011|![Image](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/DeforestPlot_randomforest.png)|![Image](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/DeforestPlot_lasso.png)|![Image](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/DeforestPlot_gradientboosting.png)|![Image](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/DeforestPlot_nn.png)|![Image](FeatureImportanceResults/2008_2009_2010_PREDICT_2011/DeforestPlot_superlearner.png)|

# Feature Importance

## 2004_2005_2006 predict 2007 feature importances
  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_lasso.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_randomforest.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_gradientboosting.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_neuralnetwork.png)  
![Random Forest](FeatureImportanceResults/2004_2005_2006_PREDICT_2007/FeatureImportance/features_superlearner.png)