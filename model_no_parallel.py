
def train_random_forest(X_train, Y_train, X_test, Y_test, FILE_PATH):
    #random forest
    pipeline = Pipeline([
                        ('scaler',StandardScaler()),
                        ('model',RandomForestRegressor(n_estimators = 500))
    ])

    search = GridSearchCV(pipeline,
                        {'model__max_depth': np.arange(3,11,8) },
                        cv = muni_cv, scoring = "neg_mean_squared_error",verbose = 3
                        )

    search.fit(X_train,Y_train)

    dump(search.best_estimator_, f'FeatureImportanceResults/{FOLDER_NAME}/ModelFits/pipeline_randomforest.joblib')

    coefficients = search.best_estimator_._final_estimator.feature_importances_
    importance = np.abs(coefficients)

    yhat = search.best_estimator_.predict(X_test)
    np.savetxt(f'FeatureImportanceResults/{FOLDER_NAME}/PredictedDeforestation/yhat_randomforest.txt', yhat, delimiter=",")
    #yhat.to_csv(f'FeatureImportanceResults/{FOLDER_NAME}/PredictedDeforestation/yhat_randomforest.txt')

    #yhat_list.append(yhat)
    randomforest_features_df = generate_results_table(coefficients, X_train.columns, 'randomforest', yhat, Y_test, FILE_PATH, normalized = True)

    return randomforest_features_df
        
def train_lasso(X_train, Y_train, X_test, Y_test, FILE_PATH):
    pipeline = Pipeline([
                    ('scaler',StandardScaler()),
                    ('model',Lasso())
    ])

    search = GridSearchCV(pipeline,
                        {'model__alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]},
                        cv = muni_cv, scoring = "neg_mean_squared_error",verbose = 3
                        )

    search.fit(X_train,Y_train)

    dump(search.best_estimator_, f'FeatureImportanceResults/{FOLDER_NAME}/ModelFits/pipeline_lasso.joblib')

    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)

    yhat = search.best_estimator_.predict(X_test)
    np.savetxt(f'FeatureImportanceResults/{FOLDER_NAME}/PredictedDeforestation/yhat_lasso.txt', yhat, delimiter=",")


    lasso_features_df = generate_results_table(coefficients, X_train.columns, 'lasso', yhat, Y_test, FILE_PATH, normalized = True)

    return lasso_features_df

def train_gradient_boost(X_train, Y_train, X_test, Y_test, FILE_PATH):
    pipeline = Pipeline([
                    ('scaler',StandardScaler()),
                    ('model',GradientBoostingRegressor(learning_rate = 0.1, min_samples_leaf = 2))
    ])

    search = GridSearchCV(pipeline,
                        {'model__n_estimators':np.arange(50, 150, 50), 'model__max_depth':np.arange(3, 5, 1)},
                        cv = muni_cv, scoring = "neg_mean_squared_error",verbose = 3
                        )

    search.fit(X_train,Y_train)

    search.best_params_

    dump(search.best_estimator_.named_steps['model'], f'FeatureImportanceResults/{FOLDER_NAME}/ModelFits/pipeline_gradientboosting.joblib')

    coefficients = search.best_estimator_.named_steps['model'].feature_importances_
    importance = np.abs(coefficients)

    yhat = search.best_estimator_.predict(X_test)
    np.savetxt(f'FeatureImportanceResults/{FOLDER_NAME}/PredictedDeforestation/yhat_gradientboosting.txt', yhat, delimiter=",")

    gradient_boosting_features_df = generate_results_table(coefficients, X_train.columns, 'gradientboosting', yhat, Y_test, FILE_PATH, normalized = True)

    return gradient_boosting_features_df

def train_neural_network(X_train, Y_train, X_test, Y_test, FILE_PATH):
    pipeline = Pipeline([
                    ('scaler',StandardScaler()),
                    ('model', MLPRegressor(activation = 'logistic', random_state=42))
    ])

    search = GridSearchCV(pipeline,
                        {'model__hidden_layer_sizes':[(50,),(100,)], 'model__alpha':np.arange(0.00001, 0.001, 0.001)},
                        cv = muni_cv, scoring = "neg_mean_squared_error",verbose = 3
                        )

    search.fit(X_train,Y_train)

    dump(search.best_estimator_, f'FeatureImportanceResults/{FOLDER_NAME}/ModelFits/pipeline_neuralnetwork.joblib')

    explainer = shap.KernelExplainer(search.best_estimator_.predict, X_train)

    shap_values = explainer.shap_values(X_test, nsamples=1000)

    shap.summary_plot(shap_values,X_test,feature_names=X_test.columns)

    feature_names = X_train.columns

    rf_resultX = pd.DataFrame(shap_values, columns = feature_names)

    vals = np.abs(rf_resultX.values).mean(0)

    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                    columns=['col_name','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                                ascending=False, inplace=True)

    yhat = search.best_estimator_.predict(X_test)
    np.savetxt(f'FeatureImportanceResults/{FOLDER_NAME}/PredictedDeforestation/yhat_neuralnetwork.txt', yhat, delimiter=",")

    #yhat.to_csv(f'FeatureImportanceResults/{FOLDER_NAME}/PredictedDeforestation/yhat_neuralnetwork.txt')
    #yhat_list.append(yhat)

    nn_features_df = generate_results_table(np.array(shap_importance.feature_importance_vals), np.array(shap_importance.col_name), 'neuralnetwork', yhat, Y_test, FILE_PATH, normalized = True)

    return nn_features_df

def train_super_learner( X_train, Y_train, X_test, Y_test, FILE_PATH, muni_cv, base_learners):
    # get models
    models = get_models(base_learners)
    # get out of fold predictions
    meta_X, meta_y = get_out_of_fold_predictions(X_train, Y_train, models, muni_cv)
    print('Meta Data Shape: ', meta_X.shape, meta_y.shape)

    fit_base_models(X_train, Y_train, models)
    meta_model = fit_meta_model(meta_X, meta_y)

    evaluate_models(X_test, Y_test, models)

    yhat = super_learner_predictions(X_test, models, meta_model)
    #yhat_list.append(yhat)
    np.savetxt(f'FeatureImportanceResults/{FOLDER_NAME}/PredictedDeforestation/yhat_superlearner.txt', yhat, delimiter=",")
    #yhat.to_csv(f'FeatureImportanceResults/{FOLDER_NAME}/PredictedDeforestation/yhat_superlearner.txt')

    # Evaluate the performance of the model
    mse = mean_squared_error(Y_test, yhat)
    print("MSE:", mse)

    #Super Learner Feature Importance

    #Random forest 
    random_forest_weighted_importance = models[0].feature_importances_ * meta_model.coef_[0]

    #Lasso 
    lasso_weighted_importance = models[1].coef_ * meta_model.coef_[1]

    #GradientBoostingRegressor
    gradient_boosting_weighted_importance = models[2].feature_importances_ * meta_model.coef_[2]

    #NeuralNetwork
    explainer = shap.KernelExplainer(models[2].predict, X_train)
    shap_values = explainer.shap_values(X_test, nsamples=100)

    feature_names = X_train.columns

    rf_resultX = pd.DataFrame(shap_values, columns = feature_names)

    vals = np.abs(rf_resultX.values).mean(0)

    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                    columns=['col_name','feature_importance_vals'])


    nn_weighted_importance = shap_importance.feature_importance_vals * meta_model.coef_[3]

    super_learner_feature_importance = np.mean([random_forest_weighted_importance, lasso_weighted_importance, gradient_boosting_weighted_importance, nn_weighted_importance], axis = 0)

    super_learner_features_df = generate_results_table(super_learner_feature_importance, X_train.columns, 'superlearner', yhat, Y_test, FILE_PATH, normalized = True)

    return super_learner_features_df

