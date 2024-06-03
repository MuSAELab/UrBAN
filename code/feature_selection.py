import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
import shap
from sklearn.preprocessing import MinMaxScaler
import depmeas #use from github
import feature_select # use from github

from regression_analysis import ml_gridsearchcv, score_mae_box, score_rmse_box, score_corr_box

def feature_selection(feature_data,
                      x_train, x_val,
                      y_train, y_val,
                      method, model='random forest',
                      predefined_split=False, 
                      refit=True):
    """
    Perform feature selection based on the specified method.

    Parameters:
    - feature_data (pd.DataFrame): DataFrame containing the feature data.
    - x_train (np.ndarray): Training feature data.
    - x_val (np.ndarray): Validation feature data.
    - y_train (np.ndarray): Training target data.
    - y_val (np.ndarray): Validation target data.
    - method (str): Feature selection method to use ('importance', 'pca', 'mrmr', 'shap').
    - model (str): Model to use for feature selection ('random forest' by default).
    - predefined_split (bool): Indicates whether to use a predefined split for cross-validation (default False).
    - refit (bool): Indicates whether to refit the model after feature selection (default True).

    Returns:
    - tuple: Validation metrics (val_mae, val_rmse, val_corr).
    """
    

    val_mae = []
    val_rmse = []
    val_corr = []

    if method == 'importance':
        
        model = RandomForestRegressor(random_state = 0)
        model.fit(x_train, y_train)
        importance = model.feature_importances_
        
        for i in range(1, len(importance)+1)[:10]:
            
            print(feature_data.iloc[:, :-5].columns[importance.argsort()[-i:][::-1]])

            x_train_i = x_train[:, importance.argsort()[-i:][::-1]]
            x_val_i = x_val[:, importance.argsort()[-i:][::-1]]

            x_train_val_i = np.concatenate([x_train_i, x_val_i], axis = 0)
            y_train_val = np.concatenate([y_train, y_val], axis = 0)
            
            
            split_index = []
            
            if predefined_split:
                
                for i in range(len(x_train_i)):
                    split_index.append(-1)
                for i in range(len(x_val_i)):
                    split_index.append(0)
                            
                grid = ml_gridsearchcv('random forest',
                                       predefined_split = predefined_split,
                                       split_index = split_index, 
                                       refit=refit
                                      )
                
                grid.fit(x_train_val_i, y_train_val)

                model = RandomForestRegressor(random_state = 0, **grid.best_params_)
                model.fit(x_train_i, y_train)
                
            else:
                
                model = ml_gridsearchcv('random forest',
                                        predefined_split = predefined_split,
                                        split_index = split_index, 
                                         refit=refit
                                       )
                model.fit(x_train_i, y_train)


            val_mae.append(score_mae_box(y_val, model.predict(x_val_i)))
            val_rmse.append(score_rmse_box(y_val, model.predict(x_val_i)))
            val_corr.append(score_corr_box(y_val, model.predict(x_val_i)))

        
    elif method == 'pca':
        
        for i in range(1, min(x_val.shape[0], x_val.shape[1]) +1)[:10]:
            
            
            
            pca_features = PCA(n_components=i, random_state=0)
            x_train_i = pca_features.fit_transform(x_train)
            x_val_i = pca_features.fit_transform(x_val)
            
            print('Explained variation per principal component: {}'.format(pca_features.explained_variance_ratio_))


            min_max_scaler = MinMaxScaler()
            x_train_i = min_max_scaler.fit_transform(x_train_i)
            x_val_i = min_max_scaler.fit_transform(x_val_i)

            x_train_val_i = np.concatenate([x_train_i, x_val_i], axis = 0)
            y_train_val = np.concatenate([y_train, y_val], axis = 0)
            
            split_index = []
            if predefined_split:
                
                for i in range(len(x_train_i)):
                    split_index.append(-1)
                for i in range(len(x_val_i)):
                    split_index.append(0)
                            
                grid = ml_gridsearchcv('random forest',
                                       predefined_split = predefined_split,
                                       split_index = split_index, 
                                       refit=refit
                                      )
                
                grid.fit(x_train_val_i, y_train_val)

                model = RandomForestRegressor(random_state = 0, **grid.best_params_)
                model.fit(x_train_i, y_train)
                
            else:
                
                model = ml_gridsearchcv('random forest',
                                        predefined_split = predefined_split,
                                        split_index = split_index, 
                                         refit=refit
                                       )
                model.fit(x_train_i, y_train)


            val_mae.append(score_mae_box(y_val, model.predict(x_val_i)))
            val_rmse.append(score_rmse_box(y_val, model.predict(x_val_i)))
            val_corr.append(score_corr_box(y_val, model.predict(x_val_i)))
        
    elif method == 'mrmr':
        
        num_features_to_select = len(feature_data.iloc[:, :-5].columns)
        K_MAX = 1000
        estimator = depmeas.mi_tau
        n_jobs = -1
        feature_ranking_idxs = feature_select.feature_select(x_train,y_train,
            num_features_to_select=num_features_to_select,K_MAX=K_MAX,
            estimator=estimator,n_jobs=n_jobs)

        num_selected_features = len(feature_ranking_idxs)

        for i in range(num_selected_features)[:10]:

            x_train_i = x_train[:,feature_ranking_idxs[0:i+1]]
            x_val_i = x_val[:,feature_ranking_idxs[0:i+1]]


            x_train_val_i = np.concatenate([x_train_i, x_val_i], axis = 0)
            y_train_val = np.concatenate([y_train, y_val], axis = 0)
            
            split_index = []
            split_index = []
            if predefined_split:
                
                for i in range(len(x_train_i)):
                    split_index.append(-1)
                for i in range(len(x_val_i)):
                    split_index.append(0)
                            
                grid = ml_gridsearchcv('random forest',
                                       predefined_split = predefined_split,
                                       split_index = split_index, 
                                       refit=refit
                                      )
                
                grid.fit(x_train_val_i, y_train_val)

                model = RandomForestRegressor(random_state = 0, **grid.best_params_)
                model.fit(x_train_i, y_train)
                
            else:
                
                model = ml_gridsearchcv('random forest',
                                        predefined_split = predefined_split,
                                        split_index = split_index, 
                                         refit=refit
                                       )
                model.fit(x_train_i, y_train)


            val_mae.append(score_mae_box(y_val, model.predict(x_val_i)))
            val_rmse.append(score_rmse_box(y_val, model.predict(x_val_i)))
            val_corr.append(score_corr_box(y_val, model.predict(x_val_i)))
        
        
    elif method == 'shap':
        
        X100 = shap.utils.sample(x_train.astype(float), 100)  # 100 instances for use as the background distribution
        grid = ml_gridsearchcv('random forest', refit=True)
        grid.fit(x_train.astype(float), y_train)
        explainer = shap.Explainer(grid.predict, X100, feature_names=feature_data.iloc[:, :-5].columns)
        shap_values = explainer(x_train.astype(float))#, max_evals=6401
        
        vals= np.abs(shap_values.values).mean(0)
                    
        for i in range(1, len(vals)+1)[:10]:
    
            print(feature_data.iloc[:, :-5].columns[vals.argsort()[-i:][::-1]])

            x_train_i = x_train[:, vals.argsort()[-i:][::-1]]

            x_val_i = x_val[:, vals.argsort()[-i:][::-1]]

            x_train_val_i = np.concatenate([x_train_i, x_val_i], axis = 0)
            y_train_val = np.concatenate([y_train, y_val], axis = 0)
            
            split_index = []
            if predefined_split:
                
                for i in range(len(x_train_i)):
                    split_index.append(-1)
                for i in range(len(x_val_i)):
                    split_index.append(0)
                            
                grid = ml_gridsearchcv('random forest',
                                       predefined_split = predefined_split,
                                       split_index = split_index, 
                                       refit=refit
                                      )
                
                grid.fit(x_train_val_i, y_train_val)

                model = RandomForestRegressor(random_state = 0, **grid.best_params_)
                model.fit(x_train_i, y_train)
                
            else:
                
                model = ml_gridsearchcv('random forest',
                                        predefined_split = predefined_split,
                                        split_index = split_index, 
                                         refit=refit
                                       )
                model.fit(x_train_i, y_train)


            val_mae.append(score_mae_box(y_val, model.predict(x_val_i)))
            val_rmse.append(score_rmse_box(y_val, model.predict(x_val_i)))
            val_corr.append(score_corr_box(y_val, model.predict(x_val_i)))

    return val_mae, val_rmse, val_corr



