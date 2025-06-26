import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import depmeas #use from github
import feature_select # use from github
from feature_select import feature_select_optimized

from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneGroupOut, KFold, GridSearchCV, PredefinedSplit, GroupKFold, RandomizedSearchCV
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, r2_score
from scipy.stats import ttest_rel, wilcoxon
import random
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
def create_random_feature(dataset, seed=0, min_fob=1, max_fob=30):
    set_seed(seed=seed)
    dataset = dataset.copy()
    dataset['fob'] = np.random.uniform(min_fob, max_fob + 1e-10, len(dataset))
    dataset['fob'] = np.clip(dataset['fob'], min_fob, max_fob)
  
    return dataset


def create_random_array(array, seed=0, min_fob=1, max_fob=30):
    set_seed(seed=seed)

    random_array = np.random.uniform(min_fob, max_fob + 1e-10, len(array))
    random_array = np.clip(random_array, min_fob, max_fob)
  
    return random_array


def clip_based_on_boxes(y_true, y_pred):
    """
    Calculate the correlation coefficient considering box constraints.

    Args:
    - y_true (ndarray): True target labels.
    - y_pred (ndarray): Predicted target labels.

    Returns:
    - r2_box (float): R squared.
    """
    
    #y_true = y_true.astype(int)
    temp = []
    
    for i in range(len(y_true)):
        if 0 < y_true[i] <= 10:
            y_box = 1
        elif 10 < y_true[i] <= 20:
            y_box = 2
        elif 20 < y_true[i] <= 30:
            y_box = 3
            
        temp.append(np.clip(y_pred[i], 10 * y_box - 10, 10 * y_box))

    return np.array(temp).flatten()


def score_mae_box(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) considering box constraints.

    Args:
    - y_true (ndarray): True target labels.
    - y_pred (ndarray): Predicted target labels.

    Returns:
    - mae_box (float): Mean Absolute Error.
    """
    temp = []
    
    for i in range(len(y_true)):
        if 0 < y_true[i] <= 10:
            y_box = 1
        elif 10 < y_true[i] <= 20:
            y_box = 2
        elif 20 < y_true[i] <= 30:
            y_box = 3
        temp.append(np.clip(y_pred[i], (10 * y_box) - 10, 10 * y_box))
        
    mae_box = mean_absolute_error(y_true, np.array(temp)) 
    
    return mae_box 

mae_box_scorer = make_scorer(score_mae_box, greater_is_better=False)

def score_rmse_box(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) considering box constraints.

    Args:
    - y_true (ndarray): True target labels.
    - y_pred (ndarray): Predicted target labels.

    Returns:
    - rmse_box (float): Root Mean Squared Error.
    """
    temp = []
    
    for i in range(len(y_true)):
        if 0 < y_true[i] <= 10:
            y_box = 1
        elif 10 < y_true[i] <= 20:
            y_box = 2
        elif 20 < y_true[i] <= 30:
            y_box = 3
        temp.append(np.clip(y_pred[i], (10 * y_box) - 10, 10 * y_box))
        
    rmse_box = mean_squared_error(y_true, np.array(temp), squared=False) 
    return rmse_box 

rmse_box_scorer = make_scorer(score_rmse_box, greater_is_better=False)

def score_corr_box(y_true, y_pred):
    """
    Calculate the correlation coefficient considering box constraints.

    Args:
    - y_true (ndarray): True target labels.
    - y_pred (ndarray): Predicted target labels.

    Returns:
    - corr_box (float): Pearson correlation coefficient.
    """
    temp = []
    
    for i in range(len(y_true)):
        if 0 < y_true[i] <= 10:
            y_box = 1
        elif 10 < y_true[i] <= 20:
            y_box = 2
        elif 20 < y_true[i] <= 30:
            y_box = 3
        temp.append(np.clip(y_pred[i], 10 * y_box - 10, 10 * y_box))
        
    corr_box = pearsonr(np.array(y_true).flatten(), np.array(temp).flatten())[0]
    
    return corr_box 

def score_r2_box(y_true, y_pred):
    """
    Calculate the correlation coefficient considering box constraints.

    Args:
    - y_true (ndarray): True target labels.
    - y_pred (ndarray): Predicted target labels.

    Returns:
    - r2_box (float): R squared.
    """
    temp = []
    
    for i in range(len(y_true)):
        if 0 < y_true[i] <= 10:
            y_box = 1
        elif 10 < y_true[i] <= 20:
            y_box = 2
        elif 20 < y_true[i] <= 30:
            y_box = 3
        temp.append(np.clip(y_pred[i], 10 * y_box - 10, 10 * y_box))
        
    r2_box = r2_score(np.array(y_true).flatten(), np.array(temp).flatten())
    
    return r2_box 

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def score_mape_box(y_true, y_pred):
    """
    Calculate the correlation coefficient considering box constraints.

    Args:
    - y_true (ndarray): True target labels.
    - y_pred (ndarray): Predicted target labels.

    Returns:
    - mape_box (float): MAPE.
    """
    temp = []
    
    for i in range(len(y_true)):
        if 0 < y_true[i] <= 10:
            y_box = 1
        elif 10 < y_true[i] <= 20:
            y_box = 2
        elif 20 < y_true[i] <= 30:
            y_box = 3
        temp.append(np.clip(y_pred[i], 10 * y_box - 10, 10 * y_box))
        
    mape_box = mean_absolute_percentage_error(np.array(y_true).flatten(), np.array(temp).flatten())
    
    return mape_box


def evaluate_model_performance(y_tests, random_predictions, model_predictions, clipped=True):
    """
    Evaluate model performance by comparing it with a random baseline using paired t-tests.

    Parameters:
    - y_tests: List of arrays containing actual values of the target variable for each iteration.
    - random_predictions: List of arrays containing random baseline predictions for each iteration.
    - model_predictions: List of arrays containing model predictions for each iteration.

    Returns:
    - Dictionary containing t-statistics, p-values, and mean ± std for MAE, RMSE, and correlation.
    """
   
    if clipped:
    # Clip predictions based on boxes
        random_predictions = [
            clip_based_on_boxes(y_test, random_pred) 
            for y_test, random_pred in zip(y_tests, random_predictions)
        ]
        model_predictions = [
            clip_based_on_boxes(y_test, model_pred) 
            for y_test, model_pred in zip(y_tests, model_predictions)
        ]
        
    # Calculate metrics for random predictions
    random_maes = [round(mean_absolute_error(y_test, random_pred), 2) for y_test, random_pred in zip(y_tests, random_predictions)]
    random_rmses = [round(mean_squared_error(y_test, random_pred, squared=False), 2) for y_test, random_pred in zip(y_tests, random_predictions)]
    random_corrs = [round(pearsonr(y_test, random_pred)[0], 2) for y_test, random_pred in zip(y_tests, random_predictions)]
    random_r2 = [round(r2_score(y_test, random_pred), 2) for y_test, random_pred in zip(y_tests, random_predictions)]
    random_mape = [round(mean_absolute_percentage_error(y_test, random_pred), 2) for y_test, random_pred in zip(y_tests, random_predictions)]

    # Calculate metrics for the model
    model_maes = [round(mean_absolute_error(y_test, model_pred), 2) for y_test, model_pred in zip(y_tests, model_predictions)]
    model_rmses = [round(mean_squared_error(y_test, model_pred, squared=False), 2) for y_test, model_pred in zip(y_tests, model_predictions)]
    model_corrs = [round(pearsonr(y_test, model_pred)[0], 2) for y_test, model_pred in zip(y_tests, model_predictions)]
    model_r2 = [round(r2_score(y_test, model_pred), 2) for y_test, model_pred in zip(y_tests, model_predictions)]
    model_mape = [round(mean_absolute_percentage_error(y_test, model_pred), 2) for y_test, model_pred in zip(y_tests, model_predictions)]

    
    #stat, p_value = wilcoxon(model_maes, random_maes, alternative='two-sided')
    # Paired t-test
    t_stat_mae, p_value_mae = ttest_rel(model_maes, random_maes)
    t_stat_rmse, p_value_rmse = ttest_rel(model_rmses, random_rmses)
    t_stat_corr, p_value_corr = ttest_rel(model_corrs, random_corrs)
    t_stat_r2, p_value_r2 = ttest_rel(model_r2, random_r2)
    t_stat_mape, p_value_mape = ttest_rel(model_mape, random_mape)

    # Print results
    print(f"Random Forest MAE: {np.mean(model_maes)} ± {np.std(model_maes)}, Random Regressor MAE: {np.mean(random_maes)} ± {np.std(random_maes)}")
    print(f"Random Forest RMSE: {np.mean(model_rmses)} ± {np.std(model_rmses)}, Random Regressor RMSE: {np.mean(random_rmses)} ± {np.std(random_rmses)}")
    print(f"Random Forest Corr: {np.mean(model_corrs)} ± {np.std(model_corrs)}, Random Regressor Corr: {np.mean(random_corrs)} ± {np.std(random_corrs)}")
    print(f"Random Forest R2: {np.mean(model_r2)} ± {np.std(model_r2)}, Random Regressor R2: {np.mean(random_r2)} ± {np.std(random_r2)}")
    print(f"Random Forest MAPE: {np.mean(model_mape)} ± {np.std(model_mape)}, Random Regressor R2: {np.mean(random_mape)} ± {np.std(random_mape)}")

    print(f"MAE t-statistic: {t_stat_mae}, p-value: {p_value_mae}")
    print(f"RMSE t-statistic: {t_stat_rmse}, p-value: {p_value_rmse}")
    print(f"Correlation t-statistic: {t_stat_corr}, p-value: {p_value_corr}")
    print(f"R2 t-statistic: {t_stat_r2}, p-value: {p_value_r2}")
    print(f"MAPE t-statistic: {t_stat_mape}, p-value: {p_value_mape}")

    # Interpretation
    if p_value_mae < 0.05:
        print("The improvement in MAE is statistically significant.")
    else:
        print("The improvement in MAE is not statistically significant.")

    if p_value_rmse < 0.05:
        print("The improvement in RMSE is statistically significant.")
    else:
        print("The improvement in RMSE is not statistically significant.")

    if p_value_corr < 0.05:
        print("The improvement in Correlation is statistically significant.")
    else:
        print("The improvement in Correlation is not statistically significant.")
        
    if p_value_r2 < 0.05:
        print("The improvement in R squared is statistically significant.")
    else:
        print("The improvement in R squared is not statistically significant.")
        
    if p_value_mape < 0.05:
        print("The improvement in MAPE is statistically significant.")
    else:
        print("The improvement in MAPE is not statistically significant.")

    # Return results as a dictionary
    return {
        "MAE": {"t-statistic": t_stat_mae, "p-value": p_value_mae, "mean_model": np.mean(model_maes), "std_model": np.std(model_maes), "mean_random": np.mean(random_maes), "std_random": np.std(random_maes)},
        "RMSE": {"t-statistic": t_stat_rmse, "p-value": p_value_rmse, "mean_model": np.mean(model_rmses), "std_model": np.std(model_rmses), "mean_random": np.mean(random_rmses), "std_random": np.std(random_rmses)},
        "Correlation": {"t-statistic": t_stat_corr, "p-value": p_value_corr, "mean_model": np.mean(model_corrs), "std_model": np.std(model_corrs), "mean_random": np.mean(random_corrs), "std_random": np.std(random_corrs)},
        "R2": {"t-statistic": t_stat_r2, "p-value": p_value_r2, "mean_model": np.mean(model_r2), "std_model": np.std(model_r2), "mean_random": np.mean(random_r2), "std_random": np.std(random_r2)},
        "MAPE": {"t-statistic": t_stat_mape, "p-value": p_value_mape, "mean_model": np.mean(model_mape), "std_model": np.std(model_mape), "mean_random": np.mean(random_mape), "std_random": np.std(random_mape)},
    }


def ml_gridsearchcv_kfold(model, cv, refit=False):
    
    """
    Perform grid search cross-validation to tune hyperparameters of a machine learning model.

    Parameters:
    - model (str): Name of the machine learning model.
    - predefined_split (bool): Whether to use predefined splits for cross-validation.
    - split_index (list): List of indices indicating the fold to which each sample belongs if predefined_split is True.
    - refit (bool): Whether to refit the best estimator using all available data.

    Returns:
    - GridSearchCV: Grid search cross-validation object.

    This function performs grid search cross-validation to tune hyperparameters of a specified machine learning model. It supports various models including Support Vector Regression (SVR), Lasso, Ridge, K-Nearest Neighbors (KNN), Random Forest, and Decision Tree. The grid of hyperparameters to search over is predefined for each model. If predefined_split is True, the function uses predefined splits for cross-validation, otherwise, it uses standard k-fold cross-validation.

    """

    score = mae_box_scorer #mae_box_scorer #'neg_mean_absolute_error' #  #'r2' #'neg_mean_absolute_error' #neg_mean_squared_error'
    
    if model == 'svr':
        param_grid = {'C': [0.1, 1, 10], 
              'gamma': [1, 0.1, 10],
              'kernel': ['rbf','linear']}
        
        grid = GridSearchCV(SVR(), param_grid, refit=refit, verbose=0, scoring=score,
                                cv=cv)
        
    elif model == 'lasso':
        param_grid = {'alpha': np.logspace(-4, -0.5, 30)} 
        
        grid = GridSearchCV(Lasso(random_state=0, max_iter=10000), param_grid, refit=refit, verbose=0,
                                scoring=score, cv=cv)
        
    elif model == 'ridge':
        param_grid = {'alpha': np.logspace(-4, -0.5, 30)} 
        grid = GridSearchCV(Ridge(random_state=0, max_iter=10000), param_grid, refit=refit, verbose=0,
                                scoring=score, cv=cv)
        
    elif model == 'knn':
        param_grid = {'n_neighbors': np.arange(1, 10)} 
        grid = GridSearchCV(KNeighborsRegressor(), param_grid, refit=refit, verbose=0, scoring=score,
                                cv=cv)
        
    elif model == 'random forest':
        
        param_grid = {'n_estimators': np.arange(100, 200, 50),
               'max_features': ['auto', 'sqrt'],
               'max_depth': np.arange(10, 30, 10),
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}
        """
        param_grid = {
                'n_estimators': np.arange(100, 1000, 100),
                'max_features': [0.1, 0.5, 'auto', 'sqrt'],
                'max_depth': np.arange(10, 100, 10),
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 10],
                'bootstrap': [True, False]
            }
        
        """
        grid = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, refit=refit, verbose=0, 
                            scoring=score, cv=cv)
        """
        n_iter_search = 100  # Adjust this based on how exhaustive you want the search to be

        grid = RandomizedSearchCV(
            RandomForestRegressor(random_state=0),
            param_distributions=param_grid,
            n_iter=n_iter_search,
            random_state=42,  # Ensures reproducibility
            refit=refit,
            verbose=0,
            scoring=score,
            cv=cv
        )
        """
    
    
    elif model == 'tree':
        param_grid = {"min_samples_split": [2, 5, 10],
                       "min_samples_leaf": [1, 2, 4]
                     }
        grid = GridSearchCV(DecisionTreeRegressor(random_state=1), param_grid, refit=refit, verbose=0,
                            scoring=score, cv=cv)
        
    return grid


def random_split_cv(feature_data, selected_columns=[], n_splits=5, model='random forest'):
    """
    Perform 5-fold nested cross-validation for model evaluation.

    Parameters:
    - feature_data (DataFrame): DataFrame containing the feature data.
    - feature (str): Name of the feature being evaluated.
    - method (str): Feature selection method (not used directly here but can be integrated).
    - num_features (int): Number of features to select.
    - n_splits (int): Number of folds for cross-validation.
    - model (str): Machine learning model to use.

    Returns:
    - Tuple: Model predictions, true labels, and performance metrics.
    """
    x_data = feature_data.loc[:, selected_columns].values
    y_data = feature_data.iloc[:,1].values

    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
   
    mae_scores = []
    rmse_scores = []
    corr_scores = []
    r2_scores = []
    mape_scores = []
    
    
    mae_scores_clipped = []
    rmse_scores_clipped = []
    corr_scores_clipped = []
    r2_scores_clipped = []
    mape_scores_clipped = []
    
    model_predictions = []
    y_tests = []

    for split_num, (outer_train_idx, outer_test_idx) in enumerate(outer_cv.split(x_data), start=1):
        
        print(f"\n### Outer Fold {split_num} ###")
        # Outer loop: train-test split
        x_train, x_test = x_data[outer_train_idx], x_data[outer_test_idx]
        y_train, y_test = y_data[outer_train_idx], y_data[outer_test_idx]
        
        """
        if len(x_train.shape) == 1:
            x_train = x_train.reshape(-1, 1)
    
        """
        inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Inner loop: GridSearchCV with cross-validation
        grid = ml_gridsearchcv_kfold(model, cv=inner_cv, refit=True)
        
        
        grid.fit(x_train, y_train)

        # Best model evaluation on the outer test set
        best_model = grid.best_estimator_
        predictions = best_model.predict(x_test)
        
        model_predictions.append(predictions)
        y_tests.append(y_test)

        clipped_pred = clip_based_on_boxes(y_test, predictions)
        
        mae, rmse, corr, r2, mape = calc_all_metrics(y_test, predictions)    
        print(f"Outer Fold Results: MAE={mae:.4f}, RMSE={rmse:.4f}, Correlation={corr:.4f}, R squared={r2:.4f}, MAPE={mape:.4f}")
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        corr_scores.append(corr)
        r2_scores.append(r2)
        mape_scores.append(mape)
        

        mae, rmse, corr, r2, mape = calc_all_metrics(y_test, clipped_pred)    
        print(f"Outer Fold Results Clipped: MAE={mae:.4f}, RMSE={rmse:.4f}, Correlation={corr:.4f}, R squared={r2:.4f}, MAPE={mape:.4f}")
        
        
        mae_scores_clipped.append(mae)
        rmse_scores_clipped.append(rmse)
        corr_scores_clipped.append(corr)
        r2_scores_clipped.append(r2)
        mape_scores_clipped.append(mape)

    # Summary of results
    results = {
        "test_mae": {"mean": np.mean(mae_scores), "std": np.std(mae_scores)},
        "test_rmse": {"mean": np.mean(rmse_scores), "std": np.std(rmse_scores)},
        "test_corr": {"mean": np.mean(corr_scores), "std": np.std(corr_scores)},
        "test_r2": {"mean": np.mean(r2_scores), "std": np.std(r2_scores)},
        "test_mape": {"mean": np.mean(mape_scores), "std": np.std(mape_scores)},
    }
    
    
    
    results_clipped = {
        "test_mae": {"mean": np.mean(mae_scores_clipped), "std": np.std(mae_scores_clipped)},
        "test_rmse": {"mean": np.mean(rmse_scores_clipped), "std": np.std(rmse_scores_clipped)},
        "test_corr": {"mean": np.mean(corr_scores_clipped), "std": np.std(corr_scores_clipped)},
        "test_r2": {"mean": np.mean(r2_scores_clipped), "std": np.std(r2_scores_clipped)},
        "test_mape": {"mean": np.mean(mape_scores_clipped), "std": np.std(mape_scores_clipped)},
    }

    print("\nFinal Results Across Outer Folds:")
    print(f'MAE: {results["test_mae"]["mean"]:.4f} ± {results["test_mae"]["std"]:.4f}')
    print(f'RMSE: {results["test_rmse"]["mean"]:.4f} ± {results["test_rmse"]["std"]:.4f}')
    print(f'Correlation: {results["test_corr"]["mean"]:.4f} ± {results["test_corr"]["std"]:.4f}')
    print(f'R squared: {results["test_r2"]["mean"]:.4f} ± {results["test_r2"]["std"]:.4f}')
    print(f'MAPE: {results["test_mape"]["mean"]:.4f} ± {results["test_mape"]["std"]:.4f}')
    
    
    
    print("\nFinal Results Clipped Across Outer Folds:")
    print(f'MAE: {results_clipped["test_mae"]["mean"]:.4f} ± {results_clipped["test_mae"]["std"]:.4f}')
    print(f'RMSE: {results_clipped["test_rmse"]["mean"]:.4f} ± {results_clipped["test_rmse"]["std"]:.4f}')
    print(f'Correlation: {results_clipped["test_corr"]["mean"]:.4f} ± {results_clipped["test_corr"]["std"]:.4f}')
    print(f'R squared: {results_clipped["test_r2"]["mean"]:.4f} ± {results_clipped["test_r2"]["std"]:.4f}')
    print(f'MAPE: {results_clipped["test_mape"]["mean"]:.4f} ± {results_clipped["test_mape"]["std"]:.4f}')
    
    

    return model_predictions, y_tests, results, results_clipped


def hive_independent_cv(feature_data, selected_columns=[], model='random forest', n_outer_folds=5, n_inner_folds=5, random_regressor=False, min_fob=1, max_fob=30, balance_data=False, normalize=False):
    """
    Perform nested cross-validation using GroupKFold for both the outer and inner loops.

    Parameters:
    - feature_data (DataFrame): DataFrame containing the feature data and a 'tag' column for hive IDs.
    - selected_columns (list): List of feature column names to use.
    - model (str): Machine learning model to use.
    - n_outer_folds (int): Number of folds for the outer cross-validation loop.
    - n_inner_folds (int): Number of folds for the inner cross-validation loop.
    - balance_data (bool): Whether to balance training data during cross-validation.

    Returns:
    - Tuple: Model predictions, true labels, and performance metrics.
    """
    mae_scores = []
    rmse_scores = []
    corr_scores = []
    r2_scores = []
    mape_scores = []

    
    mae_scores_clipped = []
    rmse_scores_clipped = []
    corr_scores_clipped = []
    r2_scores_clipped = []
    mape_scores_clipped = []
    
    
    model_predictions = []
    y_tests = []
    
    hive_results = {}

    # Outer CV: Hive-independent GroupKFold
    outer_cv = GroupKFold(n_splits=n_outer_folds)
    groups = feature_data['tag'].values  # Hive IDs
    
    x_data = feature_data.loc[:, selected_columns].values
    y_data = feature_data.iloc[:, 1].values

    for train_val_idx, test_idx in outer_cv.split(x_data, y_data, groups):
        # Split into training-validation and test sets
        x_train_val, x_test = x_data[train_val_idx], x_data[test_idx]
        y_train_val, y_test = y_data[train_val_idx], y_data[test_idx]
        test_hive_ids = np.unique(groups[test_idx])  # Test hive IDs

        print(f"Outer Loop - Test Hives: {test_hive_ids}")

        # Balancing training-validation data if enabled
        if balance_data:
            inner_groups = groups[train_val_idx]
            x_train_val, y_train_val, inner_groups = balance_training_data(
                x_train_val, y_train_val, inner_groups, n_bins=10
            )
        else:
            inner_groups = groups[train_val_idx]
            
            
        if normalize:
        
            scaler = MinMaxScaler()
            # Fit the scaler on x_train and transform
            x_train_val = scaler.fit_transform(x_train_val)
            x_test = scaler.fit_transform(x_test)
        
        if random_regressor:
            y_train_val = create_random_array(y_train_val, seed=0, min_fob=min_fob, max_fob=max_fob)
            
        # Inner CV: Hive-independent GroupKFold
        inner_cv = GroupKFold(n_splits=n_inner_folds)
        inner_folds = []
        for train_idx, val_idx in inner_cv.split(x_train_val, y_train_val, inner_groups):
            inner_folds.append((train_idx, val_idx))

        
        #inner_folds = KFold(n_splits=5, shuffle=True, random_state=42)
        
        
        # Perform hyperparameter tuning using inner folds
        grid = ml_gridsearchcv_kfold(model, cv=inner_folds, refit=True)
        grid.fit(x_train_val, y_train_val)

        # Best model evaluation on the outer test set
        best_model = grid.best_estimator_
        predictions = best_model.predict(x_test)
        
        model_predictions.append(predictions)
        y_tests.append(y_test)
        
        # Save predictions and true labels by hive ID
        for hive_id in test_hive_ids:
            hive_indices = np.where(groups[test_idx] == hive_id)
            hive_results[hive_id] = (y_test[hive_indices], predictions[hive_indices])


        clipped_pred = clip_based_on_boxes(y_test, predictions)
        
        mae, rmse, corr, r2, mape = calc_all_metrics(y_test, predictions)    
        print(f"Outer Fold Results: MAE={mae:.4f}, RMSE={rmse:.4f}, Correlation={corr:.4f}, R squared={r2:.4f}, MAPE={mape:.4f}")
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        corr_scores.append(corr)
        r2_scores.append(r2)
        mape_scores.append(mape)
        

        mae, rmse, corr, r2, mape = calc_all_metrics(y_test, clipped_pred)    
        print(f"Outer Fold Results Clipped: MAE={mae:.4f}, RMSE={rmse:.4f}, Correlation={corr:.4f}, R squared={r2:.4f}, MAPE={mape:.4f}")
        
        
        mae_scores_clipped.append(mae)
        rmse_scores_clipped.append(rmse)
        corr_scores_clipped.append(corr)
        r2_scores_clipped.append(r2)
        mape_scores_clipped.append(mape)

    # Summary of results
    results = {
        "test_mae": {"mean": np.mean(mae_scores), "std": np.std(mae_scores)},
        "test_rmse": {"mean": np.mean(rmse_scores), "std": np.std(rmse_scores)},
        "test_corr": {"mean": np.mean(corr_scores), "std": np.std(corr_scores)},
        "test_r2": {"mean": np.mean(r2_scores), "std": np.std(r2_scores)},
        "test_mape": {"mean": np.mean(mape_scores), "std": np.std(mape_scores)},
    }
    
    
    
    results_clipped = {
        "test_mae": {"mean": np.mean(mae_scores_clipped), "std": np.std(mae_scores_clipped)},
        "test_rmse": {"mean": np.mean(rmse_scores_clipped), "std": np.std(rmse_scores_clipped)},
        "test_corr": {"mean": np.mean(corr_scores_clipped), "std": np.std(corr_scores_clipped)},
        "test_r2": {"mean": np.mean(r2_scores_clipped), "std": np.std(r2_scores_clipped)},
        "test_mape": {"mean": np.mean(mape_scores_clipped), "std": np.std(mape_scores_clipped)},
    }

    print("\nFinal Results Across Outer Folds:")
    print(f'MAE: {results["test_mae"]["mean"]:.4f} ± {results["test_mae"]["std"]:.4f}')
    print(f'RMSE: {results["test_rmse"]["mean"]:.4f} ± {results["test_rmse"]["std"]:.4f}')
    print(f'Correlation: {results["test_corr"]["mean"]:.4f} ± {results["test_corr"]["std"]:.4f}')
    print(f'R squared: {results["test_r2"]["mean"]:.4f} ± {results["test_r2"]["std"]:.4f}')
    print(f'MAPE: {results["test_mape"]["mean"]:.4f} ± {results["test_mape"]["std"]:.4f}')
    
    
    
    print("\nFinal Results Clipped Across Outer Folds:")
    print(f'MAE: {results_clipped["test_mae"]["mean"]:.4f} ± {results_clipped["test_mae"]["std"]:.4f}')
    print(f'RMSE: {results_clipped["test_rmse"]["mean"]:.4f} ± {results_clipped["test_rmse"]["std"]:.4f}')
    print(f'Correlation: {results_clipped["test_corr"]["mean"]:.4f} ± {results_clipped["test_corr"]["std"]:.4f}')
    print(f'R squared: {results_clipped["test_r2"]["mean"]:.4f} ± {results_clipped["test_r2"]["std"]:.4f}')
    print(f'MAPE: {results_clipped["test_mape"]["mean"]:.4f} ± {results_clipped["test_mape"]["std"]:.4f}')

    return model_predictions, y_tests, results, results_clipped, hive_results

def calc_all_metrics(y_true, y_pred):
    
    mae = mean_absolute_error(y_true, y_pred) 
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    corr = pearsonr(y_true, y_pred)[0] 
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mae, rmse, corr, r2, mape
    
def features_selection_cv(feature, feature_data, selected_columns, n_splits, method, split, model, preprocessing):    
    # Extract x_data and y_data
    x_data = feature_data.loc[:, selected_columns].values
    y_data = feature_data.iloc[:, 1].values  # Target

    
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    splits = list(outer_cv.split(x_data))  # Convert the generator to a list
    first_split = splits[0]  # Access the first split

    # Unpack train and test indices
    outer_train_idx, outer_test_idx = first_split
    # Use the indices to create training and testing sets
    x_train = x_data[outer_train_idx]  # Training features
    y_train = y_data[outer_train_idx]  # Training labels
    
    
    
    
    ranking_filename = f"{feature}_{method}_feature_ranking_{split}_preprocessing_{preprocessing}.csv"

    # Check if the ranking file already exists
    if os.path.exists(ranking_filename):
        
        print(f"Ranking file found. Loading from {ranking_filename}")
        ranking_df = pd.read_csv(ranking_filename)
        feature_ranking_idxs = ranking_df["Feature Index"].values
        
    else:
        
        print("Ranking file not found. Computing feature rankings...")
    
        feature_ranking_idxs = np.zeros(len(selected_columns))
        
        # Select features based on the method (mRMR or SHAP)
        if method == 'mrmr':
            num_features_to_select = x_data.shape[1]  # Total number of features
            K_MAX = 5800
            estimator = depmeas.mi_tau
            n_jobs = -1
            feature_ranking_idxs = feature_select.feature_select_optimized(
                x_train,
                y_train,
                num_features_to_select=num_features_to_select,
                K_MAX=K_MAX,
                estimator=estimator,
                n_jobs=n_jobs,
            )

        elif method == 'shap':
            # SHAP
            X100 = shap.utils.sample(x_train.astype(float), 100)  # 100 instances for the background distribution
            grid = ml_gridsearchcv_kfold(model, cv=None, refit=True)
            grid.fit(x_train.astype(float), y_train)
            explainer = shap.Explainer(grid.predict, X100, feature_names=feature_data.iloc[:, 2:].columns)
            shap_values = explainer(x_train.astype(float), max_evals=12000)

            vals = np.abs(shap_values.values).mean(0)
            feature_ranking_idxs = np.argsort(vals)[::-1]  # Sort features by SHAP importance

        # Save feature ranking indices to CSV
        ranking_df = pd.DataFrame({
            "Rank": range(1, len(feature_ranking_idxs) + 1),
            "Feature Index": feature_ranking_idxs
        })
        ranking_filename = f"{feature}_{method}_feature_ranking_{split}_preprocessing_{preprocessing}.csv"
        ranking_df.to_csv(ranking_filename, index=False)
        print(f"Feature rankings saved to {ranking_filename}")
    
    mae_scores = []
    rmse_scores = []
    corr_scores = []
    r2_scores = []
    mape_scores = []

    mae_scores_std = []
    rmse_scores_std = []
    corr_scores_std = []
    r2_scores_std = []
    mape_scores_std = []

    # Loop through feature selection
    for i in range(1, min(len(x_data[0]), 30)+1):  # range(1, len(x_data[0]) + 1)# Start from 1 to select at least 1 feature
        print(f"\n### Number of Features: {i} ###")

        selected_features = feature_data.columns[2:][feature_ranking_idxs[:i]]
        

        # Run nested CV for the selected features
        print(f"Running Nested CV for {method} with {i} features...")
        
        if split=='random':
        
            _, _, _, results = random_split_cv(
                feature_data,
                selected_columns=selected_features,
                n_splits=n_splits,
                model=model
            )
            
        elif split=='independent':
            
           #model_predictions, y_tests, results, results_clipped, hive_results 
            _, _,_, results, _ = hive_independent_cv(
               feature_data,
               selected_columns=selected_features,
               model=model, n_outer_folds=3, n_inner_folds=2, balance_data=False)

        mae_scores.append(results["test_mae"]["mean"])
        rmse_scores.append(results["test_rmse"]["mean"])
        corr_scores.append(results["test_corr"]["mean"])
        r2_scores.append(results["test_r2"]["mean"])
        mape_scores.append(results["test_mape"]["mean"])

        # Append the standard deviations
        mae_scores_std.append(results["test_mae"]["std"])
        rmse_scores_std.append(results["test_rmse"]["std"])
        corr_scores_std.append(results["test_corr"]["std"])
        r2_scores_std.append(results["test_r2"]["std"])
        mape_scores_std.append(results["test_mape"]["std"])

    # Return all results
    results = {
        "mae_scores": {"mean": mae_scores, "std": mae_scores_std},
        "rmse_scores": {"mean": rmse_scores, "std": rmse_scores_std},
        "corr_scores": {"mean": corr_scores, "std": corr_scores_std},
        "r2_scores": {"mean": r2_scores, "std": r2_scores_std},
        "mape_scores": {"mean": mape_scores, "std": mape_scores_std},
    }
    
    save_results_to_csv(results, method, feature, split, preprocessing)

    return results

def balance_training_data(x_train, y_train, groups, n_bins=10, samples_per_bin=None):
    """
    Balance training data for regression by binning continuous target values.

    Parameters:
    - x_train (np.ndarray): Training features.
    - y_train (np.ndarray): Training target values.
    - groups (np.ndarray): Group labels corresponding to training data.
    - n_bins (int): Number of bins for stratification.
    - samples_per_bin (int or None): Number of samples per bin. If None, uses the minimum bin size.

    Returns:
    - Balanced x_train, y_train, and groups.
    """
    # Create a DataFrame to handle x, y, and groups together
    df = pd.DataFrame({'y': y_train, 'group': groups, 'x': list(x_train)})
    df['bin'] = pd.cut(df['y'], bins=n_bins, labels=False)

    # Determine the number of samples per bin
    bin_counts = df['bin'].value_counts()
    min_samples = bin_counts.min() if samples_per_bin is None else samples_per_bin

    # Sample from each bin
    balanced_df = df.groupby('bin').apply(
        lambda x: x.sample(min(min_samples, len(x)), random_state=42)
    ).reset_index(drop=True)

    # Extract balanced x, y, and groups
    x_balanced = np.vstack(balanced_df['x'].values)
    y_balanced = balanced_df['y'].values
    groups_balanced = balanced_df['group'].values

    return x_balanced, y_balanced, groups_balanced

def save_results_to_csv(results, method, feature, split, preprocessing='off'):
    # Extract values from the results dictionary
    mae_mean = results["mae_scores"]["mean"]
    mae_std = results["mae_scores"]["std"]
    rmse_mean = results["rmse_scores"]["mean"]
    rmse_std = results["rmse_scores"]["std"]
    corr_mean = results["corr_scores"]["mean"]
    corr_std = results["corr_scores"]["std"]
    r2_mean = results["r2_scores"]["mean"]
    r2_std = results["r2_scores"]["std"]
    mape_mean = results["mape_scores"]["mean"]
    mape_std = results["mape_scores"]["std"]

    # Create a DataFrame to store the results
    data = {
        "Number of Features": list(range(1, len(mae_mean) + 1)),  # Assuming equal length for all lists
        "MAE Mean": mae_mean,
        "MAE Std": mae_std,
        "RMSE Mean": rmse_mean,
        "RMSE Std": rmse_std,
        "Correlation Mean": corr_mean,
        "Correlation Std": corr_std,
        "R2 Mean": r2_mean,
        "R2 Std": r2_std,
        "MAPE Mean": mape_mean,
        "MAPE Std": mape_std
    }

    df = pd.DataFrame(data)

    # Save to CSV
    
    file_name = f"{feature}_{method}_{split}_preprocessing_{preprocessing}.csv"
    file_name = file_name.replace(" ", "_")  # Replace spaces with underscores
    
    df.to_csv(file_name, index=False)
    
    
def read_and_plot_results(method, feature, split, preprocessing='off'):
    """
    Reads the CSV file containing results and plots metrics with means and standard deviations.

    Parameters:
        csv_file (str): Path to the CSV file.
    """
    # Read the CSV file
    file_name = f"{feature}_{method}_{split}_preprocessing_{preprocessing}.csv"
    df = pd.read_csv(file_name)

    # Extract data
    num_features = df["Number of Features"]
    metrics = {
        "MAE": {"mean": df["MAE Mean"], "std": df["MAE Std"]},
        "RMSE": {"mean": df["RMSE Mean"], "std": df["RMSE Std"]},
        "Correlation": {"mean": df["Correlation Mean"], "std": df["Correlation Std"]},
        "R2": {"mean": df["R2 Mean"], "std": df["R2 Std"]},
        "MAPE": {"mean": df["MAPE Mean"], "std": df["MAPE Std"]},
    }

    # Plot
    plt.figure(figsize=(10, 6))
    for metric_name, metric_data in metrics.items():
        mean = metric_data["mean"]
        std = metric_data["std"]

        # Plot the mean line
        plt.plot(num_features, mean, label=f"{metric_name} Mean")
        
        # Add shaded area for std
        plt.fill_between(num_features, mean - std, mean + std, alpha=0.2, label=f"{metric_name} ± Std")

    # Plot settings
    plt.title("Metrics Across Feature Selection")
    plt.xlabel("Number of Features")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f'{feature}_{method}_{split}_preprocessing_{preprocessing}_error.png', dpi=400)
