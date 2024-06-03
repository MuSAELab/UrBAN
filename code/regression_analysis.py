import numpy as np
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.svm import SVR
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer

import shap
#from feature_selection import select_best_feature_selection_method

def get_splits(feature_data, train_hives, val_hives, test_hives):
    """
    Split the feature data into train, validation, and test sets based on the given hive tags.

    Args:
    - feature_data (DataFrame): DataFrame containing the feature data.
    - train_hives (list): List of hive tags for the training set.
    - val_hives (list): List of hive tags for the validation set.
    - test_hives (list): List of hive tags for the test set.

    Returns:
    - x_train (ndarray): Feature data for the training set.
    - y_train (ndarray): Target labels for the training set.
    - x_val (ndarray): Feature data for the validation set.
    - y_val (ndarray): Target labels for the validation set.
    - x_test (ndarray): Feature data for the test set.
    - y_test (ndarray): Target labels for the test set.
    """
    
    
    x_train = feature_data[feature_data['tag'].isin(train_hives)].iloc[:, :-5].values
    y_train = feature_data[feature_data['tag'].isin(train_hives)].iloc[:, -1].values

    x_val = feature_data[feature_data['tag'].isin(val_hives)].iloc[:, :-5].values
    y_val = feature_data[feature_data['tag'].isin(val_hives)].iloc[:, -1].values

    x_test = feature_data[feature_data['tag'].isin(test_hives)].iloc[:, :-5].values
    y_test = feature_data[feature_data['tag'].isin(test_hives)].iloc[:, -1].values
    
    return x_train, y_train, x_val, y_val, x_test, y_test




def ml_gridsearchcv(model, predefined_split=False, split_index=[], refit=False):
    
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

    score = mae_box_scorer #'neg_mean_absolute_error' # mae_box_scorer #'r2' #'neg_mean_absolute_error' #neg_mean_squared_error'
    
    if model == 'svr':
        param_grid = {'C': [0.1, 1, 10], 
              'gamma': [1, 0.1, 10],
              'kernel': ['rbf','linear']}
        if predefined_split:
            grid = GridSearchCV(SVR(), param_grid, refit=refit, verbose=0, scoring=score,
                                cv=PredefinedSplit(test_fold=split_index))
        else:              
            grid = GridSearchCV(SVR(), param_grid, refit=refit, verbose=0, scoring=score)
        
    elif model == 'lasso':
        param_grid = {'alpha': np.logspace(-4, -0.5, 30)} 
        if predefined_split:
            grid = GridSearchCV(Lasso(random_state=0, max_iter=10000), param_grid, refit=refit, verbose=0,
                                scoring=score, cv=PredefinedSplit(test_fold=split_index))
        else:
            grid = GridSearchCV(Lasso(random_state=0, max_iter=10000), param_grid, refit=refit, verbose=0, scoring=score)
        
    elif model == 'ridge':
        param_grid = {'alpha': np.logspace(-4, -0.5, 30)} 
        if predefined_split:
            grid = GridSearchCV(Ridge(random_state=0, max_iter=10000), param_grid, refit=refit, verbose=0,
                                scoring=score, cv=PredefinedSplit(test_fold=split_index))
        else:
            grid = GridSearchCV(Ridge(random_state=0, max_iter=10000), param_grid, refit=refit, verbose=0, scoring=score)
        
    elif model == 'knn':
        param_grid = {'n_neighbors': np.arange(1, 10)} 
        if predefined_split:
            grid = GridSearchCV(KNeighborsRegressor(), param_grid, refit=refit, verbose=0, scoring=score,
                                cv=PredefinedSplit(test_fold=split_index))
        else:
            grid = GridSearchCV(KNeighborsRegressor(), param_grid, refit=refit, verbose=0, scoring=score)
        
    elif model == 'random forest':
        param_grid = {'n_estimators': np.arange(100, 200, 50),
               'max_features': ['auto', 'sqrt'],
               'max_depth': np.arange(10, 30, 10),
               'min_samples_split': [2, 5, 10],
               'min_samples_leaf': [1, 2, 4],
               'bootstrap': [True, False]}
        if predefined_split:
            grid = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, refit=refit, verbose=0, 
                               scoring=score, cv=PredefinedSplit(test_fold=split_index))
        else:
            grid = GridSearchCV(RandomForestRegressor(random_state=0), param_grid, refit=refit, verbose=0, scoring=score)
        
    elif model == 'tree':
        param_grid = {"min_samples_split": [2, 5, 10],
                       "min_samples_leaf": [1, 2, 4]
                     }
        if predefined_split:
            grid = GridSearchCV(DecisionTreeRegressor(random_state=1), param_grid, refit=refit, verbose=0,
                            scoring=score, cv=PredefinedSplit(test_fold=split_index))
        else:
            grid = GridSearchCV(DecisionTreeRegressor(random_state=1), param_grid, refit=refit, verbose=0, scoring=score)
        
    return grid




def select_best_feature_selection_method(feature, feature_data, x_train, x_val, y_train, y_val, methods=['importance', 'pca', 'mrmr', 'shap'], model='random forest', predefined_split=False, refit=True):
    """
    Select the best feature selection method based on evaluation metrics.

    Parameters:
    - feature (str): Name of the feature or dimension.
    - feature_data (pd.DataFrame): DataFrame containing feature data.
    - x_train (np.ndarray): Training input features.
    - x_val (np.ndarray): Validation input features.
    - y_train (np.ndarray): Training target values.
    - y_val (np.ndarray): Validation target values.
    - methods (list): List of feature selection methods to evaluate.
    - model (str): Machine learning model to use for evaluation (default is 'random forest').
    - predefined_split (bool): Whether to use predefined splits for cross-validation (default is False).
    - refit (bool): Whether to refit the model after selecting features (default is True).

    Returns:
    - str: Best feature selection method.
    - int: Number of features selected by the best method.
    """
    split = 'rndm'
    if predefined_split:
        split = 'indpnt'        

    best_method = None
    best_num_features = None
    best_score = float('inf')

    results = {}
    scaler = MinMaxScaler()

    for method in methods:
        val_mae, val_rmse, val_corr = feature_selection(feature_data, x_train, x_val, y_train, y_val, method, model, predefined_split, refit)

        # Convert correlation to negative correlation to minimize it
        val_corr = [-corr for corr in val_corr]

        # Normalize the scores
        mae_norm = scaler.fit_transform(np.array(val_mae).reshape(-1, 1)).flatten()
        rmse_norm = scaler.fit_transform(np.array(val_rmse).reshape(-1, 1)).flatten()
        corr_norm = scaler.fit_transform(np.array(val_corr).reshape(-1, 1)).flatten()

        # Aggregate the normalized scores
        aggregate_score = mae_norm + rmse_norm + corr_norm

        # Find the best number of features for the current method
        best_index = np.argmin(aggregate_score)
        results[method] = {
            'num_features': best_index + 1,
            'mae': val_mae[best_index],
            'rmse': val_rmse[best_index],
            'corr': val_corr[best_index]
        }

        if aggregate_score[best_index] < best_score:
            best_score = aggregate_score[best_index]
            best_method = method
            best_num_features = best_index + 1

    print(f'Best Method: {best_method}')
    print(f'Number of Features: {best_num_features}')
    
    return best_method, best_num_features





def performance_on_selected_features(feature_data, x_train, x_val, x_test,  y_train, y_val, y_test, method, num_features, 
                                     model='random forest', predefined_split=False, refit=True):
    
    """
    Evaluate the performance of a machine learning model using selected features.

    Parameters:
    - feature_data (DataFrame): DataFrame containing feature data.
    - x_train (array-like): Input features for training.
    - x_val (array-like): Input features for validation.
    - x_test (array-like): Input features for testing.
    - y_train (array-like): Target labels for training.
    - y_val (array-like): Target labels for validation.
    - y_test (array-like): Target labels for testing.
    - method (str): Feature selection method ('importance', 'pca', 'mrmr', 'shap').
    - num_features (int): Number of selected features.
    - model (str, optional): Machine learning model to use. Defaults to 'random forest'.
    - predefined_split (bool, optional): Whether to use predefined splits for cross-validation. Defaults to False.
    - refit (bool, optional): Whether to refit the model after feature selection. Defaults to True.

    Returns:
    - tuple: Mean absolute error (MAE), root mean squared error (RMSE), Pearson correlation coefficient, and predictions.
    """
        
    
    i = num_features

    if method == 'importance':
        
        model = RandomForestRegressor(random_state = 0)
        model.fit(x_train, y_train)
        importance = model.feature_importances_
        

        x_train_i = x_train[:, importance.argsort()[-i:][::-1]]
        x_test_i = x_test[:, importance.argsort()[-i:][::-1]]
        x_val_i = x_val[:, importance.argsort()[-i:][::-1]]

        x_train_val_i = np.concatenate([x_train_i, x_val_i], axis = 0)
        y_train_val = np.concatenate([y_train, y_val], axis = 0)

        
    elif method == 'pca':
        
            
        pca_features = PCA(n_components=i, random_state=0)
        x_train_i = pca_features.fit_transform(x_train)
        x_val_i = pca_features.fit_transform(x_val)
        x_test_i = pca_features.fit_transform(x_test)

        min_max_scaler = MinMaxScaler()
        x_train_i = min_max_scaler.fit_transform(x_train_i)
        x_val_i = min_max_scaler.fit_transform(x_val_i)
        x_test_i = min_max_scaler.fit_transform(x_test_i)

        x_train_val_i = np.concatenate([x_train_i, x_val_i], axis = 0)
        y_train_val = np.concatenate([y_train, y_val], axis = 0)


    elif method == 'mrmr':
        
        num_features_to_select = len(feature_data.iloc[:, :-5].columns)
        K_MAX = 1000
        estimator = depmeas.mi_tau
        n_jobs = -1
        feature_ranking_idxs = feature_select.feature_select(x_train,y_train,
            num_features_to_select=num_features_to_select,K_MAX=K_MAX,
            estimator=estimator,n_jobs=n_jobs)

        num_selected_features = len(feature_ranking_idxs)


        x_train_i = x_train[:,feature_ranking_idxs[0:i]]
        x_test_i = x_test[:,feature_ranking_idxs[0:i]]
        x_val_i = x_val[:,feature_ranking_idxs[0:i]]


        x_train_val_i = np.concatenate([x_train_i, x_val_i], axis = 0)
        y_train_val = np.concatenate([y_train, y_val], axis = 0)


    elif method == 'shap':
        
        X100 = shap.utils.sample(x_train.astype(float), 100)  # 100 instances for use as the background distribution
        grid = ml_gridsearchcv('random forest', refit=True)
        grid.fit(x_train.astype(float), y_train)
        explainer = shap.Explainer(grid.predict, X100, feature_names=feature_data.iloc[:, :-5].columns)
        shap_values = explainer(x_train.astype(float))#, max_evals=6401
        
        vals= np.abs(shap_values.values).mean(0)    
                        
        x_train_i = x_train[:, vals.argsort()[-i:][::-1]]
        x_val_i = x_val[:, vals.argsort()[-i:][::-1]]
        x_test_i = x_test[:, vals.argsort()[-i:][::-1]]

        x_train_val_i = np.concatenate([x_train_i, x_val_i], axis = 0)
        y_train_val = np.concatenate([y_train, y_val], axis = 0)
        
        
    split_index = []
            
    if predefined_split:

        for i in range(len(x_train_i)):
            split_index.append(-1)
        for i in range(len(x_val_i)):
            split_index.append(0)
            
    grid = ml_gridsearchcv('random forest', predefined_split = predefined_split, split_index = split_index, refit=refit)
    grid.fit(x_train_val_i, y_train_val)

    predictions = grid.predict(x_test_i)
    mae = score_mae_box(y_test, grid.predict(x_test_i))
    rmse = score_rmse_box(y_test, grid.predict(x_test_i))
    corr = score_corr_box(y_test, grid.predict(x_test_i))
    
    print('MAE: ', mae)
    print('RMSE: ', rmse)
    print('Correlation: ', corr)

    return mae, rmse, corr, predictions



def random_split_evaluation(feature_data, feature, n_iterations=10, model='random forest'):
    """
    Evaluate the performance of a model using random splits of the dataset.

    Parameters:
    - feature_data (DataFrame): DataFrame containing the feature data.
    - feature (str): Name of the feature being evaluated.
    - n_iterations (int): Number of iterations for random splits.
    - model (str): Name of the machine learning model to use.

    Returns:
    - Tuple: A tuple containing the model predictions and true labels for each iteration.

    This function performs a random split of the dataset into training, validation, and test sets. It then selects the best feature selection method and evaluates the performance of the selected features using the specified machine learning model. This process is repeated for the specified number of iterations.

    """

    x_data = feature_data.iloc[:, :-5].values
    y_data = feature_data.iloc[:, -1].values

    mae_scores = []
    rmse_scores = []
    corr_scores = []
    model_predictions = []
    y_tests = []

    for i in range(n_iterations):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=i)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=i)
        
        method, num_features = select_best_feature_selection_method(
            feature, feature_data, x_train, x_val, y_train, y_val, 
            methods=['importance', 'pca', 'mrmr', 'shap'], model=model, predefined_split=False, refit=True)

        mae, rmse, corr, pred = performance_on_selected_features(feature_data, x_train, x_val, x_test, y_train, y_val, y_test, 
method, num_features, model=model, predefined_split=False, refit=True)
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        corr_scores.append(corr)
        model_predictions.append(pred)
        y_tests.append(y_test)

    results = {
        "MAE": {"mean": np.mean(mae_scores), "std": np.std(mae_scores)},
        "RMSE": {"mean": np.mean(rmse_scores), "std": np.std(rmse_scores)},
        "Correlation": {"mean": np.mean(corr_scores), "std": np.std(corr_scores)},
    }

    print(f'MAE: {results["MAE"]["mean"]:.4f} ± {results["MAE"]["std"]:.4f}')
    print(f'RMSE: {results["RMSE"]["mean"]:.4f} ± {results["RMSE"]["std"]:.4f}')
    print(f'Correlation: {results["Correlation"]["mean"]:.4f} ± {results["Correlation"]["std"]:.4f}')
    
    return model_predictions, y_tests

def independent_split_evaluation(feature_data, feature, initial_train_hives, initial_val_hives, initial_test_hives, n_iterations=10, model='random forest'):
    """
    Evaluate the performance of a model using independent splits of the dataset.

    Parameters:
    - feature_data (DataFrame): DataFrame containing the feature data.
    - feature (str): Name of the feature being evaluated.
    - initial_train_hives (list): List of initial training hives.
    - initial_val_hives (list): List of initial validation hives.
    - initial_test_hives (list): List of initial test hives.
    - n_iterations (int): Number of iterations for independent splits.
    - model (str): Name of the machine learning model to use.

    Returns:
    - Tuple: A tuple containing the model predictions and true labels for each iteration.

    This function performs independent splits of the dataset into training, validation, and test sets. It then selects the best feature selection method and evaluates the performance of the selected features using the specified machine learning model. This process is repeated for the specified number of iterations.

    """

    mae_scores = []
    rmse_scores = []
    corr_scores = []
    model_predictions = []
    y_tests=[]

    all_hives = initial_train_hives + initial_val_hives + initial_test_hives
    np.random.seed(0)

    for i in range(n_iterations):
        
        np.random.shuffle(all_hives)
        train_hives = all_hives[:5]
        val_hives = all_hives[5:7]
        test_hives = all_hives[7:9]
        
        print("="*40)
        print(f"Train hives: {train_hives}")
        print("-"*40)
        print(f"Validation hives: {val_hives}")
        print("-"*40)
        print(f"Test hives: {test_hives}")
        print("="*40)
                
        method, num_features = select_best_feature_selection_method(
            feature, feature_data, x_train, x_val, y_train, y_val, 
            methods=['importance', 'pca', 'mrmr', 'shap'], model=model, predefined_split=True, refit=True)

        mae, rmse, corr, pred = performance_on_selected_features(
            feature_data, x_train, x_val, x_test, y_train, y_val, y_test, 
            method, num_features, model=model, predefined_split=True, refit=True)
        
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        corr_scores.append(corr)
        model_predictions.append(pred)
        y_tests.append(y_test)

    results = {
        "MAE": {"mean": np.mean(mae_scores), "std": np.std(mae_scores)},
        "RMSE": {"mean": np.mean(rmse_scores), "std": np.std(rmse_scores)},
        "Correlation": {"mean": np.mean(corr_scores), "std": np.std(corr_scores)},
    }

    print(f'MAE: {results["MAE"]["mean"]:.4f} ± {results["MAE"]["std"]:.4f}')
    print(f'RMSE: {results["RMSE"]["mean"]:.4f} ± {results["RMSE"]["std"]:.4f}')
    print(f'Correlation: {results["Correlation"]["mean"]:.4f} ± {results["Correlation"]["std"]:.4f}')
    
    return model_predictions, y_tests




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

def random_baseline_metrics(y_tests, n_iterations=10, random_range=(0, 31)):
    """
    Generate random baseline predictions and calculate performance metrics.

    Parameters:
    - y_tests: List of arrays containing actual values of the target variable for each iteration.
    - n_iterations: Number of iterations for generating random predictions.
    - random_range: Tuple specifying the range for random predictions.

    Returns:
    - List of lists containing random predictions for each iteration.
    """
    all_mae_scores = []
    all_rmse_scores = []
    all_corr_scores = []
    all_predicted_lists = []

    np.random.seed(0)  # Set a random seed for reproducibility

    for y_test in y_tests:
        mae_scores = []
        rmse_scores = []
        corr_scores = []
        predicted_list = []

        for _ in range(n_iterations):
            predicted = np.random.uniform(random_range[0], random_range[1], len(y_test))
            mae_scores.append(score_mae_box(y_test, predicted))
            rmse_scores.append(score_rmse_box(y_test, predicted))
            corr_scores.append(score_corr_box(y_test, predicted))
            predicted_list.append(predicted)

        # Collect the predictions and metrics for this y_test
        all_mae_scores.append(mae_scores)
        all_rmse_scores.append(rmse_scores)
        all_corr_scores.append(corr_scores)
        all_predicted_lists.append(np.mean(predicted_list, axis=0))

    # Calculate average and standard deviation across all iterations
    mae_avg = np.mean([np.mean(scores) for scores in all_mae_scores])
    mae_std = np.mean([np.std(scores) for scores in all_mae_scores])
    rmse_avg = np.mean([np.mean(scores) for scores in all_rmse_scores])
    rmse_std = np.mean([np.std(scores) for scores in all_rmse_scores])
    corr_avg = np.mean([np.mean(scores) for scores in all_corr_scores])
    corr_std = np.mean([np.std(scores) for scores in all_corr_scores])

    #print(f'MAE: {mae_avg:.4f} ± {mae_std:.4f}')
    #print(f'RMSE: {rmse_avg:.4f} ± {rmse_std:.4f}')
    #print(f'Correlation: {corr_avg:.4f} ± {corr_std:.4f}')

    return all_predicted_lists



def evaluate_model_performance(y_tests, random_predictions, model_predictions):
    """
    Evaluate model performance by comparing it with a random baseline using paired t-tests.

    Parameters:
    - y_tests: List of arrays containing actual values of the target variable for each iteration.
    - random_predictions: List of arrays containing random baseline predictions for each iteration.
    - model_predictions: List of arrays containing model predictions for each iteration.

    Returns:
    - Dictionary containing t-statistics, p-values, and mean ± std for MAE, RMSE, and correlation.
    """

    # Calculate metrics for the random baseline
    random_maes = [score_mae_box(y_test, random_pred) for y_test, random_pred in zip(y_tests, random_predictions)]
    random_rmses = [score_rmse_box(y_test, random_pred) for y_test, random_pred in zip(y_tests, random_predictions)]
    random_corrs = [score_corr_box(y_test, random_pred) for y_test, random_pred in zip(y_tests, random_predictions)]
    
    # Calculate metrics for the model
    model_maes = [score_mae_box(y_test, model_pred) for y_test, model_pred in zip(y_tests, model_predictions)]
    model_rmses = [score_rmse_box(y_test, model_pred) for y_test, model_pred in zip(y_tests, model_predictions)]
    model_corrs = [score_corr_box(y_test, model_pred) for y_test, model_pred in zip(y_tests, model_predictions)]

    # Paired t-test
    t_stat_mae, p_value_mae = ttest_rel(model_maes, random_maes)
    t_stat_rmse, p_value_rmse = ttest_rel(model_rmses, random_rmses)
    t_stat_corr, p_value_corr = ttest_rel(model_corrs, random_corrs)

    # Print results
    print(f"Random Forest MAE: {np.mean(model_maes)} ± {np.std(model_maes)}, Random Regressor MAE: {np.mean(random_maes)} ± {np.std(random_maes)}")
    print(f"Random Forest RMSE: {np.mean(model_rmses)} ± {np.std(model_rmses)}, Random Regressor RMSE: {np.mean(random_rmses)} ± {np.std(random_rmses)}")
    print(f"Random Forest Corr: {np.mean(model_corrs)} ± {np.std(model_corrs)}, Random Regressor Corr: {np.mean(random_corrs)} ± {np.std(random_corrs)}")

    print(f"MAE t-statistic: {t_stat_mae}, p-value: {p_value_mae}")
    print(f"RMSE t-statistic: {t_stat_rmse}, p-value: {p_value_rmse}")
    print(f"Correlation t-statistic: {t_stat_corr}, p-value: {p_value_corr}")

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

    # Return results as a dictionary
    return {
        "MAE": {"t-statistic": t_stat_mae, "p-value": p_value_mae, "mean_model": np.mean(model_maes), "std_model": np.std(model_maes), "mean_random": np.mean(random_maes), "std_random": np.std(random_maes)},
        "RMSE": {"t-statistic": t_stat_rmse, "p-value": p_value_rmse, "mean_model": np.mean(model_rmses), "std_model": np.std(model_rmses), "mean_random": np.mean(random_rmses), "std_random": np.std(random_rmses)},
        "Correlation": {"t-statistic": t_stat_corr, "p-value": p_value_corr, "mean_model": np.mean(model_corrs), "std_model": np.std(model_corrs), "mean_random": np.mean(random_corrs), "std_random": np.std(random_corrs)},
    }


