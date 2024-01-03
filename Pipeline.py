import pandas as pd
import numpy as np
from imblearn.over_sampling import ADASYN
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import RobustScaler
from fancyimpute import IterativeImputer
from sklearn.linear_model import LinearRegression
from boruta import BorutaPy
from mrmr import mrmr_classif
from imblearn.under_sampling import RandomUnderSampler
import pickle
from sklearn.model_selection import KFold
import sklearn_train as searcher
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from itertools import compress
from skopt import BayesSearchCV


np.random.seed(1000)
rstate = 12 #random state


def gen_classifier_binary(alg):
    # if optimization == 'random':
    est_rs = 1000
    if alg == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(random_state=est_rs)

    elif alg == 'xgb':
        from xgboost import XGBClassifier
        estimator = XGBClassifier(objective='binary:logistic', booster='gbtree', nthread=4, eval_metric='auc',
                                  use_label_encoder=False, random_state=est_rs)

    elif alg == 'cat':
        from catboost import CatBoostClassifier
        cat_features = []
        estimator = CatBoostClassifier(loss_function='Logloss', nan_mode='Min',
                                       one_hot_max_size=31, random_state=est_rs, logging_level='Silent', thread_count=1)

    elif alg == 'elastic':
        from sklearn.linear_model import SGDClassifier
        estimator = SGDClassifier(loss='log', penalty='elasticnet')

    else:
        from sklearn.linear_model import SGDClassifier
        estimator = SGDClassifier(loss='log', penalty='l2')

    return estimator

def gen_parameter_space(alg, optimization):
    space = None
    x = 0
    if alg == 'rf':
        if optimization == 'random':
            if space is None:
                space = {'n_estimators': list(np.arange(50, 300, 50))} # sampled uniformly
        elif optimization == 'bayesian':
            if space is None:
                space = {'n_estimatiors': list(np.arange(50, 300, 50))}
        elif optimization == 'grid':
            if space is None:
                space = {'n_estimators': np.arange(50, 300, 50)}
    elif alg == 'xgb':
        if optimization == 'bayesian':
            if space is None:
                space = {'max_depth': randint(2, 4), 'n_estimators': randint(140, 240), #'scale_pos_weight': uniform(2, 5),
                     'learning_rate': uniform(0.015, 0.12), 'reg_alpha': uniform(1.2, 0.5),
                     'reg_lambda': uniform(1.2, 0.5), 'subsample':uniform(0.85,0.15),
                     'colsample_bytree': uniform(0.85, 0.12), 'gamma': uniform(3, 10)}
        elif optimization == 'random':
            if space is None:
                space = {'max_depth': randint(2, 4), 'n_estimators': list(np.arange(140, 240, 20)), #'scale_pos_weight': uniform(2, 5),
                     'learning_rate': uniform(0.015, 0.12), 'reg_alpha': uniform(1.2, 0.5),
                     'reg_lambda': uniform(1.2, 0.5), 'subsample':uniform(0.85,0.15),
                     'colsample_bytree': uniform(0.85, 0.12), 'gamma': uniform(3, 10)}
        else:
            if space is None:    
                space = {'max_depth': [2, 4], 'n_estimators': np.arange(140, 240, 20),
                     #'scale_pos_weight': np.arange(2, 6, 2),
                     'learning_rate': np.arange(0.085, 0.12, 0.05), 'reg_alpha': np.arange(0.5, 1.2, 0.2),
                     'reg_lambda': np.arange(0.5, 1.2, 0.2), 'subsample':np.arange(0.85,1.0),
                     'colsample_bytree': np.arange(0.85, 0.97, 0.5), 'gamma': np.arange(3, 10, 2)}

    elif alg == 'cat':
        if optimization == 'random':
            x = 1
            if space is None:
                space = {'depth': randint(1, 3), 'n_estimators': randint(90, 95),
                         'learning_rate': uniform(0.01, 0.1),
                         'scale_pos_weight': uniform(1, 3)}  # ,'subsample':uniform(0.85,0.15)}
        elif optimization == 'bayesian':
            if space is None:
                space = {'depth':[2,3,4], 'n_estimators':[80,90,100],
                         'learning_rate': [0.075, 0.085, 0.1, 0.12],
                         'scale_pos_weight': [2,3,4, 5] }  # ,'subsample':uniform(0.85,0.15)}
        elif optimization == 'grid':
            x = 2
            if space is None:
                space = {
                    'depth': [2, 3],  # Usually, depths of 3-10 are tried, but you can keep it small initially.
                    'n_estimators': [90, 100, 110],
                    'learning_rate': [0.1, 0.125, 0.155],
                    'scale_pos_weight': [2, 3],
                    'l2_leaf_reg': [1,2,3]  # Corrected the typo here.
}

    elif alg == 'elastic':
        if optimization == 'random':
            space = {'alpha': np.logspace(-5, 5, 100, endpoint=True), 'l1_ratio': np.arange(0, 1, 0.05)}
        elif optimization == 'bayesian':
            space = {'alpha': np.logspace(-5, 5, 100, endpoint=True), 'l1_ratio': np.arange(0, 1, 0.05)}
        elif optimization == 'grid':
            space = {'alpha': np.logspace(-5, 5, 100, endpoint=True), 'l1_ratio': np.arange(0, 1, 0.05)}

    return space, x


def gen_searcher(estimator, space, strategy='random'):
    searcher_rs = 256
    cv_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    if strategy == 'ray':
        search = TuneSearchCV(estimator, param_distributions=space, early_stopping=False, search_optimization="random",
                              cv=cv_inner, random_state=searcher_rs, use_gpu=False)
    elif strategy == 'bayesian':
        search = BayesSearchCV(estimator, space, n_iter=50, cv=cv_inner, n_jobs=1,random_state=searcher_rs, return_train_score=True, scoring='roc_auc')
    elif strategy == 'random':
        search = RandomizedSearchCV(estimator, space, n_iter=50, cv=cv_inner, n_jobs=4, random_state=searcher_rs, return_train_score=True, verbose=False,  scoring='roc_auc')
    elif strategy == 'grid':
        search = GridSearchCV(estimator, space, cv=cv_inner, n_jobs=4, return_train_score=True, verbose=0,  scoring='roc_auc')

    return search

          
def preprocessing(X_train, X_test, y_train, y_test):
    # Separate the data into each class
    X_train_class_0 = X_train[y_train == 0]
    X_train_class_1 = X_train[y_train == 1]
    X_test_class_0 = X_test[y_test == 0]
    X_test_class_1 = X_test[y_test == 1]

    # Handle missing values for class 0 in training data
    imputer_class_0 = IterativeImputer(estimator=LinearRegression(), initial_strategy='mean', random_state=42)
    X_train_class_0_imputed = pd.DataFrame(imputer_class_0.fit_transform(X_train_class_0), columns=X_train_class_0.columns)
    X_test_class_0_imputed = pd.DataFrame(imputer_class_0.transform(X_test_class_0), columns=X_test_class_0.columns)

    # Handle missing values for class 1 in training data
    imputer_class_1 = IterativeImputer(estimator=LinearRegression(), initial_strategy='mean', random_state=42)
    X_train_class_1_imputed = pd.DataFrame(imputer_class_1.fit_transform(X_train_class_1), columns=X_train_class_1.columns)
    X_test_class_1_imputed = pd.DataFrame(imputer_class_1.transform(X_test_class_1), columns=X_test_class_1.columns)

    # Concatenate the imputed data from each class
    X_train_imputed = pd.concat([X_train_class_0_imputed, X_train_class_1_imputed], axis=0)
    X_test_imputed = pd.concat([X_test_class_0_imputed, X_test_class_1_imputed], axis=0)

    # Apply ADASYN on the imputed training data to address class imbalance
    adasyn = ADASYN()
    X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_imputed, y_train)

    # Undersample the majority class
    class_counts = y_train_resampled.value_counts()
    min_class_count = min(class_counts)
    maj_class_label = class_counts.idxmax()
    undersampler = RandomUnderSampler(sampling_strategy={maj_class_label: min_class_count}, random_state=42)
    X_train_resampled_undersampled, y_train_resampled_undersampled = undersampler.fit_resample(X_train_resampled, y_train_resampled)

    # Apply Robust Scaler to the resampled and undersampled training data
    scaler = RobustScaler()
    X_train_resampled_undersampled_scaled = pd.DataFrame(scaler.fit_transform(X_train_resampled_undersampled), columns=X_train_resampled_undersampled.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns)

    # Drop low variance features
    low_variance_features = X_train_resampled_undersampled_scaled.columns[X_train_resampled_undersampled_scaled.var() < 0.1]  
    X_train_resampled_undersampled_scaled = X_train_resampled_undersampled_scaled.drop(columns=low_variance_features)
    X_test_scaled = X_test_scaled.drop(columns=low_variance_features)
    print("Number of features left after dropping low variance:", X_train_resampled_undersampled_scaled.shape[1])
    
    return X_train_resampled_undersampled_scaled, X_test_scaled, y_train_resampled_undersampled




def univariate_selection(X_train, y_train, k=50):
    # Select top k features based on univariate statistical tests
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)

    # Get the selected feature names
    selected_feature_indices = selector.get_support(indices=True)
    selected_features = X_train.columns[selected_feature_indices].tolist()

    return selected_features


def rfe_feature_selection(X_train, y_train, n_features=50):
    # Perform Recursive Feature Elimination (RFE)
    estimator = RandomForestClassifier()  # You can change the estimator to the desired model
    selector = RFE(estimator, n_features_to_select=n_features)
    selector.fit(X_train, y_train)

    # Get the selected feature names
    selected_feature_indices = selector.support_
    selected_features = X_train.columns[selected_feature_indices].tolist()

    return selected_features


def pca_feature_selection(X_train, n_components=50):
    # Perform Principal Component Analysis (PCA)
    pca = PCA(n_components=n_components)
    pca.fit(X_train)
    X_train_transformed = pca.transform(X_train)

    return X_train_transformed, pca


def boruta_feature_selection(X_train, y_train):
    # Perform Boruta feature selection
    rf = RandomForestClassifier(n_estimators=100)
    selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)
    selector.fit(X_train.values, y_train.values)

    # Get the selected feature names
    selected_features = X_train.columns[selector.support_].tolist()

    return selected_features



def mrmr_feature_selection(X_train, y_train, k=50):
    # Perform mRMR feature selection
    selected_features = mrmr_classif(X_train, y_train, K=k)

    return selected_features




def feature_tuning(X_train, X_test, y_train, feature_selection):
    # Perform feature selection based on the chosen method
    if feature_selection == 'univariate':
        selected_features = univariate_selection(X_train, y_train)
    elif feature_selection == 'rfe':
        selected_features = rfe_feature_selection(X_train, y_train)
    elif feature_selection == 'pca':
        X_train_transformed, pca = pca_feature_selection(X_train)
        X_test_transformed = pca.transform(X_test)  # Transform X_test using the fitted PCA
        return X_train_transformed, X_test_transformed
    elif feature_selection == 'boruta':
        selected_features = boruta_feature_selection(X_train, y_train)
    elif feature_selection == 'mrmr':
        selected_features = mrmr_feature_selection(X_train, y_train)
    else:
        raise ValueError("Invalid feature selection option")

    if len(selected_features) == 0:
        raise ValueError("No features selected")

    # Subset the selected features
    X_train_tuned = X_train[selected_features]
    X_test_tuned = X_test[selected_features]

    return X_train_tuned, X_test_tuned


def evaluate_model(model, X_test, y_test, cutoff):
    # Make predictions on the test set with probabilities
   y_pred = model.predict(X_test)
   f1 = f1_score(y_test, y_pred)
   auc_roc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

   return f1, auc_roc

def find_best_model(Features, Target, feature_selection_options, model_options, cutoffs, k=2):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    best_f1_score = 0.0
    best_auc_roc = 0.0
    best_model = None
    best_tune = None
    best_feature_selection = None
    best_cutoff = None

    # Store the results for each parameter search
    results = []

    for cutoff in cutoffs:
        print(f"Running models with cutoff = {cutoff}")

        for feature_selection in feature_selection_options:
            for model_type in model_options:
                # Initialize lists to store the results for each fold and parameter search
                f1_scores = []
                auc_roc_scores = []
                class_probs = []
                params_used = []

                for train_index, test_index in kf.split(Features, Target):
                    X_train, X_test = Features.iloc[train_index], Features.iloc[test_index]
                    y_train, y_test = Target.iloc[train_index], Target.iloc[test_index]

                    # Preprocess the data using k-fold specific function
                    X_train_resampled, X_test_scaled, y_train_resampled = preprocessing(X_train, X_test, y_train, y_test)
                    X_train_tuned, X_test_tuned = feature_tuning(X_train_resampled, X_test_scaled, y_train_resampled, feature_selection)

                    # Create and train the model
                    if model_type == 'catboost':
                        estimator = searcher.gen_classifier_binary('cat')
                        space, x = searcher.gen_parameter_space('cat','grid')
                        print(space)
                        search = searcher.gen_searcher(estimator, space, strategy='grid')
                    elif model_type == 'xgboost':
                        estimator = gen_classifier_binary('xgb')
                        space = gen_parameter_space('xgb', optimization='random')
                        search = gen_searcher(estimator, space, strategy='random')
                    elif model_type == 'randomforest':
                        estimator = gen_classifier_binary('rf')
                        space = gen_parameter_space('rf', optimization='random')
                        search = gen_searcher(estimator, space, strategy='random')
                    else:
                          raise ValueError("Invalid model type")
   
                    # Perform parameter search
                    search.fit(X_train_tuned, y_train_resampled,verbose=False)
                    
                    
                   
                    # Get the best estimator from the parameter search
                    best_model = search.best_estimator_

                    # Evaluate the model on the test set
                    f1, auc_roc = evaluate_model(best_model, X_test_tuned, y_test, cutoff)
                    f1_scores.append(f1)
                    auc_roc_scores.append(auc_roc)
                   

                    # Store the class probabilities for each instance of the parameter search
                    predicted_probabilities = best_model.predict_proba(X_test_tuned)
                    class_probs.append(predicted_probabilities)

                    # Store the parameters used in this instance of the parameter search
                    params_used.append(search.best_params_)

                # Save the results for the current parameter search
                result = {
                    'feature_selection': feature_selection,
                    'model_type': model_type,
                    'cutoff': cutoff,
                    'f1_scores': f1_scores,
                    'auc_roc_scores': auc_roc_scores,
                    'class_probs': class_probs,
                    'params_used': params_used  # Include the parameters used in this instance
                }
                results.append(result)
                # Getting training scores using learning curve
                train_sizes, train_scores, valid_scores = learning_curve(
                    best_model, X_train_tuned, y_train_resampled, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, scoring='roc_auc'
                )
                
                # Calculate mean and standard deviation for training set scores
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                
                # Calculate mean and standard deviation for validation set scores
                valid_mean = np.mean(valid_scores, axis=1)
                valid_std = np.std(valid_scores, axis=1)
                
                # Plotting
                plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
                plt.plot(train_sizes, valid_mean, color="#111111", label="Cross-validation score")
                
                # Coloring the space between standard deviations
                plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
                plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, color="#DDDDDD")
                
                plt.title(f"Learning Curve for {model_type}")
                plt.xlabel("Training Set Size")
                plt.ylabel("AUC ROC Score")
                plt.legend(loc="best")
                plt.show()


                # Check if the current model has a better F1 score than the previous best model
                if np.max(f1_scores) > best_f1_score and np.max(auc_roc_scores) > best_auc_roc:
                    best_f1_score = np.max(f1_scores)
                    best_auc_roc = np.max(auc_roc_scores)
                    best_model = model_type
                    best_tune = [X_test_tuned, X_train_tuned]
                    best_feature_selection = feature_selection
                    best_cutoff = cutoff

    print(f"Best Feature Selection: {best_feature_selection}, Best Model: {best_model}", flush=True)
    print("Best F1 Score: {:.2f}%".format(best_f1_score * 100), flush=True)
    print("Best AUC-ROC Score: {:.2f}%".format(best_auc_roc * 100), flush=True)
    print("Best Cutoff: {:.2f}".format(best_cutoff), flush=True)
    print("\n The best estimator across ALL searched params:\n",search.best_estimator_)
    print("\n The best score across ALL searched params:\n",search.best_score_)
    print("\n The best parameters across ALL searched params:\n",search.best_params_)
    print(search.scoring)
    print(flush=True)

    # Save the best model to a file (optional)
    best_model_filename = f"best_model_{best_feature_selection}_{best_model}.pkl"
    with open(best_model_filename, 'wb') as file:
        pickle.dump(best_model, file)

    return best_model, best_tune, best_feature_selection, results



if __name__ == '__main__':
    excel_data = pd.read_excel("") # removed name for sercuity reasons

    # Separate the target and features
    excel_data.drop(columns=[''], inplace=True) # removed columns name for sercuity reasons
    Target = excel_data['true_decline']
    Features = excel_data.drop(columns=['true_decline'])

    # Specify the feature selection and model options
    feature_selection_options = ['mrmr']
    model_options = ['catboost']

    # Specify the cutoffs you want to try (e.g., [0.3, 0.5, 0.7])
    cutoffs = [0.7]

    # Perform k-fold cross-validation and find the best model
    best_model, best_tune, best_feature_selection, results = find_best_model(
        Features, Target, feature_selection_options, model_options, cutoffs, k=5)
    
