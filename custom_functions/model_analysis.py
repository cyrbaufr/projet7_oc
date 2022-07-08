from sklearn.metrics import classification_report
import  matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier



def plot_confusion_matrix(y_test, y_pred):
    """
    https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
    """
    # création de la matrice de confusion
    cf_matrix = confusion_matrix(y_test, y_pred)
    # Matrice de confusion en pourcentage
    fig = sns.heatmap(cf_matrix, annot=True, 
                fmt=".0f", cmap='Blues')
    plt.ylabel("True Label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    plt.show(fig)
    print(classification_report(y_test, y_pred))
    # Calcul de la ROC AUC
    rocscore = roc_auc_score(y_test, y_pred)
    print(f'ROC AUC Score: {rocscore:.02f}')


def plot_roc_curve(fpr, tpr, label=None):
    """
    The ROC curve, modified from 
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91
    """
    plt.figure(figsize=(8,8))
    plt.title('ROC Curve')
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0,1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.legend(loc='best')


def grid_search_wrapper(refit_score='precision_score'):
    """
    fits a GridSearchCV classifier using refit_score for optimization
    prints classifier performance metrics
    """
    # stratification
    skf = StratifiedKFold(n_splits=10)
    grid_search = GridSearchCV(clf, param_grid,
                               scoring=scorers,
                               refit=refit_score,
                               cv=skf,
                               return_train_score=True,
                               n_jobs=-1)
    # fit optimized model on train set
    grid_search.fit(X_train.values, y_train.values)
    # make the predictions on test set
    y_pred = grid_search.predict(X_test.values)
    # print best parameters
    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)
    # print confusion matrix on the test data
    print('Confusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       columns=['pred_neg', 'pred_pos'],
                       index=['neg', 'pos']))
    return grid_search


def calculate_confusion_mat(sampling='ros',
                            imputer_type='median',
                            model = DecisionTreeClassifier(class_weight='balanced')):
    if sampling=='ros':
        sampler_type = RandomOverSampler(random_state=42)
        my_X_train = X_ros_train
        my_X_test = X_ros_test
        my_y_train = y_ros_train
        my_y_test = y_ros_test
        
    elif sampling=='rus':
        sampler_type = RandomUnderSampler(random_state=42)
        my_X_train = X_rus_train
        my_X_test = X_rus_test
        my_y_train = y_rus_train
        my_y_test = y_rus_test

    steps = [('impute', SimpleImputer(strategy=imputer_type)),
             ('ros', sampler_type),
             ('model', model)]
    pipeline = Pipeline(steps=steps)
    pipeline.fit(my_X_train, my_y_train)
    y_pred = pipeline.predict(my_X_test)
    print(classification_report(my_y_test, y_pred))


def calculate_scores(X, y, pipeline):
    """
    Calculate the following scores: accuracy, precision, recall, F1, ROC AUC
    
    Intput
    ------
    X: matrix with independant features
    y: target
    pipeline: use imblearn pipeline as it includes resampling step
    pipeline will have following steps: 'imputer', 'scaler', 'sampler' and 'model'
    
    Output
    ------
    Mean scores
    
    """
    print(pipeline)
    # evaluate pipeline with cross validation
    cv = RepeatedStratifiedKFold(n_splits=5,
                                 n_repeats=3,
                                 random_state=1)
    # dictionnary of scores to calculate
    scoring = {'accuracy_score': 'accuracy',
               'precision_score': 'precision',
               'recall_score': 'recall',
               'F1_score':'f1',
               'AUROC':'roc_auc'
              }
    # cross validation
    scores = cross_validate(pipeline,
                            X, y,
                            scoring=scoring,  # scoring='roc_auc',
                            cv=cv)
    # print results
    print(scores.keys())
    for key in scores.keys():
        print('Mean {}: {:.02f}'.format(key, np.mean(scores[key])))