import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, \
                            accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

## CORRELLATIONS
def get_top_absolute_corrs(df):
    corr = df.corr()
    idxs = corr.unstack().index
    nr_idx = sorted(list(set(['*'.join(sorted(list(i))) for i in idxs])))
    keep_idx = []
    for i in nr_idx:
        i = i.split('*')
        if i[0] != i[1]:
            keep_idx.append(tuple(i))
    corrs = corr.unstack().loc[keep_idx]
    corrs = pd.concat([corrs, corrs.abs()],axis=1).rename(columns={0:'corr',1:'abs'}).sort_values(by='abs',ascending=False)
    return corrs
    
def get_top_absolute_corrs_with(df, col):
    corr = df.corr()
    idxs = corr.unstack().index
    nr_idx = sorted(list(set(['*'.join(sorted(list(i))) for i in idxs])))
    keep_idx = []
    for i in nr_idx:
        i = i.split('*')
        if i[0] != i[1]:
            keep_idx.append(tuple(i))
    corrs = corr.unstack().loc[keep_idx]
    corrs = pd.concat([corrs, corrs.abs()],axis=1).rename(columns={0:'corr',1:'abs'}).sort_values(by='abs',ascending=False).reset_index()
    return corrs[corrs['level_0'].str.contains(col) | corrs['level_1'].str.contains(col)][['level_0','level_1','corr']]
    

## SCORING MODELS
def score_regression_model(model, X_train, y_train, X_test, y_test, name='', verbose=1):
    if name == '':
        name = model.__class__.__name__
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    r2_train = r2_score(y_train,y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    
    if verbose > 0:
        print('{} train r2 score: {:.2f}'.format(name, r2_train))
        print('{} test r2 score: {:.2f}'.format(name, r2_test))
        print('---------------')
        print('{} train rmse: {:.2f}'.format(name, rmse_train))
        print('{} test rmse: {:.2f}'.format(name, rmse_test))
        print('---------------')
        print('{} train mae: {:.2f}'.format(name, mae_train))
        print('{} test mae: {:.2f}'.format(name, mae_test))
    return {'name':name, 'r2_train':r2_train,'r2_test':r2_test,'rmse_train': rmse_train,
            'rmse_test': rmse_test,'mae_train': mae_train,'mae_test': mae_test}



def score_classification_model(model, X_train, y_train, X_test, y_test, name='', verbose=1):
    if name == '':
        name = model.__class__.__name__
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    y_train_proba = model.predict_proba(X_train)[:, 1]  # Probability of positive class
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_train = precision_score(y_train, y_train_pred)
    precision_test = precision_score(y_test, y_test_pred)
    recall_train = recall_score(y_train, y_train_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f1_test = f1_score(y_test, y_test_pred)
    auc_train = roc_auc_score(y_train, y_train_proba)
    auc_test = roc_auc_score(y_test, y_test_proba)
    confusion_train = confusion_matrix(y_train, y_train_pred)
    confusion_test = confusion_matrix(y_test, y_test_pred)
    
    if verbose > 0:
        print('{} train accuracy: {:.2f}'.format(name, accuracy_train))
        print('{} test accuracy: {:.2f}'.format(name, accuracy_test))
        print('---------------')
        print('{} train precision: {:.2f}'.format(name, precision_train))
        print('{} test precision: {:.2f}'.format(name, precision_test))
        print('---------------')
        print('{} train recall: {:.2f}'.format(name, recall_train))
        print('{} test recall: {:.2f}'.format(name, recall_test))
        print('---------------')
        print('{} train F1 score: {:.2f}'.format(name, f1_train))
        print('{} test F1 score: {:.2f}'.format(name, f1_test))
        print('---------------')
        print('{} train AUC: {:.2f}'.format(name, auc_train))
        print('{} test AUC: {:.2f}'.format(name, auc_test))
        print('---------------')
        print('{} train confusion matrix:\n{}'.format(name, confusion_train))
        print('{} test confusion matrix:\n{}'.format(name, confusion_test))
        
    return {'name': name,
            'accuracy_train': accuracy_train, 'accuracy_test': accuracy_test,
            'precision_train': precision_train, 'precision_test': precision_test,
            'recall_train': recall_train, 'recall_test': recall_test,
            'f1_train': f1_train, 'f1_test': f1_test,
            'auc_train': auc_train, 'auc_test': auc_test,
            'confusion_train': confusion_train, 'confusion_test': confusion_test}
