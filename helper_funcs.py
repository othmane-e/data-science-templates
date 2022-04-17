import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


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
def score_model(model, X_train, y_train, X_test, y_test, name='', verbose=1):
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