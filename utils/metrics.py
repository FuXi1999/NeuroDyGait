import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def performance_metrics(Y, pre_Y, method):
    assert Y.shape == pre_Y.shape

    if method == 'r2':
        score = list(r2_score(Y, pre_Y, multioutput='raw_values'))
    elif method == 'rmse':
        score = list(mean_squared_error(Y, pre_Y, multioutput='raw_values', squared=False))
    elif method == 'pearsonr':
        score = [pearsonr(Y[:, idx],pre_Y[:,idx])[0] for idx in range(Y.shape[1])]
    else:
        print('Error! {} is undefined.'.format(method))

    return score


def to_DataFrame(res, method, subjects_list, jointsColumns, time_step):
    res = np.asarray(res)
    avg_res = np.mean(res, axis=0, keepdims=True)
    std_res = np.var(res, axis=0, keepdims=True)

    res_fat = np.vstack([res, avg_res, std_res])

    index = [item[:-4] for item in subjects_list]+['Avg.', 'Std.']
    columns = [method+'-TS'+str(time_step)+'-'+item[1:] for item in jointsColumns]
    df = pd.DataFrame(res_fat, index=index, columns=columns)

    return df