from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from scipy import stats

def mse_rmse_mae_r2(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

def mean_ci(y_values, confidence=0.95):
    n = len(y_values)
    mean = np.mean(np.array(y_values))
    sem = stats.sem(y_values)
    h = sem * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h