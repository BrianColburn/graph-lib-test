from typing import Literal, Optional
import numpy as np
import pandas as pd

def mk_case_indices(results_df, pos_def='confidence >= 0.5', target_pos=0, target_neg=1):
    contingency_queries = {
        'Hits': f'target == {target_pos} and ({pos_def})',
        'Misses': f'target == {target_pos} and not ({pos_def})',
        'False Alarms': f'target == {target_neg} and ({pos_def})',
        'Correct Rejects': f'target == {target_neg} and not ({pos_def})'
    }
    
    indices = {label: results_df.query(query).index for label, query in contingency_queries.items()}
    
    return indices


def mk_cmat(results_df: pd.DataFrame, pos_def: str='confidence >= 0.5', **kwargs) -> np.ndarray:
    return np.array([len(vals) for vals in mk_case_indices(results_df, pos_def, **kwargs).values()]).reshape(-1,2)


def compute_metrics(cmat: np.ndarray, *rest) -> dict[str, np.double]:
    tp, fn, fp, tn = (cmat, *rest) if len(rest) == 3 else cmat.ravel()
    p = tp + fn
    n = fp + tn
    pp = tp + fp
    pn = fn + tn
    
    idk = (tp*tn - fp*fn)
    expected_correct = ((tp+fn)*(tp+fp) + (tn+fn)*(tn+fp))/(tp+fn+fp+tn)
    detection_failure_ratio = fn/(fn+tn)
    frequency_of_hits = tp/(tp + fp)
    
    results = {
        'TP': tp,
        'FN': fn,
        'FP': fp,
        'TN': tn,
        'POD': tp / p,
        'F': fp / n,
        'FAR': fp / pp,
        'CSI': tp / (p + fp),
        'Dice': 2*tp / (2*tp + fn + fp),
        'PSS': idk / (n*p),
        'HSS': (tp + tn - expected_correct) / (tp+fn+fp+tn - expected_correct),
        'ORSS': idk / (tp*tn + fp*fn),
        'CSS': frequency_of_hits - detection_failure_ratio
    }

    results['SR'] = 1-results['FAR']

    if np.isnan(results['PSS']):
        results['PSS'] = -1

    return results


def calc_threshold_metrics(
        regression_df: pd.DataFrame,
        target_name: str='target',
        prediction_cols: Optional[list[str]]=None,
        target_thresholds: Optional[np.ndarray | Literal['unique']]=None,
        regression_thresholds: Optional[np.ndarray]=None):

    if prediction_cols == None:
        prediction_cols = [col for col in regression_df.columns[[t in [float, np.float32] for t in regression_df.dtypes]]
                           if col != 'target']

    if target_thresholds is None:
        target_thresholds = np.linspace(
            start=regression_df[target_name].min(),
            stop=regression_df[target_name].max(),
            num=51,
        )
    elif target_thresholds == 'unique':
        target_thresholds = np.sort(regression_df[target_name].unique())

    if regression_thresholds is None:
        regression_thresholds = target_thresholds

    dfs = []

    for regression_name in prediction_cols:
        df = pd.DataFrame([{
            'target_threshold': target_threshold,
            'regression_threshold': regression_threshold,
            **compute_metrics(
                mk_cmat(
                    pd.concat([
                         (regression_df[target_name] <=
                          target_threshold).rename('target'),
                        regression_df[regression_name].rename('regression')], axis=1),
                    pos_def=f'regression <= {regression_threshold}',
                    target_pos=1,
                    target_neg=0))}
               for target_threshold in target_thresholds
               for regression_threshold in regression_thresholds])
        df['model'] = regression_name
        dfs.append(df)

    return pd.concat(dfs)



def calc_regression_metrics(regression_df):
    metrics = dict()
    residuals = regression_df['prediction'] - regression_df['target']
    metrics['me'] = residuals.mean()
    metrics['mae'] = residuals.abs().mean()
    metrics['mse'] = (residuals**2).mean()
    metrics['rmse'] = metrics['mse']**0.5
    return metrics


def central_freq(regression_df: pd.DataFrame, prediction_cols: Optional[list[str]]=None, thresholds=None):
    if prediction_cols == None:
        prediction_cols = [col for col in regression_df.columns[[t in [float, np.float32] for t in regression_df.dtypes]]
                           if col != 'target']
    thresholds = thresholds or [1, 2, 3, 4, np.inf]

    model_central_frequencies = []

    for model_name in prediction_cols:
        cfs = []
        df = regression_df[['target', model_name]].rename(columns={model_name: 'prediction'})
        differences = (df['prediction'] - df['target']).abs()

        for threshold in thresholds:
            mask = differences <= threshold

            cfs.append({
                'threshold': threshold,
                'mae': calc_regression_metrics(df[mask])['mae'],
                'count': mask.sum(),
                'percent': mask.mean()*100,
            })
        
        df = pd.DataFrame(cfs)
        df['model'] = model_name
        model_central_frequencies.append(df)
    
    model_central_frequencies = pd.concat(model_central_frequencies).set_index(['model', 'threshold'])

    return model_central_frequencies


def calc_ignorance_score(
        data_frame: pd.DataFrame,
        target_name: str='target',
        prediction_cols: Optional[list[str]]=None,
        log_base: float=np.e,
) -> pd.Series:

    if prediction_cols == None:
        prediction_cols = [col for col in data_frame.columns[[t in [float, np.float32] for t in data_frame.dtypes]]
                           if col != 'target']
    
    ignorance_score = pd.DataFrame({
        model_name: -np.log(data_frame[target_name]*data_frame[model_name]+(1-data_frame[target_name])*(1-data_frame[model_name]))/np.log(log_base)
        for model_name in prediction_cols
    }).mean().rename_axis('model').rename('score')

    return ignorance_score
