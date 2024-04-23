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


def calc_threshold_metrics(regression_df, target_name='target', regression_names=['regression'], target_thresholds=None, regression_thresholds=None):
    if target_thresholds is None:
        target_thresholds = [1]

    if regression_thresholds is None:
        regression_thresholds = np.linspace(0, 10, num=51)

    dfs = []

    for regression_name in regression_names:
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
