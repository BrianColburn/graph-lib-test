
def calc_threshold_metrics(regression_df, target_name='target', regression_names=['regression'], target_thresholds=None, regression_thresholds=None):
    if target_thresholds is None:
        target_thresholds = [1]

    if regression_thresholds is None:
        regression_thresholds = np.linspace(0, 10, num=51)

    dfs = []

    for regression_name in regression_names:
        df = pd.DataFrame([{'target_threshold': target_threshold, 'regression_threshold': regression_threshold, **model_utils.compute_metrics(
                mk_cmat(
                     pd.concat([(regression_df[target_name] <= target_threshold).rename('target'), regression_df[regression_name].rename('regression')], axis=1),
                     fog_def=f'regression <= {regression_threshold}',
                     target_pos=1,
                     target_neg=0))}
               for target_threshold in target_thresholds
               for regression_threshold in regression_thresholds])
        df['model'] = regression_name
        dfs.append(df)

    return pd.concat(dfs)
