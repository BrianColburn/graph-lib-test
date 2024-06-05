import functools
import inspect

import pandas as pd


def filter_kwargs(func):
    """Create a function that silently ignores any unexpected keyword arguments."""
    func_params = inspect.signature(func).parameters
    
    invalid_params = {name: param for name, param in func_params.items()
                         if param.kind == inspect.Parameter.POSITIONAL_ONLY | inspect.Parameter.VAR_KEYWORD}
    
    if len(invalid_params) > 0:
        raise ValueError(f'The `filter_kwargs` decorator can only be applied to a function with a fixed number of keyword parameters. Invalid parameters: {", ".join(invalid_params.keys())}')

    @functools.wraps(func)
    def filtering_func(*args, **available_kwargs):
        available_kwargs = available_kwargs or dict()
        common_kwargs = func_params.keys() & available_kwargs.keys()
        kwargs = {k: available_kwargs[k] for k in common_kwargs}
        return func(*args, **kwargs)
    
    return filtering_func


def input_type_analyzer(func):
    """Create a function that prints some information about its inputs."""
    func_sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        available_args = args or tuple()
        available_kwargs = kwargs or dict()

        print('Calling wrapped function with signature', func_sig)
        print(f'Possible parameters for {func.__name__}:')
        for name, param in func_sig.parameters.items():
            print(f'\t{name}: {param}')
        print(f'{func.__name__} returns {func_sig.return_annotation}')
        print('arg types:', [type(arg) for arg in args])
        print('kwarg types:', {kwarg: type(kwarg) for kwarg in kwargs})

        return func(*args, **kwargs)
    
    return wrapper


def intake_target_and_pred(func):
    """Allow the wrapped function to easily process a few different input formats."""

    func_sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*available_args, **available_kwargs):
        args = available_args
        kwargs = available_kwargs

        return func(*args, **kwargs)

    return wrapper


# HACK: This is from the "cool_turtles_doodles" notebook I sent Miranda last October.
#       Something like this is probably a good idea, so I'm putting it here for now.
def sanity_check_df_for_metrics(regression_df):
    # Fail early and don't leave the user guessing about why we failed.
    # Decided to make this a function since this is done more than once,
    # and I want to do the same check everywhere.
    if 'target' not in regression_df or 'prediction' not in regression_df:
        exception_msg = 'The regression dataframe must have "target" and "prediction" columns'
        exception_msg += f'Current columns: {regression_df.columns}\n'
    raise ValueError(exception_msg)


def boxify(ensemble_timeseries):
    if hasattr(ensemble_timeseries, 'columns'):
        ensemble_timeseries = ensemble_timeseries[ensemble_timeseries.columns[ensemble_timeseries.dtypes != object]]
    boxplot_data = pd.concat([
        ensemble_timeseries.quantile(0.25).rename('Q1'),
        ensemble_timeseries.median().rename('median'),
        ensemble_timeseries.quantile(0.75).rename('Q3')
        ], axis=1)#.rename_axis(['year', 'month'])
    boxplot_data['IQR'] = boxplot_data['Q3']-boxplot_data['Q1']
    boxplot_data['lower_fence'] = boxplot_data['Q1']-1.5*boxplot_data['IQR']
    boxplot_data['upper_fence'] = boxplot_data['Q3']+1.5*boxplot_data['IQR']
    boxplot_data['maximum'] = ensemble_timeseries.max()
    boxplot_data['minimum'] = ensemble_timeseries.min()
    #boxplot_data['timestamp'] = [f'{year}-{month:02d}-01' for year, month in boxplot_data.index]
    #boxplot_data = boxplot_data.set_index('timestamp')
    return boxplot_data
