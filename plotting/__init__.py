#scikit-learn related imports
import sklearn
import sklearn.metrics
import sklearn.calibration

from pathlib import Path

import plotly
import plotly.express as px
import plotly.graph_objs as go
import plotly.graph_objects

import numpy as np
import pandas as pd

import ipywidgets.widgets as widgets

import inspect

#import optuna
#import sqlite3
#import IPython.display

import re

def fmt_presentation_figure(fig):
    """Format presentation figure according to CBI standards."""
    return fig.update_layout(margin=dict(t=50, r=0, b=0, l=0), title_font_size=32).update_yaxes(title_font_size=24, tickfont_size=16).update_xaxes(title_font_size=24, tickfont_size=16)#.to_image(format='png', scale=2)


def mk_heatmap(regression, target, bins, axis_range=None, histfunc='freq'):
    if histfunc == 'freq':
        histfunc = lambda x: x/x.sum()
    elif histfunc == 'logfreq':
        histfunc=lambda x:np.log10(1e-3+x/x.sum())
    elif not callable(histfunc):
        raise ValueError('histfunc must be "freq", "logfreq" or callable!')

    tmp = np.histogram2d(regression, target, bins=bins)
    heatmap_data = (histfunc(tmp[0].transpose()),
                    tmp[1],
                    tmp[2])
    fig = go.Figure(data=go.Heatmap(x=heatmap_data[1], y=heatmap_data[2], z=heatmap_data[0], type='heatmap',
                                     text=[[f'x: [{xl:.2f}, {xu:.2f})<br>y: [{yl:.2f}, {yu:.2f})'
                                            for xl, xu in zip(heatmap_data[1][:-1], heatmap_data[1][1:])]
                                           for yl, yu in zip(heatmap_data[2][:-1], heatmap_data[2][1:])],
                                    coloraxis='coloraxis'
                                    ),
                     layout=dict(height=400, width=500, xaxis=dict(range=[0,11]), yaxis=dict(range=[0,11])))
    
    fig.update_layout(
        coloraxis_colorscale=[[0,"rgb(255,255,255)"], [1,"rgb(0,0,255)"]],
    )

    return fig


def plot_study_roc(studies, case=None, inputs=None, year=None, locations=None, calibration_inputs=None, options=dict()):
    opts = dict(width=500, height=500)
    opts.update(options)
    if type(studies) == list:
        multi_model_results = studies
    else:
        multi_model_results = compute_multi_model_results(studies, case, inputs, year, locations, calibration_inputs=calibration_inputs)
    curves = []
    for model_results in multi_model_results:
        try:
            if calibration_inputs is None:
                curves.append(pd.DataFrame(np.array(
                sklearn.metrics.roc_curve(1-model_results['target'],
                                          model_results['confidence_fog'])).transpose(),
                                   columns=['x','y','t']))
            else:
                curves.append(pd.DataFrame(np.array(
                sklearn.metrics.roc_curve(1-model_results['target'],
                                          model_results['confidence_fog_cal'])).transpose(),
                                   columns=['x','y','t']))
        except Exception:
            pass

    fig = plotly.graph_objs.Figure(layout=opts)

    for curve in curves:
        #threshold_indices = curve.iloc[curve['t'][1:].round(1).drop_duplicates().index]
    
        #fig.add_scatter(x=threshold_indices.values[:,0],
        #                y=threshold_indices.values[:,1],
        #                mode='markers', showlegend=False, marker=dict(color='black', opacity=0.2))
        fig.add_scatter(x=curve['x'], y=curve['y'], line=dict(width=1, color='rgba(0,0,0,0.2)'), mode='lines', showlegend=False)

    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )
    fig.update_xaxes(range=[0,1], title_text='False Positive Rate')
    fig.update_yaxes(range=[0,1], title_text='True Positive Rate')
    fig.update_layout(title=case)
    
    return fig


def plot_performance_diagram(metrics, subfig_func=None):
    fig = go.Figure(data=
            go.Contour(
                x=np.linspace(0,1),
                y=np.linspace(0,1),
                z=[[np.nan_to_num(1/(1/pod + 1/sr - 1), 0)
                    for pod in np.linspace(0, 1)]
                for ix, sr in enumerate(np.linspace(0,1))],
                contours_coloring='lines',
            )
         ).update_layout(height=500, width=500)

    if 'model' not in metrics.columns:
        metrics = metrics.assign(model='Singleton')
    
    if callable(subfig_func):
        subfig_func_param_names = inspect.signature(subfig_func).parameters.keys()
    
    for model, model_metrics in metrics.groupby('model'):
        if 'target_threshold' in metrics.columns:
            for t in metrics['target_threshold'].sort_values().unique():
                subfig = px.line(model_metrics[model_metrics['target_threshold'] == t].reset_index(), x='SR', y='POD', height=500, width=500, range_x=[
                                0, 1], range_y=[0, 1], hover_data=['index', 'CSI', 'target_threshold', 'regression_threshold'])
                if callable(subfig_func):
                    available_arguments = dict(model=model, t=t, fig=subfig)
                    arguments = {param: available_arguments[param] for param in subfig_func_param_names}
                    subfig = subfig_func(**arguments)
                fig.add_traces(
                    subfig.data
                )

        else:
            subfig = px.scatter(model_metrics.reset_index(), x='SR', y='POD', height=500, width=500, range_x=[0,1], range_y=[0,1], hover_data=['index','CSI'])
            if callable(subfig_func):
                available_arguments = dict(model=model, subfig=subfig)
                arguments = {param: available_arguments[param] for param in subfig_func_param_names}
                subfig = subfig_func(**arguments)
            fig.add_traces(
                subfig.data
            )
    
    fig.update_xaxes(title='Success Ratio').update_yaxes(title='Probability of Detection')

    return fig.update_traces(selector=dict(type='contour'), contours_coloring="none", contours_showlabels=True, name='CSI')
