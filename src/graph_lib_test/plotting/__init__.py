#scikit-learn related imports
from typing import Callable
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


def plot_roc_diagram(metrics, subfig_func=None):
    fig = go.Figure().update_layout(height=500, width=500)

    if 'model' not in metrics.columns:
        metrics = metrics.assign(model='Singleton')
    
    if callable(subfig_func):
        subfig_func_param_names = inspect.signature(subfig_func).parameters.keys()
    
    for model, model_metrics in metrics.groupby('model'):
        if 'target_threshold' in metrics.columns:
            for t in metrics['target_threshold'].sort_values().unique():
                subfig = px.line(model_metrics[model_metrics['target_threshold'] == t].reset_index(), x='F', y='POD', height=500, width=500, range_x=[
                                0, 1], range_y=[0, 1], hover_data=['index', 'PSS', 'target_threshold', 'regression_threshold', 'model'])
                if callable(subfig_func):
                    available_arguments = dict(model=model, t=t, fig=subfig)
                    arguments = {param: available_arguments[param] for param in subfig_func_param_names}
                    subfig = subfig_func(**arguments)
                fig.add_traces(
                    subfig.data
                )

        else:
            subfig = px.scatter(model_metrics.reset_index(), x='F', y='POD', height=500, width=500, range_x=[0,1], range_y=[0,1], hover_data=['index','PSS'])
            if callable(subfig_func):
                available_arguments = dict(model=model, subfig=subfig)
                arguments = {param: available_arguments[param] for param in subfig_func_param_names}
                subfig = subfig_func(**arguments)
            fig.add_traces(
                subfig.data
            )
    
    fig.update_xaxes(title='False Positive Rate').update_yaxes(title='True Positive Rate')
    
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
                                0, 1], range_y=[0, 1], hover_data=['index', 'CSI', 'target_threshold', 'regression_threshold', 'model'])
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
    
    fig = fmt_presentation_figure(fig
                                  .update_layout(
                                      title="Performance Diagram",
                                      height=500,
                                      width=580,
                                      showlegend=True)
                                  .update_xaxes(title='Success Ratio')
                                  .update_yaxes(title='Probability of Detection')
                                  .update_traces(
                                      selector=dict(type='contour'),
                                      contours_coloring="none",
                                      contours_showlabels=True,
                                      name='CSI')
                                  )
    
    return fig


def mk_taylor_diagram_info(df: pd.DataFrame, colors: Callable=None) -> pd.DataFrame:
    taylor_df = pd.DataFrame({model: dict(
        model=model,
        x=np.cos(np.arccos(np.corrcoef(df[model], df['target'])[1,0]))*df[model].std(),
        y=np.sin(np.arccos(np.corrcoef(df[model], df['target'])[1,0]))*df[model].std())
        for model in df.columns.difference({'site', 'time', 'leadtime', 'location', 'metar'})}).transpose()
    taylor_df = taylor_df.assign(length=lambda df: (df['x']**2+df['y']**2)**0.5)
    taylor_df = taylor_df.assign(rmse=lambda df: ((df['x'] - df.loc['target'].x)**2 + (df['y'] - df.loc['target'].y)**2)**0.5)

    if callable(colors):
        taylor_df['color'] = taylor_df['model'].apply(colors)

    return taylor_df


def plot_taylor_diagram(taylor_df: pd.DataFrame) -> go.Figure:
    taylor_fig = go.Figure(layout=dict(
        xaxis_range=[min(-0.1, taylor_df['x'].min()*1.1), taylor_df['x'].max()*1.1],
        yaxis_range=[min(-0.1, taylor_df['y'].min()*1.1), taylor_df['y'].max()*1.1])).update_layout(height=500, width=500)

    #taylor_fig = px.scatter(taylor_df, x='x', y='y',color='model', height=500, width=550, range_x=[-0.1, taylor_df['length'].max()*1.1], range_y=[-0.1,taylor_df['length'].max()*1.1], title='Taylor Diagram')

    for corr in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1]:
        taylor_fig = taylor_fig.add_shape(type="line",
            line_color="cyan",
            line_width=1,
            label=dict(text=f'{corr}  ', textposition="end"),
            x0=0, y0=0, x1=np.cos(np.arccos(corr))*taylor_df['length'].max()*1.05, y1=np.sin(np.arccos(corr))*taylor_df['length'].max()*1.05
        )

    for rmse in [0.5,1,1.5,2,2.25,2.5]:
        taylor_fig = taylor_fig.add_shape(type="circle",
            line_color="orange",
            line_width=1,
            label=dict(text=f'{rmse}', textposition="middle left"),
            x0=taylor_df['x']['target']-rmse, y0=taylor_df['y']['target']-rmse, x1=taylor_df['x']['target']+rmse, y1=taylor_df['y']['target']+rmse
        )

    taylor_fig = taylor_fig.add_shape(type="circle",
        line_color="black",
        line_width=1,
            x0=-taylor_df['length'].max()*1.05, y0=-taylor_df['length'].max()*1.05, x1=taylor_df['length'].max()*1.05, y1=taylor_df['length'].max()*1.05
    )

    names_in_legend = set()

    for name, row in taylor_df.iterrows():
        subfig = px.scatter(pd.DataFrame(row).transpose(), x='x', y='y', hover_data=['model', 'rmse'])
        name_in_legend = name.split("-")[0]
        subfig = subfig.update_traces(marker_color=row.get('color'), name=name_in_legend, showlegend=name_in_legend not in names_in_legend, opacity=0.5 if 'pruned' in name else (1 if 'de-' in name else 1))
        if name_in_legend not in names_in_legend: names_in_legend.add(name_in_legend)
        taylor_fig = taylor_fig.add_traces(subfig.data)
    
    taylor_fig = fmt_presentation_figure(taylor_fig.update_layout(title="Taylor Diagram").update_xaxes(title="Standard Deviation (observation)").update_yaxes(title="Standard Deviation").update_layout(height=500, width=580, showlegend=True))
    
    return taylor_fig