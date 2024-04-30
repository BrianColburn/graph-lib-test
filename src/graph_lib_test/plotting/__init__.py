#scikit-learn related imports
from typing import Callable, Collection, Optional, Sequence
from uuid import uuid4
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
    
    iterations = 0
    
    for model, model_metrics in metrics.groupby('model'):
        if 'target_threshold' in metrics.columns:
            for t in metrics['target_threshold'].sort_values().unique():
                subfig = px.line(model_metrics[model_metrics['target_threshold'] == t].reset_index(),
                                 x='F',
                                 y='POD',
                                 range_x=[0, 1],
                                 range_y=[0, 1],
                                 hover_data=['index',
                                             'PSS',
                                             'target_threshold',
                                             'regression_threshold',
                                             'model'])
                subfig.update_traces(
                    name=f'{model} (t={t})',
                    showlegend=True,
                    line_color=px.colors.DEFAULT_PLOTLY_COLORS[iterations % len(px.colors.DEFAULT_PLOTLY_COLORS)])

                if callable(subfig_func):
                    available_arguments = dict(model=model, t=t, fig=subfig)
                    arguments = {param: available_arguments[param] for param in subfig_func_param_names}
                    subfig = subfig_func(**arguments)
                fig.add_traces(
                    subfig.data
                )

                iterations += 1

        else:
            subfig = px.scatter(model_metrics.reset_index(), x='F', y='POD', height=500, width=500, range_x=[0,1], range_y=[0,1], hover_data=['index','PSS'])
            if callable(subfig_func):
                available_arguments = dict(model=model, subfig=subfig)
                arguments = {param: available_arguments[param] for param in subfig_func_param_names}
                subfig = subfig_func(**arguments)
            fig.add_traces(
                subfig.data
            )

            iterations += 1
    
    fig = fmt_presentation_figure(fig
                                  .update_layout(
                                      title="ROC Diagram",
                                      height=500,
                                      width=580,
                                      showlegend=True)
                                  .update_xaxes(
                                      title='False Positive Rate',
                                      range=[-0.01, 1.01])
                                  .update_yaxes(
                                      title='True Positive Rate',
                                      range=[0, 1.01])
                                  )
    
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
    
    iterations = 0
    
    for model, model_metrics in metrics.groupby('model'):
        if 'target_threshold' in metrics.columns:
            for t in metrics['target_threshold'].sort_values().unique():
                subfig = px.line(model_metrics[model_metrics['target_threshold'] == t].reset_index(),
                                 x='SR',
                                 y='POD',
                                 range_x=[0, 1],
                                 range_y=[0, 1],
                                 hover_data=['index',
                                             'CSI',
                                             'target_threshold',
                                             'regression_threshold',
                                             'model'])
                subfig.update_traces(
                    name=f'{model} (t={t})',
                    showlegend=True,
                    line_color=px.colors.DEFAULT_PLOTLY_COLORS[iterations % len(px.colors.DEFAULT_PLOTLY_COLORS)])

                if callable(subfig_func):
                    available_arguments = dict(model=model, t=t, fig=subfig)
                    arguments = {param: available_arguments[param] for param in subfig_func_param_names}
                    subfig = subfig_func(**arguments)
                fig.add_traces(
                    subfig.data
                )

                iterations += 1

        else:
            subfig = px.scatter(model_metrics.reset_index(), x='SR', y='POD', height=500, width=500, range_x=[0,1], range_y=[0,1], hover_data=['index','CSI'])
            if callable(subfig_func):
                available_arguments = dict(model=model, subfig=subfig)
                arguments = {param: available_arguments[param] for param in subfig_func_param_names}
                subfig = subfig_func(**arguments)
            fig.add_traces(
                subfig.data
            )

            iterations += 1
    
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
                                      hoverinfo='skip',
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
    if not (set(taylor_df.columns) >= {'model', 'x', 'y', 'length', 'rmse'}):
        taylor_df = mk_taylor_diagram_info(taylor_df)
    min_bound = taylor_df[['x','y']].min().min()
    max_bound = taylor_df[['x','y']].max().max()
    taylor_fig = go.Figure(layout=dict(
        xaxis_range=[min(-0.01, -np.log10(max_bound), min_bound*1.1), max_bound*1.1],
        yaxis_range=[min(-0.01, -np.log10(max_bound), min_bound*1.1), max_bound*1.1])).update_layout(height=500, width=500)

    #taylor_fig = px.scatter(taylor_df, x='x', y='y',color='model', height=500, width=550, range_x=[-0.1, taylor_df['length'].max()*1.1], range_y=[-0.1,taylor_df['length'].max()*1.1], title='Taylor Diagram')

    for corr in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99,1]:
        taylor_fig = taylor_fig.add_shape(type="line",
            line_color="cyan",
            line_width=1,
            label=dict(text=f'{corr}  ', textposition="end"),
            x0=0, y0=0, x1=np.cos(np.arccos(corr))*taylor_df['length'].max()*1.05, y1=np.sin(np.arccos(corr))*taylor_df['length'].max()*1.05
        )

    for rmse in np.linspace(taylor_df['rmse'].min(), taylor_df['rmse'].max(), num=4):
        taylor_fig = taylor_fig.add_shape(type="circle",
            line_color="orange",
            line_width=1,
            label=dict(text=f'{rmse:.2f}', textposition="middle left"),
            x0=taylor_df['x']['target']-rmse,
            y0=taylor_df['y']['target']-rmse,
            x1=taylor_df['x']['target']+rmse,
            y1=taylor_df['y']['target']+rmse
        )

    taylor_fig = taylor_fig.add_shape(type="circle",
        line_color="black",
        line_width=1,
            x0=-taylor_df['length'].max()*1.05,
            y0=-taylor_df['length'].max()*1.05,
            x1=taylor_df['length'].max()*1.05,
            y1=taylor_df['length'].max()*1.05
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


def plot_attributes_diagram(
        data_frame: Optional[pd.DataFrame]=None,
        y_true=None,
        y_prob: Optional[Sequence[str | float]]=None,
        n_bins=10,
) -> go.Figure:
    if y_true == None and isinstance(data_frame, pd.DataFrame) and 'target' in data_frame:
        y_true = 'target'
    
    if isinstance(y_true, str) and isinstance(data_frame, pd.DataFrame) and y_true in data_frame:
        y_true = data_frame[y_true]

    if y_prob == None and isinstance(data_frame, pd.DataFrame):
        y_prob = [col for col in data_frame.columns[data_frame.dtypes == float]
                           if col != 'target']
    
    if len(y_prob) > 0:
        sample_element = y_prob[0]
        if isinstance(sample_element, str) and isinstance(data_frame, pd.DataFrame) and set(y_prob) <= set(data_frame.columns):
            y_prob: pd.DataFrame = data_frame[y_prob]
    
    climatology = y_true.mean()

    is_binary_classification = set(y_true.unique()) == {0,1}

    observed_label = None
    forecast_label = None
    axes_range = None

    if is_binary_classification:
        observed_label, forecast_label = 'Observed Frequency', 'Forecast Probability'
        axes_range = [0,1]
        dfs = []

        for model_name in y_prob.columns:
            df = pd.DataFrame(
                    np.array(sklearn.calibration.calibration_curve(y_true, y_prob[model_name], n_bins=n_bins)).transpose(),
                    columns=['Observed Frequency', 'Forecast Probability'])
            df['model'] = model_name
            dfs.append(df)

        df = pd.concat(dfs)
    else:
        observed_label, forecast_label = 'Observed', 'Forecast'
        axes_range = [np.min(y_true), np.max(y_true)]
        bins = np.histogram_bin_edges(y_true, bins=n_bins)

        y_true_label = 'target'
        label_iterations = 0
        while y_true_label in y_prob.columns:
            if label_iterations > 10:
                raise RuntimeError('Unable to generate unique name for target values')
            y_true_label = uuid4().hex()
            label_iterations += 1

        means = y_prob.assign(**{y_true_label: y_true})
        means = means.groupby(np.digitize(y_true, bins)).mean()
        df = means.melt(id_vars=y_true_label, var_name='model').rename(columns={
            y_true_label: observed_label,
            'value': forecast_label,
        })
    

    rel_figure = px.line(df,
                         x=forecast_label,
                         y=observed_label,
                         color='model',
                         width=600,
                         height=500)

    rel_figure.add_shape(
        type='line', line=dict(dash='dash'),
        x0=axes_range[0], x1=axes_range[1], y0=axes_range[0], y1=axes_range[1]
    )
    rel_figure.add_vline(x=climatology, line=dict(dash='dash'))
    rel_figure.add_hline(y=climatology, line=dict(dash='dash'))
    rel_figure.add_shape(
        type='line',
        line=dict(width=4, color='aqua'),
        x0=axes_range[0],
        x1=axes_range[1],
        y0=(climatology-axes_range[0])/2+axes_range[0],
        y1=(axes_range[1]+climatology)/2,
    )
    #rel_figure.add_shape(
    #    type='line', line=dict(width=4, color='aqua'),
    #    x0=climatology, x1=1, y0=climatology, y1=(1-climatology)/2,
    #    layer='below'
    #)
    rel_figure.add_shape(
        type='path',
        path=f'''M{axes_range[0]} {(climatology-axes_range[0])/2+axes_range[0]}
                 L{axes_range[1]} {(axes_range[1]+climatology)/2}
                 L{axes_range[1]} {axes_range[1]}
                 L{axes_range[0]} {axes_range[0]}
                 Z
                 M{climatology} {axes_range[0]}
                 L{climatology} {axes_range[1]}
                 L{axes_range[1]} {axes_range[1]}
                 L{axes_range[0]} {axes_range[0]}
                 Z''',
        layer='below',
        fillcolor='lightblue',
        line=dict(width=0)
    )
    rel_figure.update_xaxes(range=axes_range)
    rel_figure.update_yaxes(range=axes_range)
    rel_figure.update_layout(title=f'Reliability Diagram')

    rel_figure = fmt_presentation_figure(rel_figure)

    return rel_figure
