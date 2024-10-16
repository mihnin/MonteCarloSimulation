import plotly.graph_objects as go
import numpy as np
import pandas as pd

def plot_simulation_results(simulated_data, ci_lower, ci_upper, target_column, plot_type='histogram'):
    if plot_type == 'histogram':
        return plot_histogram(simulated_data, ci_lower, ci_upper, target_column)
    elif plot_type == 'line':
        return plot_line(simulated_data, ci_lower, ci_upper, target_column)
    elif plot_type == 'box':
        return plot_box(simulated_data, ci_lower, ci_upper, target_column)
    else:
        raise ValueError("Invalid plot type. Choose 'histogram', 'line', or 'box'.")

def plot_histogram(simulated_data, ci_lower, ci_upper, target_column):
    fig = go.Figure()

    # Histogram of simulated data
    fig.add_trace(go.Histogram(
        x=simulated_data,
        name="Simulation Results",
        nbinsx=50,
        hoverinfo='y+bin',
        hoverlabel=dict(namelength=-1)
    ))

    # Confidence interval
    fig.add_vline(x=ci_lower, line_dash="dash", line_color="red", annotation_text=f"Lower CI: {ci_lower:.2f}")
    fig.add_vline(x=ci_upper, line_dash="dash", line_color="red", annotation_text=f"Upper CI: {ci_upper:.2f}")

    # Mean line
    mean = np.mean(simulated_data)
    fig.add_vline(x=mean, line_color="green", annotation_text=f"Mean: {mean:.2f}")

    fig.update_layout(
        title=f"Monte Carlo Simulation Results for {target_column} (Histogram)",
        xaxis_title=target_column,
        yaxis_title="Frequency",
        showlegend=False,
        hovermode='closest'
    )

    # Add range slider and selector
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=3, label="3y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig

def plot_line(simulated_data, ci_lower, ci_upper, target_column):
    fig = go.Figure()

    # Line plot of simulated data
    x = list(range(len(simulated_data)))
    fig.add_trace(go.Scatter(
        x=x,
        y=simulated_data,
        mode='lines',
        name="Simulation Results",
        hoverinfo='x+y',
        hoverlabel=dict(namelength=-1)
    ))

    # Confidence interval
    fig.add_hline(y=ci_lower, line_dash="dash", line_color="red", annotation_text=f"Lower CI: {ci_lower:.2f}")
    fig.add_hline(y=ci_upper, line_dash="dash", line_color="red", annotation_text=f"Upper CI: {ci_upper:.2f}")

    # Mean line
    mean = np.mean(simulated_data)
    fig.add_hline(y=mean, line_color="green", annotation_text=f"Mean: {mean:.2f}")

    fig.update_layout(
        title=f"Monte Carlo Simulation Results for {target_column} (Line Plot)",
        xaxis_title="Simulation Run",
        yaxis_title=target_column,
        showlegend=True,
        hovermode='closest'
    )

    # Add range slider and selector
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=100, label="100", step="all", stepmode="backward"),
                dict(count=500, label="500", step="all", stepmode="backward"),
                dict(count=1000, label="1000", step="all", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig

def plot_box(simulated_data, ci_lower, ci_upper, target_column):
    fig = go.Figure()

    # Box plot of simulated data
    fig.add_trace(go.Box(
        y=simulated_data,
        name="Simulation Results",
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        hoverinfo='y',
        hoverlabel=dict(namelength=-1)
    ))

    # Confidence interval
    fig.add_hline(y=ci_lower, line_dash="dash", line_color="red", annotation_text=f"Lower CI: {ci_lower:.2f}")
    fig.add_hline(y=ci_upper, line_dash="dash", line_color="red", annotation_text=f"Upper CI: {ci_upper:.2f}")

    # Mean line
    mean = np.mean(simulated_data)
    fig.add_hline(y=mean, line_color="green", annotation_text=f"Mean: {mean:.2f}")

    fig.update_layout(
        title=f"Monte Carlo Simulation Results for {target_column} (Box Plot)",
        yaxis_title=target_column,
        showlegend=False,
        hovermode='closest'
    )

    return fig

def plot_sensitivity_analysis(sensitivity_results):
    fig = go.Figure()

    for param, results in sensitivity_results.items():
        x_values, y_values = zip(*results)
        fig.add_trace(go.Scatter(
            x=x_values,
            y=y_values,
            mode='lines+markers',
            name=param,
            hoverinfo='x+y',
            hoverlabel=dict(namelength=-1)
        ))

    fig.update_layout(
        title="Sensitivity Analysis",
        xaxis_title="Parameter Value",
        yaxis_title="Simulation Mean",
        showlegend=True,
        hovermode='closest'
    )

    return fig
