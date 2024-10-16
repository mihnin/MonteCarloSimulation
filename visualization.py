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
    fig.add_trace(go.Histogram(x=simulated_data, name="Simulation Results", nbinsx=50))

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
        showlegend=False
    )

    return fig

def plot_line(simulated_data, ci_lower, ci_upper, target_column):
    fig = go.Figure()

    # Line plot of simulated data
    x = range(len(simulated_data))
    fig.add_trace(go.Scatter(x=x, y=simulated_data, mode='lines', name="Simulation Results"))

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
        showlegend=True
    )

    return fig

def plot_box(simulated_data, ci_lower, ci_upper, target_column):
    fig = go.Figure()

    # Box plot of simulated data
    fig.add_trace(go.Box(y=simulated_data, name="Simulation Results"))

    # Confidence interval
    fig.add_hline(y=ci_lower, line_dash="dash", line_color="red", annotation_text=f"Lower CI: {ci_lower:.2f}")
    fig.add_hline(y=ci_upper, line_dash="dash", line_color="red", annotation_text=f"Upper CI: {ci_upper:.2f}")

    # Mean line
    mean = np.mean(simulated_data)
    fig.add_hline(y=mean, line_color="green", annotation_text=f"Mean: {mean:.2f}")

    fig.update_layout(
        title=f"Monte Carlo Simulation Results for {target_column} (Box Plot)",
        yaxis_title=target_column,
        showlegend=False
    )

    return fig
