import plotly.graph_objects as go
import numpy as np

def plot_simulation_results(simulated_data, ci_lower, ci_upper, target_column):
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
        title=f"Monte Carlo Simulation Results for {target_column}",
        xaxis_title=target_column,
        yaxis_title="Frequency",
        showlegend=False
    )

    return fig
