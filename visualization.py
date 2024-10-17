import plotly.graph_objects as go
import plotly.express as px
import numpy as np
##import pandas as pd
import streamlit as st

def plot_simulation_results(simulated_data, ci_lower, ci_upper, target_column, plot_type='гистограмма'):
    if plot_type == 'гистограмма':
        return plot_histogram(simulated_data, ci_lower, ci_upper, target_column)
    elif plot_type == 'линейный':
        return plot_line(simulated_data, ci_lower, ci_upper, target_column)
    elif plot_type == 'ящик с усами':
        return plot_box(simulated_data, ci_lower, ci_upper, target_column)
    else:
        raise ValueError("Неверный тип графика. Выберите 'гистограмма', 'линейный' или 'ящик с усами'.")

def plot_histogram(simulated_data, ci_lower, ci_upper, target_column):
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=simulated_data,
        name="Результаты симуляции",
        nbinsx=50,
        hoverinfo='y+text',
        hovertext=[f'Значение: {x:.2f}' for x in simulated_data],
        hoverlabel=dict(namelength=-1)
    ))

    fig.add_vline(x=ci_lower, line_dash="dash", line_color="red", annotation_text=f"Нижний ДИ: {ci_lower:.2f}")
    fig.add_vline(x=ci_upper, line_dash="dash", line_color="red", annotation_text=f"Верхний ДИ: {ci_upper:.2f}")

    mean = np.mean(simulated_data)
    fig.add_vline(x=mean, line_color="green", annotation_text=f"Среднее: {mean:.2f}")

    fig.update_layout(
        title=f"Результаты симуляции Монте-Карло для {target_column} (Гистограмма)",
        xaxis_title=target_column,
        yaxis_title="Частота",
        showlegend=False,
        hovermode='closest'
    )

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1%", step="all", stepmode="backward"),
                dict(count=5, label="5%", step="all", stepmode="backward"),
                dict(count=10, label="10%", step="all", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig

def plot_line(simulated_data, ci_lower, ci_upper, target_column):
    fig = go.Figure()

    x = list(range(len(simulated_data)))
    fig.add_trace(go.Scatter(
        x=x,
        y=simulated_data,
        mode='lines',
        name="Результаты симуляции",
        hoverinfo='x+y',
        hovertext=[f'Запуск: {i+1}<br>Значение: {y:.2f}' for i, y in enumerate(simulated_data)],
        hoverlabel=dict(namelength=-1)
    ))

    fig.add_hline(y=ci_lower, line_dash="dash", line_color="red", annotation_text=f"Нижний ДИ: {ci_lower:.2f}")
    fig.add_hline(y=ci_upper, line_dash="dash", line_color="red", annotation_text=f"Верхний ДИ: {ci_upper:.2f}")

    mean = np.mean(simulated_data)
    fig.add_hline(y=mean, line_color="green", annotation_text=f"Среднее: {mean:.2f}")

    fig.update_layout(
        title=f"Результаты симуляции Монте-Карло для {target_column} (Линейный график)",
        xaxis_title="Запуск симуляции",
        yaxis_title=target_column,
        showlegend=True,
        hovermode='closest'
    )

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

    fig.add_trace(go.Box(
        y=simulated_data,
        name="Результаты симуляции",
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8,
        hoverinfo='y',
        hovertext=[f'Значение: {y:.2f}' for y in simulated_data],
        hoverlabel=dict(namelength=-1)
    ))

    fig.add_hline(y=ci_lower, line_dash="dash", line_color="red", annotation_text=f"Нижний ДИ: {ci_lower:.2f}")
    fig.add_hline(y=ci_upper, line_dash="dash", line_color="red", annotation_text=f"Верхний ДИ: {ci_upper:.2f}")

    mean = np.mean(simulated_data)
    fig.add_hline(y=mean, line_color="green", annotation_text=f"Среднее: {mean:.2f}")

    fig.update_layout(
        title=f"Результаты симуляции Монте-Карло для {target_column} (Боксплот)",
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
            hoverinfo='x+y+text',
            hovertext=[f'{param}: {x:.2f}<br>Среднее: {y:.2f}' for x, y in results],
            hoverlabel=dict(namelength=-1)
        ))

    fig.update_layout(
        title="Анализ чувствительности",
        xaxis_title="Значение параметра",
        yaxis_title="Среднее значение симуляции",
        showlegend=True,
        hovermode='closest'
    )

    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1%", step="all", stepmode="backward"),
                dict(count=5, label="5%", step="all", stepmode="backward"),
                dict(count=10, label="10%", step="all", stepmode="backward"),
                dict(step="all")
            ])
        )
    )

    return fig

@st.cache_data
def create_correlation_heatmap(correlation_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.index,
        hoverongaps=False,
        hoverinfo='x+y+z',
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))

    fig.update_layout(
        title="Тепловая карта корреляций",
        xaxis_title="Переменные",
        yaxis_title="Переменные",
        height=600,
        width=800
    )

    return fig

def plot_correlation_heatmap(correlation_matrix):
    fig = create_correlation_heatmap(correlation_matrix)
    return fig

def plot_3d_scatter(data, x_col, y_col, z_col):
    fig = go.Figure(data=[go.Scatter3d(
        x=data[x_col],
        y=data[y_col],
        z=data[z_col],
        mode='markers',
        marker=dict(
            size=5,
            color=data[z_col],
            colorscale='Viridis',
            opacity=0.8
        ),
        hoverinfo='text',
        hovertext=[f'{x_col}: {x:.2f}<br>{y_col}: {y:.2f}<br>{z_col}: {z:.2f}' for x, y, z in zip(data[x_col], data[y_col], data[z_col])]
    )])

    fig.update_layout(
        title=f"3D Точечный график: {x_col} vs {y_col} vs {z_col}",
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col
        ),
        hovermode='closest'
    )

    return fig
