import numpy as np
import pandas as pd
from scipy import stats

def run_monte_carlo_simulation(data, num_simulations, confidence_level, trend=None, seasonality=None, multi_var=False):
    if multi_var:
        return run_multi_variable_simulation(data, num_simulations, confidence_level)
    
    if isinstance(data, pd.Series):
        data = data.values

    if trend:
        data = apply_trend(data, trend)
    
    if seasonality:
        data = apply_seasonality(data, seasonality)

    mean = np.mean(data)
    std = np.std(data)

    simulated_data = np.random.normal(mean, std, size=(len(data), num_simulations))
    simulated_means = np.mean(simulated_data, axis=0)

    results = {
        'mean': np.mean(simulated_means),
        'median': np.median(simulated_means),
        'std': np.std(simulated_means),
        'ci_lower': np.percentile(simulated_means, (100 - confidence_level) / 2),
        'ci_upper': np.percentile(simulated_means, 100 - (100 - confidence_level) / 2),
        'simulated_data': simulated_means
    }

    return results

def apply_trend(data, trend):
    if trend == 'linear':
        x = np.arange(len(data))
        slope, intercept, _, _, _ = stats.linregress(x, data)
        trend_line = slope * x + intercept
        return data + trend_line
    elif trend == 'exponential':
        x = np.arange(len(data))
        _, intercept, _, _, _ = stats.linregress(x, np.log(data))
        trend_line = np.exp(intercept) * np.exp(0.1 * x)  # Assuming 10% growth rate
        return data * trend_line
    else:
        return data

def apply_seasonality(data, seasonality):
    seasons = len(seasonality)
    seasonal_factors = np.tile(seasonality, len(data) // seasons + 1)[:len(data)]
    return data * seasonal_factors

def run_multi_variable_simulation(data, num_simulations, confidence_level):
    corr_matrix = data.corr()
    means = data.mean()
    stds = data.std()

    simulated_data = {}
    for col in data.columns:
        simulated_data[col] = np.random.normal(means[col], stds[col], size=(len(data), num_simulations))

    for i in range(num_simulations):
        for j, col1 in enumerate(data.columns):
            for k, col2 in enumerate(data.columns):
                if j != k:
                    simulated_data[col1][:, i], simulated_data[col2][:, i] = apply_correlation(
                        simulated_data[col1][:, i], simulated_data[col2][:, i], corr_matrix.loc[col1, col2]
                    )

    results = {}
    for col in data.columns:
        simulated_means = np.mean(simulated_data[col], axis=0)
        results[col] = {
            'mean': np.mean(simulated_means),
            'median': np.median(simulated_means),
            'std': np.std(simulated_means),
            'ci_lower': np.percentile(simulated_means, (100 - confidence_level) / 2),
            'ci_upper': np.percentile(simulated_means, 100 - (100 - confidence_level) / 2),
            'simulated_data': simulated_means
        }

    return results

def apply_correlation(x, y, correlation):
    z = stats.norm.ppf(stats.rankdata(y)/(len(y)+1))
    return x, stats.norm.ppf(stats.rankdata(correlation * x + np.sqrt(1 - correlation**2) * z)/(len(y)+1))
