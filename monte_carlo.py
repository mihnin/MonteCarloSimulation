import numpy as np
import pandas as pd
from scipy import stats

def run_monte_carlo_simulation(data, num_simulations, confidence_level, trend=None, seasonality=None, multi_var=False, custom_distribution=None, correlation_matrix=None):
    if multi_var:
        return run_multi_variable_simulation(data, num_simulations, confidence_level, custom_distribution, correlation_matrix)
    
    if isinstance(data, pd.Series):
        data = data.values

    if trend:
        data = apply_trend(data, trend)
    
    if seasonality:
        data = apply_seasonality(data, seasonality)

    mean = np.mean(data)
    std = np.std(data)

    if custom_distribution:
        simulated_data = generate_custom_distribution(custom_distribution, mean, std, size=(len(data), num_simulations))
    else:
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

def generate_custom_distribution(distribution, mean, std, size):
    if distribution == 'normal':
        return np.random.normal(mean, std, size=size)
    elif distribution == 'lognormal':
        return np.random.lognormal(mean, std, size=size)
    elif distribution == 'uniform':
        return np.random.uniform(mean - std * np.sqrt(3), mean + std * np.sqrt(3), size=size)
    else:
        raise ValueError("Unsupported distribution type")

def run_multi_variable_simulation(data, num_simulations, confidence_level, custom_distribution=None, correlation_matrix=None):
    if correlation_matrix is None:
        correlation_matrix = data.corr()
    else:
        # Convert correlation_matrix to float64
        correlation_matrix = correlation_matrix.astype(np.float64)
    
    means = data.mean()
    stds = data.std()

    simulated_data = {}
    for col in data.columns:
        if custom_distribution:
            simulated_data[col] = generate_custom_distribution(custom_distribution, means[col], stds[col], size=(len(data), num_simulations))
        else:
            simulated_data[col] = np.random.normal(means[col], stds[col], size=(len(data), num_simulations))

    # Apply correlations
    cholesky = np.linalg.cholesky(correlation_matrix)
    for i in range(num_simulations):
        correlated_data = np.dot(cholesky, np.column_stack([simulated_data[col][:, i] for col in data.columns]).T).T
        for j, col in enumerate(data.columns):
            simulated_data[col][:, i] = correlated_data[:, j]

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
