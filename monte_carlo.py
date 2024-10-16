import numpy as np
import pandas as pd
from scipy import stats

def run_monte_carlo_simulation(data, num_simulations, confidence_level, trend=None, seasonality=None, multi_var=False, custom_distribution=None, correlation_matrix=None, distribution_params=None, external_factors=None):
    if multi_var:
        return run_multi_variable_simulation(data, num_simulations, confidence_level, custom_distribution, correlation_matrix, distribution_params, external_factors)
    
    if isinstance(data, pd.Series):
        data = data.values

    if trend:
        data = apply_trend(data, trend)
    
    if seasonality:
        data = apply_seasonality(data, seasonality)

    mean = np.mean(data)
    std = np.std(data)

    if custom_distribution:
        simulated_data = generate_custom_distribution(custom_distribution, mean, std, size=(len(data), num_simulations), params=distribution_params)
    else:
        simulated_data = np.random.normal(mean, std, size=(len(data), num_simulations))

    if external_factors:
        simulated_data = apply_external_factors(simulated_data, external_factors)

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
    x = np.arange(len(data))
    if trend == 'linear':
        slope, intercept, _, _, _ = stats.linregress(x, data)
        trend_line = slope * x + intercept
        return data + trend_line
    elif trend == 'exponential':
        _, intercept, _, _, _ = stats.linregress(x, np.log(data))
        trend_line = np.exp(intercept) * np.exp(0.1 * x)  # Assuming 10% growth rate
        return data * trend_line
    else:
        return data

def apply_seasonality(data, seasonality):
    seasons = len(seasonality)
    seasonal_factors = np.tile(seasonality, len(data) // seasons + 1)[:len(data)]
    return data * seasonal_factors

def generate_custom_distribution(distribution, mean, std, size, params=None):
    if distribution == 'normal':
        if params:
            return np.random.normal(params.get('loc', mean), params.get('scale', std), size=size)
        return np.random.normal(mean, std, size=size)
    elif distribution == 'lognormal':
        if params:
            return np.random.lognormal(params.get('mean', np.log(mean)), params.get('sigma', std/mean), size=size)
        return np.random.lognormal(np.log(mean), std/mean, size=size)
    elif distribution == 'uniform':
        if params:
            return np.random.uniform(params.get('low', mean - std * np.sqrt(3)), params.get('high', mean + std * np.sqrt(3)), size=size)
        return np.random.uniform(mean - std * np.sqrt(3), mean + std * np.sqrt(3), size=size)
    else:
        raise ValueError("Unsupported distribution type")

def apply_external_factors(simulated_data, external_factors):
    for factor, impact in external_factors.items():
        simulated_data *= impact
    return simulated_data

def run_multi_variable_simulation(data, num_simulations, confidence_level, custom_distribution=None, correlation_matrix=None, distribution_params=None, external_factors=None):
    if correlation_matrix is None:
        correlation_matrix = data.corr()
    else:
        correlation_matrix = correlation_matrix.astype(np.float64)
    
    means = data.mean()
    stds = data.std()

    simulated_data = {}
    for col in data.columns:
        if custom_distribution:
            col_params = distribution_params.get(col, {}) if distribution_params else None
            simulated_data[col] = generate_custom_distribution(custom_distribution, means[col], stds[col], size=(len(data), num_simulations), params=col_params)
        else:
            simulated_data[col] = np.random.normal(means[col], stds[col], size=(len(data), num_simulations))

    # Apply correlations
    cholesky = np.linalg.cholesky(correlation_matrix)
    for i in range(num_simulations):
        correlated_data = np.dot(cholesky, np.column_stack([simulated_data[col][:, i] for col in data.columns]).T).T
        for j, col in enumerate(data.columns):
            simulated_data[col][:, i] = correlated_data[:, j]

    if external_factors:
        for col in simulated_data:
            simulated_data[col] = apply_external_factors(simulated_data[col], external_factors.get(col, {}))

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

def perform_sensitivity_analysis(data, num_simulations, confidence_level, base_params, sensitivity_params):
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]  # Use the first column for sensitivity analysis
    
    base_result = run_monte_carlo_simulation(data, num_simulations, confidence_level, **base_params)
    sensitivity_results = {}

    for param, range_values in sensitivity_params.items():
        param_results = []
        for value in range_values:
            current_params = base_params.copy()
            if param in current_params:
                current_params[param] = value
            result = run_monte_carlo_simulation(data, num_simulations, confidence_level, **current_params)
            param_results.append((value, result['mean']))
        sensitivity_results[param] = param_results

    return base_result, sensitivity_results
