import numpy as np

def run_monte_carlo_simulation(data, num_simulations, confidence_level):
    mean = data.mean()
    std = data.std()

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
