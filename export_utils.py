import pandas as pd
from io import BytesIO

def export_results_to_excel(results):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='openpyxl')

    if isinstance(results, dict) and 'simulated_data' in results:
        # Single variable simulation
        df = pd.DataFrame({
            'Simulated Data': results['simulated_data'],
            'Statistics': [
                f"Mean: {results['mean']:.2f}",
                f"Median: {results['median']:.2f}",
                f"Standard Deviation: {results['std']:.2f}",
                f"CI Lower: {results['ci_lower']:.2f}",
                f"CI Upper: {results['ci_upper']:.2f}"
            ]
        })
        df.to_excel(writer, sheet_name='Simulation Results', index=False)
    elif isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
        # Multi-variable simulation
        for variable, var_results in results.items():
            df = pd.DataFrame({
                'Simulated Data': var_results['simulated_data'],
                'Statistics': [
                    f"Mean: {var_results['mean']:.2f}",
                    f"Median: {var_results['median']:.2f}",
                    f"Standard Deviation: {var_results['std']:.2f}",
                    f"CI Lower: {var_results['ci_lower']:.2f}",
                    f"CI Upper: {var_results['ci_upper']:.2f}"
                ]
            })
            df.to_excel(writer, sheet_name=f'{variable} Results', index=False)
    else:
        raise ValueError("Invalid results format")

    writer.save()
    output.seek(0)
    return output
