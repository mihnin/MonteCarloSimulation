import pandas as pd
from io import BytesIO

def export_results_to_excel(results):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if isinstance(results, dict) and 'simulated_data' in results:
            # Single variable simulation
            df = pd.DataFrame({
                'Simulated Data': results['simulated_data']
            })
            df.to_excel(writer, sheet_name='Simulation Results', index=False)
            
            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Standard Deviation', 'CI Lower', 'CI Upper'],
                'Value': [
                    results['mean'],
                    results['median'],
                    results['std'],
                    results['ci_lower'],
                    results['ci_upper']
                ]
            })
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        elif isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            # Multi-variable simulation
            for variable, var_results in results.items():
                df = pd.DataFrame({
                    'Simulated Data': var_results['simulated_data']
                })
                df.to_excel(writer, sheet_name=f'{variable} Results', index=False)
                
                stats_df = pd.DataFrame({
                    'Statistic': ['Mean', 'Median', 'Standard Deviation', 'CI Lower', 'CI Upper'],
                    'Value': [
                        var_results['mean'],
                        var_results['median'],
                        var_results['std'],
                        var_results['ci_lower'],
                        var_results['ci_upper']
                    ]
                })
                stats_df.to_excel(writer, sheet_name=f'{variable} Statistics', index=False)
        
        else:
            raise ValueError("Invalid results format")

    output.seek(0)
    return output
