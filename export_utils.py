import pandas as pd
from io import BytesIO

def export_results_to_excel(results):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        if isinstance(results, dict) and 'simulated_data' in results:
            # Симуляция одной переменной
            df = pd.DataFrame({
                'Симулированные данные': results['simulated_data']
            })
            df.to_excel(writer, sheet_name='Результаты симуляции', index=False)
            
            stats_df = pd.DataFrame({
                'Статистика': ['Среднее', 'Медиана', 'Стандартное отклонение', 'Нижний ДИ', 'Верхний ДИ'],
                'Значение': [
                    results['mean'],
                    results['median'],
                    results['std'],
                    results['ci_lower'],
                    results['ci_upper']
                ]
            })
            stats_df.to_excel(writer, sheet_name='Статистика', index=False)
        
        elif isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
            # Симуляция нескольких переменных
            for variable, var_results in results.items():
                df = pd.DataFrame({
                    'Симулированные данные': var_results['simulated_data']
                })
                df.to_excel(writer, sheet_name=f'Результаты {variable}', index=False)
                
                stats_df = pd.DataFrame({
                    'Статистика': ['Среднее', 'Медиана', 'Стандартное отклонение', 'Нижний ДИ', 'Верхний ДИ'],
                    'Значение': [
                        var_results['mean'],
                        var_results['median'],
                        var_results['std'],
                        var_results['ci_lower'],
                        var_results['ci_upper']
                    ]
                })
                stats_df.to_excel(writer, sheet_name=f'Статистика {variable}', index=False)
        
        else:
            raise ValueError("Неверный формат результатов")

    output.seek(0)
    return output
