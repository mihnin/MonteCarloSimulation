import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import requests
from data_processing import load_data, preprocess_data
from monte_carlo import run_monte_carlo_simulation, perform_sensitivity_analysis
from visualization import plot_simulation_results, plot_sensitivity_analysis, plot_correlation_heatmap, plot_3d_scatter
from export_utils import export_results_to_excel
from database import save_analysis, get_all_analyses, get_analysis

st.set_page_config(page_title="Анализ бизнес-данных", page_icon="assets/favicon.png", layout="wide")

def fetch_api_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data)
        return df
    except requests.RequestException as e:
        st.error(f"Ошибка при получении данных из API: {str(e)}")
        return None

def plot_scatter_matrix(results):
    df = pd.DataFrame({k: v['simulated_data'] for k, v in results.items()})
    fig = px.scatter_matrix(df, title="Результаты многомерной симуляции")
    fig.update_traces(diagonal_visible=False)
    return fig

def main():
    st.title("Анализ бизнес-данных с помощью симуляции Монте-Карло")

    # Тест функциональности сохранения и извлечения
    st.subheader("Тест функциональности сохранения и извлечения")
    if st.button("Запустить тест"):
        # Сохранение тестового анализа
        test_config = {
            'data_source': 'test',
            'use_multi_var': False,
            'target_column': 'Продажи',
            'num_simulations': 1000,
            'confidence_level': 95
        }
        test_results = {
            'mean': 1000,
            'median': 950,
            'std': 100,
            'ci_lower': 800,
            'ci_upper': 1200,
            'simulated_data': list(np.random.normal(1000, 100, 1000))
        }
        save_analysis("Тестовый анализ", test_config, test_results)
        st.success("Тестовый анализ успешно сохранен!")

        # Извлечение и отображение сохраненного анализа
        analyses = get_all_analyses()
        if analyses:
            test_analysis = get_analysis(analyses[-1][0])  # Получение последнего сохраненного анализа
            st.write("Извлеченный тестовый анализ:")
            st.json(test_analysis)
        else:
            st.error("Не удалось извлечь сохраненный анализ.")

    # Боковая панель для навигации
    page = st.sidebar.selectbox("Выберите страницу", ["Запустить новый анализ", "Просмотреть сохраненные анализы"])

    if page == "Запустить новый анализ":
        run_new_analysis()
    else:
        view_saved_analyses()

def run_new_analysis():
    # Выбор источника данных
    data_source = st.radio("Выберите источник данных", ["Загрузить файл", "API", "Использовать тестовые данные"])

    df = None
    if data_source == "Загрузить файл":
        # Загрузка файла
        uploaded_file = st.file_uploader("Загрузите ваш файл Excel или CSV", type=["xlsx", "xls", "csv"])

        if uploaded_file is not None:
            try:
                # Загрузка и предобработка данных
                df = load_data(uploaded_file)
                if df is not None:
                    st.success("Данные успешно загружены!")
                    st.subheader("Предварительный просмотр данных")
                    st.write(df.head())
                else:
                    st.error("Не удалось загрузить файл. Пожалуйста, проверьте формат файла и попробуйте снова.")
            except ValueError as e:
                st.error(str(e))
                st.info("Используются тестовые данные в качестве резервного варианта. Вы можете попробовать загрузить файл снова.")
                df = pd.read_csv('test_data.csv', encoding='utf-8-sig')
        else:
            st.info("Файл не загружен. Используются тестовые данные.")
            df = pd.read_csv('test_data.csv', encoding='utf-8-sig')
    elif data_source == "API":
        # Получение данных из API
        api_url = st.text_input("Введите URL API")
        if api_url:
            df = fetch_api_data(api_url)
            if df is not None:
                st.success("Данные успешно получены из API!")
                st.subheader("Предварительный просмотр данных")
                st.write(df.head())
    else:
        # Использование тестовых данных
        try:
            df = pd.read_csv('test_data.csv', encoding='utf-8-sig')
            st.info("Используются тестовые данные.")
            st.subheader("Предварительный просмотр данных")
            st.write(df.head())
        except Exception as e:
            st.error(f"Не удалось загрузить тестовые данные: {str(e)}")

    if df is not None:
        # Preprocess data
        numeric_columns = preprocess_data(df)

        # Настройка симуляции Монте-Карло
        st.subheader("Настройка симуляции Монте-Карло")
        
        # Опция многопеременной симуляции
        use_multi_var = st.checkbox("Запустить многопеременную симуляцию")
        
        if use_multi_var:
            target_columns = st.multiselect("Выберите целевые столбцы для симуляции", numeric_columns, default=numeric_columns)
        else:
            target_column = st.selectbox("Выберите целевой столбец для симуляции", numeric_columns)
        
        num_simulations = st.slider("Количество симуляций", min_value=100, max_value=10000, value=1000, step=100)
        confidence_level = st.slider("Уровень доверия (%)", min_value=80, max_value=99, value=95, step=1)

        # Расширенные настройки
        st.subheader("Расширенные настройки")
        use_trend = st.checkbox("Применить анализ тренда")
        trend_type = None
        if use_trend:
            trend_type = st.selectbox("Select trend type", ["linear", "exponential"])

        use_seasonality = st.checkbox("Применить сезонные корректировки")
        seasonality = None
        if use_seasonality:
            seasons = st.number_input("Количество сезонов", min_value=2, max_value=12, value=4)
            seasonality = [st.number_input(f"Фактор сезона {i+1}", min_value=0.1, max_value=2.0, value=1.0, step=0.1) for i in range(seasons)]

        # Custom distribution selection
        custom_distribution = st.selectbox("Выберите распределение", ["нормальное", "логнормальное", "равномерное"])
        
        # Distribution parameters
        st.subheader("Параметры распределения")
        if custom_distribution == "нормальное":
            loc = st.number_input("Расположение (среднее)", value=0.0)
            scale = st.number_input("Масштаб (стандартное отклонение)", value=1.0, min_value=0.1)
            distribution_params = {"loc": loc, "scale": scale}
        elif custom_distribution == "логнормальное":
            mean = st.number_input("Среднее логарифма", value=0.0)
            sigma = st.number_input("Стандартное отклонение логарифма", value=1.0, min_value=0.1)
            distribution_params = {"mean": mean, "sigma": sigma}
        else:  # равномерное
            low = st.number_input("Нижняя граница", value=0.0)
            high = st.number_input("Верхняя граница", value=1.0)
            distribution_params = {"low": low, "high": high}

        # Внешние факторы
        st.subheader("Внешние факторы")
        use_external_factors = st.checkbox("Применить внешние факторы")
        external_factors = {}
        if use_external_factors:
            num_factors = st.number_input("Количество внешних факторов", min_value=1, max_value=5, value=1)
            for i in range(num_factors):
                factor_name = st.text_input(f"Название фактора {i+1}", value=f"Фактор {i+1}")
                factor_impact = st.slider(f"Влияние фактора {i+1}", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
                external_factors[factor_name] = factor_impact

        # Correlation matrix
        if use_multi_var:
            st.subheader("Correlation Matrix")
            correlation_matrix = pd.DataFrame(index=target_columns, columns=target_columns)
            for i, col1 in enumerate(target_columns):
                for j, col2 in enumerate(target_columns):
                    if i <= j:
                        if i == j:
                            correlation_matrix.loc[col1, col2] = 1.0
                        else:
                            correlation = st.number_input(f"Correlation between {col1} and {col2}", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)
                            correlation_matrix.loc[col1, col2] = correlation
                            correlation_matrix.loc[col2, col1] = correlation
            st.write(correlation_matrix)

        # Анализ чувствительност
        st.subheader("Анализ чувствительности")
        run_sensitivity = st.checkbox("Запустить анализ чувствительности")
        sensitivity_params = {}
        if run_sensitivity:
            num_params = st.number_input("Количество параметров для анализа чувствительности", min_value=1, max_value=5, value=1)
            for i in range(num_params):
                param_name = st.text_input(f"Название параметра {i+1}", value=f"Параметр {i+1}")
                param_min = st.number_input(f"Минимальное значение {param_name}", value=0.5)
                param_max = st.number_input(f"Максимальное значение {param_name}", value=1.5)
                param_steps = st.number_input(f"Количество шагов для {param_name}", min_value=2, max_value=10, value=5)
                sensitivity_params[param_name] = np.linspace(param_min, param_max, param_steps)

        # Выбор типа графика
        graph_type = st.selectbox("Выберите тип графика", ["гистограмма", "линейный", "ящик с усами"])

        if st.button("Запустить симуляцию"):
            # Запуск смуляции Монте-Карло
            if use_multi_var:
                results = run_monte_carlo_simulation(df[target_columns], num_simulations, confidence_level, trend=trend_type, seasonality=seasonality, multi_var=True, custom_distribution=custom_distribution, correlation_matrix=correlation_matrix, distribution_params=distribution_params, external_factors=external_factors)
            else:
                results = run_monte_carlo_simulation(df[target_column], num_simulations, confidence_level, trend=trend_type, seasonality=seasonality, custom_distribution=custom_distribution, distribution_params=distribution_params, external_factors=external_factors)

            # Отображение результатов
            st.subheader("Результаты симуляции")
            if use_multi_var:
                for col, col_results in results.items():
                    st.write(f"Результаты для {col}:")
                    st.write(f"Среднее: {col_results['mean']:.2f}")
                    st.write(f"Медиана: {col_results['median']:.2f}")
                    st.write(f"Стандартное отклонение: {col_results['std']:.2f}")
                    st.write(f"{confidence_level}% оверительный интервал: ({col_results['ci_lower']:.2f}, {col_results['ci_upper']:.2f})")
                    
                    # Визуализация результатов
                    fig = plot_simulation_results(col_results['simulated_data'], col_results['ci_lower'], col_results['ci_upper'], col, plot_type=graph_type)
                    st.plotly_chart(fig, use_container_width=True)

                # Добавление матрицы рассеяния для многопеременных симуляций
                st.subheader("Матрица рассеяния многопеременной симуляции")
                scatter_matrix_fig = plot_scatter_matrix(results)
                st.plotly_chart(scatter_matrix_fig, use_container_width=True)

                # Добавление тепловой карты корреляции
                st.subheader("Тепловая карта корреляции")
                heatmap_fig = plot_correlation_heatmap(correlation_matrix)
                st.plotly_chart(heatmap_fig, use_container_width=True)

                # Добавление 3D графика рассеяния, если есть по крайней мере 3 переменных
                if len(target_columns) >= 3:
                    st.subheader("3D график рассеяния")
                    x_col = st.selectbox("Выберите переменную для оси X", target_columns)
                    y_col = st.selectbox("Выберите переменную для оси Y", [col for col in target_columns if col != x_col])
                    z_col = st.selectbox("Выберите переменную для оси Z", [col for col in target_columns if col not in [x_col, y_col]])
                    
                    scatter_3d_data = pd.DataFrame({col: results[col]['simulated_data'] for col in [x_col, y_col, z_col]})
                    scatter_3d_fig = plot_3d_scatter(scatter_3d_data, x_col, y_col, z_col)
                    st.plotly_chart(scatter_3d_fig, use_container_width=True)
            else:
                st.write(f"Среднее: {results['mean']:.2f}")
                st.write(f"Медиана: {results['median']:.2f}")
                st.write(f"Стандартное отклонение: {results['std']:.2f}")
                st.write(f"{confidence_level}% Доверительный интервал: ({results['ci_lower']:.2f}, {results['ci_upper']:.2f})")

                # Визуализация результатов
                fig = plot_simulation_results(results['simulated_data'], results['ci_lower'], results['ci_upper'], target_column, plot_type=graph_type)
                st.plotly_chart(fig, use_container_width=True)

            # Запуск анализа чувствительности, если выбран
            if run_sensitivity:
                st.subheader("Результаты анализа чувствительности")
                base_params = {
                    'trend': trend_type,
                    'seasonality': seasonality,
                    'custom_distribution': custom_distribution,
                    'distribution_params': distribution_params,
                    'external_factors': external_factors
                }
                sensitivity_data = df[target_column] if not use_multi_var else df[target_columns].iloc[:, 0]
                base_result, sensitivity_results = perform_sensitivity_analysis(
                    sensitivity_data,
                    num_simulations,
                    confidence_level,
                    base_params,
                    sensitivity_params
                )

                fig_sensitivity = plot_sensitivity_analysis(sensitivity_results)
                st.plotly_chart(fig_sensitivity, use_container_width=True)

            # Сохранение анализа
            st.subheader("Сохранить ализ")
            analysis_name = st.text_input("Введите название для этого анализа")
            if st.button("Сохранить анализ"):
                if analysis_name:
                    config = {
                        'data_source': data_source,
                        'use_multi_var': use_multi_var,
                        'target_columns' if use_multi_var else 'target_column': target_columns if use_multi_var else target_column,
                        'num_simulations': num_simulations,
                        'confidence_level': confidence_level,
                        'trend_type': trend_type,
                        'seasonality': seasonality,
                        'custom_distribution': custom_distribution,
                        'distribution_params': distribution_params,
                        'external_factors': external_factors,
                        'sensitivity_params': sensitivity_params
                    }
                    save_analysis(analysis_name, config, results)
                    st.success(f"Анализ '{analysis_name}' успешно сохранен!")
                else:
                    st.warning("Пожалуйста, введите название для анализа перед сохранением.")

            # Экспорт результатов
            excel_file = export_results_to_excel(results)
            st.download_button(
                label="Экспортировать результаты в Excel",
                data=excel_file,
                file_name="результаты_симуляции.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

def view_saved_analyses():
    st.subheader("Saved Analyses")
    analyses = get_all_analyses()
    if analyses:
        selected_analysis = st.selectbox("Select an analysis to view", analyses, format_func=lambda x: f"{x[1]} ({x[2]})")
        if selected_analysis:
            analysis = get_analysis(selected_analysis[0])
            if analysis:
                st.write(f"Analysis Name: {analysis['name']}")
                st.write(f"Date: {analysis['date']}")
                st.subheader("Configuration")
                st.json(analysis['config'])
                st.subheader("Results")
                st.json(analysis['results'])

                # Визуализация результатов
                if analysis['config']['use_multi_var']:
                    for col, col_results in analysis['results'].items():
                        st.write(f"Results for {col}:")
                        fig = plot_simulation_results(col_results['simulated_data'], col_results['ci_lower'], col_results['ci_upper'], col, plot_type='гистограмма')
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    target_column = analysis['config']['target_column']
                    fig = plot_simulation_results(analysis['results']['simulated_data'], analysis['results']['ci_lower'], analysis['results']['ci_upper'], target_column, plot_type='гистограмма')
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No saved analyses found.")

    st.sidebar.title("About")
    st.sidebar.info("This application performs Monte Carlo simulations on uploaded business data or data fetched from an API to analyze and visualize potential outcomes.")

if __name__ == "__main__":
    main()
