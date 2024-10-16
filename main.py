import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_processing import load_data, preprocess_data
from monte_carlo import run_monte_carlo_simulation, perform_sensitivity_analysis
from visualization import plot_simulation_results, plot_sensitivity_analysis
from export_utils import export_results_to_excel

st.set_page_config(page_title="Business Data Analysis", page_icon="assets/favicon.png", layout="wide")

def main():
    st.title("Business Data Analysis with Monte Carlo Simulation")

    # File upload
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        # Load and preprocess data
        df = load_data(uploaded_file)
    else:
        # Load test data if no file is uploaded
        df = pd.read_csv('test_data.csv')
        st.info("Using test data. You can upload your own file above.")

    if df is not None:
        st.success("Data loaded successfully!")
        st.subheader("Data Preview")
        st.write(df.head())

        # Preprocess data
        numeric_columns = preprocess_data(df)

        # Monte Carlo simulation configuration
        st.subheader("Monte Carlo Simulation Configuration")
        
        # Multi-variable simulation option
        use_multi_var = st.checkbox("Run multi-variable simulation")
        
        if use_multi_var:
            target_columns = st.multiselect("Select target columns for simulation", numeric_columns, default=numeric_columns)
        else:
            target_column = st.selectbox("Select target column for simulation", numeric_columns)
        
        num_simulations = st.slider("Number of simulations", min_value=100, max_value=10000, value=1000, step=100)
        confidence_level = st.slider("Confidence level (%)", min_value=80, max_value=99, value=95, step=1)

        # Advanced options
        st.subheader("Advanced Options")
        use_trend = st.checkbox("Apply trend analysis")
        trend_type = None
        if use_trend:
            trend_type = st.selectbox("Select trend type", ["linear", "exponential"])

        use_seasonality = st.checkbox("Apply seasonal adjustments")
        seasonality = None
        if use_seasonality:
            seasons = st.number_input("Number of seasons", min_value=2, max_value=12, value=4)
            seasonality = [st.number_input(f"Season {i+1} factor", min_value=0.1, max_value=2.0, value=1.0, step=0.1) for i in range(seasons)]

        # Custom distribution selection
        custom_distribution = st.selectbox("Select distribution", ["normal", "lognormal", "uniform"])
        
        # Distribution parameters
        st.subheader("Distribution Parameters")
        if custom_distribution == "normal":
            loc = st.number_input("Location (mean)", value=0.0)
            scale = st.number_input("Scale (standard deviation)", value=1.0, min_value=0.1)
            distribution_params = {"loc": loc, "scale": scale}
        elif custom_distribution == "lognormal":
            mean = st.number_input("Mean of log", value=0.0)
            sigma = st.number_input("Standard deviation of log", value=1.0, min_value=0.1)
            distribution_params = {"mean": mean, "sigma": sigma}
        else:  # uniform
            low = st.number_input("Lower bound", value=0.0)
            high = st.number_input("Upper bound", value=1.0)
            distribution_params = {"low": low, "high": high}

        # External factors
        st.subheader("External Factors")
        use_external_factors = st.checkbox("Apply external factors")
        external_factors = {}
        if use_external_factors:
            num_factors = st.number_input("Number of external factors", min_value=1, max_value=5, value=1)
            for i in range(num_factors):
                factor_name = st.text_input(f"Factor {i+1} name", value=f"Factor {i+1}")
                factor_impact = st.slider(f"Factor {i+1} impact", min_value=0.5, max_value=1.5, value=1.0, step=0.1)
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

        # Sensitivity analysis
        st.subheader("Sensitivity Analysis")
        run_sensitivity = st.checkbox("Run sensitivity analysis")
        sensitivity_params = {}
        if run_sensitivity:
            num_params = st.number_input("Number of parameters for sensitivity analysis", min_value=1, max_value=5, value=1)
            for i in range(num_params):
                param_name = st.text_input(f"Parameter {i+1} name", value=f"Param {i+1}")
                param_min = st.number_input(f"{param_name} minimum value", value=0.5)
                param_max = st.number_input(f"{param_name} maximum value", value=1.5)
                param_steps = st.number_input(f"{param_name} number of steps", min_value=2, max_value=10, value=5)
                sensitivity_params[param_name] = np.linspace(param_min, param_max, param_steps)

        # Graph type selection
        graph_type = st.selectbox("Select graph type", ["histogram", "line", "box"])

        if st.button("Run Simulation"):
            # Run Monte Carlo simulation
            if use_multi_var:
                results = run_monte_carlo_simulation(df[target_columns], num_simulations, confidence_level, trend=trend_type, seasonality=seasonality, multi_var=True, custom_distribution=custom_distribution, correlation_matrix=correlation_matrix, distribution_params=distribution_params, external_factors=external_factors)
            else:
                results = run_monte_carlo_simulation(df[target_column], num_simulations, confidence_level, trend=trend_type, seasonality=seasonality, custom_distribution=custom_distribution, distribution_params=distribution_params, external_factors=external_factors)

            # Display results
            st.subheader("Simulation Results")
            if use_multi_var:
                for col, col_results in results.items():
                    st.write(f"Results for {col}:")
                    st.write(f"Mean: {col_results['mean']:.2f}")
                    st.write(f"Median: {col_results['median']:.2f}")
                    st.write(f"Standard Deviation: {col_results['std']:.2f}")
                    st.write(f"{confidence_level}% Confidence Interval: ({col_results['ci_lower']:.2f}, {col_results['ci_upper']:.2f})")
                    
                    # Visualize results
                    fig = plot_simulation_results(col_results['simulated_data'], col_results['ci_lower'], col_results['ci_upper'], col, plot_type=graph_type)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.write(f"Mean: {results['mean']:.2f}")
                st.write(f"Median: {results['median']:.2f}")
                st.write(f"Standard Deviation: {results['std']:.2f}")
                st.write(f"{confidence_level}% Confidence Interval: ({results['ci_lower']:.2f}, {results['ci_upper']:.2f})")

                # Visualize results
                fig = plot_simulation_results(results['simulated_data'], results['ci_lower'], results['ci_upper'], target_column, plot_type=graph_type)
                st.plotly_chart(fig, use_container_width=True)

            # Run sensitivity analysis if selected
            if run_sensitivity:
                st.subheader("Sensitivity Analysis Results")
                base_result, sensitivity_results = perform_sensitivity_analysis(df[target_column] if not use_multi_var else df[target_columns], num_simulations, confidence_level, {
                    "trend": trend_type,
                    "seasonality": seasonality,
                    "custom_distribution": custom_distribution,
                    "distribution_params": distribution_params,
                    "external_factors": external_factors
                }, sensitivity_params)

                fig_sensitivity = plot_sensitivity_analysis(sensitivity_results)
                st.plotly_chart(fig_sensitivity, use_container_width=True)

            # Export results button
            excel_file = export_results_to_excel(results)
            st.download_button(
                label="Export Results to Excel",
                data=excel_file,
                file_name="simulation_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    st.sidebar.title("About")
    st.sidebar.info("This application performs Monte Carlo simulations on uploaded business data to analyze and visualize potential outcomes.")

if __name__ == "__main__":
    main()
