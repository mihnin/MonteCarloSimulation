import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_processing import load_data, preprocess_data
from monte_carlo import run_monte_carlo_simulation
from visualization import plot_simulation_results
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

        use_multi_var = st.checkbox("Run multi-variable simulation")

        # Graph type selection
        graph_type = st.selectbox("Select graph type", ["histogram", "line", "box"])

        if st.button("Run Simulation"):
            # Run Monte Carlo simulation
            if use_multi_var:
                results = run_monte_carlo_simulation(df[numeric_columns], num_simulations, confidence_level, multi_var=True)
            else:
                results = run_monte_carlo_simulation(df[target_column], num_simulations, confidence_level, trend=trend_type, seasonality=seasonality)

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
