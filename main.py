import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_processing import load_data, preprocess_data
from monte_carlo import run_monte_carlo_simulation
from visualization import plot_simulation_results

st.set_page_config(page_title="Business Data Analysis", page_icon="assets/favicon.png", layout="wide")

def main():
    st.title("Business Data Analysis with Monte Carlo Simulation")

    # File upload
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

    if uploaded_file is not None:
        # Load and preprocess data
        df = load_data(uploaded_file)
        if df is not None:
            st.success("File uploaded successfully!")
            st.subheader("Data Preview")
            st.write(df.head())

            # Preprocess data
            numeric_columns = preprocess_data(df)

            # Monte Carlo simulation configuration
            st.subheader("Monte Carlo Simulation Configuration")
            target_column = st.selectbox("Select target column for simulation", numeric_columns)
            num_simulations = st.slider("Number of simulations", min_value=100, max_value=10000, value=1000, step=100)
            confidence_level = st.slider("Confidence level (%)", min_value=80, max_value=99, value=95, step=1)

            if st.button("Run Simulation"):
                # Run Monte Carlo simulation
                results = run_monte_carlo_simulation(df[target_column], num_simulations, confidence_level)

                # Display results
                st.subheader("Simulation Results")
                st.write(f"Mean: {results['mean']:.2f}")
                st.write(f"Median: {results['median']:.2f}")
                st.write(f"Standard Deviation: {results['std']:.2f}")
                st.write(f"{confidence_level}% Confidence Interval: ({results['ci_lower']:.2f}, {results['ci_upper']:.2f})")

                # Visualize results
                fig = plot_simulation_results(results['simulated_data'], results['ci_lower'], results['ci_upper'], target_column)
                st.plotly_chart(fig, use_container_width=True)

    st.sidebar.title("About")
    st.sidebar.info("This application performs Monte Carlo simulations on uploaded business data to analyze and visualize potential outcomes.")

if __name__ == "__main__":
    main()
