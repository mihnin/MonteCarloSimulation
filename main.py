import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from data_processing import load_data, preprocess_data
from monte_carlo import run_monte_carlo_simulation, perform_sensitivity_analysis
from visualization import plot_simulation_results, plot_sensitivity_analysis
from export_utils import export_results_to_excel

st.set_page_config(page_title="Business Data Analysis", page_icon="assets/favicon.png", layout="wide")

def main():
    st.title("Business Data Analysis with Monte Carlo Simulation")

    # File upload
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["xlsx", "xls", "csv"])

    df = None
    if uploaded_file is not None:
        try:
            # Load and preprocess data
            df = load_data(uploaded_file)
            if df is not None:
                st.success("Data loaded successfully!")
                st.subheader("Data Preview")
                st.write(df.head())
            else:
                st.error("Failed to load the uploaded file. Please check the file format and try again.")
        except Exception as e:
            st.error(f"An error occurred while loading the file: {str(e)}")
    else:
        # Load test data if no file is uploaded
        try:
            df = pd.read_csv('test_data.csv')
            st.info("Using test data. You can upload your own file above.")
        except Exception as e:
            st.error(f"Failed to load test data: {str(e)}")

    if df is not None:
        # Preprocess data
        numeric_columns = preprocess_data(df)

        # Rest of the code remains the same
        # ...

if __name__ == "__main__":
    main()
