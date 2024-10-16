import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate dates for the last 100 days
end_date = datetime.now().date()
start_date = end_date - timedelta(days=99)
dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Generate random sales and expenses data
np.random.seed(42)  # for reproducibility
sales = np.random.uniform(1000, 5000, 100)
expenses = np.random.uniform(800, 4000, 100)

# Calculate profit
profit = sales - expenses

# Create the DataFrame
df = pd.DataFrame({
    'Date': dates,
    'Sales': sales,
    'Expenses': expenses,
    'Profit': profit
})

# Save the DataFrame as a CSV file
df.to_csv('test_data.csv', index=False)
print("Test data saved to test_data.csv")
