import pandas as pd

# Load cleaned dataset
df = pd.read_csv("ecogig_cleaned_data.csv")

# -------------------------------
# 1. Delay by Condition
# -------------------------------
print("Delay by Condition:\n", df.groupby('Condition')['Delay'].mean())

# -------------------------------
# 2. Delay by Vehicle
# -------------------------------
print("\nDelay by Vehicle:\n", df.groupby('Vehicle_Type')['Delay'].mean())

# -------------------------------
# 3. City Performance
# -------------------------------
print("\nDelay by City:\n", df.groupby('City')['Delay'].mean())

# -------------------------------
# 4. Delay vs Rating
# -------------------------------
print("\nCorrelation (Delay vs Rating):\n", df[['Delay','Customer_Rating']].corr())

# -------------------------------
# 5. Late Percentage
# -------------------------------
late_percent = (df['Delivery_Status'] == 'Late').mean() * 100
print("\nLate Delivery %:", late_percent)

# -------------------------------
# 6. Cost vs Distance
# -------------------------------
print("\nCost vs Distance Correlation:\n", df[['Distance_km','Delivery_Cost']].corr())