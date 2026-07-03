import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("ecogig_raw_data.csv")

# -------------------------------
# 1. Initial Inspection
# -------------------------------
print(df.info())
print(df.isnull().sum())

# -------------------------------
# 2. Handle Missing Values
# -------------------------------
df['Distance_km'] = df['Distance_km'].fillna(df['Distance_km'].median())
df['Customer_Rating'] = df['Customer_Rating'].fillna(df['Customer_Rating'].mean())

# -------------------------------
# 3. Clean Delivery Cost
# -------------------------------
df['Delivery_Cost'] = df['Delivery_Cost'].replace('[₹,]', '', regex=True).astype(float)

# -------------------------------
# 4. Clean and Standardize Date
# -------------------------------
df['Date'] = df['Date'].astype(str).str.strip()
df['Date'] = df['Date'].str.replace(r"[#']", "", regex=True)

# First pass (day-first)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

# Second pass (month-first for remaining)
mask = df['Date'].isnull()
df.loc[mask, 'Date'] = pd.to_datetime(df.loc[mask, 'Date'], errors='coerce', dayfirst=False)

# Fill remaining missing dates
df['Date'] = df['Date'].ffill()

# -------------------------------
# 5. Standardize Text Columns
# -------------------------------
df['City'] = df['City'].str.strip().str.lower()
df['Vehicle_Type'] = df['Vehicle_Type'].str.strip().str.title()
df['Weather_Condition'] = df['Weather_Condition'].str.strip().str.title()

# -------------------------------
# 6. Fix City Names
# -------------------------------
city_map = {
    'bom': 'Mumbai',
    'bombay': 'Mumbai',
    'mumbai': 'Mumbai',
    'bnglr': 'Bangalore',
    'bangalore': 'Bangalore',
    'bengaluru': 'Bangalore',
    'delhi': 'Delhi',
    'new delhi': 'Delhi'
}

df['City'] = df['City'].replace(city_map)
df['City'] = df['City'].str.title()

# -------------------------------
# 7. Rename Condition Column
# -------------------------------
df.rename(columns={'Weather_Condition': 'Condition'}, inplace=True)

# Merge similar categories
df['Condition'] = df['Condition'].replace({
    'Heavy Rain': 'Rain'
})

# -------------------------------
# 8. Remove Logical Errors
# -------------------------------
df = df[(df['Distance_km'] > 0) &
        (df['Estimated_Time_min'] > 0) &
        (df['Actual_Time_min'] > 0)]

df['Actual_Time_min'] = df['Actual_Time_min'].replace(999, np.nan)
df['Actual_Time_min'] = df['Actual_Time_min'].fillna(df['Actual_Time_min'].median())
# -------------------------------
# 9. Create Delay Feature
# -------------------------------
df['Delay'] = df['Actual_Time_min'] - df['Estimated_Time_min']

# Cap outliers instead of removing (better approach)
upper_limit = df['Delay'].quantile(0.99)
df['Delay'] = df['Delay'].clip(upper=upper_limit)

# -------------------------------
# 10. Delivery Status (3 categories)
# -------------------------------
df['Delivery_Status'] = df['Delay'].apply(
    lambda x: 'Late' if x > 0 else ('On-Time' if x == 0 else 'Early')
)

# -------------------------------
# 11. Time-based Features
# -------------------------------
df['Month'] = df['Date'].dt.month
df['Weekday'] = df['Date'].dt.day_name()

# -------------------------------
# 12. Remove Duplicates
# -------------------------------
df.drop_duplicates(inplace=True)

# -------------------------------
# 13. Final Checks
# -------------------------------
print(df.info())
print(df.isnull().sum())
print(df.describe())

print("Unique Cities:", df['City'].unique())
print("Unique Conditions:", df['Condition'].unique())
print("Delivery Status Distribution:\n", df['Delivery_Status'].value_counts())

# -------------------------------
# 14. Save Cleaned Dataset
# -------------------------------
df.to_csv("ecogig_cleaned_data.csv", index=False)