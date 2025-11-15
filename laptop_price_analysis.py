# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate

# Show all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load dataset
data = pd.read_csv('laptop.csv')

# Display as table (first 10 rows)
print(tabulate(data.head(10), headers='keys', tablefmt='psql'))

#  Data Preprocessing and Feature Engineering
# Clean and extract numerical RAM from 'Ram' column
data['Ram_GB'] = data['Ram'].str.extract(r'(\d+)').astype(float)
# Convert ROM to storage in GB, handling TB values
def convert_storage(rom):
    if pd.isnull(rom): return np.nan
    rom_str = str(rom)
    if 'TB' in rom_str:
        return float(rom_str.replace('TB', '').strip()) * 1024
    return float(rom_str.replace('GB', '').strip())

data['Storage_GB'] = data['ROM'].apply(convert_storage)

# Convert price to numeric (handle errors)
data['price'] = pd.to_numeric(data['price'], errors='coerce')

# Convert categorical columns to string type
data['brand'] = data['brand'].astype(str)
data['GPU'] = data['GPU'].astype(str)
data['OS'] = data['OS'].astype(str)
data['spec_rating'] = pd.to_numeric(data['spec_rating'], errors='coerce')
# Drop rows missing key features
data = data.dropna(subset=['price', 'Ram_GB', 'Storage_GB', 'brand', 'GPU', 'OS', 'spec_rating'])

# Exploratory Data Analysis (EDA)
# Plot distribution of prices
sns.histplot(data['price'], bins=30, kde=True)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Count')
plt.show()

# Boxplot price vs brand
plt.figure(figsize=(10,6))
sns.boxplot(x='brand', y='price', data=data)
plt.xticks(rotation=45)
plt.title('Price by Brand')
plt.show()

# Scatter plot RAM vs price
sns.scatterplot(x='Ram_GB', y='price', data=data, hue='brand')
plt.title('Price vs RAM')
plt.show()

# Scatter plot storage vs price
sns.scatterplot(x='Storage_GB', y='price', data=data, hue='brand')
plt.title('Price vs Storage')
plt.show()

# Correlation heatmap of numerical features
sns.heatmap(data[['price', 'Ram_GB', 'Storage_GB']].corr(), annot=True)
plt.title('Feature Correlation')
plt.show()

# Model Training and Evaluation
# Prepare features and target
features_to_use = ['Ram_GB', 'Storage_GB', 'brand', 'GPU', 'OS', 'spec_rating']
X = data[features_to_use]
X = pd.get_dummies(X, columns=['brand', 'GPU', 'OS'], drop_first=True)
y = data['price']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot Actual vs Predicted
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Pick a real row for sanity test
test_row = data.iloc[0]
print("\nSanity check - Predicting actual row from dataset")
print("Actual price:", test_row['price'])

input_dict = {col: [test_row[col]] for col in features_to_use}
input_df = pd.DataFrame(input_dict)
input_df = pd.get_dummies(input_df).reindex(columns=X.columns, fill_value=0)
print("Input feature vector:")
print(input_df)
print("Predicted price:", model.predict(input_df)[0])

#  Print Feature Importances
importances = model.feature_importances_
print("\nFeature importances:")
for col, imp in zip(X.columns, importances):
    print(f"{col}: {imp}")

#  User Input Prediction
brands = sorted(data['brand'].unique())
gpus = sorted(data['GPU'].unique())
oses = sorted(data['OS'].unique())
ram_min, ram_max = data['Ram_GB'].min(), data['Ram_GB'].max()
storage_min, storage_max = data['Storage_GB'].min(), data['Storage_GB'].max()
spec_rating_min, spec_rating_max = data['spec_rating'].min(), data['spec_rating'].max()

print("\n--- Laptop Price Predictor ---")
print(f"RAM range: {ram_min} - {ram_max} GB")
print(f"Storage range: {storage_min} - {storage_max} GB")
print(f"Spec rating range: {spec_rating_min} - {spec_rating_max}")
print("Available brands:", brands)
print("Available GPUs:", gpus)
print("Available OSes:", oses)

ram_gb = float(input("Enter RAM size (GB): "))
storage_gb = float(input("Enter Storage size (GB): "))
spec_rating = float(input("Enter spec rating: "))
while True:
    brand = input("Enter brand: ")
    if brand in brands: break
    print("Brand not found.")
while True:
    gpu = input("Enter GPU: ")
    if gpu in gpus: break
    print("GPU not found.")
while True:
    os = input("Enter OS: ")
    if os in oses: break
    print("OS not found.")

user_dict = {
    'Ram_GB': [ram_gb],
    'Storage_GB': [storage_gb],
    'spec_rating': [spec_rating],
    'brand': [brand],
    'GPU': [gpu],
    'OS': [os]
}
user_df = pd.DataFrame(user_dict)
user_df = pd.get_dummies(user_df).reindex(columns=X.columns, fill_value=0)
print("\nUser input feature vector:")
print(user_df)
user_pred = model.predict(user_df)[0]
print(f"\nPredicted Price: {user_pred:,.2f}")
