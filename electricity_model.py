import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Load the data
df = pd.read_csv('electricity_data.csv')

# 2. Reshape data for the model
X = df[['Square_Footage']]
y = df['Monthly_kWh']

# 3. Create and train the model
model = LinearRegression()
model.fit(X, y)

# 4. Predict for a new house (e.g., 2000 sq ft)
prediction = model.predict([[2000]])
print(f"Predicted consumption for 2000 sq ft: {prediction[0]:.2f} kWh")