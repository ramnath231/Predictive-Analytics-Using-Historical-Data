import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("sales_data.csv")

# Convert date
df['Date'] = pd.to_datetime(df['Date'])

# Create month number
df['Month_Number'] = range(1, len(df) + 1)

# Features and target
X = df[['Month_Number']]
y = df['Sales']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Future prediction
future_months = np.array([[13], [14], [15], [16], [17], [18]])

future_sales = model.predict(future_months)

print("\nFuture Sales Forecast:")

for month, sale in zip(range(13, 19), future_sales):
    print(f"Month {month}: {sale:.2f}")

# Historical graph
plt.figure(figsize=(8,5))
plt.plot(df['Month_Number'], df['Sales'])
plt.title("Historical Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()

# Forecast graph
plt.figure(figsize=(8,5))
plt.plot(df['Month_Number'], df['Sales'], label="Historical")

plt.plot(
    range(13,19),
    future_sales,
    label="Forecast"
)

plt.legend()
plt.title("Future Sales Prediction")
plt.show()