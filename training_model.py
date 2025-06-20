import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor #Stochastic Gradient Descent
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

# Sample dataset
df = pd.read_csv('sales_data.csv')
df.sort_values(by='Units Sold', inplace=True)
df.drop(['Store ID', 'Product ID'], axis=1, inplace=True)

"""
df["Date"]=pd.to_datetime(df["Date"])
df['Weekday'] = df['Date'].dt.day_of_week
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
"""

print(df.head())
print(df.describe())
print(df.isnull().sum())

df = df.select_dtypes(include=['number'])

print(df.cov())
print(df.corr())

# Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Heatmap")
plt.show()

plt.scatter(df['Price'], df['Demand'], color='blue')
plt.xlabel('Price of the Product')
plt.ylabel('Demand of the Product')
plt.title('Price vs Demand')
plt.grid(True)
plt.show()


# Split data
y = df['Demand']
#X = df[['Units Sold']]
X = df.drop('Demand', axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01, random_state=42))
])
pipeline.fit(X_train, y_train)

# Save pipeline
joblib.dump(pipeline, 'pipeline_model.pkl')

from sklearn.metrics import mean_squared_error, r2_score
loaded_pipeline = joblib.load('pipeline_model.pkl')
y_test_pred = loaded_pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print("MSE:", mse)
print("R² Score:", r2)

