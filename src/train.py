from sklearn.datasets import fetch_california_housing
import pandas as pd

# 1. Load the dataset
data = fetch_california_housing()

# 2. Convert it into a Pandas DataFrame so it's easy to read
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target # This is our "Target" (the price)

# 3. Print the first 5 rows to see what we are working with
print("Dataset loaded successfully!")
print(df.head())

# 4. Check if there are any missing values
print("\n--- Missing Values Check ---")
print(df.isnull().sum())

# 5. Get a statistical summary
print("\n--- Statistical Summary ---")
print(df.describe())

# 6. See how features correlate with the house value
correlations = df.corr()
print("\n--- Correlation with House Value ---")
print(correlations['MedHouseVal'].sort_values(ascending=False))
import matplotlib.pyplot as plt

# 7. Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['MedInc'], df['MedHouseVal'], alpha=0.5)
plt.title('Income vs House Value')
plt.xlabel('Median Income')
plt.ylabel('House Value')
plt.show()

from sklearn.model_selection import train_test_split

# 8. Define X (The Features/Inputs) and y (The Target/Price)
# We drop the price column from X because we want the model to guess it
X = df.drop('MedHouseVal', axis=1) 
y = df['MedHouseVal']

# 9. Split the data
# test_size=0.2 means 20% goes to the 'Exam' (Testing set)
# random_state=42 ensures the split is the same every time you run the code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n--- Step 3: Data Split Successful ---")
print(f"Total rows in dataset: {len(df)}")
print(f"Rows for Training (80%): {X_train.shape[0]}")
print(f"Rows for Testing  (20%): {X_test.shape[0]}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 10. Initialize the model
model = LinearRegression()

# 11. Train the model (this is where the "learning" happens!)
model.fit(X_train, y_train)

print("\n--- Step 4: Model Training Complete ---")

# 12. Make predictions on the test set
y_pred = model.predict(X_test)

# 13. Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared Score: {r2:.4f}")

from sklearn.linear_model import Ridge

# 14. Initialize and train the Ridge model
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# 15. Evaluate Ridge
ridge_pred = ridge_model.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)

print(f"\n--- Step 6: Ridge Regression ---")
print(f"Ridge R-squared Score: {ridge_r2:.4f}")

from sklearn.linear_model import Lasso

# 16. Initialize and train the Lasso model
# alpha is the 'penalty' strength. 0.1 is a standard starting point.
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# 17. Evaluate Lasso
lasso_pred = lasso_model.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_pred)

print(f"\n--- Step 17: Lasso Regression ---")
print(f"Lasso R-squared Score: {lasso_r2:.4f}")

import joblib
import os

# 18. Create a folder to store your saved models
os.makedirs('models', exist_ok=True)

# 19. Save the Ridge model (our current leader)
# We use .pkl (pickle) format to save the trained 'brain' of the model
joblib.dump(ridge_model, 'models/house_price_model.pkl')

# 20. Save the list of feature names
# The web app needs to know the exact order of columns (MedInc, HouseAge, etc.)
joblib.dump(X.columns.tolist(), 'models/features.pkl')

print("\n--- Step 20: Model Saved Successfully ---")