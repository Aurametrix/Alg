from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Separate predictors (X) and target variable (y)
X = data.drop(columns=['HowSat'])
y = data['HowSat']

# Convert categorical columns to numerical if necessary and handle missing values
X = pd.get_dummies(X, drop_first=True)
X = X.fillna(X.median())

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Apply LassoCV for feature selection
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

# Extract the selected features
selected_features = X.columns[(lasso.coef_ != 0)]

selected_features



import matplotlib.pyplot as plt

# Visualize the selected features
plt.figure(figsize=(10, 8))
plt.barh(selected_features, lasso.coef_[lasso.coef_ != 0])
plt.xlabel('Coefficient Value')
plt.ylabel('Selected Features')
plt.title('Selected Features from Lasso Regression')
plt.grid(True)
plt.tight_layout()
plt.show()
