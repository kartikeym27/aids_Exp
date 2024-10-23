# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (replace with your retail dataset)
# Example dataset: 'sales_data.csv'
# The dataset should include features like store size, number of products, promotion, etc.
df = pd.read_csv('sales_data.csv')

# Display the first few rows
df.head()

# Data Preprocessing
# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)

# Feature selection (choosing relevant features)
X = df[['Store_Size', 'Number_of_Products', 'Promotion', 'Holiday_Season']]  # Example features
y = df['Sales']  # Target variable

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a Neural Network
model = Sequential()

# Adding the input layer and first hidden layer
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))

# Adding second hidden layer
model.add(Dense(units=64, activation='relu'))

# Adding the output layer
model.add(Dense(units=1, activation='linear'))

# Compile the neural network
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Plot training history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R2 Score: {r2}')

# Visualize predictions vs actual
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()
