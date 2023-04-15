# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the flight price data
flight_data = pd.read_csv('flight_data.csv')

# Preprocess the data
# Drop unnecessary columns
flight_data.drop(['flight_number', 'flight_name'], axis=1, inplace=True)
# Convert the date column to datetime format
flight_data['date'] = pd.to_datetime(flight_data['date'])
# Extract the day, month, and year from the date column
flight_data['day'] = flight_data['date'].dt.day
flight_data['month'] = flight_data['date'].dt.month
flight_data['year'] = flight_data['date'].dt.year
# Encode the categorical columns using one-hot encoding
flight_data = pd.get_dummies(flight_data, columns=['airline', 'source', 'destination'])

# Split the data into training and testing sets
X = flight_data.drop('price', axis=1)
y = flight_data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print('Root mean squared error:', rmse)

# Make predictions on new data
new_data = pd.DataFrame({
    'date': ['2023-06-01'],
    'airline': ['Indigo'],
    'source': ['Delhi'],
    'destination': ['Mumbai'],
    'day': [1],
    'month': [6],
    'year': [2023]
})
new_data['date'] = pd.to_datetime(new_data['date'])
new_data['day'] = new_data['date'].dt.day
new_data['month'] = new_data['date'].dt.month
new_data['year'] = new_data['date'].dt.year
new_data = pd.get_dummies(new_data, columns=['airline', 'source', 'destination'])
new_price = model.predict(new_data)[0]
print('Predicted flight price:', new_price)
