import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request, jsonify

train = pd.read_csv(r'train.csv')

# Assuming your dataset is stored in a pandas DataFrame called 'data'
# X contains the features (independent variables) and y contains the target variable (dependent variable)
X_train = train[['season', 'holiday', 'temp', 'atemp', 'humidity', 'windspeed']]
y_train = train['count']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training accuracy: {train_score}")
print(f"Testing accuracy: {test_score}")

new = Flask(__name__, template_folder='views')


@new.route('/')
def hello_world():
    return render_template('pn.html')


@new.route('/predict', methods=['POST', 'GET'])
def predict():
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400
    # Get the form data
    form_data = request.json

    # Extract the input values
    try:
        # Convert numerical inputs to float
        temperature = int(form_data['temp'])
        atm_temperature = int(form_data['a_temp'])
        humidity = int(form_data['humidity'])
        wind_speed = float(form_data['wind'])

        # Map categorical inputs to numerical values
        season_mapping = {'summer': 1, 'spring': 2, 'winter': 3, 'rainy': 4}
        season = season_mapping.get(form_data['seasons'], 0)  # Default to 0 if season not found

        day_mapping = {'holiday': 1,'work': 0}
        day = day_mapping.get(form_data['day'], 0)  # Default to 0 if day not found
    except ValueError:
        return jsonify({'error': 'Invalid input data. Ensure all numeric values are properly formatted'}), 400

    input_data = pd.DataFrame({
        'season': [season],
        'holiday': [day],
        'temp': [temperature],
        'atemp': [atm_temperature],
        'humidity': [humidity],
        'windspeed': [wind_speed]
    })
    # Perform prediction using your machine learning model
    prediction = model.predict(input_data)

    # Convert the prediction to a JSON serializable format
    prediction = prediction.tolist()  # Convert ndarray to list

    # Return the prediction as JSON
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    new.debug = True
    new.run()
