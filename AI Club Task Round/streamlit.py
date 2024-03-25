import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Load the logistic regression model from the pickle file
with open('logistic_regression_model.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)

# Define a scaler for feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Function to make predictions
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Scale the input features
    scaled_features = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    # Predict the species
    prediction = logistic_regression_model.predict(scaled_features)
    # Decode the prediction
    species = iris.target_names[prediction[0]]
    return species

# Front-end layout
st.title('Iris Species Predictor')

# Input fields for user to enter feature values
sepal_length = st.slider('Sepal Length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.slider('Sepal Width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.slider('Petal Length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.slider('Petal Width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

# Predict button
if st.button('Predict'):
    # Make prediction
    species_prediction = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.write(f'Predicted Species: {species_prediction}')
