import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import TweedieRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_processing import *

import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

"""
Model File for Analysing the results for premium pricing using the tweedie distribution

"""

base_dir = Path(__file__)
data_dir = f"./data"


def load_df():
    """ Load the processed dataframe. This will eventually come from the data_processing.py module """
    return pd.read_csv(os.path.join(data_dir, 'data.csv'), delimiter="|")

def glm_model(df):
    """ Tweedie Distribution """

    # Split into training and testing sets
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit Tweedie Regressor with power=1.5 (Compound Poisson-Gamma)
    tweedie_model = TweedieRegressor(power=1.5, alpha=0.1, link='log', max_iter=1000)
    tweedie_model.fit(X_train, y_train)

    # Predict on test data
    y_pred = tweedie_model.predict(X_test)

    # Evaluate model performance
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-Squared (R2): {r2:.2f}")

    # Plot actual vs predicted claims
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, max(y_test)], [0, max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
    plt.xlabel("Actual Claim Amounts")
    plt.ylabel("Predicted Claim Amounts")
    plt.title("Actual vs Predicted Claims (Tweedie Regressor)")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    df = load_df()
    
    glm_model(df)

