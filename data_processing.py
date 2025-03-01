import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

base_dir = Path(__file__)
data_dir = f"./data"

def path_exists(filepath):
    """ Check if the path exists """
    if os.path.exists(filepath):
        print("Path Exists!")
    else:
        print("Try Again")

def construct_dataset():
    """Construct a fake dataset to price the pure premium."""
    
    # Simulate dataset
    np.random.seed(42)
    n_samples = 1000

    # Predictor variables
    exposure = np.random.uniform(0.1, 1, size=n_samples)                                                # Policy exposure
    age = np.random.randint(18, 80, size=n_samples)                                                     # Age of policyholder
    vehicle_value = np.random.uniform(5000, 50000, size=n_samples)                                      # Value of the vehicle
    region = np.random.choice(['Urban', 'Suburban', 'Rural'], size=n_samples)                           # Region
    vehicle_type = np.random.choice(['Sedan', 'SUV', 'Truck', 'Sports'], size=n_samples)                # Vehicle type
    driving_experience = np.random.randint(0, 30, size=n_samples)                                       # Years of driving experience
    credit_score = np.random.randint(300, 850, size=n_samples)                                          # Credit score

    # Encode categorical variables for simplicity
    region_encoded = pd.get_dummies(region, drop_first=True)
    vehicle_type_encoded = pd.get_dummies(vehicle_type, drop_first=True)

    # True premium (target variable) with Tweedie distribution
    true_premium = (
                    100 
                    + 0.05 * vehicle_value 
                    + 0.3 * exposure 
                    + 0.2 * age 
                    - 0.1 * driving_experience 
                    + 0.01 * (850 - credit_score)                                                       # Higher credit score lowers premium
                    + region_encoded.mul([10, 20]).sum(axis=1)                                          # Region effect
                    + vehicle_type_encoded.mul([50, 30, 70]).sum(axis=1)                                # Vehicle type effect
    )
    
    # Simulate claims based on true premium
    claim_amount = np.random.poisson(lam=true_premium) * np.random.uniform(0, 1, n_samples)

    # Combine predictors into a DataFrame
    df = pd.DataFrame({
        'exposure': exposure,
        'age': age,
        'vehicle_value': vehicle_value,
        'driving_experience': driving_experience,
        'credit_score': credit_score,
        'region': region,
        'vehicle_type': vehicle_type,
        'claim_amount': claim_amount
    })
    
    # Save to CSV
    data_dir = "data"
    Path(data_dir).mkdir(exist_ok=True)
    df.to_csv(Path(data_dir) / "data.csv", sep="|", index=False)
    
    print(f"Dataset saved to {Path(data_dir) / 'data.csv'}")
    
def split_categorical_numerics(df):
    """
    Split the input dataframe into categorical and numerical dataframes.

    This function separates the columns of the input dataframe into two separate
    dataframes based on their data types: one for categorical data and another
    for numerical data.
    Parameters:
    df (pd.DataFrame): Input dataframe containing mixed data types.

    Returns:
    tuple: A tuple containing two dataframes:
        - categorical_df (pd.DataFrame): Dataframe containing only categorical columns
        - numerical_df (pd.DataFrame): Dataframe containing only numerical columns
    """
    
    numeric_features = [feature for feature in df.columns if df[feature].dtype != 'O']
    categorical_features = [feature for feature in df.columns if df[feature].dtype == 'O']

    return numeric_features, categorical_features

def calculate_vif(df):
    """
    Calculate Variance Inflation Factor (VIF) for each feature in a dataframe.

    Parameters:
        df (pd.DataFrame): Dataframe containing only numerical features.

    Returns:
        pd.DataFrame: Dataframe containing features and their corresponding VIF values.
    """
    vif_data = pd.DataFrame()
    vif_data["Feature"] = df.columns
    vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif_data

def scaling_predictions(df):
    """ Use the Central Limit Theorem to scale the predictors variables """

    nu = ['', '', '', '', '', '', '', '', '',] # list of the columns that will need to be scaled

    sc = StandardScaler()
    
    df_scaled = sc.fit_transform(df[nu])

    return df_scaled    
    
def apply_label_encoder():
    """ transform categorical data to numeric values """



