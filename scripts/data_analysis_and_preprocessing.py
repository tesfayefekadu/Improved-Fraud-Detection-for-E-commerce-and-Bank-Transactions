import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import socket
import struct

def handle_missing_values(df):
    """Handles missing values in the DataFrame by filling or dropping."""
    # Check missing values
    missing_values = df.isnull().sum()
    # For now, drop rows with missing values (can be modified for specific imputation)
    df_cleaned = df.dropna()
    return df_cleaned, missing_values

def remove_duplicates(df):
    """Removes duplicate rows in the DataFrame."""
    return df.drop_duplicates()

import pandas as pd

def correct_data_types(df):
    # Convert signup_time and purchase_time to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')

    # Convert ip_address to integer (if applicable)
    df['ip_address'] = df['ip_address'].fillna(0).astype(int)

    # Ensure age and purchase_value are integers (if required)
    df['age'] = df['age'].fillna(0).astype(int)
    df['purchase_value'] = df['purchase_value'].fillna(0).astype(int)

    # Convert categorical features (device_id, source, browser, sex) to category types for optimization
    df['device_id'] = df['device_id'].astype('category')
    df['source'] = df['source'].astype('category')
    df['browser'] = df['browser'].astype('category')
    df['sex'] = df['sex'].astype('category')

    return df


def univariate_analysis(df):
    """Performs univariate analysis such as histograms or count plots."""
    summary_stats = df.describe()  # Summary statistics
    return summary_stats

def bivariate_analysis(df):
    """Performs bivariate analysis between different features and the target."""
    # Select only numeric columns for correlation analysis
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    # Calculate correlation matrix for numeric columns only
    correlation_matrix = numeric_df.corr()
    
    return correlation_matrix


# Function to plot the correlation matrix as a heatmap
def plot_correlation_matrix(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])

    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()

    plt.figure(figsize=(12, 8))
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def convert_ip_to_int(ip_df):
    """Convert lower_bound_ip_address and upper_bound_ip_address to integer format."""
    # Convert the lower and upper IP address columns to integers
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(float).astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(float).astype(int)
    
    return ip_df

def merge_ip_country(fraud_df, ip_df):
    """Merge fraud data with IP address ranges based on IP address range matching."""
    # Ensure fraud_df's ip_address is integer
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)
    
    # Merge by checking if fraud_df's ip_address is between lower and upper bound
    merged_df = pd.merge_asof(
        fraud_df.sort_values('ip_address'), 
        ip_df.sort_values('lower_bound_ip_address'),
        left_on='ip_address',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Check that the ip_address is within the range of the matched lower and upper bounds
    merged_df = merged_df[(merged_df['ip_address'] >= merged_df['lower_bound_ip_address']) & 
                          (merged_df['ip_address'] <= merged_df['upper_bound_ip_address'])]
    
    return merged_df


# Function to perform feature engineering
def feature_engineering(df):
    # Calculate time-to-purchase in seconds
    df['time_to_purchase'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds()

    # Extract time-based features
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek

    # Transaction frequency and velocity
    # Transaction frequency: number of transactions per user
    df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')

    # Transaction velocity: purchase value divided by time to purchase
    df['transaction_velocity'] = df['purchase_value'] / df['time_to_purchase'].replace(0, np.nan)

    return df

def encode_categorical_features(df):
    """Encodes categorical features into numerical values."""
    df_encoded = pd.get_dummies(df, columns=['source', 'browser', 'sex'])
    return df_encoded

def normalize_and_scale(df, feature_columns):
    """Normalizes and scales selected features."""
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    df[feature_columns] = scaler.fit_transform(df[feature_columns])
    return df