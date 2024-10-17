import pandas as pd

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

def correct_data_types(df):
    """Correct data types for fraud dataset (especially timestamps)."""
    # Convert signup_time and purchase_time to datetime
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
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


def convert_ip_to_int(ip_df):
    """Converts IP address ranges to integer for merging."""
    ip_df['lower_bound_ip_address'] = ip_df['lower_bound_ip_address'].astype(int)
    ip_df['upper_bound_ip_address'] = ip_df['upper_bound_ip_address'].astype(int)
    return ip_df

def merge_ip_country(fraud_df, ip_df):
    """Merges fraud data with IP address to country mapping using IP address range."""
    # Ensure ip_address is integer
    fraud_df['ip_address'] = fraud_df['ip_address'].astype(int)
    
    # Perform the merge using a condition for matching IP address ranges
    merged_df = pd.merge_asof(fraud_df.sort_values('ip_address'), 
                              ip_df.sort_values('lower_bound_ip_address'),
                              left_on='ip_address', 
                              right_on='lower_bound_ip_address', 
                              direction='backward')
    
    return merged_df


def feature_engineering(fraud_df):
    """Creates additional features for fraud detection."""
    # Calculate time differences between signup and purchase
    fraud_df['time_to_purchase'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds()
    
    # Create time-based features
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    return fraud_df

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