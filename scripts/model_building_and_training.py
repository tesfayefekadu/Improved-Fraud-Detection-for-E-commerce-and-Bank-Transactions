from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf


# Function to separate features and target for credit card data
def separate_features_target_creditcard(df):
    X = df.drop(columns=['Class'])  # Features
    y = df['Class']  # Target
    return X, y

# Function to separate features and target for fraud data
def separate_features_target_fraud(df):
    X = df.drop(columns=['class'])  # Features
    y = df['class']  # Target
    return X, y
# Function to split the data into train and test sets
def split_train_test(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Function to train and evaluate models
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model.__class__.__name__} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    return model