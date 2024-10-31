# Improved Fraud Detection for E-commerce and Bank Transactions

This project focuses on detecting fraudulent transactions in e-commerce and banking environments using machine learning. It incorporates model training, explainability, deployment, API development, Docker containerization, and a dashboard for visualization and monitoring.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Sources](#data-sources)
3. [Project Structure](#project-structure)
4. [Environment Setup](#environment-setup)
5. [Model Training and Explainability](#model-training-and-explainability)
6. [API Development and Deployment](#api-development-and-deployment)
7. [Dashboard](#dashboard)
8. [Usage](#usage)
9. [Future Work](#future-work)

---

## Project Overview

The project uses machine learning models to detect fraud in:
- **E-commerce Transactions**
- **Bank Transactions**

Key features include:
- **Data Cleaning and Feature Engineering**
- **Model Training and Evaluation**
- **Explainability (using SHAP and LIME)**
- **API Development and Deployment with Docker**
- **Dashboard for Real-Time Monitoring and Visualization**

---

## Data Sources

- **E-commerce transactions** from `Fraud_Data.csv`
- **Bank transactions** from `creditcard.csv`
- **IP to Country Mapping** from `IpAddress_to_Country.csv`

---

## Project Structure

```plaintext
.
├── data                    # Data files
├── notebooks               # Jupyter notebooks for model training and explainability
├── app                     # Flask app for model deployment
│   ├── serve_model.py      # Flask application file
│   ├── Dockerfile          # Dockerfile for containerizing the Flask app
│   └── requirements.txt    # Python dependencies
├── dashboard               # Code for the dashboard
└── README.md               # Project documentation
```

## Environment Setup
\`\`\`
git clone https://github.com/tesfayefekadu/Improved-Fraud-Detection-for-E-commerce-and-Bank-Transactions

cd Improved-Fraud-Detection-for-E-commerce-and-Bank-Transactions
\`\`\`

### Set Up Virtual Environment
\`\`\`
python -m venv venv
source venv/bin/activate  # On Linux/MacOS
venv\Scriptsctivate  # On Windows
\`\`\`

### Install Dependencies
\`\`\`
pip install -r requirements.txt
\`\`\`

## Model Training and Explainability
## Data Preprocessing
### Data Cleaning: 
Handle missing values, duplicates, and incorrect entries.
### Feature Engineering: 
Create features relevant to fraud detection, e.g., time_to_purchase, hour_of_day, etc.
## Model Training
Run the Jupyter notebooks in the notebooks directory to:

1,Load and clean the data.

2,Train multiple models such as Decision Tree, Random Forest, MLP, etc.

3.Save trained models in .pkl files for deployment.

### Explainability
SHAP: Generates summary plots, force plots, and dependence plots for understanding model decisions.
LIME: Explains individual predictions with feature importance.

## API Development and Deployment
### Setting Up the Flask API
#### Create the Flask App

In app/serve_model.py, the API exposes endpoints for model predictions.

Define API Endpoints

/predict: Receives transaction data and returns a fraud prediction.
/explain: Provides SHAP or LIME explanations for a given instance.
Run the Flask API

Copy code
flask run --port 5000

## Dashboard
The dashboard provides visual insights into the data, model performance, and incoming predictions. It uses charts and tables to monitor trends, fraud detection patterns, and system metrics.

Starting the Dashboard
Ensure dependencies are installed.
Run the dashboard application

\`\`\`
python dashboard/app.py
\`\`\`

## Future Work
1,Extend Model Explainability for new models or additional SHAP plots.

2,Dashboard Enhancements to include real-time fraud detection insights.

3,Integrate MLOps for model versioning and tracking improvements over time.

## Acknowledgments
This project is part of an AI Mastery Training Program with Kifiya AI, focusing on real-world data science and machine learning applications in fraud detection.

