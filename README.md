# Improved-Fraud-Detection-for-E-commerce-and-Bank-Transactions


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
