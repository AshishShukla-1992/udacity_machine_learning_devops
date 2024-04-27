# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

<br/>

# Project Description
Develop a python package adhering to coding standards(PEP8) and best practices. This package trains ML models to identify credit card customers who are at high risk of churn. Random Forest and Logistic regression techinques are used to create the ML models. Framework used for development,testing and validation of ML models is sklearn. This project demonstrates use of clean coding practices.

<br/>

# Files and Data Description

Root Directory Structure:

- Python files
    - churn_library.py --> code for developing and running churn ml model
    - churn_script_logging_and_tests.py --> pytest file to validate churn_library

- Folders
    - data --> Raw data files
    - images
        - eda --> Plots of Exploratory Data Analysis from raw data
        - results --> Plots of evaluation metrics
    - models --> Trained ML models in pkl format
    - logs --> logs of churn_library and pytest

<br/>

# Running the files

Create virtual environment:

```bash
conda create -n udacity python=3.6.3
```

Install Python dependencies

1. activate virtual environment
   `conda activate udacity`

2. run `pip install -r requirements.txt` 

<br>

To run the scripts

```bash
python churn_library.py
```

```bash
pytest churn_script_logging_and_tests.py
```
