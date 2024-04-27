# library doc string
""" This is the churn library.
    This library processes the user data and produces the images and models
    Author: Ashish Kumar Shukla
"""

# import libraries
import logging
import os
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


# Create a logger instance with a unique name
logger = logging.getLogger('my_logger')

# Set the logging level
logger.setLevel(logging.DEBUG)

# Create a FileHandler and set its filename
LOG_FILE = 'logs/churn_library.log'
file_handler = logging.FileHandler(LOG_FILE)

# Optionally, set the logging level for the FileHandler
file_handler.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set the formatter for the FileHandler
file_handler.setFormatter(formatter)

# Add the FileHandler to the logger
logger.addHandler(file_handler)

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            _df: pandas dataframe
    '''
    try:
        logger.info("Importing datat from : %s", pth)
        _df = pd.read_csv(pth)
        return _df
    except Exception as excep:
        logger.error(
            "Error occured while reading the data from %s: %s",
            pth,
            str(excep))
        return None


def perform_eda(_df):
    '''
    perform eda on _df and save figures to images folder
    input:
            _df: pandas dataframe

    output:
            None
    '''

    logger.info("Performing EDA")

    _df['Churn'] = _df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    columns = [
        "Churn",
        "Customer_Age",
        "Marital_Status",
        "Total_Trans_Ct",
        "Heatmap"]

    for column in columns:
        plt.figure(figsize=(20, 10))
        if column in ("Churn", "Customer_Age"):
            _df[column].hist()
        elif column == "Marital_Status":
            _df[column].value_counts('normalize').plot(kind='bar')
        elif column == "Total_Trans_Ct":
            sns.histplot(_df[column], stat='density', kde=True)
        elif column == "Heatmap":
            sns.heatmap(_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.title(column)
        plt.savefig(f"./images/eda/{column}.png")
        plt.close()


def encoder_helper(_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            _df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
            [optional argument that could be used for naming variables or index y column]

    output:
            _df: pandas dataframe with new columns for
    '''

    logger.info("Encoding data")

    for category in category_lst:
        # Calculate mean of response variable grouped by the category column
        category_churn_proportion = _df.groupby(category)[response].mean()

        # Map the category_churn_proportion to the category values in _df
        _df[category + "_" +
            response] = _df[category].map(category_churn_proportion)

    return _df


def perform_feature_engineering(_df, response):
    '''
    input:
              _df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    logger.info("Feature Engineering")

    y = _df[response]
    X = pd.DataFrame()
    encoder__df = encoder_helper(_df, cat_columns, response)
    X[keep_cols] = encoder__df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    logger.info("Generating Classification Reports")

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))

    # Set the title for the second subplot
    axs[0].set_title(
        'Random Forest Train',
        fontsize=10,
        fontproperties='monospace')
    # Display classification report for test data
    axs[0].text(0.5, 0.5, str(classification_report(y_train, y_train_preds_rf)),
                fontsize=10, fontproperties='monospace', ha='left', va='top')

    # Set the title for the first subplot
    axs[1].set_title(
        'Random Forest Test',
        fontsize=10,
        fontproperties='monospace')
    # Display classification report for training data
    axs[1].text(0.5, 0.5, str(classification_report(y_test, y_test_preds_rf)),
                fontsize=10, fontproperties='monospace', ha='left', va='top')

    # Hide the axes for both subplots
    for ax in axs:
        ax.axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(
        './images/results/random_forest_classifier.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(5, 8))

    # Set the title for the second subplot
    axs[0].set_title(
        'Logistic Regression Train',
        fontsize=10,
        fontproperties='monospace')
    # Display classification report for test data
    axs[0].text(0.5, 0.5, str(classification_report(y_test, y_test_preds_lr)),
                fontsize=10, fontproperties='monospace', ha='left', va='top')

    # Set the title for the first subplot
    axs[1].set_title(
        'Logistic Regression Test',
        fontsize=10,
        fontproperties='monospace')
    # Display classification report for training data
    axs[1].text(0.5, 0.5, str(classification_report(y_train, y_train_preds_lr)),
                fontsize=10, fontproperties='monospace', ha='left', va='top')

    # Hide the axes for both subplots
    for ax in axs:
        ax.axis('off')

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(
        './images/results/logistic_regression_classifier.png',
        dpi=300,
        bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    logger.info("Generating Feature Importance Reports")

    # Calculate feature importances
    importances = model.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    plt.close()


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    logger.info("Training Model")

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference:
    # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    logger.info("saving best model")
    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # plots
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lrc_plot = plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    # lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_plot2_result.png')
    plt.close

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_train,
        './images/results/feature_importance.png')


if __name__ == "__main__":

    # Import data
    raw_data_frame = import_data('./data/bank_data.csv')

    # Perform EDA
    perform_eda(raw_data_frame)

    # train test split
    X_train_data, X_test_data, y_train_data, y_test_data = perform_feature_engineering(
        raw_data_frame, response='Churn')

    # train models
    train_models(X_train_data, X_test_data, y_train_data, y_test_data)
