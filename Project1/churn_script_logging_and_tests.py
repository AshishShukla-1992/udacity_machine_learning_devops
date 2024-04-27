# unit_tests doc string
""" This module is used to test the churn library.
    There are set of test cases which need to pass, it is used for validation.
    Author: Ashish Kumar Shukla
"""

import os
import logging
import pytest
import churn_library as cls

# Create a logger instance with a unique name
logger = logging.getLogger('my_logger_2')

# Set the logging level
logger.setLevel(logging.DEBUG)

# Create a FileHandler and set its filename
LOG_FILE = 'logs/unit_test.log'
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


cat_columns_churn = [
    'Gender_Churn',
    'Education_Level_Churn',
    'Marital_Status_Churn',
    'Income_Category_Churn',
    'Card_Category_Churn'
]

cat_columns = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]


@pytest.fixture(scope="module")
def raw_data():
    '''
    raw data import fixture
    '''
    try:
        path = "./data/bank_data.csv"
        _df = cls.import_data(path)
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    return _df


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        _df = cls.import_data("./data/bank_data.csv")
        logger.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert _df.shape[0] > 0
        assert _df.shape[1] > 0
    except AssertionError as err:
        logger.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(raw_data):
    '''
    test perform eda function
    '''
    cls.perform_eda(raw_data)
    image_file_names = [
        'Churn.png',
        'Customer_Age.png',
        'Marital_Status.png',
        'Total_Trans_Ct.png',
        'Heatmap.png']
    try:
        for file in image_file_names:
            file_path = './images/eda/' + file
            assert os.path.exists(file_path) is True

        logger.info("Testing EDA: SUCCESS")
    except AssertionError as err:
        logger.error(
            "Testing EDA: The plots don't exists")
        raise err


def test_encoder_helper(raw_data):
    '''
    test encoder helper
    '''
    try:
        encoded_df = cls.encoder_helper(raw_data, cat_columns, 'Churn')
        assert set(cat_columns_churn).issubset(
            set(encoded_df.columns.tolist()))
        logger.info("Encoder Helper Test Success")
    except AssertionError as assertion_error:
        logger.error("Encoder Helper Test : %s", assertion_error)
        raise assertion_error


def test_perform_feature_engineering(raw_data):
    '''
    test perform_feature_engineering
    '''
    try:
        x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
            raw_data, response='Churn')
        assert x_train.shape[0] > 0 and x_test.shape[0] > 0
        assert len(y_train) > 0 and len(y_test) > 0
        logger.info("Feature Engineering Test Successful")
    except AssertionError as exception:
        logger.error("Feature Engineering Test Failed : %s", exception)
        raise exception

    return x_train, x_test, y_train, y_test


def test_train_models(raw_data):
    '''
    test train_models
    '''

    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        raw_data, response='Churn')

    cls.train_models(x_train, x_test, y_train, y_test)

    image_file_names = [
        'feature_importance.png',
        'logistic_regression_classifier.png',
        'random_forest_classifier.png',
        'roc_plot2_result.png']
    model_file_names = ['logistic_model.pkl', 'rfc_model.pkl']
    try:
        for file in image_file_names:
            file_path = './images/results/' + file
            assert os.path.exists(file_path) is True

        for file in model_file_names:
            file_path = './models/' + file
            assert os.path.exists(file_path) is True

        logger.info("Testing Train Models SUCCESS")
    except AssertionError as err:
        logger.error(
            "Testing Trains Model Failed: The plots don't exists")
        raise err


if __name__ == "__main__":
    test_import()
    test_eda(raw_data)
    test_encoder_helper(raw_data)
    test_perform_feature_engineering(raw_data)
    test_train_models(raw_data)
