# **Design Document**: Transaction Fraud Detection

### Carina Carino

### 02/25/2024
___

## Motivation

___

> Identifying fraudulent transactions is crucial for mitigating financial losses, protecting personal information, and maintaining customers' trust.
It is imperative to accurately detect and prevent fraudulent activities especially as financial transactions become increasingly digital.
Traditional methods of fraud detection hinged on rule-based systems and have struggled to keep up with sophisticated fraud tactics. This project aims to develop and optimize a machine learning-based (ML) model that can identify fraudulent transactions with high precision and recall. 
Machine learning models have the capacity to analyze vast datasets and discern complex relationships and patterns, making it particularly well-suited to identifying fraud transactions.
While no system is perfect and some degree of error is inevitable, we recognize that we can tolerate false positives to a certain extent in pursuit of minimizing the number of false negatives. 
Model deployment is feasible by creating an API with Flask and testing it in Postman.
___

## Requirements
___
> Our primary goal is to develop a cutting-edge machine learning-based system capable of accurately detecting fraudulent transactions. 
The success of this project will be measured by the ability of our model to achieve high precision and recall in identifying fraudulent transactions, 
effectively minimizing false negatives to avoid overlooking genuine fraud cases while maintaining a tolerable level of false positives.

* #### System Assumptions
  * The system assumes that fraudulent transactions exhibit identifiable patterns that can be learned and recognized by a machine learning model.
  * It is assumed that transaction data are of sufficient quality for detailed analysis.

 * #### System Requirements
    * The system requires access to comprehensive and relevant transactional data for training.
      * Non-fraudulent
      * Fraudulent
    * The system also requires computational resource that can handle vast dataset.

 * ##### Possible Harms:
   * False Positives: Incorrectly flagging legitimate transactions as fraudulent can lead to customer dissatisfaction and operational overhead.
   * False Negatives: Failing to detect actual fraud allows fraudulent activities to succeed, resulting in financial losses for both the institution and its customers.
   * Privacy Concern: Analysis of large volumes of personal data raise privacy issues.
   * Bias and Discrimination: The system might exhibit discriminatory behavior if the training data is biased. 
 * ##### Causes:
   * Poor data quality
   * Overfitting or underfitting
   * Improper feature engineering
___

## Implementation

___
### Methods
* **Data Engineering Pipeline:** automates the extraction, transformation, and loading (ETL) process for data preprocessing in machine learning workflows.
  - `extract`: Reads CSV data.
  - `transform`: Processes data by:
    - Converting date strings to datetime objects.
    - Calculating ages and extracting date components.
    - Computing distances based on latitude and longitude.
    - Creating combined city-state columns.
    - Encoding categorical variables and applying one-hot encoding.
    - Dropping unnecessary columns for modeling.
  - `load`: Saves the transformed data to a CSV file.
  - `save_state` & `load_state`: Allows saving and loading of the pipeline's state (label encoders and one-hot columns) for consistent preprocessing across different sessions.
> The class encapsulates common data preprocessing steps, such as handling dates, calculating age, extracting date parts, calculating distances, encoding categorical variables, and dropping unnecessary columns, into separate methods. This modular design enhances code readability, maintainability, and reusability. By automating these steps, it streamlines the ETL process, reducing the likelihood of errors and ensuring consistency across different datasets and machine learning tasks.
___
* **Dataset Partitioning:**  designed for organizing and preparing a dataset for fraud detection models.
  - **Data Splitting:**
    - Automatically divides the dataset into training and test sets upon initialization.
    - Further splits the training set into actual training and validation sets if a validation size is specified.
  - **Data Retrieval Methods:**
    - `get_training_data()`: Retrieves the actual training data, excluding the validation subset.
    - `get_validation_data()`: Returns the validation dataset.
    - `get_testing_data()`: Provides the testing dataset.
    - `get_k_fold_data(k)`: Generates indices for performing k-fold cross-validation on the entire dataset.
> The class utilizes the `train_test_split function` from scikit-learn for initial data splitting and provides methods to access these splits as well as to generate indices for k-fold cross-validation using `KFold`.
> The class allows for the specification of a random seed (random_state), ensuring reproducibility across different runs. 
> This consistency is crucial for obtaining consistent results in machine learning experiments.
___
* **Metrics Pipeline:** designed for evaluating fraud detection model, using precision and recall.
  - **Metric Calculation Methods:**
    - `precision()`: Calculates the precision of predictions against actual labels.
    - `recall()`: Computes the recall of predictions against actual labels.
  - **Report Generation:**
    - `generate_report()`: Produces a report detailing precision and recall. The report is saved to a text file within the `results` directory.

  > Precision and recall were used to account for the disproportionate class distribution. 
    This imbalance in class distribution can make accuracy an unreliable metric, as a model can simply predict all transactions as non-fraudulent and achieve high accuracy but would fail to detect any fraudulent transactions.
    High recall minimizes false negatives,  or actual fraudulent transactions that were not correctly identified.
    High precision minimizes false positives, or legitimate transactions that were flagged as fraudulent.
    Balancing recall and precision ensures a robust fraud detection system capable of accurately identifying fraudulent transactions while minimizing unnecessary alerts for legitimate transactions.

### Deployment Strategy
* **Fraud Detector Model:** encapsulates the process of fraud detection model training, evaluation, and prediction. Flask facilitates end-to-end handling from data preprocessing to making predictions via an API.
  - **Data Preprocessing:** utilizes the `ETL_Pipeline` class for extracting and transforming data, ensuring it is in the correct format for model training and predictions.
  - **Model Training and Loading:**
    - `load_and_train_model`: Checks for an existing trained model and scaler; if found, loads them for immediate use. Else, it proceeds to train a new model using the provided dataset path.
    - `train()`: Trains the XGBoost classifier on the processed training data
    - `test()`: Evaluates the trained model on a separate test dataset.
  - **API Endpoint (`/predict`):** Offers a prediction endpoint that processes incoming JSON data through the ETL pipeline and model to return fraud detection predictions.
      - Returns a prediction indicating "Fraud" or "Not Fraud".
- **Local Testing:** Test the Flask application locally. Ensure the `/predict` endpoint correctly processes input data and returns the expected prediction.
- **Setting Up Postman for API Testing**
  - **Method:** `POST`
    - **URL:** `http://0.0.0.0:8786/predict`
      - **Body:** 
        - Select `raw`
        - Choose `JSON` from the drop down menu
        - Input the JSON you want to send. For example:
        ```
            {
                "trans_date_trans_time": "2022-10-12 14:32:21",
                "cc_num": 2703186189652095,
                "merchant": "fraud_Kilback LLC",
                "category": "shopping_net",
                "amt": 105.89,
                "first": "John",
                "last": "Doe",
                "sex": "M",
                "street": "123 Elm Street",
                "city": "Washington",
                "state": "DC",
                "zip": 62704,
                "lat": 39.7817,
                "long": -89.6508,
                "city_pop": 116250,
                "job": "IT trainer",
                "dob": "1990-05-19",
                "trans_num": "e9c2d8a2bb342bc446df5f578cddf8ac",
                "unix_time": 1634055141,
                "merch_lat": 39.7957,
                "merch_long": -89.6433
            }
        ```     
        - Hit `Send`
