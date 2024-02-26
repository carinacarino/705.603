# Case Satudy: Transaction Fraud Detection
___
This project aims to design, develop, and deploy a machine learning based model for detecting fraudulent transactions. 
___
## Features:
  - Data set extraction, transformaion, & loading
  - Data splitting into train, test, and/or validate
  - Predict fraud in transaction data using `XGBoost` classifier.
  - Evaluate model using `Precision` and `Recall`
  - Provides  a `/predict` endpoint in the Flask app for making predictions
  - - Returns a prediction indicating "Fraud" or "Not Fraud".

## Libraries
All required libraries are listed on requirements.txt

## How to use
### Creating docker image
```
docker buildx build -t ""<docker_user_name>/705.603:Fraud_Detection_1" --platform linux/amd64,linux/arm64 --push .
```
### Local test
```
docker run -v <host directory>:/output ccarino/705.603:Fraud_Detection_1
```
### Setting Up Postman for API Testing
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


  

  
