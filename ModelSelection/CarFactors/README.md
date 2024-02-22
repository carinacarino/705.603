# CarFactors Service

This is a Flask-based microservice for predicting the duration a car will be listed based on various factors.


### Install dependencies:
```sh
pip install -r requirements.txt
```

### Run Flask application:
```sh
python carfactors_service.py
```

### **API Endpoints**
### GET /stats: Get statistics about the trained model.
### GET /infer: Make a prediction about the duration a car will be listed based on specified 

### Example usage using Postman
```sh
# Get statistics
Set the request type to GET.
Enter the url:
http://localhost:8786/stats
```

```sh
# Make a prediction
Set the request type to GET.
Enter the url:
"http://localhost:8786/infer?transmission=automatic&color=red&odometer=85000&year=2012&bodytype=sedan&price=15000"
```