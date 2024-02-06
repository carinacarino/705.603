# ML_Microservice_Example
This microservice serves as an example of leveraging machine learning to enhance decision-making in targeted advertising campaigns. It predicts whether an individual is a good candidate for advertising based on their age and salary. The prediction model is built on a **Random Forest** algorithm, trained using data from the 'Social_Network_Ads.csv' dataset.

Built with Flask, this microservice offers a simple web interface for interacting with the prediction model.
https://flask.palletsprojects.com/en/3.0.x/

## Features
- **Performance Stats**: Access model performance statistics via `/stats`.
- **Inference Determination**: Make predictions by submitting age and salary parameters through `/infer`.


Required libraries are in requirements.txt
