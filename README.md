# Python-TensorFlow-Crypto-Prediction
The primary objective is to use TensorFlow to create trained neural network models that will predict future cryptocurrency values.
Secondary objective is to work on code organization and separation of configuration files.

# DO NOT USE THIS CODE FOR ACTUAL TRADING

## This is a proof of concept project
This project is a proof of concept and is not intended to be used for actual trading.  
The code is not optimized for speed or accuracy.  
The code is not optimized for memory usage.  
The code is not optimized for security.  
The code is not optimized for anything.

## Prerequisites
1. Python 3.6
2. TensorFlow
3. Keras
4. Pandas
5. Numpy
6. Matplotlib

## Getting Started
1. Clone the repository
2. Install the required packages
3. Change the config.json using the connection string to your database
4. Run the code
5. View the results


# Execution
This project integrates a bigger solution that involves the Blazor ML.NET for Crypto Analysis project.
Here is the GitHub link to that project: https://github.com/leandroamorimlagoa/blazor-MLNet-cripto-analysis
This service have a background service responsible for getting the data from the API and saving it to a database.


## Training
Run `python crypto-model-training.py`
For each crypto model trained, a trained model file will be created at the "models" folder.
The file name will be the crypto symbol at the end with 3 letters, like for Bitcoin: "crypto-model-btc.h5"

## Prediction
Run `python crypto-prediction.py`
A matplotlib graph will be created for each crypto symbol at the "graphs" folder.

# Future Improvements
The prediction should register the results at the database, so the Blazor ML.NET for Crypto Analysis project can use the results to decide what are the best options to invest in a show period of time

The models gerenated here will be executed by ML.NET, so the Blazor ML.NET for Crypto Analysis project.

# Credits
Developer: Leandro Amorim Lagoa 
Email: leamorim@outlook.com
Linkeding: https://www.linkedin.com/in/leandrolagoa/
