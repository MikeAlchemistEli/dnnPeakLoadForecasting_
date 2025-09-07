##Energy Forecasting with DNN

This is my first project using Python, TensorFlow and Keras.
I used the UCI Appliances Energy Prediction dataset uploaded on Github by LuisM78 on February 14 2017 to try and predict the energy use of appliances at home.

The idea is simple, to train a Deep Neural Network (DNN) so it can forecast how much load the appliances will use at different times of the day.

##WHat the code does
* Reads the UCI dataset (CSV file)
* Creates some extra features like hour of the day, weekend/weekday
* Prepares the data for training
* Builds a small neural network
* Trains it to predict appliance energy load
* Shows a plot of actual vs predicted load values
* Prints some error results like RMSE and MAE

##How to run
* Download or clone this project
* Make sure you have Python IDE installed (I used PyCharm 3.13).
* Install the libraries you need
* Run the script in PyCharm

Notes: This is my first attempt at using deep learning for forecastin, so the code is still basic. Any suggestions to make it even better are welcome.

MEKA
