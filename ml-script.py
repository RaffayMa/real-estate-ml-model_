import pandas as pd # for data wrangling
import matplotlib.pyplot as plt # for tree visualization 
import kagglehub as kgh
from sklearn.tree import DecisionTreeRegressor # DSA
from sklearn.model_selection import train_test_split # for data point split
from sklearn.metrics import mean_absolute_error # for model validation



#*********
# Data Wrangling  (R, Pandas-python)
#*********

 # Reading the csv
Titanic_file_path = '/Users/muhammadraffaymagoon/Desktop/Projects/real-estate-ml-model_/Titanic-Dataset.xls'

data = pd.read_csv(Titanic_file_path) # read tthe csv to the program
print(data.head) # View as a dataframe

data.plot() # Graph visual
plt.show()


# Read & Store the CSV
# Clean and create dataframe
# Split data into training and validation data
# Create the X (stores the input attribute) and Y (stores the predictive Attribute) VARs 
 

#********
# Create & Fit 
#*********

 # seed the tree for random splits
 # Fit the training data frame to a regressor

#*********
# Predict
#*********

# Make predictions using the validation data 

#*********
# Evaluate
#*********

# Mean Absoulte Error (MAE) 
# Use the prediction data and the validation data to generate the accuracy/MAE