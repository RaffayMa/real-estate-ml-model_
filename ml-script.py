import pandas as pd # for data wrangling
import matplotlib.pyplot as plt # for tree visualization 
import kagglehub as kgh
from sklearn.tree import DecisionTreeRegressor # DSA
from sklearn.model_selection import train_test_split # for data point split
from sklearn.metrics import mean_absolute_error # for model validation



#*********
# Data Wrangling  (R, Pandas-python)
#*********

 # Dowlaoding the Sample set
path = kgh.dataset_download("yasserh/titanic-dataset")

print("Path to dataset files:", path)

df = pd.read_csv(os.path.join(path, 'train.csv'))
print(df.head())


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