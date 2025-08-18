What the project ?
_This is my first Machine learning model, for the purpose of learning, I'll be trying to draw a template for a findamental Model : Import , Create, Fit, Predict, Evaluate_


Whats the underlying data structure ? Read the data structure ? 
*A Decision Tree.*

Research & define the base design ?
*None - backend focused for learning.*

Research & define the tech Stack ?
_python : sickit learn_





**SELF ROUGH NOTES**
Research & define the software architecture ? 

1. Create a Git repository
 done 

2. Initialize local folders/repo
 done 

3. Connect local to remote repo
  done


  SAMPLE CODE
  ```# Code you have previously used to load data
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[feature_columns]

# Specify Model
iowa_model = DecisionTreeRegressor()
# Fit Model
iowa_model.fit(X, y)

print("First in-sample predictions:", iowa_model.predict(X.head()))
print("Actual target values for those homes:", y.head().tolist())

# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex4 import *
print("Setup Complete")```
