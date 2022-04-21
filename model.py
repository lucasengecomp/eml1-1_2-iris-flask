import numpy as np # To manipulate data as arrays
from sklearn import datasets
from sklearn.linear_model import LogisticRegression # Classification model

# import some data to play with
iris = datasets.load_iris()


X = iris.iloc[:, 0:-1] # Extracting the features/independent variables
y = iris.iloc[:, -1] # Extracting the target/dependent variable

logreg = LogisticRegression(max_iter=1000) # Initializing the Logistic Regression model
logreg.fit(X, y) # Fitting the model

# Function for classification based on inputs
def classify(a, b, c, d):
    arr = np.array([a, b, c, d]) # Convert to numpy array
    arr = arr.astype(np.float64) # Change the data type to float
    query = arr.reshape(1, -1) # Reshape the array
    prediction = variety_mappings[logreg.predict(query)[0]] # Retrieve from dictionary
    return prediction # Return the prediction

