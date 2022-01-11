import numpy as np
import pandas as pd

# Preprocessing Input data
data = pd.read_csv('ex1data1.csv')
x = data.iloc[:, [0,1]]
X = x.to_numpy()
y = data.iloc[:, [2]]
Y= y.to_numpy()

# Building the model
x_1 = 0
x_0 = 0
theta = np.array([[x_0], [x_1]])

alpha = 0.01  # The learning Rate
epochs = 1500  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

# Performing Gradient Descent 
for i in range(epochs): 
    error = np.matmul(X,theta) - Y
    theta = theta - ((alpha/epochs) * np.matmul(np.transpose(X) , error))

    print((1/(2*n)) * sum((np.matmul(X,theta) - Y)**2))

print("theta")   
print (theta)