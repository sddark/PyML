import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as op

#sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))

#cost function
def cost_function(theta, X, Y):
    m,n = X.shape
    theta = theta.reshape((n,1))
    z = np.matmul(X,theta)
    error = sigmoid(z)
    return 1/n * (sum(-Y*np.log(error) - ((1-Y)* np.log(1-error))))

#gradient function
def gradient_function(theta, X, Y):
    m,n = X.shape
    theta = theta.reshape((n,1))
    z = np.matmul(X,theta)
    error = sigmoid(z)
    return 1/n * (np.matmul(np.transpose(X),(error - Y)))



data = pd.read_csv('ex2data1.csv')

#visualize the data
data0 = data[data['y']==0]
data1 = data[data['y']==1]

plt.scatter(data0.iloc[:, [0]],data0.iloc[:, [1]])
plt.scatter(data1.iloc[:, [0]],data1.iloc[:, [1]])

#
plt.show()

# Preprocessing Input data
x = data.iloc[:, [0,1]]
X = x.to_numpy()
y = data.iloc[:, [2]] 
Y= y.to_numpy()

# Building the model
m,n = X.shape
theta = np.zeros(n)
theta1 = theta.reshape((n,1))
alpha = 0.1  # The learning Rate
epochs = 1500  # The number of iterations to perform gradient descent

#n = len(X) # Number of elements in X

z = np.matmul(X,theta)
error = sigmoid(z)
#theta = 1/n * (np.matmul(np.transpose(X),(error - Y)))

#cost = 1/n * sum(-Y*np.log(error) - ((1-Y)* np.log(1-error)))
print(1/n * sum(-Y*np.log(error) - ((1-Y)* np.log(1-error))))

result = op.minimize(fun = cost_function, x0 = theta1, args = (X,Y), method = 'TNC')

print(result)
