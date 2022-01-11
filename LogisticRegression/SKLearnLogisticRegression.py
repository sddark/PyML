import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#retrieve data
data = pd.read_csv('ex2data1.csv')

#visualize the data
data0 = data[data['y']==0]
data1 = data[data['y']==1]

plt.scatter(data0.iloc[:, [0]],data0.iloc[:, [1]])
plt.scatter(data1.iloc[:, [0]],data1.iloc[:, [1]])

#plot
plt.show()


# Preprocessing Input data
x = data.iloc[:, [0,1]]
#X = x.to_numpy()
y = data.iloc[:, 2] 
#Y= y.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print(">>>", logreg.coef_)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))



# Retrieve the model parameters.
b = logreg.intercept_[0]
w1, w2 = logreg.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

# Plot the data and the classification with the decision boundary.
xmin, xmax = 27, 105
ymin, ymax = 27, 105
xd = np.array([xmin, xmax])
yd = m*xd + c
plt.plot(xd, yd, 'k', lw=1, ls='--')

plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

#add scatter plots
plt.scatter(data0.iloc[:, [0]],data0.iloc[:, [1]])
plt.scatter(data1.iloc[:, [0]],data1.iloc[:, [1]])


plt.show()