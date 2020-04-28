import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('ex1data1.txt', header = None)

X = data[0]
y = data[1]

plt.scatter(X, y)

m = len(y)

theta = np.zeros((2,1))

data_n = data.values
X = np.append(np.ones((m,1)), data_n[:,0].reshape(m,1),axis = 1)
y = data_n[:,1].reshape(m,1)

alpha = 0.01
iterations  = 1500;

def cost(X,y,theta):
    m = len(y)
    pridiction = X.dot(theta)
    squard_error = (pridiction - y)**2    

    return 1/(2*m) * np.sum(squard_error)

def gradient(X,y,theta,alpha,iterations):
    m = len(y)
    i = 0
    while i < iterations:
        pridiction = X.dot(theta)
        squard_error = (pridiction - y)
        theta -= alpha*(1/m)*((X.T).dot(squard_error))
        i += 1
    return theta

def plot_liner_fit(X,y,theta):
    plt.plot(X[:,1],X.dot(theta),'r-')

print('Cost: ')
print(cost(X, y, theta))
print('Theta: ')
print(gradient(X, y, theta, alpha, iterations))
print('Plot liner fit: ')
print(plot_liner_fit(X, y, theta))
print('Predict values for population sizes of 35,000 and 70,000')
predict  = np.array([1, 3.5])
print(predict.dot(theta)*10000)
