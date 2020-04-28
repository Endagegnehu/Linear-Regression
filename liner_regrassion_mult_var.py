import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data =  pd.read_csv('ex1data2.txt', header = None)

X = data[[0,1]]
X_normal = data[[0,1]]
y = data[2]

m = len(y)

data_n = data.values


theta = np.zeros((3,1))
alpha = 0.01
iterations = 4000

def feature_normalization(X):
    X_normal = X 
    mean = np.mean(X_normal)
    std  = np.std(X_normal)    
    nominator = X_normal - mean
    
    X_normal = nominator / std
    X = X_normal
    return X,mean,std 

X2, mean, std = feature_normalization(X)
X = np.append(np.ones((m,1)),np.array(X2).reshape(m,2),axis = 1)
X_normal = np.append(np.ones((m,1)),np.array(X_normal).reshape(m,2),axis = 1)
y = data_n[:,2].reshape(m,1)

def cost(X,y,theta):
    m = len(y)
    pridiction = X.dot(theta)
    squard_error = (pridiction - y)**2    

    return 1/(2*m) * np.sum(squard_error)

cost_grad = cost(X,y,theta)

def gradientDecent(X,y,alpha,theta,iterations):
    m = len(y)
    J_history = np.zeros((iterations,1))
    m = len(y)
    i = 0
    while i < iterations:
        pridiction = X.dot(theta)
        squard_error = (pridiction - y)
        theta -= alpha*(1/m)*((X.T).dot(squard_error))
        J_history[i] = cost(X, y, theta) 
        i += 1
    return [J_history,theta] 

gradient ,theta  = gradientDecent(X,y,alpha,theta,iterations)

iterations = np.arange(iterations)

def plot_gradientDecent(gradient, iterations):
    plt.plot(iterations, gradient)
    plt.xlabel('Iterations')
    plt.ylabel('$J(\Theta)$')
    plt.title('Gradient Descent Convergence for multiple  Varable')

def normalEquation(X_normal,y):
    theta = (np.linalg.inv((X_normal.T).dot(X_normal))).dot(X_normal.T).dot(y)
    return theta;

theta_normal_equ = normalEquation(X_normal,y)

prdct = np.array((1650,3))
prdct_norm = np.array((1,1650,3))
prdct = (prdct - mean)/std
prdct = np.append(np.ones((1,1)),np.array(prdct).reshape(1,2),axis = 1)

price  = prdct.dot(theta)

print(plot_gradientDecent(gradient, iterations))
print('Cost of Gradient Descent: ')
print(cost_grad)
print('Theta of gradient descent: ')
print(theta)
print('Pridiction using gradient descent for 1650 sq-ft, 3 br house: ')
print(price)
print('Theta of normal equation: ')
print(theta_normal_equ)
print('Pridiction using normal equation for 1650 sq-ft, 3 br house:')
print(prdct_norm.dot(theta_normal_equ))