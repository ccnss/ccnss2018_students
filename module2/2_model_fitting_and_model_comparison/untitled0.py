# -*- coding: utf-8 -*-
"""
Created on Sun Jul  1 03:38:44 2018

@author: User
"""
np.random.seed(0)

# Generate the data
beta1 = 1
beta0 = 0.5
sigma = 1
xdata = np.random.randn(100)
eps   = sigma * np.random.randn(100)
ydata = beta1*xdata + beta0 + eps

# Compute the analytical solutions
beta1_est = np.cov(xdata,ydata)[0,1] / np.var(xdata)
beta0_est = np.mean(ydata) - beta1_est*np.mean(xdata)
print('Analytical B0: ' + str(beta0_est), ', B1: ' + str(beta1_est))


print(np.cov(xdata,ydata))
print(np.var(ydata))

# Function that returns the MSE
def get_MSE(params):   
    y_hat = params[1]*xdata + params[0]    
    return np.sum((ydata-y_hat)**2)

# Optimization (recovering the parameters)
initial_guess = [2,1]
res = minimize(get_MSE, initial_guess)
res.x
print('Optimization B0: ' + str(res.x[0]), ', B1: ' + str(res.x[1]))

plt.plot(xdata, ydata, 'o')
xs = np.array([xdata.min(), xdata.max()])
plt.plot(xs, beta0_est + beta1_est*xs,'-.b',LineWidth=2,label='Analytical Solution')
plt.plot(xs, res.x[0] + res.x[1]*xs,'--r',LineWidth=2,label='Optimization Solution')
plt.ylabel('Y');
plt.xlabel('X');