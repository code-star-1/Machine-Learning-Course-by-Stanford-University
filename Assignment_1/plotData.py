import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('/Users/pptem/PycharmProjects/ML-Assignments/resources/Assignment_1/ex1data1.txt', delimiter=',')
X = data.shape[0]  #number of training example
Y = data.shape[1]  #number of features
print('Dimensions of data ', X, Y)

population = data[:, 0]
profit = data[:, 1]

plt.plot(population, profit, 'o')
plt.xlabel('Population of cities in 10,000s')
plt.ylabel('Profit in $10,000')
plt.show()


