import numpy as np
import matplotlib.pyplot as plt


def calculateCost(X, Y, weight, bias):
	return ((weight * X + bias - Y) ** 2).sum() / len(X)


def numericalDifferentiationWeight(f, x, y, weight, bias):
	deltaW = 1e-7
	upCost = f(x, y, weight + deltaW, bias)
	diff = (upCost - f(x, y, weight - deltaW, bias)) / (deltaW* 2)
	return diff, upCost

def numericalDifferentiationBias(f, x, y, weight, bias):
	deltaB = 1e-7
	upCost = f(x, y, weight, bias + deltaB)
	diff = (upCost - f(x, y, weight, bias - deltaB)) / (deltaB * 2)
	return diff, upCost


def gradientDescent(X, Y, weight, bias, learningRate, epoch):
	diff_list = []
	cost_list = []

	for i in range(epoch):
		diffW = numericalDifferentiationWeight(calculateCost, X, Y, weight, bias)
		weight = weight - learningRate * diffW[0]
		diffB = numericalDifferentiationBias(calculateCost, X, Y, weight, bias)
		bias = bias - learningRate * diffB[0]

		diff_list.append(diffB[0])
		cost_list.append(diffB[1])

	return weight, diff_list, cost_list

X = 2 * np.random.rand(100, 1)
Y = 8 + 5 * X + np.random.randn(100, 1)
w = np.random.randn(2, 1)

learningRate = 0.01
epoch = 1000

result = gradientDescent(X, Y, w[0], w[1], learningRate, epoch)

print(result)
# plt.plot(X, Y, 'p')
plt.plot(range(epoch), result[2])
plt.show()
