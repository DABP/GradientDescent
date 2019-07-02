import numpy as np
import matplotlib.pyplot as plt


def calculateCost(X, Y, weight, bias):
	return ((weight * X + bias - Y) ** 2).sum() / len(X)


def numericalDifferentiation(f, x, y, weight, bias):
	deltaX = 1e-7
	upCost = f(x + deltaX, y, weight, bias)
	diff = (upCost - f(x - deltaX, y, weight, bias)) / (deltaX * 2)

	return diff, upCost


def gradientDescent(X, Y, weight, bias, learningRate, epoch):
	diff_list = []
	cost_list = []

	for i in range(epoch):
		diff = numericalDifferentiation(calculateCost, X, Y, weight, bias)
		if diff[0] < 0:
			weight = weight - learningRate * diff[0]

		else:
			weight = weight + learningRate * diff[0]

		diff_list.append(diff[0])
		cost_list.append(diff[1])
	return weight, diff_list, cost_list

X = 2 * np.random.rand(100, 1)
Y = 8 + 5 * X + np.random.randn(100, 1)
w = np.random.randn(2, 1)

learningRate = 0.01
epoch = 10000

result = gradientDescent(X, Y, w[0], w[1], learningRate, epoch)

print(result)
# plt.plot(X, Y, 'p')
plt.plot(range(epoch), result[2])
plt.show()
