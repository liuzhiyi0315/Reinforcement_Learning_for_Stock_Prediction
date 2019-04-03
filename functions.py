import numpy as np
import math


# prints formatted price
def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


# returns the vector containing stock data from a fixed file
def getStockDataVec(key):
    vec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        vec.append(float(line.split(",")[4]))

    return vec


# returns the sigmoid
def sigmoid(x):
    return 1 / (1 + math.exp(-x))


# returns an an n-day state representation ending at time t
def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]  # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])


class PercentagePrinter:
    def __init__(self, text, num_all, blocks=50):
        self.text = text
        self.num_all = num_all
        self.blocks = blocks

    def print(self, index, state_text=None):
        percentage = int(100 * index / self.num_all)
        if state_text is not None:
            print("\r{} {} {}% ".format(self.text, state_text, percentage), end='')
        else:
            print("\r{} {}% ".format(self.text, percentage), end='')
        _percentage = int(percentage / (100 / self.blocks))
        for p in range(0, _percentage):
            print('🀫', end='')
        for p in range(self.blocks - _percentage):
            print('🀆', end='')
        print("", end='', flush=True)

    def final_print(self, state_text=None):
        if state_text is not None:
            print("\r{} {} {}% ".format(self.text, state_text, 100), end='')
        else:
            print("\r{} {}% ".format(self.text, 100), end='')
        print("")
