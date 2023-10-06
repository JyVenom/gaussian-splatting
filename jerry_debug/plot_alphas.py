import matplotlib.pyplot as plt
import numpy as np


def run():
    data = np.load("../alphas.npz")["alpha"]
    plt.hist(data, bins=100)
    plt.show()


if __name__ == '__main__':
    run()
