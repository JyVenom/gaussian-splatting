import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm


def run():
    print("Load")
    data = loadmat("../jerry_out/depth_alphas_trans_depths2.mat")
    depth = data["depth"]
    alphas = data["alphas"]
    trans = data["trans"]
    depths2 = data["depths2"]

    print("Compute")
    depth3 = np.zeros((800, 800))
    depth4 = np.zeros((800, 800))
    for i in tqdm(range(800)):
        for j in range(800):
            T = 1.0
            for k in range(150):
                depth3[i][j] += depths2[k, i, j] * alphas[k, i, j] * T
                T = T * (1.0 - alphas[k, i, j])
                depth4[i][j] += depths2[k, i, j] * alphas[k, i, j] * trans[k, i, j]
                if abs(depth3[i][j] - depth4[i][j]) > 0.01:
                    print("ERROR!")

    print("Draw")
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    axs[0].imshow(depth.squeeze(0), cmap='gray')
    axs[0].set_title('GT')
    axs[0].axis('off')
    axs[1].imshow(depth3, cmap='gray')
    axs[1].set_title('A')
    axs[1].axis('off')
    axs[2].imshow(depth4, cmap='gray')
    axs[2].set_title('T')
    axs[2].axis('off')
    plt.show()


if __name__ == "__main__":
    run()
