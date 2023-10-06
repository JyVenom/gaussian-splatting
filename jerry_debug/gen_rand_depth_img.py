import numpy as np
from PIL import Image as im


def run():
    array = (np.random.random(size=(800, 800, 3)) * 256).astype(np.uint8)
    # array[..., 3] = 255

    data = im.fromarray(array)
    data.show()
    data.save('depth.png')


if __name__ == "__main__":
    run()
