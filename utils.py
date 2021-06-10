import numpy as np


def add_noise(img, noise_type="gaussian"):
    row, col = 28, 28
    img = np.array(img)
    img = img.astype(np.float32)

    if noise_type == "gaussian":
        loc = 0
        var = 0.01
        sigma = var ** 0.5
        noise = np.random.normal(loc, sigma, img.shape)
        noise = noise.reshape(row, col)
        img = img + noise
        if img.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        img = np.clip(img, low_clip, 1.0)
        img = np.uint8(img * 255)
        return img

    if noise_type == "speckle":
        noise = np.random.randn(row, col)
        noise = noise.reshape(row, col)
        img = img + img * noise
        img = np.uint8(img)
        return img


def add_sp_noise(img, prob=0.9):
    output = np.zeros(img.shape, np.uint8)
    thres = 1 - prob
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            rdn = np.random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = img[i][j]
    return output
