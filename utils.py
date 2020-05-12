import matplotlib.pyplot as plt


def show_censored_img(x, y):
    img = x.copy()

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i, j, k] = img[i, j, k] if y[i, j] else 0

    plt.imshow(img)
    plt.show()
