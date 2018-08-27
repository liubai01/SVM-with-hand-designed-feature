import numpy as np
from matplotlib.colors import ListedColormap
import random
a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
b = random.sample(a, 5)


def visualize_db(X, y):
    """
    load design matrix X, sample 5 of each category(1 and 8).
    Visualize them by matplotlib.
    :param X: numpy.array-[n, 1, 28, 28]
    """
    import matplotlib.pyplot as plt
    X0 = X[y == 0]
    X8 = X[y == 8]

    i0 = random.sample(range(X0.shape[0]), 5)
    i8 = random.sample(range(X8.shape[0]), 5)
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        img = X0[i0[i]][0].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title("1")
        plt.axis('off')
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        img = X8[i8[i]][0].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title("8")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def visualize_pred_result(model, X_org, X_feat, y):
    import pylab
    import matplotlib.pyplot as plt
    pylab.rcParams['figure.figsize'] = (15.0, 8.0) # 显示大小
    X0_feat = X_feat[y == 0]
    X8_feat = X_feat[y == 8]

    X0_org = X_org[y == 0]
    X8_org = X_org[y == 8]
    
    i0 = random.sample(range(X0_feat.shape[0]), 5)
    i8 = random.sample(range(X8_feat.shape[0]), 5)

    tmp_X_feat = []
    for i in i0:
        tmp_X_feat.append(X0_feat[i])
    y0 = model.predict(tmp_X_feat)

    tmp_X_feat = []
    for i in i8:
        tmp_X_feat.append(X8_feat[i])
    y8 = model.predict(tmp_X_feat)

    for i in range(5):
        plt.subplot(2, 5, i + 1)
        img = X0_org[i0[i]][0].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title("g.t.: {} | pred.: {}".format(0, y0[i]))
        plt.axis('off')
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        img = X8_org[i8[i]][0].reshape(28, 28)
        plt.imshow(img, cmap='gray')
        plt.title("g. t.: {} | pred.: {}".format(8, y8[i]))
        plt.axis('off')
    plt.show()


def visualize_feat_1d(X, y):
    import matplotlib.pyplot as plt
    sample_num = y.shape[0]
    x1 = X.reshape(-1)

    for i in range(sample_num):
        if y[i] == 8:
            plt.scatter(x1[i], 0, c="r", alpha=0.1)
        else:
            plt.scatter(x1[i], 0, c="b", alpha=0.1)

    plt.show()


def visualize_feat_2d(X, y):
    import matplotlib.pyplot as plt
    sample_num = y.shape[0]

    for i in range(sample_num):
        if y[i] == 8:
            plt.scatter(X[i, 0], X[i, 1], c="r", alpha=0.1)
        else:
            plt.scatter(X[i, 0], X[i, 1], c="b", alpha=0.1)

    plt.show()


def visualize_model(model, X, y):
    import matplotlib.pyplot as plt
    label = y
    # hyperparameter for boardry
    x = X[:, 0]
    y = X[:, 1]
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    width = x_max - x_min
    height = y_max - y_min
    x_min -= width / 4
    x_max += width / 4
    y_min -= height / 4
    y_max += height / 4
    step = 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max + step, step),
                         np.arange(y_min, y_max + step, step)
                         )
    z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Create color maps
    cmap_light = ListedColormap(['#6495ED', '#FFAAAA'])

    z = z.reshape(xx.shape)

    plt.pcolormesh(xx, yy, z, cmap=cmap_light)

    sample_num = label.shape[0]
    for i in range(sample_num):
        if label[i] == 8:
            plt.scatter(X[i, 0], X[i, 1], c="r", alpha=0.3)
        else:
            plt.scatter(X[i, 0], X[i, 1], c="b", alpha=0.3)

    plt.axis([x_min, x_max, y_min, y_max])
    plt.show()
