import matplotlib.pyplot as plt
import numpy as np
import sklearn.linear_model

from mpl_toolkits.mplot3d import Axes3D
import sys
from pathlib import Path

if Path(__file__).absolute().parents[0].as_posix() not in sys.path:
    sys.path.append(Path(__file__).absolute().parents[0].as_posix())

def get_pred_elevation():
    pass


def main():
    X_train = np.random.rand(2000).reshape(1000, 2) * 60
    y_train = (X_train[:, 0] ** 2) + (X_train[:, 1] ** 2)
    X_test = np.random.rand(200).reshape(100, 2) * 60
    y_test = (X_test[:, 0] ** 2) + (X_test[:, 1] ** 2)
    print(X_train)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_train[:, 0], X_train[:, 1], y_train, marker='.', color='red')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    model = sklearn.linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("MAE: {}".format(np.abs(y_test - y_pred).mean()))
    print("RMSE: {}".format(np.sqrt(((y_test - y_pred) ** 2).mean())))

    coefs = model.coef_
    intercept = model.intercept_

    w, h = 30, 61
    xs = np.tile(np.arange(w), (w, 1))
    print(xs)
    ys = np.tile(np.arange(w), (w, 1)).T
    zs = xs * coefs[0] + ys * coefs[1] + intercept
    print("Equation: y = {:.2f} + {:.2f}x1 + {:.2f}x2".format(intercept, coefs[0],
                                                              coefs[1]))

    ax.plot_surface(xs,ys,zs, alpha=0.5)
    plt.show()


if __name__ == '__main__':
    main()
