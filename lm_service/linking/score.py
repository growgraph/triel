import numpy as np
from scipy.interpolate import CubicSpline


class BoundedCubicSpline:
    eps = 1e-4

    def __init__(self, data, a=0.0, b=1.0, n_grid=20):
        self.a = a
        self.b = b
        self.spline: CubicSpline
        self.grid = np.linspace(self.eps, 100 - self.eps, n_grid + 1)
        _data = np.array(data)
        self.fit(_data)

    @staticmethod
    def inv_sigmoid(data, a=0.0, b=1.0):
        data0 = BoundedCubicSpline.scale(data, a, b)
        return np.log(data0 / (1.0 - data0))

    @staticmethod
    def sigmoid(data, a=0.0, b=1.0):
        data0 = 1.0 / (1.0 + np.exp(-data))
        return BoundedCubicSpline.unscale(data0, a, b)

    @staticmethod
    def unscale(data, a=0.0, b=1.0):
        # [0, 1] -> [a, b]
        return a + (b - a) * data

    @staticmethod
    def scale(data, a=0.0, b=1.0):
        # [a, b] -> [0, 1]
        return (data - a) / (b - a)

    def fit(self, data):
        # [a, b] -> R
        rdata = BoundedCubicSpline.inv_sigmoid(data, self.a, self.b)
        # percentile of data
        pcts = np.percentile(rdata, self.grid)

        # [0, 1] -> R
        r_pct_grid = BoundedCubicSpline.inv_sigmoid(1e-2 * self.grid, 0.0, 1.0)
        self.spline = CubicSpline(pcts, r_pct_grid)

    def predict(self, data):
        rdata = self.inv_sigmoid(data, self.a, self.b)
        r_pcts = self.spline(rdata)

        # R -> [0, 1]
        pcts = BoundedCubicSpline.sigmoid(r_pcts, 0.0, 1.0)

        return pcts
