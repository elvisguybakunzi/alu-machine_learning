#!/usr/bin/env python3
"""
0x03. Hyperparameter Tuning
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """Bayesian optimization on a noiseless 1D Gaussian process"""

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """init of BO"""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = (np.linspace(start=bounds[0], stop=bounds[1],
                                num=ac_samples)[:, np.newaxis])
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ calculates the next best sample location """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize is True:
            mu_sample_opt = np.min(self.gp.Y)
        else:
            mu_sample_opt = np.max(self.gp.Y)

        with np.errstate(divide="warn"):
            imp = mu_sample_opt - mu - self.xsi
            z = imp / sigma
            ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            ei[ei == 0.0] = 0.0
        X = self.X_s[np.argmax(ei, axis=0)]
        return X, ei

    def optimize(self, iterations=100):
        """The optimization step"""
        x_searched = []
        for _ in range(iterations):
            x, _ = self.acquisition()
            y = self.f(x)
            if x in x_searched:
                break
            x_searched.append(x)
            self.gp.update(x, y)
        self.gp.X = self.gp.X[:-1]
        if self.minimize:
            y_opt = np.min(self.gp.Y, keepdims=True)
            return self.gp.X[np.argmin(self.gp.Y)], y_opt[0]
        else:
            y_opt = np.max(self.gp.Y, keepdims=True)
            return self.gp.X[np.argmax(self.gp.Y)], y_opt[0]
