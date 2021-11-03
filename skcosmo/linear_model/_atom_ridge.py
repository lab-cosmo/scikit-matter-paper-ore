import numpy as np
from sklearn.base import (
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.linear_model import Ridge

class AtomRidge(MultiOutputMixin, RegressorMixin):
    """ Solves a regularized linear regression problem for a model meant
    to describe a set of atom-centered properties. Difference with conventional
    Ridge involves mainly the definition of regularization, that is structured
    as an "error tolerance". This also supports computing combinations of
    multiple models, that are fit jointly. """

    def __init__(self, sigma, delta, *, solver='auto'):
        if not hasattr(delta, "__len__"):
            delta = [delta]
        self.delta = delta
        self.sigma = sigma
        self.solver = solver

    def fit(self, X, y):
        if not hasattr(X, "__len__"):
            X = [X]

        # enforces multi-output shape for y
        moy = y
        if len(y.shape)==1:
            moy = y.reshape((-1,1))

        if not(len(X) == len(self.delta)):
            raise ValueError(f"Number of model features  {len(X)} doesn't match number of delta parameters {len(self.delta)}.")
        normalization = []
        for x in X:
            normalization.append(np.sqrt((x**2).sum(axis=1).mean()))
        self.normalization_ = np.asarray(normalization)

        X = np.hstack([X[i]*self.delta[i]/normalization[i] for i in range(len(X))])

        # scales by sigma - this works whether sigma is a scalar or a Ntrain-sized array
        X = (X.T / np.sqrt(self.sigma)).T
        moy = (moy.T / np.sqrt(self.sigma)).T

        ridge = Ridge(alpha=1, fit_intercept=False, solver=self.solver)
        ridge.fit(X, moy)
        self.coef_ = ridge.coef_

        # restores whatever the user expects in terms of output shape
        if len(y.shape)==1:
            self.coef_ = self.coef_.squeeze()

    def predict(self, X):

        if not hasattr(X, "__len__"):
            X = [X]
        X = np.hstack([X[i]*self.delta[i]/self.normalization_[i] for i in range(len(X))])
        return X@self.coef_










