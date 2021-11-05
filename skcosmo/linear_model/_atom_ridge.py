import numpy as np
from sklearn.base import (
    MultiOutputMixin,
    RegressorMixin,
    BaseEstimator
)
from sklearn.linear_model import Ridge

class AtomRidge(BaseEstimator, MultiOutputMixin, RegressorMixin):
    """ Solves a regularized linear regression problem for a model meant
    to describe a set of atom-centered properties. Difference with conventional
    Ridge involves mainly the definition of regularization, that is structured
    as an "error tolerance". This also supports computing combinations of
    multiple models, that are fit jointly. """

    def __init__(self, delta, sigma, *, feature_groups=None, solver='auto'):
        self.delta = delta
        self.sigma = sigma
        self.solver = solver
        self.feature_groups = feature_groups

    def fit(self, X, y, sample_sigmas=None):

        # parameter validation
        if not hasattr(self.delta, "__len__"):
            self.delta = [self.delta]
        if self.feature_groups is None:
           feature_groups.feature_groups = [slice(None)]
        if len(self.feature_groups) != len(self.delta):
            raise ValueError(f"Number of models  {len(self.feature_groups)} doesn't match number of delta parameters {len(self.delta)}.")

        # enforces multi-output shape for y
        moy = y
        if len(y.shape)==1:
            moy = y.reshape((-1,1))

        if self.feature_groups is None:
            self.feature_groups = [slice(None)]
        normalization = []
        for g in self.feature_groups:
            normalization.append(np.sqrt((X[:,g]**2).sum(axis=1).mean()))
        self.normalization_ = np.asarray(normalization)

        nX = np.hstack([X[:,g]*self.delta[ig]/self.normalization_[ig] for ig, g in enumerate(self.feature_groups)])

        # scales by sigma - this works whether sigma is a scalar or a Ntrain-sized array
        if sample_sigmas is None:
            sample_sigmas = self.sigma
        nX = (nX.T / np.sqrt(sample_sigmas)).T
        moy = (moy.T / np.sqrt(sample_sigmas)).T

        ridge = Ridge(alpha=1, fit_intercept=False, solver=self.solver)
        ridge.fit(nX, moy)
        self.coef_ = ridge.coef_

        # restores whatever the user expects in terms of output shape
        if len(y.shape)==1:
            self.coef_ = self.coef_.squeeze()

    def predict(self, X):

        nX = np.hstack([X[:,g]*self.delta[ig]/self.normalization_[ig] for ig, g in enumerate(self.feature_groups)])
        return nX@self.coef_










