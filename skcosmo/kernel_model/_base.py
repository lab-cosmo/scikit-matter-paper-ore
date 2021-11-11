import numpy as np

from sklearn.base import (
    BaseEstimator, TransformerMixin
)

class RKHSTransform(BaseEstimator, TransformerMixin):
    """
    Solves the sparse GPR problem using an explicit RKHS trick
    """

    def __init__(self, jitter=0, relative_jitter=True):

        self.jitter = jitter
        self.relative_jitter = relative_jitter

    def fit(self, KBB):

        self.vB_, self.UBB_ = np.linalg.eigh(KBB)
        self.vB_ = np.flip(self.vB_)
        self.UBB_ = np.flip(self.UBB_, axis=1)
        self.jitter_ = self.jitter
        if self.relative_jitter:
            self.jitter_ *= self.vB_[0]

        self.nB_ = len(np.where(self.vB_ > self.jitter_)[0])
        self.PKPhi_ = self.UBB_[:, :self.nB_] * 1 / np.sqrt(self.vB_[:self.nB_])

    def transform(self, KTB):
        return KTB @ self.PKPhi_
