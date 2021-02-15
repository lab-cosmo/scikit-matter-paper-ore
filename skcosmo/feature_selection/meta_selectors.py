import numpy as np
from sklearn.utils.validation import check_is_fitted


class _MetaSelector:
    def __init__(
        self,
        selector,
        n_features_to_select,
    ):
        self.selector_ = selector
        self.n_features_to_select_ = n_features_to_select

    def _update_selector(self, n_features_to_select):
        self.selector_.n_features_to_select = n_features_to_select

    def _n_total(self, X):
        return X.shape[-1]


class BlockSelector(_MetaSelector):
    def __init__(
        self,
        selector,
        n_features_to_select,
        max_block_size,
    ):
        super().__init__(selector, n_features_to_select)

        self.max_block_size_ = max(max_block_size, n_features_to_select + 1)
        self.block_size = max(max_block_size, n_features_to_select + 1)

    def _i_block(self, X):
        n_total = self._n_total(X)
        n_block = int(n_total / self.max_block_size_)
        return np.array_split(np.arange(n_total), n_block)

    def fit(self, X, y=None):
        i_blocks = self._i_block(X)
        i_stacked = np.zeros(self.n_features_to_select_)

        X_block = X[:, i_blocks[0]]

        self.selector_.fit(X_block, y)
        i_stacked = self.selector_.selected_idx_

        for idx in i_blocks[1:]:
            this_idx = np.sort(np.concatenate((i_stacked, idx)))
            X_block = X[:, this_idx]

            self.selector_.fit(X_block, y)
            i_stacked = this_idx[self.selector_.selected_idx_]

        self.selected_idx_ = i_stacked
        self.support_mask_ = np.zeros(X.shape[1], dtype=bool)
        self.support_mask_[self.selected_idx_] = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["support_mask_"])
        return X[:, self.support_mask_]


class RollingSelector(_MetaSelector):
    def __init__(self, selector, n_features_to_select=None):
        super().__init__(selector, None)
        self.n_blocks_seen_ = 0
        self.n_selected_ = 0
        self.blocked_masks_ = []
        self.blocked_idx_ = []
        self.n_samples_seen_ = None

    def fit(self, X, y=None, n_features_to_select=None, warm_start=False):

        self._update_selector(self.n_selected_ + n_features_to_select)

        if not warm_start:
            self.selector_.fit(X, y)
            self.n_samples_seen_ = X.shape[0]
        else:
            if X.shape[0] != self.n_samples_seen_:
                raise ValueError(
                    f"This feature set does not contain values for all previously fit samples. Expecting {self.n_samples_seen_}, got {X.shape[0]}"
                )

            self.selector_.selected_idx_[: self.n_selected_] = np.arange(
                self.n_selected_
            )

            self.selector_.support_mask_ = np.zeros(
                (X.shape[0], self.n_selected_ + n_features_to_select), dtype=bool
            )
            self.selector_.support_mask_[: self.n_selected_] = True
            self.selector_.fit(np.hstack((self.selector_.X_selected_, X.copy())), y)

        self.n_blocks_seen_ += 1
        self.blocked_idx_.append(
            self.selector_.selected_idx_[self.n_selected_ :].copy() - self.n_selected_
        )
        new_support_mask = np.zeros((X.shape[1]), dtype=bool)
        new_support_mask[
            self.selector_.selected_idx_[self.n_selected_ :].copy() - self.n_selected_
        ] = True
        self.blocked_masks_.append(new_support_mask)
        self.n_selected_ += n_features_to_select
        return self

    def transform(self, X, y=None):
        check_is_fitted(self, ["blocked_idx_"])

        if np.asarray(X, dtype=object).shape[0] != self.n_blocks_seen_:
            raise ValueError(
                f"We are expecting {self.n_blocks_seen_} feature sets for X, got {X.shape[0]}."
            )
        return np.hstack([x[:, idx] for x, idx in zip(X, self.blocked_idx_)])
