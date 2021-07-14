from functools import cached_property

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from sklearn.utils import check_X_y


class SemiSupervisedIsolationForest(IsolationForest):
    X = None
    y = None

    def fit(self, X, y, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=['csc'], y_numeric=True)
        self.X = X
        self.y = y
        super().fit(X, y=None, sample_weight=sample_weight)

    @cached_property
    def leaf_impacts_(self):
        impact_idx = self.y != 0
        sample_impact = self.y[impact_idx]

        leaf_impacts_ = []
        for tree, features in zip(self.estimators_, self.estimators_features_):
            X_subset = self.X[impact_idx, features] if self._max_features != self.X.shape[1] else self.X[impact_idx]
            leaves_index = tree.apply(X_subset)
            impacts = lil_matrix((1, tree.tree_.n_node_samples.shape[0]), dtype=self.y.dtype)
            impacts[0, leaves_index] = sample_impact
            leaf_impacts_.append(impacts)

        return leaf_impacts_

    def _compute_score_samples(self, X, subsample_features):
        scores = super()._compute_score_samples(X, subsample_features)

        depths = np.zeros(X.shape[0], order="f")
        for tree, features, impacts in zip(self.estimators_, self.estimators_features_, self.leaf_impacts_):
            X_subset = X[:, features] if subsample_features else X
            leaves_index = tree.apply(X_subset)
            depths += - np.ravel(impacts[0, leaves_index].toarray())

        scores *= 2 ** (
                -depths
                / (len(self.estimators_)
                   * _average_path_length([self.max_samples_]))
        )
        return scores
