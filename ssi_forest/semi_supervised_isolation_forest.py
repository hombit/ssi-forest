from functools import partial
from typing import Callable, Union

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage
from scipy.sparse import lil_matrix
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from sklearn.utils import check_X_y

try:
    from functools import cached_property
except ImportError:
    from backports.cached_property import cached_property


_LABEL_LEAF_REDUCER_T = Callable[[NDArray[np.float64]], float]
_LABEL_LEAVES_REDUCER_T = Callable[[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]], NDArray[np.float64]]


class SemiSupervisedIsolationForest(IsolationForest):
    """
    Biased Isolation Forest

    The variant of the Isolation Forest (Liu et al 2008) which introduces
    artificial changes of depths of leaves containing labeled data.

    Parameters
    ----------
    label_reducer : str or callable, optional
        How to reduce multiple labels accrued in a single leaf. Could be:
            1) a callable with signature `f(labels)` where `labels`
            is a 1-D array built from `y` values, the callable must return
            a single floating number
            2) a string with a name of pre-defined reducer, could be one of:
            - 'random' (default) gives random label
            - 'mean' gives mean of all values
            - 'sum' gives sum of all values
            - 'absmax' gives label with maximum absolute value
    """

    X = None
    y = None

    def __init__(self, *, label_reducer: Union[str, _LABEL_LEAF_REDUCER_T] = 'random', **iforest_kwargs):
        super().__init__(**iforest_kwargs)
        self.rng = np.random.default_rng(iforest_kwargs.get('random_state', None))
        self.label_reducer: _LABEL_LEAVES_REDUCER_T = self._get_label_reducer(label_reducer)

    def _mean_reducer(self,
                      values: NDArray[np.float64],
                      indices: NDArray[np.int64],
                      unique_indices: NDArray[np.int64]) -> NDArray[np.float64]:
        return ndimage.mean(
            values,
            labels=indices,
            index=unique_indices,
        )

    def _sum_reducer(self,
                     values: NDArray[np.float64],
                     indices: NDArray[np.int64],
                     unique_indices: NDArray[np.int64]) -> NDArray[np.float64]:
        return ndimage.sum_labels(
            values,
            labels=indices,
            index=unique_indices,
        )

    @staticmethod
    def _reducer(values: NDArray[np.float64],
                 indices: NDArray[np.int64],
                 unique_indices: NDArray[np.int64],
                 func: _LABEL_LEAF_REDUCER_T) -> NDArray[np.float64]:
        return ndimage.labeled_comprehension(
            values,
            labels=indices,
            index=unique_indices,
            func=func,
            out_dtype=np.float64,
            default=None,
        )

    def _absmax_reducer(self,
                        values: NDArray[np.float64],
                        indices: NDArray[np.int64],
                        unique_indices: NDArray[np.int64]) -> NDArray[np.float64]:
        return self._reducer(values, indices, unique_indices, func=lambda x: x[np.argmax(np.abs(x))])

    def _random_reducer(self,
                        values: NDArray[np.float64],
                        indices: NDArray[np.int64],
                        unique_indices: NDArray[np.int64]) -> NDArray[np.float64]:
        return self._reducer(values, indices, unique_indices, func=lambda x: self.rng.choice(x))

    def _get_label_reducer(self, x: Union[str, _LABEL_LEAF_REDUCER_T]) -> _LABEL_LEAVES_REDUCER_T:
        if callable(x):
            return partial(self._reducer, func=x)
        return getattr(self, f'_{x}_reducer')

    def fit(self, X, y, sample_weight=None):
        """
        Fit estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficiency.

        y : {array-like} of shape (n_samples,)
            Input anomality labels, use zero for non-labeled data, positive
            values for known anomalies and negative values for known
            non-anomalies. It is assumed that most of the data is unlabeled,
            so the most of the elements must be zero. The typical absolute
            values of labels are between unity and `np.log2(y.shape[0]) / 2`

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y, accept_sparse=['csc'], y_numeric=True)
        if np.all(y == 0):
            raise ValueError('All labels are zero, use scikit.ensemble.IsolationForest instead')
        self.X = X
        self.y = y
        return super().fit(X, y=None, sample_weight=sample_weight)

    @cached_property
    def leaf_impacts_(self):
        impact_idx = self.y != 0
        sample_impact = self.y[impact_idx]

        leaf_impacts_ = []
        for tree, features in zip(self.estimators_, self.estimators_features_):
            X_subset = self.X[impact_idx, features] if self._max_features != self.X.shape[1] else self.X[impact_idx]
            leaves_index = tree.apply(X_subset)
            unique_leaves_index = np.unique(leaves_index)
            impacts = lil_matrix((1, tree.tree_.n_node_samples.shape[0]), dtype=self.y.dtype)
            impacts[0, unique_leaves_index] = self.label_reducer(sample_impact, leaves_index, unique_leaves_index)
            leaf_impacts_.append(impacts)

        return leaf_impacts_

    def _compute_score_samples(self, X, subsample_features):
        scores = super()._compute_score_samples(X, subsample_features)

        depths = np.zeros(X.shape[0], order="f")
        for tree, features, impacts in zip(self.estimators_, self.estimators_features_, self.leaf_impacts_):
            X_subset = X[:, features] if subsample_features else X
            leaves_index = tree.apply(X_subset)
            depths += -np.ravel(impacts[0, leaves_index].toarray())

        scores *= 2 ** (
                -depths
                / (len(self.estimators_)
                   * _average_path_length([self.max_samples_]))
        )
        return scores
