import numpy as np
from sklearn.model_selection import  StratifiedKFold


class StratifiedKFoldReg(StratifiedKFold):
    """

    This class generate cross-validation partitions
    for regression setups, such that these partitions
    resemble the original sample distribution of the
    target variable.

    """

    def split(self, X, y, groups=None):
        n_samples = len(y)

        # Number of labels to discretize our target variable,
        # into bins of quasi equal size
        n_labels = int(np.round(n_samples / self.n_splits))

        # Assign a label to each bin of n_splits points
        y_labels_sorted = np.concatenate([np.repeat(ii, self.n_splits) for ii in range(n_labels)])

        # Get number of points that would fall
        # out of the equally-sized bins
        mod = np.mod(n_samples, self.n_splits)

        # Find unique idxs of first unique label's ocurrence
        _, labels_idx = np.unique(y_labels_sorted, return_index=True)

        # sample randomly the label idxs to which assign the
        # the mod points
        rand_label_ix = np.random.choice(labels_idx, mod, replace=False)

        # insert these at the beginning of the corresponding bin
        y_labels_sorted = np.insert(y_labels_sorted, rand_label_ix, y_labels_sorted[rand_label_ix])

        # find each element of y to which label corresponds in the sorted
        # array of labels
        map_labels_y = dict()
        for ix, label in zip(np.argsort(y), y_labels_sorted):
            map_labels_y[ix] = label

        # put labels according to the given y order then
        y_labels = np.array([map_labels_y[ii] for ii in range(n_samples)])

        return super().split(X, y_labels, groups)