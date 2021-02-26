import numpy as np

from .simple_fps import FPS

VORONOI_CUTOFF_FRACTION = 1.0 / 6.0


class VoronoiFPS(FPS):
    """
    Base Class defined for Voronoi FPS methods

    :param initialize: predetermined index; if None provided, first index selected
                 is 0
    :type selected_: int, None
    """

    def _init_greedy_search(self, X, y, n_to_select):

        n_features = X.shape[1]

        # voronoi polyhedron location of each index
        self.vlocation_of_idx = np.full(n_features, -1)

        # indices in each voronoi polyhedron
        self.idx_in_voronoi = [np.array([]) for n in range(self.n_features_to_select)]
        self.idx_in_voronoi[0] = np.arange(n_features, dtype=int)

        # number of points in each voronoi polyhedron (VP)
        self.number_in_voronoi = np.full(self.n_features_to_select, 0)
        self.number_in_voronoi[0] = n_features

        # furthest point in from each VP center
        self.furthest_point = np.full(self.n_features_to_select, -1, dtype=int)

        # quarter of the square distance between new selected point and previously
        # selected points
        self.sel_d2q_ = np.zeros(self.n_features_to_select, float)

        super()._init_greedy_search(X, y, n_to_select)

    def _continue_greedy_search(self, X, y, n_to_select):
        """ Continues the search. Prepares an array to store the selected features. """

        super()._continue_greedy_search(X, y, n_to_select)

        n_pad = n_to_select - self.n_selected_
        for n in range(n_pad):
            self.idx_in_voronoi.append(np.array([]))
        self.number_in_voronoi = np.pad(
            self.number_in_voronoi, (0, n_pad), "constant", constant_values=0
        )
        self.sel_d2q_ = np.pad(self.sel_d2q_, (0, n_pad), "constant", constant_values=0)
        self.furthest_point = np.concatenate(
            (self.furthest_point, np.zeros(n_pad, dtype=int))
        )

    def _get_active(self, X, last_selected):
        # because we haven't called super() yet, self.n_selected_ has not been
        # incremented yet

        f_active = np.zeros(self.n_features_to_select, bool)

        """must compute distance of the new point to all the previous FPS. Some
            of these might have been computed already, but bookkeeping could be
            worse that recomputing (TODO: verify!)
            """
        # calculation in a single block
        self.sel_d2q_[: self.n_selected_] = (
            self.norms_[self.selected_idx_[: self.n_selected_]]
            + self.norms_[last_selected]
            - 2 * (self.X_selected_[:, : self.n_selected_].T @ X[:, last_selected])
        ) * 0.25

        for ic in range(self.n_selected_):
            # empty voronoi, no need to consider it
            if self.number_in_voronoi[ic] > 1:
                r2 = self.haussdorf_[self.furthest_point[ic]]
                
                if self.sel_d2q_[ic] < r2:
                    # these voronoi cells need to be updated
                    f_active[ic] = True

        return f_active

    def _calculate_distances(self, X, last_selected, **kwargs):
        # n_selected has not been incremented, so index of new voronoi is
        # n_selected
        self.eligible_[last_selected] = False
        self.haussdorf_[last_selected] = 0.0

        if self.n_selected_ == 0:
            self.haussdorf_ = super()._calculate_distances(X, last_selected)
            updated_points = np.arange(X.shape[-1], dtype=int)

        else:
            f_active = self._get_active(X, last_selected)
            if len(np.where(f_active)[0]) > 0:
                if (
                    np.sum(self.number_in_voronoi[f_active]) / X.shape[1]
                    > VORONOI_CUTOFF_FRACTION
                ):
                    new_dist = super()._calculate_distances(X, last_selected)
                    updated_points = np.where(new_dist < self.haussdorf_)[0]
                    np.minimum(self.haussdorf_, new_dist, self.haussdorf_)
                    f_active = np.full(self.n_selected_, False)
                    f_active[self.vlocation_of_idx[updated_points]] = True
                elif len(np.where(f_active)[0]) > 0:
                    new_dist = self.haussdorf_.copy()
                    active_points = np.concatenate(
                        np.array(self.idx_in_voronoi, dtype=object)[f_active]
                    )

                    new_dist[active_points] = (
                        self.norms_[active_points]
                        + self.norms_[last_selected]
                        - 2 * X[:, last_selected].T @ X[:, active_points]
                    )
                    updated_points = np.concatenate(
                        (np.where(new_dist < self.haussdorf_)[0], [last_selected])
                    )
                    np.minimum(self.haussdorf_, new_dist, self.haussdorf_)

                old_voronoi_loc = list(set(self.vlocation_of_idx[updated_points]))
                old_voronoi_loc.append(self.vlocation_of_idx[last_selected])
                for v in old_voronoi_loc:
                    self.eligible_[self.furthest_point[v]] = False

                    self.idx_in_voronoi[v] = np.array(
                        np.setdiff1d(
                            np.setdiff1d(self.idx_in_voronoi[v], updated_points),
                            self.selected_idx_[
                                self.selected_idx_ != self.selected_idx_[v]
                            ],
                        ),
                        dtype=int,
                    )
                    self.number_in_voronoi[v] = len(self.idx_in_voronoi[v])
                    self.furthest_point[v] = self.idx_in_voronoi[v][
                        np.argmax(self.haussdorf_[self.idx_in_voronoi[v]])
                    ]
                    self.eligible_[self.furthest_point[v]] = True
            else:
                updated_points = np.array([])

        self.vlocation_of_idx[last_selected] = self.n_selected_
        if len(updated_points) > 0:
            self.vlocation_of_idx[updated_points] = self.n_selected_
        self.idx_in_voronoi[self.n_selected_] = np.array(
            np.unique(np.concatenate(([last_selected], updated_points))), dtype=int
        )
        self.number_in_voronoi[self.n_selected_] = len(
            self.idx_in_voronoi[self.n_selected_]
        )

        my_idx = self.idx_in_voronoi[self.n_selected_]
        furthest = self.haussdorf_[my_idx].argmax()
        self.furthest_point[self.n_selected_] = my_idx[furthest]
        self.eligible_[:] = False
        self.eligible_[self.furthest_point[: self.n_selected_ + 1]] = True
        self.eligible_[self.selected_idx_[: self.n_selected_ + 1]] = False
        self.haussdorf_[self.selected_idx_] = 0.0

        assert self.vlocation_of_idx[last_selected] == self.n_selected_
        return self.haussdorf_
