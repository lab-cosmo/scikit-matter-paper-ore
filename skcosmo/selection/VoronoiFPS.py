# -*- coding: utf-8 -*-
"""

This module contains Farthest Point Sampling (FPS) classes for sub-selecting
features or samples from given datasets. Each class supports a Principal
Covariates Regression (PCov)-inspired variant, using a mixing parameter and
target values to bias the selections.

Authors: Rose K. Cersonsky
         Michele Ceriotti

"""

"""
    /* ---------------------------------------------------------------------- */
    std::tuple<Eigen::ArrayXi, Eigen::ArrayXd, Eigen::ArrayXd, Eigen::ArrayXi,
               Eigen::ArrayXd>
    select_fps_voronoi(const Eigen::Ref<const RowMatrixXd> & feature_matrix,
                       int n_sparse, int i_first_point) {
      // number of inputs
      int n_inputs = feature_matrix.rows();
      // number of features
      int n_features = feature_matrix.cols();

      // defaults to full sorting of the inputs
      if (n_sparse == 0) {
        n_sparse = n_inputs;
      }

      // TODO(ceriottm) <- use the exception mechanism
      // for librascal whatever it is
      if (n_sparse > n_inputs) {
        throw std::runtime_error("Cannot FPS more inputs than those provided");
      }

      // return arrays
      // FPS indices
      auto sparse_indices = Eigen::ArrayXi(n_sparse);
      // minmax distances^2
      auto sparse_minmax_d2 = Eigen::ArrayXd(n_sparse);
      // size^2 of Voronoi cells
      auto voronoi_r2 = Eigen::ArrayXd(n_sparse);
      // assignment of points to Voronoi cells
      auto voronoi_indices = Eigen::ArrayXi(n_inputs);
      // work arrays
      // index of the maximum-d2 point in each cell
      auto voronoi_i_far = Eigen::ArrayXd(n_sparse);
      // square moduli of inputs
      auto feature_x2 = Eigen::ArrayXd(n_inputs);
      // list of distances^2 to latest FPS point
      auto list_new_d2 = Eigen::ArrayXd(n_inputs);
      // list of minimum distances^2 to each input
      auto list_min_d2 = Eigen::ArrayXd(n_inputs);
      // flags for "active" cells
      auto f_active = Eigen::ArrayXi(n_sparse);
      // list of dist^2/4 to previously selected points
      auto list_sel_d2q = Eigen::ArrayXd(n_sparse);
      // feaures of the latest FPS point
      auto feature_new = Eigen::VectorXd(n_features);
      // matrix of the features for the active point selection
      auto feature_sel = RowMatrixXd(n_sparse, n_features);
      int i_new{};
      double d2max_new{};
      // computes the squared modulus of input points
      feature_x2 = feature_matrix.=.rowwise().sum(); - diagonal elements of kernel/covariance matrix

      // initializes arrays taking the first point provided in input
      sparse_indices(0) = i_first_point;
      //  distance square to the selected point
      list_new_d2 =
          feature_x2 + feature_x2(i_first_point) -
          2 * (feature_matrix * feature_matrix.row(i_first_point).transpose())
                  .array();
      list_min_d2 = list_new_d2;  // we only have this point....
      voronoi_r2 = 0.0;
      voronoi_indices = 0;
      // picks the initial Voronoi radius and the farthest point index
      voronoi_r2(0) = list_min_d2.maxCoeff(&voronoi_i_far(0));

      feature_sel.row(0) = feature_matrix.row(i_first_point);


#ifdef DO_TIMING
      // timing code
      double tmax{0}, tactive{0}, tloop{0};
      int64_t ndist_eval{0}, npoint_skip{0}, ndist_active{0};
      auto gtstart = hrclock::now();
#endif
      for (int i = 1; i < n_sparse; ++i) {
#ifdef DO_TIMING
        auto tstart = hrclock::now();
#endif
        /*
         * find the maximum minimum distance and the corresponding point.  this
         * is our next FPS. The maxmin point must be one of the voronoi
         * radii. So we pick it from this smaller array. Note we only act on the
         * first i items as the array is filled incrementally picks max dist and
         * index of the cell
         */
        d2max_new = voronoi_r2.head(i).maxCoeff(&i_new);
        // the actual index of the fartest point
        i_new = voronoi_i_far(i_new);
#ifdef DO_TIMING
        auto tend = hrclock::now();
        tmax +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart)
                .count();
#endif
        // store properties of the new FPS selection
        sparse_indices(i) = i_new;
        sparse_minmax_d2(i - 1) = d2max_new;
        feature_new = feature_matrix.row(i_new);
        /*
         * we store indices of the selected features because we can then compute
         * some of the distances with contiguous array operations
         */
        feature_sel.row(i) = feature_new;

        /*
         * now we find the "active" Voronoi cells, i.e. those
         * that might change due to the new selection.
         */
        f_active = 0;

#ifdef DO_TIMING
        tstart = hrclock::now();
        ndist_active += i;
#endif
        /*
         * must compute distance of the new point to all the previous FPS.  some
         * of these might have been computed already, but bookkeeping could be
         * worse that recomputing (TODO: verify!)
         *
        list_sel_d2q.head(i) =
            feature_x2(i_new) -
            2 * (feature_sel.topRows(i) * feature_new).array();
        for (ssize_t j = 0; j < i; ++j) {
          list_sel_d2q(j) += feature_x2(sparse_indices(j));
        }
        list_sel_d2q.head(i) *= 0.25;  // triangle inequality: voronoi_r < d/2
        for (ssize_t j = 0; j < i; ++j) {
          /*
           * computes distances to previously selected points and uses triangle
           * inequality to find which voronoi sets might be affected by the
           * newly selected point divide by four so we don't have to do that
           * later to speed up the bound on distance to the new point
           */
          if (list_sel_d2q(j) < voronoi_r2(j)) {
            f_active(j) = 1;
            //! size of active cells will have to be recomputed
            voronoi_r2(j) = 0;
#ifdef DO_TIMING
          } else {
            ++npoint_skip;
#endif
          }
        }

#ifdef DO_TIMING
        tend = hrclock::now();
        tactive +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart)
                .count();

        tstart = hrclock::now();
#endif

        for (ssize_t j = 0; j < n_inputs; ++j) {
          int voronoi_idx_j = voronoi_indices(j);
          // only considers "active" points
          if (f_active(voronoi_idx_j) > 0) {
            /*
             * check if we can skip this check for point j. this is a tighter
             * bound on the distance, since |x_j-x_sel|<rvoronoi_sel
             *
            if (list_sel_d2q(voronoi_idx_j) < list_min_d2(j)) {
              double d2_j = feature_x2(i_new) + feature_x2(j) -
                            2 * feature_new.dot(feature_matrix.row(j));
              /*
               * we have to reassign point j to the new selection. also, the
               * voronoi center is actually that of the new selection
               */
              if (d2_j < list_min_d2(j)) {
                list_min_d2(j) = d2_j;
                voronoi_indices(j) = voronoi_idx_j = i;
              }
            }
            // also must update the voronoi radius
            if (list_min_d2(j) > voronoi_r2(voronoi_idx_j)) {
              voronoi_r2(voronoi_idx_j) = list_min_d2(j);
              // stores the index of the FP of the cell
              voronoi_i_far(voronoi_idx_j) = j;
            }
          }
        }

#ifdef DO_TIMING
        tend = hrclock::now();
        tloop +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(tend - tstart)
                .count();
#endif
      }
      sparse_minmax_d2(n_sparse - 1) = 0;

#ifdef DO_TIMING
      auto gtend = hrclock::now();

      std::cout << "Skipped " << npoint_skip << " FPS centers of "
                << n_sparse * (n_sparse - 1) / 2 << " - "
                << npoint_skip * 100. / (n_sparse * (n_sparse - 1) / 2)
                << "%\n";
      std::cout << "Computed " << ndist_eval << " distances rather than "
                << n_inputs * n_sparse << " - "
                << ndist_eval * 100. / (n_inputs * n_sparse) << " %\n";

      std::cout << "Time total "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(gtend -
                                                                        gtstart)
                           .count() *
                       1e-9
                << "\n";
      std::cout << "Time looking for max " << tmax * 1e-9 << "\n";
      std::cout << "Time looking for active " << tactive * 1e-9 << " with "
                << ndist_active << " distances\n";
      std::cout << "Time general loop " << tloop * 1e-9 << "\n";
#endif

      return std::make_tuple(sparse_indices, sparse_minmax_d2, list_min_d2,
                             voronoi_indices, voronoi_r2);
    }


"""

import numpy as np
from skcosmo.pcovr.pcovr_distances import pcovr_covariance, pcovr_kernel
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from skcosmo.utils import get_progress_bar


class GreedySelector(SelectorMixin, MetaEstimatorMixin, BaseEstimator):
    """ Selects features or samples in an iterative way """

    def __init__(self, n_select=None, support=None, kernel=None):
        self.support_ = support  # we can pass on a np.array[bool] as support parameter (which we will choose)
        self.n_select_ = n_select
        self.n_selected_ = 0

        if kernel is None:
            kernel = lambda x: x
        self.kernel_ = kernel  # TODO implement support of providing a kernel function, sklearn style

    def _get_support_mask(self):
        check_is_fitted(self)
        return self.support_


class SimpleFPS(GreedySelector):
    def fit(self, X, initial=0):

        self.support_ = np.zeros(X.shape[0], int)
        self.select_distance_ = np.zeros(X.shape[0], float)
        self.norms_ = (X ** 2).sum(axis=1)

        # first point
        self.support_[0] = initial

        # distance of all points to the selected point
        self.haussdorf_ = self.norms_ + self.norms_[initial] - 2 * X[initial] @ X.T

        for i in range(1, self.n_select_):
            # finds the point with maximum haussdorf distance
            isel = self.haussdorf_.argmax()

            # updates support and tracks maximum minimum distance to selection
            self.support_[i] = isel
            self.select_distance_[i - 1] = self.haussdorf_[isel]

            # distances of all points to the new point
            idistance = self.norms_ + self.norms_[isel] - 2 * X[isel] @ X.T
            # updates haussdorf distances
            self.haussdorf_ = np.minimum(self.haussdorf_, idistance)

        self.n_selected_ = self.n_select_


def _calc_distances_(K, ref_idx, idxs=None):
    """
    Calculates the distance between points in ref_idx and idx

    Assumes

    .. math::
        d(i, j) = K_{i,i} - 2 * K_{i,j} + K_{j,j}

    : param K : distance matrix, must contain distances for ref_idx and idxs
    : type K : array

    : param ref_idx : index of reference points
    : type ref_idx : int

    : param idxs : indices of points to compute distance to ref_idx
                   defaults to all indices in K
    : type idxs : list of int, None

    """
    if idxs is None:
        idxs = range(K.shape[0])
    return np.array(
        [np.real(K[j][j] - 2 * K[j][ref_idx] + K[ref_idx][ref_idx]) for j in idxs]
    )


from time import time

class SimpleVoronoiFPS(GreedySelector):
    """
    Base Class defined for Voronoi FPS selection methods

    :param idx: predetermined index; if None provided, first index selected
                 is 0
    :type idx: int, None
    """

    def fit(self, X, initial=0):
        """Method for FPS select based upon a product of the input matrices

        Parameters
        ----------
        :param X: working matrix
        :type X: np.array(n_samples, n_features)

        :param initial: initial choosed point for algorithm
        :type initial: int

        Returns
        -------
        idx: list of n selections
        """

        self.idx = [initial]
        self.norms_ = (X ** 2).sum(axis=1)

        # distance of all points to the selected point
        self.haussdorf_ = self.norms_ + self.norms_[initial] - 2 * X[initial] @ X.T
    
        # assignment points to Voronoi cell (initially we have 1 Voronoi cell)
        # this is the list of the index of the selected point that is the center of the
        # Voronoi cell to which each point in X belongs to
        self.voronoi_number = np.full(self.haussdorf_.shape[0], 0)                

        if self.n_select_ <= 0:
            raise ValueError("You must call select(n) with n > 0.")

        # tracks selected points
        self.Xsel_ = np.zeros((self.n_select_,X.shape[1]),float)
        self.Xsel_[0] = X[initial]
        self.nsel_= np.zeros(self.n_select_,float)
        self.nsel_[0] = self.norms_[initial]

        # index of the maximum - d2 point in each voronoi cell
        # this is the index of the point which is farthest from the center in eac
        # voronoi cell.         
        self.voronoi_i_far = np.zeros(self.n_select_, int)
        self.voronoi_i_far[0] = np.argmax(self.haussdorf_)
        self.voronoi_np = np.zeros(self.n_select_, int)
        self.voronoi_np[0] = X.shape[0]

        # define the voronoi_r2 for the idx point
        # this is the maximum distance from the center for each of the cells
        self.voronoi_r2 = np.zeros(self.n_select_, float)
        self.voronoi_r2[0] = self.haussdorf_[self.voronoi_i_far[0]]
        
        f_active = np.zeros(self.n_select_, bool)        
        sel_d2q = np.zeros(self.n_select_, float)
        f_mask = np.zeros(X.shape[0], bool)
        
        time1 = 0
        time2 = 0
        time3 = 0
        time4 = 0
        time5 = 0
        time6 = 0
        tnsel = np.zeros(self.n_select_, int)
        # Loop over the remaining points...        
        for i in range(len(self.idx), self.n_select_ - 1):
            """Find the maximum minimum (maxmin) distance and the corresponding point. This
            is our next FPS. The maxmin point must be one of the Voronoi
            radii. So we pick it from this smaller array. Note we only act on the
            first i items as the array is filled incrementally picks max dist and
            index of the cell
            """
            
            # the new farthest point must be one of the "farthest from its cell" points
            # so we don't need to loop over all points to find it
            # print("Voronoi size ", self.voronoi_np[:i])
            c_new = self.voronoi_r2[:i].argmax()            
            i_new = self.voronoi_i_far[c_new]
            
            time1 -= time()
            f_active[:i] = False
            nsel = 0
            """must compute distance of the new point to all the previous FPS. Some
               of these might have been computed already, but bookkeeping could be
               worse that recomputing (TODO: verify!)
            """
            # calculation in a single block
            sel_d2q[:i] = (self.nsel_[:i] + self.norms_[i_new] - 2 * X[i_new] @ self.Xsel_[:i].T) * 0.25
            
            for ic in range(i):                
                # empty voronoi, no need to consider it
                if self.voronoi_np[ic] > 1 and sel_d2q[ic] < self.voronoi_r2[ic]:
                    # these voronoi cells need to be updated
                    f_active[ic] = True
                    self.voronoi_r2[ic] = 0
                    nsel += self.voronoi_np[ic]
            f_active[i] = True
            tnsel[i] = nsel
            time1 += time()
         
            if nsel > len(X)//6:
                # it's better to do a standard update.... 
                time4 -= time()                
                all_dist =  (self.norms_+ self.norms_[i_new] - 2 * X[i_new] @ X.T )                 
                time4 += time()
                time2 -= time()
                
                #print(f_active[:i])            
                l_update = np.where(all_dist < self.haussdorf_)[0]
                self.haussdorf_[l_update] = all_dist[l_update]
                for j in l_update:
                    self.voronoi_np[self.voronoi_number[j]] -= 1
                self.voronoi_number[l_update] = i
                self.voronoi_np[i] = len(l_update)
                time2 += time()                
                time6 -= time()                
                                
                for ic in np.where(f_active)[0]:
                    jc = np.where(self.voronoi_number == ic)[0]
                    if len(jc) == 0:
                        continue
                    self.voronoi_i_far[ic] = jc[np.argmax(self.haussdorf_[jc])]
                    self.voronoi_r2[ic] = self.haussdorf_[self.voronoi_i_far[ic]]

                #print(nsel, len(X),self.voronoi_np[:(i+1)])                
                # ~ for j in range(self.haussdorf_.shape[0]):
                    # ~ # check only "active" cells
                    # ~ jcell = self.voronoi_number[j]
                    # ~ if f_active[jcell]:
                        # ~ # if this point was assigned to the new cell, we need update data for this polyhedra.
                        # ~ # Vice versa, we need to update the data for the cell, because we set voronoi_r2 as zero
                        # ~ if self.haussdorf_[j] > self.voronoi_r2[jcell]:
                            # ~ #print(j, i_new, i, self.voronoi_number[j])
                            # ~ self.voronoi_r2[jcell] = self.haussdorf_[j]
                            # ~ self.voronoi_i_far[jcell] = j
                #print("r2", jcell, self.voronoi_r2[:i]) 
                time6 += time()                
            else:
                time3-=time()
                for j in range(self.haussdorf_.shape[0]):
                    # check only "active" cells
                    if f_active[self.voronoi_number[j]]:
                        # check, can this point be in a new polyhedron or not
                        if sel_d2q[self.voronoi_number[j]] < self.haussdorf_[j]:
                            time5 -= time()
                            d2_j = (self.norms_[j] + self.norms_[i_new] 
                                     - 2 * X[j] @ X[i_new].T )                        
                            time5 += time()
                            # assign a point to the new polyhedron
                            if self.haussdorf_[j] > d2_j:
                                self.haussdorf_[j] = d2_j
                                self.voronoi_np[self.voronoi_number[j]] -= 1
                                self.voronoi_number[j] = i
                                self.voronoi_np[i] += 1
                        # if this point was assigned to the new cell, we need update data for this polyhedron.
                        # vice versa, we need to update the data for the cell, because we set voronoi_r2 as zero
                        if self.haussdorf_[j] > self.voronoi_r2[self.voronoi_number[j]]:
                            self.voronoi_r2[self.voronoi_number[j]] = self.haussdorf_[j]
                            self.voronoi_i_far[self.voronoi_number[j]] = j
                time3+=time()
            self.idx.append(i_new)
            self.Xsel_[i] = X[i_new]
            self.nsel_[i] = self.norms_[i_new]
        self.support_ = np.array(
            [True if i in self.idx else False for i in range(self.haussdorf_.shape[0])]
        )
        print("Timing: ", time1, " s " , time2, " s " , time6, " s ", time4, " s ", time3-time5, " s ", time5, " s ")
        #print("NSel: ", tnsel)
        return self, tnsel

    def calc_distance(self, idx_1, idx_2=None):
        """
            Abstract method to be used for calculating the distances
            between two indexed points. Should be overwritten if default
            functionality is not desired

        : param idx_1 : index of first point to use
        : type idx_1 : int

        : param idx_2 : index of first point to use; if None, calculates the
                        distance between idx_1 and all points
        : type idx_2 : list of int or None
        """
        return _calc_distances_(self.product, idx_1, idx_2)


class SampleVoronoiFPS(SimpleVoronoiFPS):
    """

    For sample selection, traditional FPS employs a row-wise Euclidean
    distance, which can be expressed using the Gram matrix
    :math:`\\mathbf{K} = \\mathbf{X} \\mathbf{X}^T`

    .. math::
        \\operatorname{d}_r(i, j) = K_{ii} - 2 K_{ij} + K_{jj}.

    When mixing < 1, this will use PCov-FPS, where a modified Gram matrix is
    used to express the distances

    .. math::
        \\mathbf{\\tilde{K}} = \\alpha \\mathbf{XX}^T +
        (1 - \\alpha)\\mathbf{\\hat{Y}\\hat{Y}}^T

    :param idxs: predetermined indices; if None provided, first index selected
                 is random
    :type idxs: list of int, None

    :param X: Data matrix :math:`\\mathbf{X}` from which to select a
                   subset of the `n` rows
    :type X: array of shape (n x m)

    :param mixing: mixing parameter, as described in PCovR as
                  :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param progress_bar: option to use `tqdm <https://tqdm.github.io/>`_
                         progress bar to monitor selections
    :type progress_bar: boolean

    :param tol: threshold below which values will be considered 0,
                      defaults to 1E-12
    :type tol: float

    :param Y: array to include in biased selection when mixing < 1;
              required when mixing < 1, throws AssertionError otherwise
    :type Y: array of shape (n x p), optional when :math:`{\\alpha = 1}`

    """

    def __init__(self, X, mixing=1.0, tol=1e-12, Y=None, **kwargs):

        self.mixing = mixing
        self.tol = tol

        if mixing < 1:
            try:
                self.A, self.Y = check_X_y(X, Y, copy=True, multi_output=True)
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.A, self.Y = check_array(X, copy=True), None

        self.product = pcovr_kernel(self.mixing, self.A, self.Y)
        super().__init__(tol=tol, **kwargs)


class FeatureVoronoiFPS(SimpleVoronoiFPS):
    """

    For feature selection, traditional FPS employs a column-wise Euclidean
    distance, which can be expressed using the covariance matrix
    :math:`\\mathbf{C} = \\mathbf{X} ^ T \\mathbf{X}`

    .. math::
        \\operatorname{d}_c(i, j) = C_{ii} - 2 C_{ij} + C_{jj}.

    When mixing < 1, this will use PCov-FPS, where a modified covariance matrix
    is used to express the distances

    .. math::
        \\mathbf{\\tilde{C}} = \\alpha \\mathbf{X}^T\\mathbf{X} +
        (1 - \\alpha)(\\mathbf{X}^T\\mathbf{X})^{-1/2}\\mathbf{X}^T
        \\mathbf{\\hat{Y}\\hat{Y}}^T\\mathbf{X}(\\mathbf{X}^T\\mathbf{X})^{-1/2}

    :param idxs: predetermined indices; if None provided, first index selected
                 is random
    :type idxs: list of int, None

    :param X: Data matrix :math:`\\mathbf{X}` from which to select a
                   subset of the `n` columns
    :type X: array of shape (n x m)

    :param mixing: mixing parameter, as described in PCovR as
                   :math:`{\\alpha}`, defaults to 1
    :type mixing: float

    :param progress_bar: option to use `tqdm <https://tqdm.github.io/>`_
                         progress bar to monitor selections
    :type progress_bar: boolean

    :param tol: threshold below which values will be considered 0,
                      defaults to 1E-12
    :type tol: float

    :param Y: array to include in biased selection when mixing < 1;
              required when mixing < 1, throws AssertionError otherwise
    :type Y: array of shape (n x p), optional when :math:`{\\alpha = 1}`


    """

    def __init__(self, X, mixing=1.0, tol=1e-12, Y=None, **kwargs):

        self.mixing = mixing
        self.tol = tol

        if mixing < 1:
            try:
                self.A, self.Y = check_X_y(X, Y, copy=True, multi_output=True)
            except AssertionError:
                raise Exception(r"For $\alpha < 1$, $Y$ must be supplied.")
        else:
            self.A, self.Y = check_array(X, copy=True), None
        self.product = pcovr_covariance(self.mixing, self.A, self.Y, rcond=self.tol)
        super().__init__(tol=tol, **kwargs)
