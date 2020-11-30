import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
import umap
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from NeuralNetworkRegressors import Neural_Regression


class subset_genes_Dim_Reduced_ElasticNet(ElasticNet):

    def __init__(self, alpha=1.0, l1_ratio=0.7, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic',
                 subset_min=15, subset_fold=15, subset_logT=True, dimreduced=True, ndim=3, dimmethod='pca',
                 convfpkmToTpm=False, verbose=True):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.convfpkmToTpm = convfpkmToTpm
        self.verbose = verbose
        self.dimreduced = dimreduced
        self.dimension = ndim
        self.dimmethod = dimmethod
        self.reducer = None

        super(subset_genes_Dim_Reduced_ElasticNet, self).__init__(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, max_iter=max_iter,
            copy_X=copy_X, tol=tol, warm_start=warm_start, positive=positive,
            random_state=random_state, selection=selection)

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log2(data + 1)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    def _dimension_reduction(self, data, type='train'):
        embedding = None

        if type== 'test' and self.reducer is not None:
            data = pd.DataFrame(self.sc.transform(data))
            embedding = self.reducer.transform(data)
            embedding = normalize(embedding,axis=0)

        if type=='train':
            if self.dimmethod == 'pca':
                self.reducer = PCA(n_components=self.dimension)
            if self.dimmethod == 'umap':
                self.reducer = umap.UMAP(n_components=self.dimension)
            self.sc = StandardScaler()
            data = pd.DataFrame(self.sc.fit_transform(data))
            embedding = self.reducer.fit_transform(data)
            embedding = normalize(embedding,axis=0)

        return pd.DataFrame(embedding).fillna(0)


    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=True, these_genes=np.array(False)):
        if these_genes.any():  # just get these_genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
        else:  # calculate the these_genes list based on the other parameters

            genes = self._expr_levels(data, logTrans=False)  # operate selection criteria on non-log FPKM
            eps = 0.001
            has_start_gt = genes > self.subset_min
            mvt = genes.min()
            mvt[mvt < eps] = eps
            has_fold_change = (genes.max() / mvt) >= self.subset_fold

            aset = (has_start_gt.any(axis=0) & has_fold_change)
            these_genes = aset[aset].index  # get list of gene names that meet criteria
            genes = self._expr_levels(data, logTrans=self.subset_logT)  # do the final op on user-selected xform

        subgenes = genes.loc[:, these_genes]
        if verbose:
            print(
                'using {} genes in subset requiring a max TPM > {} and > {}fold change between max and min samples '.format(
                    len(these_genes), self.subset_min, self.subset_fold))

        return subgenes


    def predict(self, X):
        """Perform regression on samples in X.
        For an one-class model, +1 (inlier) or -1 (outlier) is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).
        Returns
        -------
        y_pred : array, shape (n_samples,)
        """
        check_is_fitted(self, 'genecolumns_')
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        if self.dimreduced is True and len(self.genecolumns_) >= self.dimension:
            X_sub = self._dimension_reduction(X_sub,type='test')

       # X_sub = super(subset_genes_LinRegr, self)._validate_for_predict(X_sub)
        return super(subset_genes_Dim_Reduced_ElasticNet, self).predict(X_sub)

    def fit(self, X, y, check_input=True):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'subset_logT')
        check_is_fitted(self, 'convfpkmToTpm')
        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        if self.dimreduced is True and len(self.genecolumns_) >= self.dimension:
            print('Selected Genes',len(self.genecolumns_))
            X_sub = self._dimension_reduction(X_sub,type='train')
        super(subset_genes_Dim_Reduced_ElasticNet, self).fit(X_sub, y, check_input)
        return self


class subset_genes_Dim_Reduced_LinRegr(LinearRegression):

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True,
                 n_jobs=1, subset_min=1500, subset_fold=1500, subset_logT=True, dimreduced=True, ndim=3, dimmethod='pca',
                 convfpkmToTpm=False, verbose=False):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.convfpkmToTpm = convfpkmToTpm
        self.dimreduced = dimreduced
        self.dimension = ndim
        self.dimmethod = dimmethod
        self.verbose = verbose

        super(subset_genes_Dim_Reduced_LinRegr, self).__init__(
            fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log2(data + 1)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    def _dimension_reduction(self, data, type='train'):
        embedding = None

        if type == 'test' and self.reducer is not None:
            #data = pd.DataFrame(self.sc.transform(data))
            embedding = self.reducer.transform(data)
            #embedding = self.post_sc.transform(embedding)

        if type == 'train':
            if self.dimmethod == 'pca':
                self.reducer = PCA(n_components=self.dimension)
            if self.dimmethod == 'umap':
                self.reducer = umap.UMAP(n_components=self.dimension)
            #self.sc = StandardScaler()
            #data = pd.DataFrame(self.sc.fit_transform(data))
            embedding = self.reducer.fit_transform(data)
            #self.post_sc = StandardScaler()
            #embedding = self.post_sc.fit_transform(embedding)

        return pd.DataFrame(embedding).fillna(0)


    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=True, these_genes=np.array(False)):
        if these_genes.any():  # just get these_genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
        else:  # calculate the these_genes list based on the other parameters

            genes = self._expr_levels(data, logTrans=False)  # operate selection criteria on non-log FPKM
            eps = 0.001
            has_start_gt = genes > self.subset_min
            mvt = genes.min()
            mvt[mvt < eps] = eps
            has_fold_change = (genes.max() / mvt) > self.subset_fold

            aset = (has_start_gt.any(axis=0) & has_fold_change)
            these_genes = aset[aset].index  # get list of gene names that meet criteria
            genes = self._expr_levels(data, logTrans=self.subset_logT)  # do the final op on user-selected xform

        subgenes = genes.loc[:, these_genes]
        if verbose:
            print(
                'using {} genes in subset requiring a max TPM > {} and > {}fold change between max and min samples '.format(
                    len(these_genes), self.subset_min, self.subset_fold))
        return subgenes

    def predict(self, X):
        """Perform regression on samples in X.
        For an one-class model, +1 (inlier) or -1 (outlier) is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).
        Returns
        -------
        y_pred : array, shape (n_samples,)
        """
        check_is_fitted(self, 'genecolumns_')
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        if self.dimreduced is True:
            X_sub = self._dimension_reduction(X_sub,type='test')
            print(X_sub)
        X_sub = super(subset_genes_Dim_Reduced_LinRegr, self)._validate_for_predict(X_sub)
        return super(subset_genes_Dim_Reduced_LinRegr, self).predict(X_sub)

    def fit(self, X, y, sample_weight=None):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'subset_logT')
        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        if self.dimreduced is True:
            X_sub = self._dimension_reduction(X_sub,type='train')
        super(subset_genes_Dim_Reduced_LinRegr, self).fit(X_sub, y, sample_weight)
        return self


class subset_genes_Dim_Reduced_SVR(SVR):

    def __init__(self, kernel='poly', degree=2, gamma='auto', coef0=0.0,
                 tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 cache_size=200, verbose=True, max_iter=-1,
                 subset_min=200, subset_fold=100, subset_logT=True,
                  dimreduced=True, ndim=3, dimmethod='pca',):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.dimreduced = dimreduced
        self.dimmethod = dimmethod
        self.dimension = ndim
        self.reduced = None

        super(subset_genes_Dim_Reduced_SVR, self).__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, epsilon=epsilon, verbose=verbose,
            shrinking=shrinking, cache_size=cache_size, max_iter=max_iter)

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log2(data + 1)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    def _dimension_reduction(self, data, type='train'):
        embedding = None

        if type == 'test' and self.reducer is not None:
            data = pd.DataFrame(self.sc.transform(data))
            embedding = self.reducer.transform(data)
            embedding = self.post_sc.transform(embedding)

        if type == 'train':
            if self.dimmethod == 'pca':
                self.reducer = PCA(n_components=self.dimension)
            if self.dimmethod == 'umap':
                self.reducer = umap.UMAP(n_components=self.dimension)
            self.sc = StandardScaler()
            data = pd.DataFrame(self.sc.fit_transform(data))
            embedding = self.reducer.fit_transform(data)
            self.post_sc = StandardScaler()
            embedding = self.post_sc.fit_transform(embedding)

        return pd.DataFrame(embedding).fillna(0)

    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=True, these_genes=np.array(False)):
        if these_genes.any():  # just get these_genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
        else:  # calculate the these_genes list based on the other parameters

            genes = self._expr_levels(data, logTrans=False)  # operate selection criteria on non-log FPKM
            eps = 0.001
            has_start_gt = genes > self.subset_min
            mvt = genes.min()
            mvt[mvt < eps] = eps
            has_fold_change = (genes.max() / mvt) > self.subset_fold

            aset = (has_start_gt.any(axis=0) & has_fold_change)
            these_genes = aset[aset].index  # get list of gene names that meet criteria
            genes = self._expr_levels(data, logTrans=self.subset_logT)  # do the final op on user-selected xform

        subgenes = genes.loc[:, these_genes]
        if verbose:
            print(
                'using {} genes in subset requiring a max TPM > {} and > {}fold change between max and min samples '.format(
                    len(these_genes), self.subset_min, self.subset_fold))
        return subgenes

    def predict(self, X):
        """Perform regression on samples in X.
        For an one-class model, +1 (inlier) or -1 (outlier) is returned.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of X is
            (n_samples_test, n_samples_train).
        Returns
        -------
        y_pred : array, shape (n_samples,)
        """
        check_is_fitted(self, 'genecolumns_')
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        if self.dimreduced is True:
            X_sub = self._dimension_reduction(X_sub, type='test')
        print(X_sub)
        X_sub = super(subset_genes_Dim_Reduced_SVR, self)._validate_for_predict(X_sub)
        predict = self._sparse_predict if self._sparse else self._dense_predict
        return predict(X_sub)

    def fit(self, X, y, sample_weight=None):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'subset_logT')
        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        if self.dimreduced is True:
            X_sub = self._dimension_reduction(X_sub, type='train')

        super(subset_genes_Dim_Reduced_SVR, self).fit(X_sub, y, sample_weight)
        return self


class subset_genes_Dim_Reduced_GPR(GaussianProcessRegressor):

    def __init__(self, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                 n_restarts_optimizer=0, normalize_y=True, copy_X_train=False, random_state=None,
                 subset_min=50, subset_fold=50, verbose=False,dosubset = True, subset_logT=True,
                 dimreduced=True, ndim=3, dimmethod='pca'):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.verbose = verbose
        self.dimreduced = dimreduced
        self.dimension = ndim
        self.dimmethod = dimmethod
        self.dosubset = dosubset
        self.reducer = None

        super(subset_genes_Dim_Reduced_GPR, self).__init__(
            kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0, normalize_y=True, copy_X_train=False, random_state=None)

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log2(data + 1)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=False, these_genes=np.array(False)):
        if these_genes.any():  # just get these_genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
        else:  # calculate the these_genes list based on the other parameters

            genes = self._expr_levels(data, logTrans=False)  # operate selection criteria on non-log FPKM
            eps = 0.001
            has_start_gt = genes >= self.subset_min
            mvt = genes.min()
            mvt[mvt < eps] = eps
            has_fold_change = (genes.max() / mvt) >= self.subset_fold

            aset = (has_start_gt.any(axis=0) & has_fold_change)
            these_genes = aset[aset].index  # get list of gene names that meet criteria
            genes = self._expr_levels(data, logTrans=self.subset_logT)  # do the final op on user-selected xform

        subgenes = genes.loc[:, these_genes]
        if verbose:
            print(
                'using {} genes in subset requiring a max TPM > {} and > {}fold change between max and min samples '.format(
                    len(these_genes), self.subset_min, self.subset_fold))
        return subgenes

    def _dimension_reduction(self, data, type='train'):
        embedding = None

        if type == 'test' and self.reducer is not None:
            data = self.sc.transform(data)
            embedding = self.reducer.transform(data)
           # embedding = normalize(embedding)

        if type == 'train':
            if self.dimmethod == 'pca':
                self.reducer = PCA(n_components=self.dimension)
            if self.dimmethod == 'umap':
                self.reducer = umap.UMAP(n_components=self.dimension)
            self.sc = StandardScaler()
            data = self.sc.fit_transform(data)
            embedding = self.reducer.fit_transform(data)
            #embedding = normalize(embedding)

        return pd.DataFrame(embedding).fillna(0)

    def predict(self, X):

        check_is_fitted(self, 'genecolumns_')
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub,type='test')

        #print(X_sub)
        return super(subset_genes_Dim_Reduced_GPR, self).predict(X_sub,return_std=True)

    def fit(self, X, y, sample_weight=None):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'subset_logT')
        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub,type='train')
        super(subset_genes_Dim_Reduced_GPR, self).fit(X_sub, y)
        return self

class subset_genes_dim_Reduced_Neural_Regression(Neural_Regression):
    def __init__(self, learning_rate = 0.0001, num_epochs = 1000,
                subset_min=50, subset_fold=50, verbose=False,dosubset = True, subset_logT=True,
                 dimreduced=True, ndim=3, dimmethod='pca'):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.verbose = verbose
        self.dimreduced = dimreduced
        self.dimension = ndim
        self.dimmethod = dimmethod
        self.dosubset = dosubset
        self.reducer = None

        super(subset_genes_dim_Reduced_Neural_Regression,self).__init__(in_size=self.dimension, out_size=1, learning_rate=learning_rate,num_epochs=num_epochs)


    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log2(data + 1)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=False, these_genes=np.array(False)):
        if these_genes.any():  # just get these_genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
        else:  # calculate the these_genes list based on the other parameters

            genes = self._expr_levels(data, logTrans=False)  # operate selection criteria on non-log FPKM
            eps = 0.001
            has_start_gt = genes >= self.subset_min
            mvt = genes.min()
            mvt[mvt < eps] = eps
            has_fold_change = (genes.max() / mvt) >= self.subset_fold

            aset = (has_start_gt.any(axis=0) & has_fold_change)
            these_genes = aset[aset].index  # get list of gene names that meet criteria
            genes = self._expr_levels(data, logTrans=self.subset_logT)  # do the final op on user-selected xform

        subgenes = genes.loc[:, these_genes]
        if verbose:
            print(
                'using {} genes in subset requiring a max TPM > {} and > {}fold change between max and min samples '.format(
                    len(these_genes), self.subset_min, self.subset_fold))
        return subgenes

    def _dimension_reduction(self, data, type='train'):
        embedding = None

        if type == 'test' and self.reducer is not None:
            data = self.sc.transform(data)
            embedding = self.reducer.transform(data)
            #embedding = normalize(embedding)

        if type == 'train':
            if self.dimmethod == 'pca':
                self.reducer = PCA(n_components=self.dimension)
            if self.dimmethod == 'umap':
                self.reducer = umap.UMAP(n_components=self.dimension)
            self.sc = StandardScaler()
            data = self.sc.fit_transform(data)
            embedding = self.reducer.fit_transform(data)
            embedding = normalize(embedding)

        return pd.DataFrame(embedding).fillna(0)

    def predict(self, X):
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub,type='test')

        #print(X_sub)
        return super(subset_genes_dim_Reduced_Neural_Regression, self).predict(X_sub)

    def fit(self, X, y, sample_weight=None):

        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub,type='train')
        super(subset_genes_dim_Reduced_Neural_Regression, self).fit(X_sub, y)
        return self
