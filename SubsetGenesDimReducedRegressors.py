import numpy as np
import pandas as pd
import os
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.linear_model import LinearRegression, ElasticNet, ARDRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
import umap
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from NeuralNetworkRegressors import Neural_Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# some of the classes in this file are modified from https://github.com/jasongfleischer/Predicting-age-from-the-transcriptome-of-human-dermal-fibroblasts/releases

class subset_genes_Dim_Reduced_ElasticNet(ElasticNet):

    def __init__(self, alpha=1.0, l1_ratio=0.7, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 varying_genes = False, flavor='seurat', topgene = 10,
                 random_state=None, selection='cyclic',
                 subset_min=15, subset_fold=15, subset_logT=True, dimreduced=True, dimension=3, dimmethod='pca',
                 convfpkmToTpm=False, verbose=True):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.convfpkmToTpm = convfpkmToTpm
        self.verbose = verbose
        self.dimreduced = dimreduced
        self.dimension = dimension
        self.dimmethod = dimmethod
        self.varying_genes = varying_genes
        self.scanpymethod = flavor
        self.topgene = topgene
        self.reducer = None

        super(subset_genes_Dim_Reduced_ElasticNet, self).__init__(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=fit_intercept,
            normalize=normalize, precompute=precompute, max_iter=max_iter,
            copy_X=copy_X, tol=tol, warm_start=warm_start, positive=positive,
            random_state=random_state, selection=selection)

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log1p(data)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    def _dimension_reduction(self, data, type='train'):
        embedding = None

        if type == 'test' and self.reducer is not None:
            data = pd.DataFrame(self.sc.transform(data))
            embedding = self.reducer.transform(data)
            #embedding = normalize(embedding, axis=0)

        if type == 'train':
            if self.dimmethod == 'pca':
                self.reducer = PCA(n_components=self.dimension)
            if self.dimmethod == 'umap':
                self.reducer = umap.UMAP(n_components=self.dimension)
            self.sc = StandardScaler()
            data = pd.DataFrame(self.sc.fit_transform(data))
            embedding = self.reducer.fit_transform(data)
            #embedding = normalize(embedding, axis=0)

        return pd.DataFrame(embedding).fillna(0)

    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=True, these_genes=np.array(False)):
        if self.varying_genes is True: # Use scanpy selected genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
            with open(os.path.join('data', f'{self.scanpymethod}_bulk_highly_variable_top{self.topgene}.txt'), 'r') as f:
                these_genes = [int(x) for x in f.read().split(' ')]
        else:
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
        print('In Predict')
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        if self.dimreduced is True and len(self.genecolumns_) >= self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='test')

        # X_sub = super(subset_genes_LinRegr, self)._validate_for_predict(X_sub)
        return super(subset_genes_Dim_Reduced_ElasticNet, self).predict(X_sub)

    def fit(self, X, y, check_input=True):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'subset_logT')
        check_is_fitted(self, 'convfpkmToTpm')
        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        print(len(self.genecolumns_),self.dimension)
        if self.dimreduced is True and len(self.genecolumns_) >= self.dimension:
            print('Selected Genes', len(self.genecolumns_))
            X_sub = self._dimension_reduction(X_sub, type='train')
        super(subset_genes_Dim_Reduced_ElasticNet, self).fit(X_sub, y, check_input)
        return self

class subset_genes_Dim_Reduced_SVR(SVR):

    def __init__(self, kernel='poly', degree=2, gamma='auto', coef0=0.0,
                 tol=1e-3, C=1.0, epsilon=0.1, shrinking=True,
                 varying_genes=False, flavor='seurat', topgene=10,
                 cache_size=200, verbose=True, max_iter=-1,
                 subset_min=15, subset_fold=15, subset_logT=True,
                 dimreduced=True, dimension=3, dimmethod='pca', ):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.dimreduced = dimreduced
        self.dimmethod = dimmethod
        self.dimension = dimension
        self.varying_genes = varying_genes
        self.scanpymethod = flavor
        self.topgene = topgene
        self.reduced = None

        super(subset_genes_Dim_Reduced_SVR, self).__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, epsilon=epsilon, verbose=verbose,
            shrinking=shrinking, cache_size=cache_size, max_iter=max_iter)

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log1p(data)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    def _dimension_reduction(self, data, type='train'):
        embedding = None

        if type == 'test' and self.reducer is not None:
            data = pd.DataFrame(self.sc.transform(data))
            embedding = self.reducer.transform(data)
            # embedding = self.post_sc.transform(embedding)

        if type == 'train':
            if self.dimmethod == 'pca':
                self.reducer = PCA(n_components=self.dimension)
            if self.dimmethod == 'umap':
                self.reducer = umap.UMAP(n_components=self.dimension)
            self.sc = StandardScaler()
            data = pd.DataFrame(self.sc.fit_transform(data))
            embedding = self.reducer.fit_transform(data)
        # self.post_sc = StandardScaler()
        # embedding = self.post_sc.fit_transform(embedding)

        return pd.DataFrame(embedding).fillna(0)

    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=True, these_genes=np.array(False)):
        if self.varying_genes is True: # Use scanpy selected genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
            with open(os.path.join('data', f'{self.scanpymethod}_bulk_highly_variable_top{self.topgene}.txt'), 'r') as f:
                these_genes = [int(x) for x in f.read().split(' ')]
        else:
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
        print(self.genecolumns_)
        if self.dimreduced is True:
            X_sub = self._dimension_reduction(X_sub, type='train')

        super(subset_genes_Dim_Reduced_SVR, self).fit(X_sub, y, sample_weight)
        return self


class subset_genes_dim_Reduced_Neural_Regression(Neural_Regression):
    def __init__(self, learning_rate=0.0001, num_epochs=1000,
                 subset_min=50, subset_fold=50, verbose=False,
                 subset_logT=True,
                 varying_genes=False, flavor='seurat', topgene=10,
                 dimreduced=True, dimension=3, dimmethod='pca'):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.verbose = verbose
        self.dimreduced = dimreduced
        self.dimension = dimension
        self.dimmethod = dimmethod
        self.varying_genes = varying_genes
        self.scanpymethod = flavor
        self.topgene = topgene

        super(subset_genes_dim_Reduced_Neural_Regression, self).__init__(in_size=self.dimension, out_size=1,
                                                                         learning_rate=learning_rate,
                                                                         num_epochs=num_epochs)

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log1p(data)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=False, these_genes=np.array(False)):
        if self.varying_genes is True: # Use scanpy selected genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
            with open(os.path.join('data', f'{self.scanpymethod}_bulk_highly_variable_top{self.topgene}.txt'), 'r') as f:
                these_genes = [int(x) for x in f.read().split(' ')]
        else:
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
            if self.varying_genes is True:
                print('using {} most varying genes from seurat',len(these_genes))
            else:
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
            # embedding = normalize(embedding)

        return pd.DataFrame(embedding).fillna(0)

    def predict(self, X):
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='test')

        pred = super(subset_genes_dim_Reduced_Neural_Regression, self).predict(X_sub)
        print('Predicted',pred)
        return pred

    def fit(self, X, y, sample_weight=None):

        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='train')
        super(subset_genes_dim_Reduced_Neural_Regression, self).fit(X_sub, y)
        return self


class subset_genes_Dim_Reduced_GPR(GaussianProcessRegressor):

    def __init__(self, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b',
                 n_restarts_optimizer=0, normalize_y=True, copy_X_train=False, random_state=None,
                 varying_genes=False, flavor='seurat', topgene=10,
                 subset_min=50, subset_fold=50, verbose=False, dosubset=True, subset_logT=True,
                 dimreduced=True, dimension=3, dimmethod='pca'):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.verbose = verbose
        self.dimreduced = dimreduced
        self.dimension = dimension
        self.dimmethod = dimmethod
        self.dosubset = dosubset
        self.varying_genes = varying_genes
        self.scanpymethod = flavor
        self.topgene = topgene
        self.reducer = None

        super(subset_genes_Dim_Reduced_GPR, self).__init__(
            kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b',
            n_restarts_optimizer=0, normalize_y=True, copy_X_train=False, random_state=None)

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log1p(data)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=False, these_genes=np.array(False)):
        if self.varying_genes is True: # Use scanpy selected genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
            with open(os.path.join('data', f'{self.scanpymethod}_bulk_highly_variable_top{self.topgene}.txt'), 'r') as f:
                these_genes = [int(x) for x in f.read().split(' ')]
        else:
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
            # embedding = normalize(embedding)

        return pd.DataFrame(embedding).fillna(0)

    def predict(self, X):

        check_is_fitted(self, 'genecolumns_')
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='test')

        X_sub = normalize(X_sub)
        return super(subset_genes_Dim_Reduced_GPR, self).predict(X_sub, return_std=True)

    def fit(self, X, y, sample_weight=None):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'subset_logT')
        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='train')
        X_sub = normalize(X_sub)
        super(subset_genes_Dim_Reduced_GPR, self).fit(X_sub, y)
        return self


class subset_genes_dim_Reduced_ARDRegression(ARDRegression):
    def __init__(self, n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6,
                 lambda_1=1.e-6, lambda_2=1.e-6, compute_score=False,
                 threshold_lambda=1.e+4, fit_intercept=True, normalize=False,
                 copy_X=True, varying_genes=False, scanpymethod='seurat', topgene=10,
                 verbose=False, subset_min=15, subset_fold=15,
                 subset_logT=True, dimreduced=False, dimension=3, dimmethod='pca',
                 convfpkmToTpm=False):

        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.convfpkmToTpm = convfpkmToTpm
        self.dimreduced = dimreduced
        self.dimension = dimension
        self.dimmethod = dimmethod
        self.verbose = verbose
        self.varying_genes = varying_genes
        self.scanpymethod = scanpymethod
        self.topgene = topgene


        super(subset_genes_dim_Reduced_ARDRegression, self).__init__(
            n_iter=300, tol=1.e-3, alpha_1=1.e-6, alpha_2=1.e-6,
            lambda_1=1.e-6, lambda_2=1.e-6, compute_score=False,
            threshold_lambda=1.e+4, fit_intercept=True, normalize=False,
            copy_X=True, verbose=False
        )

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log1p(data)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    def _dimension_reduction(self, data, type='train'):
        embedding = None

        if type == 'test' and self.reducer is not None:
            data = pd.DataFrame(self.sc.transform(data))
            embedding = self.reducer.transform(data)

        if type == 'train':
            if self.dimmethod == 'pca':
                self.reducer = PCA(n_components=self.dimension)
            if self.dimmethod == 'umap':
                self.reducer = umap.UMAP(n_components=self.dimension)
            self.sc = StandardScaler()
            data = pd.DataFrame(self.sc.fit_transform(data))
            embedding = self.reducer.fit_transform(data)

        return pd.DataFrame(embedding).fillna(0)

    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=False, these_genes=np.array(False)):
        if self.varying_genes is True: # Use scanpy selected genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
            with open(os.path.join('data', f'{self.scanpymethod}_bulk_highly_variable_top{self.topgene}.txt'), 'r') as f:
                these_genes = [int(x) for x in f.read().split(' ')]
        else:
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
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='test')
        X_sub = normalize(X_sub)
        return super(subset_genes_dim_Reduced_ARDRegression, self).predict(np.array(X_sub), return_std=False)

    def fit(self, X, y, sample_weight=None):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'subset_logT')
        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='train')
        X_sub = normalize(X_sub)
        super(subset_genes_dim_Reduced_ARDRegression, self).fit(X_sub, y)
        return self


class subset_genes_Dim_Reduced_DecisionTreeRegressor(DecisionTreeRegressor):
    def __init__(self, criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
                 min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0,
                 varying_genes=False, flavor='seurat', topgene=10,
                 subset_min=50, subset_fold=50, verbose=True, dosubset=True, subset_logT=True,
                 dimreduced=True, dimension=3, dimmethod='pca'):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.verbose = verbose
        self.dimreduced = dimreduced
        self.dimension = dimension
        self.dimmethod = dimmethod
        self.dosubset = dosubset
        self.varying_genes = varying_genes
        self.scanpymethod = flavor
        self.topgene = topgene
        self.reducer = None

        super(subset_genes_Dim_Reduced_DecisionTreeRegressor, self).__init__(
            criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
            min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None, presort='deprecated', ccp_alpha=0.0,
        )

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log1p(data)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=False, these_genes=np.array(False)):
        if self.varying_genes is True: # Use scanpy selected genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
            with open(os.path.join('data', f'{self.scanpymethod}_bulk_highly_variable_top{self.topgene}.txt'), 'r') as f:
                these_genes = [int(x) for x in f.read().split(' ')]
        else:
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
            # embedding = normalize(embedding)

        return pd.DataFrame(embedding).fillna(0)

    def predict(self, X):

        check_is_fitted(self, 'genecolumns_')
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='test')

        # print(X_sub)
        return super(subset_genes_Dim_Reduced_DecisionTreeRegressor, self).predict(X_sub)

    def fit(self, X, y):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'subset_logT')
        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='train')
        super(subset_genes_Dim_Reduced_DecisionTreeRegressor, self).fit(X_sub, y)
        return self

class subset_genes_Dim_Reduced_RandomForestRegressor(RandomForestRegressor):
    def __init__(self, n_estimators=100,  criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False,
                 n_jobs=None, random_state=None, verbose=True, warm_start=False, ccp_alpha=0.0, max_samples=None,
                 varying_genes=False, flavor='seurat', topgene=10,
                 subset_min=50, subset_fold=50,  dosubset=True, subset_logT=True,
                 dimreduced=True, dimension=3, dimmethod='pca'):
        self.subset_min = subset_min
        self.subset_fold = subset_fold
        self.subset_logT = subset_logT
        self.verbose = verbose
        self.dimreduced = dimreduced
        self.dimension = dimension
        self.dimmethod = dimmethod
        self.dosubset = dosubset
        self.varying_genes = varying_genes
        self.scanpymethod = flavor
        self.topgene = topgene
        self.reducer = None

        super(subset_genes_Dim_Reduced_RandomForestRegressor, self).__init__(
            n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1,
            min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, bootstrap=True, oob_score=False,
            n_jobs=None, random_state=None, verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None,
        )

    def _expr_levels(self, data, logTrans=False):
        if logTrans:
            return np.log1p(data)
        else:
            return data

    def get_subset_genes(self, data):
        return self._subset_genes(data, verbose=self.verbose)

    # this version of subset genes assumes data is ONLY genes with no metadata columns in a pd.DataFrame
    def _subset_genes(self, data, verbose=False, these_genes=np.array(False)):
        if self.varying_genes is True: # Use scanpy selected genes
            genes = self._expr_levels(data, logTrans=self.subset_logT)
            with open(os.path.join('data', f'{self.scanpymethod}_bulk_highly_variable_top{self.topgene}.txt'), 'r') as f:
                these_genes = [int(x) for x in f.read().split(' ')]
        else:
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
            # embedding = normalize(embedding)

        return pd.DataFrame(embedding).fillna(0)

    def predict(self, X):

        check_is_fitted(self, 'genecolumns_')
        X_sub = self._subset_genes(X, these_genes=self.genecolumns_)
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='test')

        pred = super(subset_genes_Dim_Reduced_RandomForestRegressor, self).predict(X_sub)
        return pred

    def fit(self, X, y):
        check_is_fitted(self, 'subset_min')
        check_is_fitted(self, 'subset_fold')
        check_is_fitted(self, 'subset_logT')
        X_sub = self._subset_genes(X, verbose=self.verbose)
        self.genecolumns_ = X_sub.columns
        if self.dimreduced is True and len(self.genecolumns_) > self.dimension:
            X_sub = self._dimension_reduction(X_sub, type='train')
        super(subset_genes_Dim_Reduced_RandomForestRegressor, self).fit(X_sub, y)
        return self
