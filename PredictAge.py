from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
import gzip
import pandas as pd
import sys
from SubsetGenesDimReducedRegressors import subset_genes_dim_Reduced_Neural_Regression, subset_genes_Dim_Reduced_GPR, \
    subset_genes_Dim_Reduced_SVR
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, RationalQuadratic, ConstantKernel
from ModelTest import ModelRunner
from sklearn.linear_model import LinearRegression, ElasticNet

from sklearn.preprocessing import StandardScaler, normalize


def load_file(filename):
    if '.csv' in filename:
        return load_csv(filename)
    elif '.pkl' in filename:
        return load_pickle(filename)
    elif '.gz' in filename:
        return load_gz(filename)


def load_pickle(filename):
    file_to_load = open(filename, 'rb')
    return pickle.load(file_to_load)


def load_gz(filename):
    with gzip.open(filename, 'rb') as f:
        df = pd.read_csv(f, sep='\t')
    return df


def load_csv(filename):
    df = pd.read_csv(filename).dropna()
    return df


class PredictAge(object):
    def __init__(self, metadatafile, genenamefile, celldatafile=None, bulk_datafile=None, targetfile=None):
        self.meta_data_df = load_file(metadatafile)
        self.gene_name = load_file(genenamefile)
        self.cell_data = load_file(celldatafile)
        self.bulk_data = load_file(bulk_datafile)
        self.target_data = load_file(targetfile)

    def process_datasets(self):
        self._generate_control_refined(self.meta_data_df)
        self._filter_bulk_data()
        self._filter_target_data()
        self._filter_cell_data()

    def _generate_control_refined(self, meta_data_df):
        disease_col_name = 'Disease_Identity'
        cellcategory_col_name = 'CellType_Category'
        subject_col_name = 'Subject_Identity'
        self.meta_data_df['Index'] = meta_data_df.index
        self.control_metadata_df = self.meta_data_df[self.meta_data_df[disease_col_name] == 'Control'][
            self.meta_data_df[cellcategory_col_name] != 'Multiplet']
        self.control_metadata_df.index = self.control_metadata_df[subject_col_name]
        self.control_subjects = self.control_metadata_df[subject_col_name].unique()
        self.control_refined_list = self.control_metadata_df['Index'].tolist()

    def _filter_bulk_data(self):
        self.bulk_data = self.bulk_data.iloc[:, 1:]
        self.bulk_data = self.bulk_data.apply(lambda x: x * 1000000 / sum(x), axis=0).fillna(0)

    def _filter_target_data(self):
        age_col = 'Age'
        subject_col = 'Subject'
        self.target_data.index = self.target_data[subject_col]
        self.target_data = self.target_data.loc[self.control_subjects]
        self.ages = self.target_data[age_col]

    def _filter_cell_data(self):
        cell_identity_col_name = 'Manuscript_Identity'
        self.cell_list = self.control_metadata_df[cell_identity_col_name].unique()

        for cell in self.cell_list:
            cell_df = self.cell_data[cell].apply(lambda x: x * 1000000 / sum(x), axis=0).fillna(0)
            for subject in self.control_subjects:
                if subject not in cell_df.columns.values:
                    cell_df[subject] = [0 for i in range(cell_df.shape[0])]
            self.cell_data[cell] = cell_df.loc[:, self.control_subjects]  # Rearrange the subject names in columns

    def train_test_models(self):
        self.process_datasets()
        pred_age_features = dict()
        pred_age_std = dict()
        for cell in self.cell_list:
            print(cell)
            features = self.cell_data[cell].T
            targets = self.ages

            kernel = DotProduct(sigma_0=1.0) + WhiteKernel(noise_level=0.1)

            clf = subset_genes_Dim_Reduced_GPR(kernel=kernel, verbose=False, normalize_y=True, subset_min=200,
                                               subset_fold=200, ndim=5, subset_logT=True, dimmethod='pca')
            # clf = subset_genes_Dim_Reduced_SVR(kernel='rbf',degree=5,subset_min=1500,subset_fold=1500,subset_logT=True,ndim=5)
            # clf = subset_genes_Neural_Regression(subset_min=200, verbose=True,subset_fold=200,ndim=5,subset_logT=True)
            drm = ModelRunner(model=clf, features=features, targets=targets)
            pred_age_features[cell] = drm.get_predicted_age()
            pred_age_std[cell] = drm.get_prediction_std()

            print(pred_age_features[cell])
            break


        pred_age_df = pd.DataFrame(pred_age_features).set_index(self.control_subjects)
        pred_age_df.to_csv('Pred_age_df_genes.csv')

        pred_age_std_df = pd.DataFrame(pred_age_std).set_index(self.control_subjects)
        pred_age_std_df.to_csv('Pred_age_std_genes.csv')


if __name__ == '__main__':
    dir = ''
    sys.stdout = open("GaussianProcessForEachCellIdentity.txt", "w")
    predictage = PredictAge(metadatafile=dir + 'GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz',
                            genenamefile=dir + 'GSE136831_AllCells.GeneIDs.txt.gz',
                            celldatafile=dir + 'All37CellTypeGeneExpression.pkl',
                            bulk_datafile=dir + 'Control_BulkRNA-seq.csv',
                            targetfile=dir + 'IPF Control Information.csv')

    predictage.train_test_models()

    sys.stdout.close()
