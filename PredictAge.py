import pickle
import gzip
import pandas as pd
import sys
from SubsetGenesDimReducedRegressors import subset_genes_Dim_Reduced_ElasticNet, subset_genes_Dim_Reduced_DecisionTreeRegressor,\
    subset_genes_Dim_Reduced_GPR, subset_genes_Dim_Reduced_RandomForestRegressor, subset_genes_Dim_Reduced_SVR, subset_genes_dim_Reduced_ARDRegression, subset_genes_dim_Reduced_Neural_Regression
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, RationalQuadratic, ConstantKernel
from ModelTest import ModelRunner
from CellExpertEnsembleRegressor import CellExpertStackingRegressor

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
        self._filter_target_data()
        self._filter_bulk_data()
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
        self.bulk_data = self.bulk_data.loc[:, self.control_subjects]  # Rearrange the subject names in columns

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
            cell_df = self.cell_data[cell].apply(lambda x: x*10000 / sum(x), axis=0).fillna(0)
            for subject in self.control_subjects:
                if subject not in cell_df.columns.values:
                    cell_df[subject] = [0 for i in range(cell_df.shape[0])]
            self.cell_data[cell] = cell_df.loc[:, self.control_subjects]  # Rearrange the subject names in columns

    def run_bulk_predictor(self):
        self.process_datasets()
        features = self.bulk_data.T
        print(features)
        targets = self.ages
        clf = subset_genes_Dim_Reduced_DecisionTreeRegressor(subset_min=500, subset_fold=500, dimreduced=False, subset_logT=True )
        drm = ModelRunner(model=clf, features=features, targets=targets)
        predicted_ages = drm.get_predicted_age()

        print(predicted_ages)


    def run_individual_cell_experts(self):
        self.process_datasets()
        pred_age_features = dict()
        pred_age_std = dict()
        for cell in self.cell_list:
            print(cell)
            features = self.cell_data[cell].T
            targets = self.ages

            kernel = 0.5**2*DotProduct(sigma_0=1.0)**2 + 0.5*WhiteKernel(noise_level=0.1)

            clf = subset_genes_Dim_Reduced_GPR(kernel=kernel, verbose=False, normalize_y=True, subset_min=15, varying_genes=True, topgene=50,  subset_fold=15, dimreduced=False, subset_logT=True, dimmethod='pca')
            #clf = subset_genes_Dim_Reduced_SVR(kernel='rbf',degree=3,subset_min=15,subset_fold=15, subset_logT=True, dimreduced=True,ndim=5)
            #clf = subset_genes_dim_Reduced_Neural_Regression(subset_min=15, verbose=False,subset_fold=15,varying_genes=True,topgene=50,dimreduced=True,dimension=5,subset_logT=True)
            #clf = subset_genes_dim_Reduced_ARDRegression(subset_min=15, verbose=False, subset_fold=15,varying_genes=True, topgene=100, dimreduced=False,dimension=10, subset_logT=True)
            #clf = subset_genes_Dim_Reduced_RandomForestRegressor(subset_logT=True, subset_min=15, varying_genes=True,verbose=True, dimreduced=True, dimension=10, subset_fold=15)
            drm = ModelRunner(model=clf, features=features, targets=targets,name=f'{cell}GPRRegressor')
            pred_age_features[cell] = drm.get_predicted_age()
            drm.draw_prediction_line(dir='results')

            pred_age_std[cell] = drm.get_prediction_std()
            #print(pred_age_features[cell])

           # print(pred_age_std[cell])

        pred_age_df = pd.DataFrame(pred_age_features).set_index(self.control_subjects)
        pred_age_df.to_csv('SeuratGenesGPR_Pred_age_df_genes.csv')

        pred_age_std_df = pd.DataFrame(pred_age_std).set_index(self.control_subjects)
        pred_age_std_df.to_csv('SeuratGenesGPR_Pred_age_std_genes.csv')

    def run_ensemble_of_cell_experts(self):
        self.process_datasets()
        features = self.cell_data
        targets = self.ages
        base = subset_genes_dim_Reduced_ARDRegression(subset_logT=True,dimreduced=False, scanpymethod='seurat',varying_genes=True,topgene=50)
        clf = CellExpertStackingRegressor(base_estimator=base,
                                           celllist=['VE_Capillary_B','VE_Venous','VE_Arterial','VE_Peribronchial'])
        drm = ModelRunner(model=clf, features=features, targets=targets, name='ARDRegressor')
        drm.leave_one_out_cross_validation_ensemble()
        print(drm.pred_age)


if __name__ == '__main__':
    dir = 'C:/CPCB/Ziv Bar Joseph/'
    #sys.stdout = open("SeuratGenesGPRForEachCellIdentity.txt", "w")
    predictage = PredictAge(metadatafile=dir + 'GSE136831_AllCells.Samples.CellType.MetadataTable.txt.gz',
                            genenamefile=dir + 'GSE136831_AllCells.GeneIDs.txt.gz',
                            celldatafile=dir + 'data/All37CellTypeGeneExpression.pkl',
                            bulk_datafile=dir + 'data/Control_BulkRNA-seq.csv',
                            targetfile=dir + 'IPF Control Information.csv')

    predictage.run_ensemble_of_cell_experts()

   # sys.stdout.close()
