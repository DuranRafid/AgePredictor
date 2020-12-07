from sklearn.ensemble import StackingRegressor, VotingRegressor,RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import ARDRegression
import numpy as np
import copy

class CellExpertStackingRegressor():
    def __init__(self, base_estimator, final_estimator=AdaBoostRegressor(random_state=42,n_estimators=10),
                 celllist=None):  # Hence base_estimator will be a subset gene dim reduced regressor
        self.celllist = celllist
        self.base_estimator = base_estimator
        self.final_estimator = final_estimator
        self.base_estimators = [copy.deepcopy(self.base_estimator) for i in range(len(self.celllist))]

    def fit(self, X, y,trainlist):  # Hence X is allcelldict, trainlist is the dataset to be trained
        predMat = np.zeros((len(y[trainlist]),len(self.celllist)))
        for i in range(len(self.celllist)):
            cell = self.celllist[i]
            features = X[cell].T
            targets = y[trainlist]
            preds = cross_val_predict(copy.deepcopy(self.base_estimators[i]), features.iloc[trainlist, :], targets,cv=3)
            self.base_estimators[i].fit(features.iloc[trainlist, :], targets)
            predMat[:,i] = preds
        predMat = normalize(predMat)
        self.final_estimator.fit(predMat, targets)

    def predict(self, X, testlist):  # Predict the output from final estimator
        preds = self.transpose(X, testlist)
        preds = normalize(preds)
        return self.final_estimator.predict(preds)

    def transpose(self, X, testlist):
        preds = []
        for i in range(len(self.celllist)):
            cell = self.celllist[i]
            features = X[cell].T
            features_mask = features.sum(axis=1) / 10000
            pred = self.base_estimators[i].predict(features.iloc[testlist,:]) * features_mask[testlist]
            preds.append(pred)
        return np.array(preds).reshape(1,-1)

    def set_droplist(self, X, testlist):  # X is cell data
        self.droplist = []
        for cell in self.celllist:
            print(cell)
            features = X[cell].T
            features_mask = features.sum(axis=1)
            if features_mask[testlist] == 0:
                self.droplist.append(cell)
        return self.droplist
