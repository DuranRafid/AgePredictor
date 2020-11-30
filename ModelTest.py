import numpy as np
from sklearn.metrics import mean_absolute_error as score_MAE
from sklearn.metrics import mean_squared_error as score_MSE
from sklearn.metrics import median_absolute_error as score_MED
from sklearn.metrics import r2_score as score_R2
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut

def inverse_indicator(value):
    if round(value.values[0]) == 0:
        return -1
    return value

class ModelRunner(object):
    def __init__(self, model, features, targets):
        self.model = model
        self.features = features
        self.features_mask = features.sum(axis=1) / 1000000
        self.targets = targets
        self.model_run = False


    def leave_one_out_cross_validation(self):
        plot_cval = LeaveOneOut()
        self.true_age = []
        self.pred_age = []
        self.pred_std = []
        for train, test in plot_cval.split(self.features):
            self.model.fit(self.features.iloc[train, :], self.targets[train])
            self.pred_age.append(self.model.predict(self.features.iloc[test, :])[0] * self.features_mask[test])
            #std is only for gaussian process regression
            self.pred_std.append(self.model.predict(self.features.iloc[test, :])[1] * inverse_indicator(self.features_mask[test]))
            self.true_age.append(self.targets[test]  * self.features_mask[test])

        actual_pred_age = [x for x in np.array(self.pred_age).flatten() if x != 0]
        actual_true_age = [x for x in np.array(self.true_age).flatten() if x != 0]

        print('Number of samples', len(actual_pred_age))
        errstr = "MAE:{:3.1f} MED:{:3.1f} R2:{:3.2f}".format(score_MAE(actual_true_age, actual_pred_age),
                                                             score_MED(actual_true_age, actual_pred_age),
                                                             score_R2(actual_true_age, actual_pred_age))
        print(errstr)
        self.model_run = True

    def get_predicted_age(self):
        if self.model_run is False:
            self.leave_one_out_cross_validation()
        return np.array(self.pred_age).flatten()

    def get_prediction_std(self):
        if self.model_run is False:
            self.leave_one_out_cross_validation()
        return np.array(self.pred_std).flatten()
