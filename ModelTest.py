import numpy as np
from sklearn.metrics import mean_absolute_error as score_MAE
from sklearn.metrics import mean_squared_error as score_MSE
from sklearn.metrics import median_absolute_error as score_MED
from sklearn.metrics import r2_score as score_R2
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut
import matplotlib.pyplot as plt
import seaborn as sns
import os

def inverse_indicator(value):
    if round(value.values[0]) == 0:
        return -1
    return value

class ModelRunner(object):
    def __init__(self, model, features, targets, name = None):
        self.model = model
        self.features = features
        self.targets = targets
        self.name = name
        self.model_run = False


    def leave_one_out_cross_validation(self):
        plot_cval = LeaveOneOut()
        features_mask = self.features.sum(axis=1) / 10000
        features_mask = features_mask.fillna(0)
        self.true_age = []
        self.pred_age = []
        self.pred_std = []
        for train, test in plot_cval.split(self.features):
            self.model.fit(self.features.iloc[train, :], self.targets[train])
            self.pred_age.append(self.model.predict(self.features.iloc[test, :])[0]* features_mask[test])
            #std is only for bayesian models
            self.pred_std.append(self.model.predict(self.features.iloc[test, :])[1] * inverse_indicator(features_mask[test]))
            self.true_age.append(self.targets[test]  * features_mask[test])

        self.actual_pred_age = [x for x in np.array(self.pred_age).flatten() if x != 0]
        self.actual_true_age = [round(x) for x in np.array(self.true_age).flatten() if x != 0]

        print('Number of samples', len(self.actual_pred_age))
        self.errstr = "MAE:{:3.1f} MED:{:3.1f} R2:{:3.2f}".format(score_MAE(self.actual_true_age, self.actual_pred_age),
                                                             score_MED(self.actual_true_age, self.actual_pred_age),
                                                             score_R2(self.actual_true_age, self.actual_pred_age))
        print(self.errstr)

        self.model_run = True

    def leave_one_out_cross_validation_ensemble(self):
        plot_cval = LeaveOneOut()
        self.true_age = []
        self.pred_age = []
        self.pred_std = []

        for train, test in plot_cval.split(self.targets):
            self.model.fit(self.features, self.targets, train)
            self.pred_age.append(self.model.predict(self.features,test))
            # std is only for bayesian models
           # self.pred_std.append(self.model.predict(self.features,test)[1])
            self.true_age.append(self.targets[test])

        self.actual_pred_age = [x for x in np.array(self.pred_age).flatten() if x != 0]
        self.actual_true_age = [round(x) for x in np.array(self.true_age).flatten() if x != 0]

        print('Number of samples', len(self.actual_pred_age))
        self.errstr = "MAE:{:3.1f} MED:{:3.1f} R2:{:3.2f}".format(score_MAE(self.actual_true_age, self.actual_pred_age),
                                                                  score_MED(self.actual_true_age, self.actual_pred_age),
                                                                  score_R2(self.actual_true_age, self.actual_pred_age))
        print(self.errstr)

        self.model_run = True
        pass

    def draw_prediction_line(self, dir = ''):
        sns.set_style('white')
        # sns.set_palette( sns.xkcd_palette(['nice blue','faded red']))

        fig1 = plt.figure(figsize=(3.5, 3.5))
        ax1 = fig1.add_subplot(111)
        sns.regplot(self.true_age, self.pred_age, ax=ax1)
        ax1.plot([0, 100], [0, 100], 'k:')
        ax1.text(0.99, 0.03, self.errstr,
                 horizontalalignment='right',
                 verticalalignment='bottom',
                 transform=ax1.transAxes)
        # plt.title(name)
        plt.axis('equal')
        ax1.set_xlabel('True age (years)'), ax1.set_ylabel('Predicted age (years)');
        sns.despine()
        plt.tight_layout()
        plt.draw()
        plt.savefig(dir+os.sep+'fig_plot_predictions_'+self.name, dpi=400)

    def get_predicted_age(self):
        if self.model_run is False:
            self.leave_one_out_cross_validation()
        return np.array(self.pred_age).flatten()

    def get_prediction_std(self):
        if self.model_run is False:
            self.leave_one_out_cross_validation()
        return np.array(self.pred_std).flatten()
