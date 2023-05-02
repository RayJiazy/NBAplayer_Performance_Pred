import pandas as pd
import numpy as np

from FeatureCatcher import FeatureCatcher
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



class MultiPredModelSVR:
    def __init__(self, pca=False, standerization=False):
        self.pca = pca
        self.standerization = standerization
        self.model_pts = SVR(kernel='rbf', C=10, epsilon=0.2, gamma=0.001)
        self.model_reb = SVR(kernel='rbf', C=10, epsilon=0.2, gamma=0.001)
        self.model_ast = SVR(kernel='rbf', C=10, epsilon=0.2, gamma=0.001)

    def get_RMSE_loss(self, pred, y):
        return np.sqrt(np.sum(np.power((pred - y), 2)) / len(pred))

    def get_MAE_loss(self, pred, y):
        return np.sum(np.abs(pred - y)) / len(pred)

    def models_train(self, x_pts, x_reb, x_ast, y_pts, y_reb, y_ast):
        print("=========================")
        self.model_pts.fit(x_pts, y_pts)
        self.model_reb.fit(x_reb, y_reb)
        self.model_ast.fit(x_ast, y_ast)
        pred_pts = self.model_pts.predict(x_pts)
        pred_reb = self.model_reb.predict(x_reb)
        pred_ast = self.model_ast.predict(x_ast)
        rmse_loss_pts = self.get_RMSE_loss(pred_pts, y_pts)
        rmse_loss_reb = self.get_RMSE_loss(pred_reb, y_reb)
        rmse_loss_ast = self.get_RMSE_loss(pred_ast, y_ast)
        rmse_total = rmse_loss_pts + rmse_loss_reb + rmse_loss_ast
        print(
            f'RMSE loss on pts of transet:{rmse_loss_pts}, RMSE loss on reb of transet:{rmse_loss_reb}, RMSE loss on ast of transet:{rmse_loss_ast}')
        print(f'transet RMSE loss in total:{rmse_total}')

    def models_test(self, x_pts, x_reb, x_ast, y_pts, y_reb, y_ast):
        print("========================")
        pred_pts = self.model_pts.predict(x_pts)
        pred_reb = self.model_reb.predict(x_reb)
        pred_ast = self.model_ast.predict(x_ast)
        rmse_loss_pts = self.get_RMSE_loss(pred_pts, y_pts)
        rmse_loss_reb = self.get_RMSE_loss(pred_reb, y_reb)
        rmse_loss_ast = self.get_RMSE_loss(pred_ast, y_ast)
        rmse_total = rmse_loss_pts + rmse_loss_reb + rmse_loss_ast

        mae_loss_pts = self.get_MAE_loss(pred_pts, y_pts)
        mae_loss_reb = self.get_MAE_loss(pred_reb, y_reb)
        mae_loss_ast = self.get_MAE_loss(pred_ast, y_ast)
        mae_total = mae_loss_pts + mae_loss_reb + mae_loss_ast

        print(
            f'RMSE loss on pts of testset:{rmse_loss_pts}, RMSE loss on reb of testset:{rmse_loss_reb}, RMSE loss on ast of testset:{rmse_loss_ast}')
        print(f'testset RMSE loss in total:{rmse_total}')
        print(f'testset MAE loss in total:{mae_total}')
        return rmse_total, mae_total

    def predict(self,x_pts, x_reb, x_ast):
        pred_pts = self.model_pts.predict(x_pts)
        pred_reb = self.model_reb.predict(x_reb)
        pred_ast = self.model_ast.predict(x_ast)

        prediction = np.hstack((np.hstack((pred_pts, pred_reb)), pred_ast))
        return prediction

    def cross_validation(self, X_train, y_train, ga_list):
        rmse_his = []
        rmse_min = float('inf')
        best_gamma = None
        for gamma in ga_list:
            kf = KFold(n_splits=5, shuffle=True)
            rmse_k_fold = []
            for i, (train_index, val_index) in enumerate(kf.split(X_train)):
                print('======================')
                print(f"Fold {i + 1}:")
                x_t, x_val = X_train[train_index], X_train[val_index]
                y_t, y_val = y_train[train_index], y_train[val_index]
                model = SVR(kernel='rbf', C=10, epsilon=0.2, gamma=gamma)
                model.fit(x_t, y_t)

                pred_t = model.predict(x_t)
                pred_val = model.predict(x_val)

                rmse_loss_t = self.get_RMSE_loss(pred_t, y_t)
                rmse_loss_val = self.get_RMSE_loss(pred_val, y_val)
                rmse_k_fold.append(rmse_loss_val)

                print(f'transet RMSE loss:{rmse_loss_t}, valset RMSE loss:{rmse_loss_val}')
                if rmse_loss_val < rmse_min:
                    rmse_min = rmse_loss_val
                    best_gamma = gamma
            rmse_his.append(rmse_k_fold)
        return rmse_his, best_gamma



#split input and label
def split_input(df):
    return df.iloc[:-1]

def split_label(df):
    return df.iloc[-1]

def get_input_label(df):
    df_xtrain = df.groupby('block_id').apply(lambda x: split_input(x))
    df_label = df.groupby('block_id').apply(lambda x: split_label(x))
    df_xtrain.reset_index(drop = True, inplace = True)
    df_xtrain.reset_index(drop = True, inplace = True)
    return df_xtrain, df_label

def split_train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df_data = pd.read_csv('data/trainset.csv')
    df_xtrain, df_label = get_input_label(df_data)

    fc = FeatureCatcher(df_xtrain=df_xtrain, df_label=df_label)
    feature_pts, feature_reb, feature_ast, label_pts, label_reb, label_ast = fc.get_feature_labels()
    x_pts_train, x_pts_test, y_pts_train, y_pts_test = split_train_test(feature_pts, label_pts)
    x_reb_train, x_reb_test, y_reb_train, y_reb_test = split_train_test(feature_reb, label_reb)
    x_ast_train, x_ast_test, y_ast_train, y_ast_test = split_train_test(feature_ast, label_ast)

    svr_model = MultiPredModelSVR()
    svr_model.models_train(x_pts_train, x_reb_train, x_ast_train, y_pts_train, y_reb_train, y_ast_train)
    svr_model.models_test(x_pts_test, x_reb_test, x_ast_test, y_pts_test, y_reb_test, y_ast_test)



