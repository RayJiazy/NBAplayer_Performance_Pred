from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import KFold
from GetFeature import get_feature_data
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
class NonLinearRegressor:
    def __init__(self, regress_type, grid_search_param, pca=False):
        self.pca = pca
        self.regress_type = regress_type
        self.grid_search_param = grid_search_param
        if regress_type == "RandomForest":
            self.is_val = True
            self.model_pts = RandomForestRegressor()
            self.model_reb = RandomForestRegressor()
            self.model_ast = RandomForestRegressor()
        elif regress_type == "DecisionTree":
            self.is_val = True
            self.model_pts = DecisionTreeRegressor(max_depth=15,min_samples_leaf=70)
            self.model_reb = DecisionTreeRegressor(max_depth=15,min_samples_leaf=70)
            self.model_ast = DecisionTreeRegressor(max_depth=15,min_samples_leaf=70)
        else:
            self.is_val = True
            self.model_pts = XGBRegressor()
            self.model_reb = XGBRegressor()
            self.model_ast = XGBRegressor()

    def get_RMSE_loss(self, pred, y):
        return np.sqrt(mean_squared_error(y,pred))

    def get_MAE_loss(self, pred, y):
        return mean_absolute_error(y,pred,)

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

    def cross_validation_rf(self, X_train, y_train, tree_list):
        rmse_his = []
        rmse_min = float('inf')
        best_tree_num = None
        for tree in tree_list:
            kf = KFold(n_splits=5, shuffle=True)
            rmse_k_fold = []
            for i, (train_index, val_index) in enumerate(kf.split(X_train)):
                # print('======================')
                # print(f"Fold {i + 1}:")
                x_t, x_val = X_train[train_index], X_train[val_index]
                y_t, y_val = y_train[train_index], y_train[val_index]
                model = RandomForestRegressor(n_estimators=tree,max_features='sqrt')
                model.fit(x_t, y_t)

                pred_t = model.predict(x_t)
                pred_val = model.predict(x_val)

                rmse_loss_t = self.get_RMSE_loss(pred_t, y_t)
                rmse_loss_val = self.get_RMSE_loss(pred_val, y_val)
                rmse_k_fold.append(rmse_loss_val)

                # print(f'transet RMSE loss:{rmse_loss_t}, valset RMSE loss:{rmse_loss_val}')
                if rmse_loss_val < rmse_min:
                    rmse_min = rmse_loss_val
                    best_tree_num = tree
            rmse_his.append(rmse_k_fold)
        return rmse_his, best_tree_num

    def gridsearch(self,param_grid, estimator, X, y):
        grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
        grid_search.fit(X, y)
        return grid_search.best_params_

    def best_hyperparameter(self,pts_X_train, pts_y_train,reb_X_train, reb_y_train,ast_X_train, ast_y_train, param_grid):
        if self.is_val:
            params_pts = None
            params_reb = None
            params_ast = None
            if self.regress_type == "RandomForest":
                params_pts = self.gridsearch(param_grid, RandomForestRegressor(),pts_X_train, pts_y_train)
                self.model_pts = RandomForestRegressor(max_depth=params_pts["max_depth"],
                                                       min_samples_leaf=params_pts["min_samples_leaf"],
                                                       min_samples_split=params_pts["min_samples_split"],
                                                       n_estimators=params_pts["n_estimators"])
                params_reb = self.gridsearch(param_grid, RandomForestRegressor(), reb_X_train, reb_y_train)
                self.model_reb = RandomForestRegressor(max_depth=params_reb["max_depth"],
                                                       min_samples_leaf=params_reb["min_samples_leaf"],
                                                       min_samples_split=params_reb["min_samples_split"],
                                                       n_estimators=params_reb["n_estimators"])
                params_ast = self.gridsearch(param_grid, RandomForestRegressor(), ast_X_train, ast_y_train)
                self.model_ast = RandomForestRegressor(max_depth=params_ast["max_depth"],
                                                       min_samples_leaf=params_ast["min_samples_leaf"],
                                                       min_samples_split=params_ast["min_samples_split"],
                                                       n_estimators=params_ast["n_estimators"])
            elif self.regress_type == "DecisionTree":
                params_pts = self.gridsearch(param_grid, DecisionTreeRegressor(),pts_X_train, pts_y_train)
                self.model_pts = DecisionTreeRegressor(max_depth=params_pts["max_depth"],
                                                       min_samples_leaf=params_pts["min_samples_leaf"],
                                                       min_samples_split=params_pts["min_samples_split"])
                params_reb = self.gridsearch(param_grid, DecisionTreeRegressor(), reb_X_train, reb_y_train)
                self.model_reb = DecisionTreeRegressor(max_depth=params_reb["max_depth"],
                                                       min_samples_leaf=params_reb["min_samples_leaf"],
                                                       min_samples_split=params_reb["min_samples_split"])
                params_ast = self.gridsearch(param_grid, DecisionTreeRegressor(), ast_X_train, ast_y_train)
                self.model_ast = DecisionTreeRegressor(max_depth=params_ast["max_depth"],
                                                       min_samples_leaf=params_ast["min_samples_leaf"],
                                                       min_samples_split=params_ast["min_samples_split"])
            print(f"The best parameters of {self.regress_type}_pts: {params_pts}")
            print(f"The best parameters of {self.regress_type}_reb: {params_reb}")
            print(f"The best parameters of {self.regress_type}_ast: {params_ast}")

def get_result(classifier_type,param_grid,pts_X_train, pts_X_test, reb_X_train, reb_X_test, ast_X_train, ast_X_test, pts_y_train, reb_y_train, ast_y_train, pts_y_test, reb_y_test, ast_y_test):
    classifier = NonLinearRegressor(classifier_type,param_grid)
    print(f"The result of {classifier_type}:")
    classifier.best_hyperparameter(pts_X_train, pts_y_train,reb_X_train, reb_y_train,ast_X_train, ast_y_train,param_grid)
    classifier.models_train(pts_X_train,reb_X_train,ast_X_train,pts_y_train,reb_y_train,ast_y_train)
    classifier.models_test(pts_X_test,reb_X_test,ast_X_test,pts_y_test,reb_y_test,ast_y_test)

# if __name__ == "__main__":
#     param_grid_rf = {
#         'n_estimators': [10,100,500],
#         'min_samples_split' : [2,3,4,5,6,7,8,9,10],
#         "min_samples_leaf": [2,3,4,5,6,7,8,9,10],
#         'max_depth': [1,2,3,4,5]
#     }
#     param_grid_dt = {
#         'min_samples_split' : [2,3,4,5,6,7,8,9,10],
#         "min_samples_leaf": [2,3,4,5,6,7,8,9,10],
#         'max_depth': [1,2,3,4,5]
#     }
#     # get_result("RandomForest", param_grid_rf, *get_feature_data(is_load=False))
#     get_result("DecisionTree", param_grid_dt, *get_feature_data(is_load=False))


def get_test_res(estimator,model_type,train_X,train_y,test_X,test_y):
    estimator.fit(train_X,train_y)
    pred = estimator.predict(test_X)
    rmse_loss = np.sqrt(mean_squared_error(test_y,pred))
    mae_loss = mean_absolute_error(test_y,pred)
    print(f"RMSE of {model_type}: {round(rmse_loss, 3)}")
    print(f"MAE of {model_type}: {round(mae_loss, 3)}")

pts_X_train, pts_X_test, reb_X_train, reb_X_test, ast_X_train, ast_X_test, \
pts_y_train, reb_y_train, ast_y_train, pts_y_test, reb_y_test, ast_y_test = get_feature_data(is_load=False,is_standard=True,is_augment=True)
#
# n_comp = 8
# pca_pts_train = PCA(n_components=n_comp)
# pca_pts_train.fit(pts_X_train)
# pts_X_train = pca_pts_train.transform(pts_X_train)
#
# pca_reb_train = PCA(n_components=n_comp)
# pca_reb_train.fit(reb_X_train)
# reb_X_train = pca_reb_train.transform(reb_X_train)
#
# pca_ast_train = PCA(n_components=n_comp)
# pca_ast_train.fit(ast_X_train)
# ast_X_train = pca_ast_train.transform(ast_X_train)
#
# pca_pts_test = PCA(n_components=n_comp)
# pca_pts_test.fit(pts_X_test)
# pts_X_test = pca_pts_test.transform(pts_X_test)
#
# pca_reb_test = PCA(n_components=n_comp)
# pca_reb_test.fit(reb_X_test)
# reb_X_test = pca_reb_test.transform(reb_X_test)
#
# pca_ast_test = PCA(n_components=n_comp)
# pca_ast_test.fit(ast_X_test)
# ast_X_test = pca_ast_test.transform(ast_X_test)

rf_pts = RandomForestRegressor(max_depth=5,min_samples_leaf=5,min_samples_split=7,n_estimators=100)
rf_reb = RandomForestRegressor(max_depth=5,min_samples_leaf=6,min_samples_split=4,n_estimators=100)
rf_ass = RandomForestRegressor(max_depth=5,min_samples_leaf=6,min_samples_split=8,n_estimators=100)
print("The result of RandomForest:")
get_test_res(rf_pts,"pts",pts_X_train,pts_y_train,pts_X_test,pts_y_test)
get_test_res(rf_pts,"reb",reb_X_train,reb_y_train,reb_X_test,reb_y_test)
get_test_res(rf_pts,"ast",ast_X_train,ast_y_train,ast_X_test,ast_y_test)

dt_pts = DecisionTreeRegressor(max_depth=5,min_samples_leaf=9,min_samples_split=5)
dt_reb = DecisionTreeRegressor(max_depth=5,min_samples_leaf=8,min_samples_split=2)
dt_ass = DecisionTreeRegressor(max_depth=5,min_samples_leaf=5,min_samples_split=2)

print("The result of DecisionTree:")
get_test_res(dt_pts,"pts",pts_X_train,pts_y_train,pts_X_test,pts_y_test)
get_test_res(dt_pts,"reb",reb_X_train,reb_y_train,reb_X_test,reb_y_test)
get_test_res(dt_pts,"ast",ast_X_train,ast_y_train,ast_X_test,ast_y_test)