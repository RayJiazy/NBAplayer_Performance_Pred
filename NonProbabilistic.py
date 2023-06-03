from sklearn.linear_model import Ridge,LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import KFold
from GetFeature import get_feature_data
from sklearn.decomposition import PCA
import numpy as np
class NonProbabilistic:
    def __init__(self, regress_type, pca=False):
        self.pca = pca
        if regress_type == "ridge":
            self.is_val = True
            self.model_pts = Ridge(alpha=1.0)
            self.model_reb = Ridge(alpha=1.0)
            self.model_ast = Ridge(alpha=1.0)
        else:
            self.is_val = False
            self.model_pts = LinearRegression()
            self.model_reb = LinearRegression()
            self.model_ast = LinearRegression()

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

    def cross_validation(self, X_train, y_train, alpha_list):
        rmse_his = []
        rmse_min = float('inf')
        best_alpha = None
        for alpha in alpha_list:
            kf = KFold(n_splits=5, shuffle=True)
            rmse_k_fold = []
            for i, (train_index, val_index) in enumerate(kf.split(X_train)):
                x_t, x_val = X_train[train_index], X_train[val_index]
                y_t, y_val = y_train[train_index], y_train[val_index]
                model = Ridge(alpha=alpha)
                model.fit(x_t, y_t)

                pred_t = model.predict(x_t)
                pred_val = model.predict(x_val)

                rmse_loss_t = self.get_RMSE_loss(pred_t, y_t)
                rmse_loss_val = self.get_RMSE_loss(pred_val, y_val)
                rmse_k_fold.append(rmse_loss_val)

                if rmse_loss_val < rmse_min:
                    rmse_min = rmse_loss_val
                    best_alpha = alpha
            rmse_his.append(rmse_k_fold)
        return rmse_his, best_alpha
    def best_hyperparameter(self,pts_X_train, pts_y_train,reb_X_train, reb_y_train,ast_X_train, ast_y_train,alpha_list):
        if self.is_val:
           _, alpha_pts = self.cross_validation(pts_X_train, pts_y_train,alpha_list)
           print(f"The best alpha of pts: {alpha_pts}")
           self.model_pts = Ridge(alpha=alpha_pts)
           _, alpha_reb = self.cross_validation(reb_X_train, reb_y_train,alpha_list)
           print(f"The best alpha of reb: {alpha_reb}")
           self.model_reb = Ridge(alpha=alpha_reb)
           _, alpha_ast = self.cross_validation(ast_X_train, ast_y_train,alpha_list)
           print(f"The best alpha of ast: {alpha_ast}")
           self.model_ast = Ridge(alpha=alpha_ast)

def get_test_res(estimator,model_type,train_X,train_y,test_X,test_y):
    estimator.fit(train_X,train_y)
    pred = estimator.predict(test_X)
    rmse_loss = np.sqrt(mean_squared_error(test_y,pred))
    mae_loss = mean_absolute_error(test_y,pred)
    print(f"RMSE of {model_type}: {round(rmse_loss, 3)}")
    print(f"MAE of {model_type}: {round(mae_loss, 3)}")
alphas = [0.001,0.01,0.1,0.5,1,10]
pts_X_train, pts_X_test, reb_X_train, reb_X_test, ast_X_train, ast_X_test, \
pts_y_train, reb_y_train, ast_y_train, pts_y_test, reb_y_test, ast_y_test = get_feature_data(is_load=False,is_standard=True,is_augment=True)
# n_comp = 3
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



ridge = NonProbabilistic("ridge")
print("The result of RidgeRegression:")
ridge.best_hyperparameter(pts_X_train, pts_y_train,reb_X_train, reb_y_train,ast_X_train, ast_y_train,alphas)
ridge.models_train(pts_X_train,reb_X_train,ast_X_train,pts_y_train,reb_y_train,ast_y_train)
ridge.models_test(pts_X_test,reb_X_test,ast_X_test,pts_y_test,reb_y_test,ast_y_test)
linear = NonProbabilistic("linear")
print("The result of LinearRegression:")
linear.models_train(pts_X_train,reb_X_train,ast_X_train,pts_y_train,reb_y_train,ast_y_train)
linear.models_test(pts_X_test,reb_X_test,ast_X_test,pts_y_test,reb_y_test,ast_y_test)
rg_pts = Ridge(alpha=0.5)
rg_reb = Ridge(alpha=0.001)
rg_ast = Ridge(alpha=0.1)
print("The result of RidgeRegression:")
get_test_res(rg_pts,"pts",pts_X_train,pts_y_train,pts_X_test,pts_y_test)
get_test_res(rg_pts,"reb",reb_X_train,reb_y_train,reb_X_test,reb_y_test)
get_test_res(rg_pts,"ast",ast_X_train,ast_y_train,ast_X_test,ast_y_test)

lr_pts = LinearRegression()
lr_reb = LinearRegression()
lr_ast = LinearRegression()
print("The result of LinearRegression:")
get_test_res(lr_pts,"pts",pts_X_train,pts_y_train,pts_X_test,pts_y_test)
get_test_res(lr_pts,"reb",reb_X_train,reb_y_train,reb_X_test,reb_y_test)
get_test_res(lr_pts,"ast",ast_X_train,ast_y_train,ast_X_test,ast_y_test)