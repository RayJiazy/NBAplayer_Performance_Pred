from src.FeatureCatcher import *
from DataProcessor import DataProcessor
from sklearn.preprocessing import StandardScaler
def split_input(df):
    return df.iloc[:-1]

def split_label(df):
    return df.iloc[-1]

def get_input_label(df,column):
    df_xtrain = df.groupby(column).apply(lambda x: split_input(x))
    df_label = df.groupby(column).apply(lambda x: split_label(x))
    df_xtrain.reset_index(drop = True, inplace = True)
    df_xtrain.reset_index(drop = True, inplace = True)
    return df_xtrain, df_label

def multi_normalization(pts_X_train, reb_X_train, ast_X_train, pts_X_test, reb_X_test, ast_X_test):
    scaler_pts = StandardScaler()
    scaler_reb = StandardScaler()
    scaler_ast = StandardScaler()
    scaler_pts.fit(pts_X_train)
    scaler_reb.fit(reb_X_train)
    scaler_ast.fit(ast_X_train)
    pts_X_train_norm = scaler_pts.transform(pts_X_train)
    pts_X_test_norm = scaler_pts.transform(pts_X_test)
    reb_X_train_norm = scaler_reb.transform(reb_X_train)
    reb_X_test_norm = scaler_reb.transform(reb_X_test)
    ast_X_train_norm = scaler_ast.transform(ast_X_train)
    ast_X_test_norm = scaler_ast.transform(ast_X_test)
    return pts_X_train_norm, pts_X_test_norm, reb_X_train_norm, reb_X_test_norm, ast_X_train_norm, ast_X_test_norm

def get_feature_data(is_standard=True,is_load=False,is_augment=True):
    if is_augment:
        column = "block_id"
    else:
        column = "ID"
    if is_load:
        data_processor = DataProcessor()
        data_processor.process_raw_data()
    df_train = pd.read_csv(f"../dataset/trainset_{column}.csv")
    df_test = pd.read_csv(f"../dataset/testset_{column}.csv")
    train_X, train_y = get_input_label(df_train,column)
    test_X, test_y = get_input_label(df_test,column)
    fc_train = FeatureCatcher(column, df_xtrain=train_X, df_label=train_y)
    fc_test = FeatureCatcher(column, df_xtrain=test_X, df_label=test_y)
    pts_X_train, reb_X_train, ast_X_train, pts_y_train, reb_y_train, ast_y_train = fc_train.get_feature_labels()
    pts_X_test, reb_X_test, ast_X_test, pts_y_test, reb_y_test, ast_y_test = fc_test.get_feature_labels()
    if is_standard:
        pts_X_train, pts_X_test, reb_X_train, reb_X_test, ast_X_train, ast_X_test = multi_normalization(pts_X_train, reb_X_train, ast_X_train, pts_X_test, reb_X_test, ast_X_test)
    return pts_X_train, pts_X_test, reb_X_train, reb_X_test, ast_X_train, ast_X_test, \
           pts_y_train, reb_y_train, ast_y_train,pts_y_test, reb_y_test, ast_y_test
