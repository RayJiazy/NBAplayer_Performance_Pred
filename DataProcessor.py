import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
pd.pandas.set_option('display.max_columns', None)

class DataProcessor:
    def __init__(self, raw_path="../raw_data/all_seasons.csv", dataset_path="../dataset/", min_num=3, block_size=5,train_ratio=0.8):
        self.raw_path = raw_path
        self.dataset_path = dataset_path
        self.min_num = min_num
        self.block_size = block_size
        self.train_ratio = train_ratio
        self.df = None
        self.unique_values_player = None
        self.init_data()

    def init_data(self):
        self.df = pd.read_csv(self.raw_path)
        self.df.drop(columns=self.df.columns[0], inplace=True)

    def code_player(self, df):
s        unique_values_player = df[["player_name", "player_height", "player_weight"]].drop_duplicates()
        unique_values_player = unique_values_player.reset_index(drop=True)
        unique_values_player.reset_index(inplace=True)
        unique_values_player.rename(columns={"index": "ID"}, inplace=True)
        self.unique_values_player = unique_values_player


    def clean_data(self, df, unique_values_player, history_num=3):
        merged_df = pd.merge(unique_values_player, df, on=["player_name","player_height","player_weight"])
        merged_df.drop(columns=["player_name","player_height","player_weight","draft_year","draft_round","draft_number"], inplace=True)
        merged_df.drop(columns=["team_abbreviation","college","country"],inplace=True)
        filtered_df = merged_df.groupby('ID').filter(lambda x:len(x)>=history_num)
        filtered_df.sort_values(by=["ID","season"],ascending=[True,True])
        filtered_df.drop(columns=["season"],inplace=True)
        filtered_df["season_no"] = filtered_df.groupby("ID").cumcount()+1
        filtered_df["total_season"] = filtered_df.groupby("ID")["ID"].transform("count")
        dataset = filtered_df.reset_index(drop=True)
        return dataset

    def data_augmentation(self, data, subset_rows=5):
        subblocks = []
        cnt = 0
        for attribute,group in data.groupby("ID"):
            len_group = len(group)
            group = group.reset_index(drop=True)
            if len_group > subset_rows:
                for i in range(len_group-subset_rows+1):
                    subgroup = group.loc[i:i+subset_rows-1].copy()
                    subgroup["block_id"] = cnt
                    subblocks.append(subgroup)
                    cnt += 1
            else:
                group["block_id"] = cnt
                subblocks.append(group)
                cnt += 1
        augmented_data = pd.concat(subblocks).reset_index(drop=True)
        return augmented_data

    def splite_data(self, data, column):
        data_id = data["ID"].unique()
        n = len(data_id)
        np.random.shuffle(data_id)
        train_ids = data_id[0:int(n*self.train_ratio)]
        test_ids = data_id[int(n*self.train_ratio):]
        train_set = [data[data["ID"] == train_id] for train_id in train_ids]
        train_set = pd.concat(train_set)
        test_set = [data[data["ID"] == test_id] for test_id in test_ids]
        test_set = pd.concat(test_set)
        train_set.to_csv(os.path.join(self.dataset_path,f"trainset_{column}.csv"), index=False)
        test_set.to_csv(os.path.join(self.dataset_path,f"testset_{column}.csv"), index=False)

    def process_raw_data(self):
        self.code_player(self.df)
        dataset = self.clean_data(self.df, self.unique_values_player, self.min_num)
        augmented_data = self.data_augmentation(dataset, self.block_size)
        self.splite_data(augmented_data, "block_id")
        self.splite_data(dataset, "ID")

def get_X(df):
    return pd.DataFrame(df[0:-1])
def get_y(df):
    return pd.DataFrame(df[len(df)-1:len(df)])
def load_data(column):
    train_data = pd.read_csv(f"../dataset/trainset_{column}.csv",index_col=False)
    test_data = pd.read_csv(f"../dataset/testset_{column}.csv",index_col=False)
    X_train = train_data.groupby(column).apply(lambda x:get_X(x)).reset_index(drop=True)
    y_train = train_data.groupby(column).apply(lambda x:get_y(x)).reset_index(drop=True)
    X_test = test_data.groupby(column).apply(lambda x:get_X(x)).reset_index(drop=True)
    y_test = test_data.groupby(column).apply(lambda x:get_y(x)).reset_index(drop=True)
    return X_train,y_train,X_test,y_test

def build_features(df):
    age = df.age.values[-1]
    gp_mean = np.mean(df.gp.values)
    pts_mean = np.mean(df.pts.values)
    pts_last1year= df.pts.values[-1]
    pts_last2year = df.pts.values[-2]
    net_mean = np.mean(df.net_rating.values)
    ts_mean = np.mean(df.ts_pct.values)
    usg_mean = np.mean(df.usg_pct.values)
    n_season = df['season_no'].values[-1]
    data = np.array([[age, gp_mean, pts_mean, pts_last1year, pts_last2year, net_mean,ts_mean, usg_mean, n_season]])
    return pd.DataFrame(data)

def extract_feature(column,X_train,y_train,X_test,y_test):
    df_feature_train = X_train.sort_values([column],ascending=[True]).groupby(column).apply(lambda x: build_features(x))
    x_feature_train = df_feature_train.values
    labels_train = y_train.sort_values([column],ascending=[True]).pts.values

    df_feature_test = X_test.sort_values([column],ascending=[True]).groupby(column).apply(lambda x: build_features(x))
    x_feature_test = df_feature_test.values
    labels_test = y_test.sort_values([column],ascending=[True]).pts.values
    return x_feature_train,labels_train,x_feature_test,labels_test




def get_processed_data(column):
    x_feature_train, labels_train, x_feature_test, labels_test = extract_feature(column, *load_data(column))
    scaler = StandardScaler()
    scaler.fit(x_feature_train)
    x_feature_train = scaler.transform(x_feature_train)
    x_feature_test = scaler.transform(x_feature_test)
    return x_feature_train, labels_train, x_feature_test, labels_test


def data_pipeline(is_augmented):
    if is_augmented:
        column = "block_id"
    else:
        column = "ID"
    x_feature_train, labels_train, x_feature_test, labels_test = get_processed_data(column)
    return x_feature_train, labels_train, x_feature_test, labels_test

