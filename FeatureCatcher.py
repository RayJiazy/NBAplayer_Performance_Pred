import pandas as pd
import numpy as np


class FeatureCatcher:
    def __init__(self, df_xtrain, df_label):
        self.df_xtrain = df_xtrain
        self.df_label = df_label

    def get_feature_labels(self):
        feature_pts = self.df_xtrain.groupby('block_id').apply(lambda x: self.build_features_pts(x)).values
        feature_reb = self.df_xtrain.groupby('block_id').apply(lambda x: self.build_features_rebs(x)).values
        feature_ast = self.df_xtrain.groupby('block_id').apply(lambda x: self.build_features_ast(x)).values
        label_pts = self.df_label['pts'].values
        label_reb = self.df_label['reb'].values
        label_ast = self.df_label['ast'].values
        return feature_pts, feature_reb, feature_ast, label_pts, label_reb, label_ast

    def build_features_pts(self, df):
        age = df.age.values[-1]
        gp_mean = np.mean(df.gp.values)
        pts_mean = np.mean(df.pts.values)
        pts_last1year = df.pts.values[-1]
        pts_last2year = df.pts.values[-2]
        net_rmean = np.mean(df.net_rating.values)
        ts_mean = np.mean(df.ts_pct.values)
        usg_mean = np.mean(df.usg_pct.values)
        n_season = df['season_no'].values[-1]

        data = np.array(
            [[age, gp_mean, pts_mean, pts_last1year, pts_last2year, net_rmean, ts_mean, usg_mean, n_season]])
        return pd.DataFrame(data)

    def build_features_rebs(self, df):
        age = df.age.values[-1]
        gp_mean = np.mean(df.gp.values)
        rebs_mean = np.mean(df.reb.values)
        rebs_last1year = df.reb.values[-1]
        rebs_last2year = df.reb.values[-2]
        net_rmean = np.mean(df.net_rating.values)
        oreb_mean = np.mean(df.oreb_pct.values)
        dreb_mean = np.mean(df.dreb_pct.values)
        n_season = df['season_no'].values[-1]

        data = np.array(
            [[age, gp_mean, rebs_mean, rebs_last1year, rebs_last2year, net_rmean, oreb_mean, dreb_mean, n_season]])
        return pd.DataFrame(data)

    def build_features_ast(self, df):
        age = df.age.values[-1]
        gp_mean = np.mean(df.gp.values)
        ast_mean = np.mean(df.ast.values)
        ast_last1year = df.ast.values[-1]
        ast_last2year = df.ast.values[-2]
        net_rmean = np.mean(df.net_rating.values)
        usg_mean = np.mean(df.usg_pct.values)
        ast_pct_mean = np.mean(df.ast_pct.values)
        n_season = df['season_no'].values[-1]

        data = np.array(
            [[age, gp_mean, ast_mean, ast_last1year, ast_last2year, net_rmean, usg_mean, ast_pct_mean, n_season]])
        return pd.DataFrame(data)