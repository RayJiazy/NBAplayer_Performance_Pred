import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from FeatureCatcher import FeatureCatcher
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class MLP_5layer(nn.Module):
    def __init__(self, n_input, n_hidden1,n_hidden2,n_hidden3,n_hidden4):  # Define layers in the constructor
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, n_hidden3)
        self.fc4 = nn.Linear(n_hidden3, n_hidden4)
        self.fc5 = nn.Linear(n_hidden4, 1)

    def forward(self, x):  # Define forward pass in the forward method
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.fc5(x)
        return x

class MLP_6layer(nn.Module):
    def __init__(self, n_input):  # Define layers in the constructor
        super().__init__()
        self.fc1 = nn.Linear(n_input, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 15)
        self.fc4 = nn.Linear(15, 8)
        self.fc5 = nn.Linear(8, 5)
        self.fc6 = nn.Linear(5, 1)

    def forward(self, x):  # Define forward pass in the forward method
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x

class MLP_3layer(nn.Module):
    def __init__(self, n_input, n_hidden1, n_hidden2):  # Define layers in the constructor
        super().__init__()
        self.fc1 = nn.Linear(n_input, n_hidden1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc3 = nn.Linear(n_hidden2, 1)

    def forward(self, x):  # Define forward pass in the forward method
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class MultiPredModelMLP:
    def __init__(self, epoch, lr_pts, lr_reb, lr_ast, reg_val_pts, reg_val_reb, reg_val_ast):
        self.epoch_pts = epoch
        self.epoch_reb = epoch
        self.epoch_ast = epoch
        self.lr_pts = lr_pts
        self.lr_reb = lr_reb
        self.lr_ast = lr_ast
        self.reg_val_pts = reg_val_pts
        self.reg_val_reb = reg_val_reb
        self.reg_val_ast = reg_val_ast
        self.device = torch.device("cpu")
        self.loss_func_pts = nn.MSELoss()
        self.loss_func_reb = nn.MSELoss()
        self.loss_func_ast = nn.MSELoss()
        self.model_pts = MLP_5layer(n_input=9, n_hidden1=20, n_hidden2=20,n_hidden3=10,n_hidden4=5)
        self.model_reb = MLP_3layer(n_input=9, n_hidden1=5, n_hidden2=5)
        self.model_ast = MLP_3layer(n_input=9, n_hidden1=5, n_hidden2=5)
        # self.model_pts = MLP_6layer(n_input=9)
        # self.model_reb = MLP_6layer(n_input=9)
        # self.model_ast = MLP_6layer(n_input=9)

    def models_train(self, train_loader_pts, train_loader_reb, train_loader_ast):
        print('=====================================')
        print('start train pts model')
        optimizer_pts = torch.optim.SGD(self.model_pts.parameters(), lr=self.lr_pts, weight_decay=self.reg_val_pts)
        self.train(self.model_pts, self.epoch_pts, train_loader_pts, optimizer_pts, self.loss_func_pts)
        print('=====================================')
        print('start train reb model')
        optimizer_reb = torch.optim.SGD(self.model_reb.parameters(), lr=self.lr_reb, weight_decay=self.reg_val_reb)
        self.train(self.model_reb, self.epoch_reb, train_loader_reb, optimizer_reb, self.loss_func_reb )
        print('=====================================')
        print('start train ast model')
        optimizer_ast = torch.optim.SGD(self.model_ast.parameters(), lr=self.lr_ast, weight_decay=self.reg_val_ast)
        self.train(self.model_ast, self.epoch_ast, train_loader_ast, optimizer_ast, self.loss_func_ast )

    def models_test(self, test_loader_pts, test_loader_reb, test_loader_ast):
        loss_func = nn.MSELoss()
        test_pts_mse = self.test(self.model_pts, test_loader_pts, self.loss_func_pts)
        test_reb_mse = self.test(self.model_reb, test_loader_reb, self.loss_func_reb)
        test_ast_mse = self.test(self.model_ast, test_loader_ast, self.loss_func_ast)
        print(f'test pts rmse:{np.sqrt(test_pts_mse)}, test reb rmse:{np.sqrt(test_reb_mse)}, test ast rmse:{np.sqrt(test_ast_mse)}')
        return [np.sqrt(test_pts_mse), np.sqrt(test_reb_mse), np.sqrt(test_ast_mse)]

    def models_predict(self, test_loader_pts, test_loader_reb, test_loader_ast):
        pred_pts = self.predict(self.model_pts, test_loader_pts)
        pred_reb = self.predict(self.model_reb, test_loader_reb)
        pred_ast = self.predict(self.model_ast, test_loader_ast)

        pred_pts, pred_reb, pred_ast = np.array(pred_pts).reshape(len(pred_pts),1), np.array(pred_reb).reshape(len(pred_reb),1), np.array(pred_ast).reshape(len(pred_ast),1)
        prediction = np.hstack((np.hstack((pred_pts, pred_reb)), pred_ast))
        return prediction

    def train(self, model, epoch_num, trainloader, optimizer, loss_func):
        model.train()
        model.to(self.device)
        train_loss_all = []
        for epoch in range(epoch_num):
            train_loss = 0
            train_num = 0
            for step, (x, y) in enumerate(trainloader):
                x, y = x.to(self.device), y.view(len(y),1).to(self.device)# Move batch to device
                optimizer.zero_grad()
                output = model(x)
                loss = loss_func(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
                train_num += x.size(0)
            train_loss_all.append(train_loss / train_num)
            if epoch % 10 == 0:
                print(f'epoch:{epoch},MSE loss:{train_loss / train_num}')

    def test(self, model, testloader, loss_func):
        model.eval()  # set model to evaluation mode
        running_loss = 0
        num = 0
        with torch.no_grad():  # no need to compute gradients for testing
            for step, (x, y) in enumerate(testloader):
                x, y = x.to(self.device), y.view(len(y),1).to(self.device)
                output = model(x)
                # if step == 1:
                #     print(output[:10], y[:10])
                loss = loss_func(output, y)  # Compute loss
                running_loss += loss.item() * x.size(0)
                num += x.size(0)
        return running_loss / num

    def predict(self, model, testloader):
        model.eval()  # set model to evaluation mode
        prediciton = []
        with torch.no_grad():  # no need to compute gradients for testing
            for step, (x, y) in enumerate(testloader):
                x, y = x.to(self.device), y.view(len(y),1).to(self.device)
                output = model(x).cpu().numpy().tolist()
                prediciton += output
        return prediciton


    def get_RMSE_loss(self, pred, y):
        return np.sqrt(np.sum(np.power((pred - y), 2)) / len(pred))

    def get_MAE_loss(self, pred, y):
        return np.sum(np.abs(pred - y)) / len(pred)



#split input and label
def split_input(df):
    return df.iloc[:-1]

def split_label(df):
    return df.iloc[-1]

def get_input_label(df):
    df_xtrain = df.groupby('block_id').apply(lambda x: split_input(x))
    df_label = df.groupby('block_id').apply(lambda x: split_label(x))
    df_xtrain.reset_index(drop=True, inplace=True)
    df_xtrain.reset_index(drop=True, inplace=True)
    return df_xtrain, df_label

def split_train_test(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def KFold_cross_validation(x_pts_train, y_pts_train, x_reb_train, y_reb_train, x_ast_train, y_ast_train):
    kfold = KFold(n_splits=20, shuffle=True)
    mse_loss_val = []
    reg_list = [0.1, 1e-2, 1e-3, 1e-4]
    lr_list = [0.1, 1e-2, 1e-3, 1e-4]
    for n in range(4):
        reg_val = reg_list[n]
        # lr = lr_list[n]
        mse_history = []
        count = 0
        for train_index, val_index in kfold.split(x_pts_train, y_pts_train):
            count += 1
            print('=================A ROUND FOR K-FOLD VALIDATION==========')
            #dataset
            x_pts_t, y_pts_t = x_pts_train[train_index], y_pts_train[train_index]  # training set
            x_pts_v, y_pts_v = x_pts_train[val_index], y_pts_train[val_index]  # val set
            x_reb_t, y_reb_t = x_reb_train[train_index], y_reb_train[train_index]
            x_reb_v, y_reb_v = x_reb_train[val_index], y_reb_train[val_index]
            x_ast_t, y_ast_t = x_ast_train[train_index], y_ast_train[train_index]
            x_ast_v, y_ast_v = x_ast_train[val_index], y_ast_train[val_index]

            x_pts_t, x_pts_v, y_pts_t, y_pts_v = torch.from_numpy(x_pts_t.astype(np.float32)), torch.from_numpy(x_pts_v.astype(np.float32)), torch.from_numpy(y_pts_t.astype(np.float32)), torch.from_numpy(y_pts_v.astype(np.float32))
            x_reb_t, x_reb_v, y_reb_t, y_reb_v = torch.from_numpy(x_reb_t.astype(np.float32)), torch.from_numpy(x_reb_v.astype(np.float32)), torch.from_numpy(y_reb_t.astype(np.float32)), torch.from_numpy(y_reb_v.astype(np.float32))
            x_ast_t, x_ast_v, y_ast_t, y_ast_v = torch.from_numpy(x_ast_t.astype(np.float32)), torch.from_numpy(x_ast_v.astype(np.float32)), torch.from_numpy(y_ast_t.astype(np.float32)), torch.from_numpy(y_ast_v.astype(np.float32))

            train_data_pts = Data.TensorDataset(x_pts_t, y_pts_t)
            test_data_pts = Data.TensorDataset(x_pts_v, y_pts_v)
            train_data_reb = Data.TensorDataset(x_reb_t, y_reb_t)
            test_data_reb = Data.TensorDataset(x_reb_v, y_reb_v)
            train_data_ast = Data.TensorDataset(x_ast_t, y_ast_t)
            test_data_ast = Data.TensorDataset(x_ast_v, y_ast_v)

            train_loader_pts = Data.DataLoader(dataset=train_data_pts, batch_size=64)
            test_loader_pts = Data.DataLoader(dataset=test_data_pts, batch_size=64)
            train_loader_reb = Data.DataLoader(dataset=train_data_reb, batch_size=64)
            test_loader_reb = Data.DataLoader(dataset=test_data_reb, batch_size=64)
            train_loader_ast = Data.DataLoader(dataset=train_data_ast, batch_size=64)
            test_loader_ast = Data.DataLoader(dataset=test_data_ast, batch_size=64)

            # Parameters for the model
            lr = 1e-3
            MLP_model = MultiPredModelMLP(50, lr, lr, lr, reg_val, reg_val, reg_val)
            MLP_model.models_train(train_loader_pts, train_loader_reb, train_loader_ast)
            val_loss = MLP_model.models_test(test_loader_pts, test_loader_reb, test_loader_ast)

            mse_history.append(val_loss)
        mse_loss_val.append(mse_history)
    return mse_loss_val




if __name__ == "__main__":
    df_data = pd.read_csv('data/trainset.csv')
    df_xtrain, df_label = get_input_label(df_data)

    fc = FeatureCatcher(df_xtrain=df_xtrain, df_label=df_label)
    feature_pts, feature_reb, feature_ast, label_pts, label_reb, label_ast = fc.get_feature_labels()

    x_pts_train, x_pts_test, y_pts_train, y_pts_test = split_train_test(feature_pts, label_pts)
    x_reb_train, x_reb_test, y_reb_train, y_reb_test = split_train_test(feature_reb, label_reb)
    x_ast_train, x_ast_test, y_ast_train, y_ast_test = split_train_test(feature_ast, label_ast)

    #standerization
    scaler = StandardScaler()
    scaler.fit(x_pts_train)
    x_pts_train, x_pts_test = scaler.transform(x_pts_train), scaler.transform(x_pts_test)
    scaler = StandardScaler()
    scaler.fit(x_reb_train)
    x_reb_train, x_reb_test = scaler.transform(x_reb_train), scaler.transform(x_reb_test)
    scaler = StandardScaler()
    scaler.fit(x_ast_train)
    x_ast_train, x_ast_test = scaler.transform(x_ast_train), scaler.transform(x_ast_test)

    # #cross validation
    # mse_loss_val = KFold_cross_validation(x_pts_train, y_pts_train, x_reb_train, y_reb_train, x_ast_train, y_ast_train)


    #numpy.ndarray to torch.tensor
    x_pts_train, x_pts_test, y_pts_train, y_pts_test = torch.from_numpy(x_pts_train.astype(np.float32)), torch.from_numpy(x_pts_test.astype(np.float32)), torch.from_numpy(y_pts_train.astype(np.float32)), torch.from_numpy(y_pts_test.astype(np.float32))
    x_reb_train, x_reb_test, y_reb_train, y_reb_test = torch.from_numpy(x_reb_train.astype(np.float32)), torch.from_numpy(x_reb_test.astype(np.float32)), torch.from_numpy(y_reb_train.astype(np.float32)), torch.from_numpy(y_reb_test.astype(np.float32))
    x_ast_train, x_ast_test, y_ast_train, y_ast_test = torch.from_numpy(x_ast_train.astype(np.float32)), torch.from_numpy(x_ast_test.astype(np.float32)), torch.from_numpy(y_ast_train.astype(np.float32)), torch.from_numpy(y_ast_test.astype(np.float32))

    train_data_pts = Data.TensorDataset(x_pts_train, y_pts_train)
    test_data_pts = Data.TensorDataset(x_pts_test, y_pts_test)
    train_data_reb = Data.TensorDataset(x_reb_train, y_reb_train)
    test_data_reb = Data.TensorDataset(x_reb_test, y_reb_test)
    train_data_ast = Data.TensorDataset(x_ast_train, y_ast_train)
    test_data_ast = Data.TensorDataset(x_ast_test, y_ast_test)


    train_loader_pts = Data.DataLoader(dataset=train_data_pts, batch_size=64)
    test_loader_pts = Data.DataLoader(dataset=test_data_pts, batch_size=64)
    train_loader_reb = Data.DataLoader(dataset=train_data_reb, batch_size=64)
    test_loader_reb = Data.DataLoader(dataset=test_data_reb, batch_size=64)
    train_loader_ast = Data.DataLoader(dataset=train_data_ast, batch_size=64)
    test_loader_ast = Data.DataLoader(dataset=test_data_ast, batch_size=64)

    print(len(train_loader_pts), len(x_pts_train))

    MLP_model = MultiPredModelMLP(1000, 1e-2,1e-2,1e-2,1e-3,1e-3,1e-3)
    MLP_model.models_train(train_loader_pts, train_loader_reb, train_loader_ast)
    MLP_model.models_test(test_loader_pts, test_loader_reb, test_loader_ast)
    # prediction = MLP_model.models_predict(test_loader_pts, test_loader_reb, test_loader_ast)