import pandas as pd
import os
import numpy as np
from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim.adam import Adam
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
import math
from sklearn.cluster import KMeans

class Data(Dataset):
    def __init__(self, train_path, test_path, seq_length):
        self.train_df = pd.read_csv(train_path)
        self.test_df = pd.read_csv(test_path)
        self.seq_length = seq_length
        self.batch = 128

    def data_scaling(self):
        train_df = self.train_df
        test_df = self.test_df
        train_n, _ = train_df.shape
        test_n, _ = test_df.shape

        df_all = pd.concat([train_df, test_df], ignore_index=True)

        scaler_x = MinMaxScaler()
        scaler_x.fit(df_all.iloc[:, 2:7])
        df_all.iloc[:, 2:7] = scaler_x.transform(df_all.iloc[:, 2:7])

        scaler_y = MinMaxScaler()
        scaler_y.fit(df_all.iloc[:, [1]])
        df_all.iloc[:, [1]] = scaler_y.transform(df_all.iloc[:, [1]])

        imputer = SimpleImputer()
        train_imputer = imputer.fit(df_all.iloc[:train_n, 1:7])
        test_imputer = imputer.fit(df_all.iloc[:-test_n, 1:7])

        train_df = train_imputer.transform(df_all.iloc[:train_n, 1:7])
        test_df = test_imputer.transform(df_all.iloc[:-test_n, 1:7])

        return train_df, test_df, scaler_x, scaler_y

    def build_dataset(self, data, seq_length, is_test=False):
        X = []
        Y = []
        if is_test:
            length = 365 + seq_length - 1
        else:
            length = len(data)
        for i in range(0, length - seq_length):
            _x = data[i:i+seq_length, :]
            _y = data[i+seq_length, [0]]
            X.append(_x)
            Y.append(_y)
        return np.array(X), np.array(Y)

    def get_dataset(self):
        train_df, test_df, scaler_x, scaler_y = self.data_scaling()
        train_X, train_y = self.build_dataset(train_df, self.seq_length, is_test = False)
        test_X, test_y = self.build_dataset(test_df, self.seq_length, is_test=True)
        return train_X, train_y, test_X, test_y, scaler_x, scaler_y

# create the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, seq_len, layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = layers, batch_first = True)
        self.fc = nn.Linear(hidden_size, output_size, bias = True)
    
    def reset_hidden_state(self):
        self.hidden = (torch.zeros(self.layers, self.seq_len, self.hidden_size),
                      torch.zeros(self.layers, self.seq_len, self.hidden_size))
      
    def forward(self, x):
        h0 = torch.zeros(4, x.size(0), self.hidden_size)
        c0 = torch.zeros(4, x.size(0), self.hidden_size)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        out = self.fc(out[:,-1])
        return out

class ParallelLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_models):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_models = num_models
        self.layers = 10
        self.seq_len = 12
        self.lstm_models = LSTMModel(input_size=input_size, hidden_size=100, output_size=1, seq_len = self.seq_len, layers = self.layers)
        self.fc = nn.Linear(hidden_size * num_models, output_size)
    
    def reset_hidden_state(self):
        self.hidden = (torch.zeros(self.layers, self.seq_len, self.hidden_size),
                      torch.zeros(self.layers, self.seq_len, self.hidden_size))

    def forward(self, p, g, c, h):
        out0 = self.lstm_models(p)
        out1 = self.lstm_models(g)
        out2 = self.lstm_models(c)
        out3 = self.lstm_models(h)
        processed_inputs = torch.cat([out0, out1, out2, out3], dim=0)
        out = self.fc(torch.squeeze(processed_inputs))
        return out
    
class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, x):
        # x shape: (batch_size, seq_length, num_basins*input_size)
        attention_weights = nn.functional.softmax(x, dim=1)  # Compute attention weights
        attended_input = attention_weights * x  # Multiply input by attention weights
        return attended_input


class RiverLevelPredictor(nn.Module):
    def __init__(self, num_basins, input_size, hidden_size, num_layers, output_size):
        super(RiverLevelPredictor, self).__init__()
        self.num_basins = num_basins
        self.linear = nn.Linear(2, 1)  # Combine stage and flux data for all basins
        self.attention = Attention()
        self.lstm = nn.LSTM(input_size * num_basins, hidden_size, num_layers, batch_first=True)  # Add the linear output to the input
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def reset_hidden_state(self):
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        return (torch.zeros(self.num_layers, self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.hidden_size).to(device))

    def forward(self, x):
        # x shape: (batch_size, num_basins, seq_length, input_size)
        batch_size, num_basins, seq_length, input_size = x.size()

        # Extract stage and flux data (index 0 and 1)
        stage_and_flux = x[:, :, :, 0:2].reshape(batch_size, num_basins, seq_length, -1)

        # Apply linear layer to combine stage and flux data
        linear_out = self.linear(stage_and_flux)
        linear_out = linear_out.view(batch_size, num_basins, seq_length, -1)

        # Concatenate current stage, precipitation, temperature, humid, heat index and linear output
        current_stage = x[:, :, :, 0].unsqueeze(-1)
        precip = x[:, :, :, 2].unsqueeze(-1)
        temp = x[:, :, :, 3].unsqueeze(-1)
        humid = x[:, :, :, 4].unsqueeze(-1)
        heat = x[:, :, :, 5].unsqueeze(-1)

        lstm_input = torch.cat([current_stage, precip, temp, humid, heat, linear_out], dim=3)

        # LSTM layers
        lstm_input = lstm_input.permute(0, 2, 1, 3)  # shape: (batch_size, seq_length, num_basins, input_size)
        lstm_input = lstm_input.reshape(batch_size, seq_length, -1)  # shape: (batch_size, seq_length, num_basins * input_size)
        
        # Apply attention
        attended_input = self.attention(lstm_input)

        out, _ = self.lstm(attended_input)
        
        # Fully connected layer
        out = self.fc(out[:, -1, :])

        return out

class RiverLevelPredictorWithoutAtt(nn.Module):
    def __init__(self, num_basins, input_size, hidden_size, num_layers, output_size):
        super(RiverLevelPredictorWithoutAtt, self).__init__()
        self.num_basins = num_basins
        self.linear = nn.Linear(2, 1)  # Combine stage and flux data for all basins
        self.lstm = nn.LSTM(input_size * num_basins, hidden_size, num_layers, batch_first=True)  # Add the linear output to the input
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def reset_hidden_state(self):
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        return (torch.zeros(self.num_layers, self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.hidden_size).to(device))

    def forward(self, x):
        # x shape: (batch_size, num_basins, seq_length, input_size)
        batch_size, num_basins, seq_length, input_size = x.size()

        # Extract stage and flux data (index 0 and 1)
        stage_and_flux = x[:, :, :, 0:2].reshape(batch_size, num_basins, seq_length, -1)

        # Apply linear layer to combine stage and flux data
        linear_out = self.linear(stage_and_flux)
        linear_out = linear_out.view(batch_size, num_basins, seq_length, -1)

        # Concatenate current stage, precipitation, temperature, humid, heat index and linear output
        current_stage = x[:, :, :, 0].unsqueeze(-1)
        precip = x[:, :, :, 2].unsqueeze(-1)
        temp = x[:, :, :, 3].unsqueeze(-1)
        humid = x[:, :, :, 4].unsqueeze(-1)
        heat = x[:, :, :, 5].unsqueeze(-1)

        lstm_input = torch.cat([current_stage, precip, temp, humid, heat, linear_out], dim=3)

        # LSTM layers
        lstm_input = lstm_input.permute(0, 2, 1, 3)  # shape: (batch_size, seq_length, num_basins, input_size)
        lstm_input = lstm_input.reshape(batch_size, seq_length, -1)  # shape: (batch_size, seq_length, num_basins * input_size)

        out, _ = self.lstm(lstm_input)
        
        # Fully connected layer
        out = self.fc(out[:, -1, :])

        return out

class BasicDataset(Dataset):
    def __init__(self, x, y):
        super(BasicDataset, self).__init__()
        self.x = x
        self.y = y
        
    def __getitem__(self, index):
        # 현재 인덱스에 해당하는 모든 유역의 데이터를 반환
        x_item = self.x[:, index, :, :]
        y_item = self.y[index]
        return x_item, y_item

    def __len__(self):
        return self.x.shape[1]  # num_samples

def get_data(input_dir, seq_length):
    # train, test dataset
    paldang_dataset = Data(train_path= input_dir + '/train/revised_pal17-21.csv', test_path= input_dir + '/test/revised_pal22.csv', seq_length=seq_length)
    p_train_X, p_train_y, p_test_X, p_test_y, p_scaler_x, p_scaler_y = paldang_dataset.get_dataset()

    gwang_dataset = Data(train_path= input_dir + '/train/revised_gwang17-21.csv', test_path= input_dir + '/test/revised_gwang22.csv', seq_length=seq_length)
    g_train_X, g_train_y, g_test_X, g_test_y, g_scaler_x, g_scaler_y = gwang_dataset.get_dataset()

    cheong_dataset = Data(train_path= input_dir + '/train/revised_cheong17-21.csv', test_path= input_dir + '/test/revised_cheong22.csv', seq_length=seq_length)
    c_train_X, c_train_y, c_test_X, c_test_y, c_scaler_x, c_scaler_y = cheong_dataset.get_dataset()

    haeng_dataset = Data(train_path= input_dir + '/train/revised_haeng17-21.csv', test_path= input_dir + '/test/revised_haeng22.csv', seq_length=seq_length)
    h_train_X, h_train_y, h_test_X, h_test_y, h_scaler_x, h_scaler_y = haeng_dataset.get_dataset()

    x_scaler = h_scaler_x
    y_scaler = h_scaler_y

    # change into float tensor type dataset
    p_train_X, g_train_X, c_train_X, h_train_X = torch.FloatTensor(p_train_X), torch.FloatTensor(g_train_X), torch.FloatTensor(c_train_X), torch.FloatTensor(h_train_X)
    p_train_y, g_train_y, c_train_y, h_train_y = torch.FloatTensor(p_train_y), torch.FloatTensor(g_train_y), torch.FloatTensor(c_train_y), torch.FloatTensor(h_train_y)
    
    p_test_X, g_test_X, c_test_X, h_test_X = torch.FloatTensor(p_test_X), torch.FloatTensor(g_test_X), torch.FloatTensor(c_test_X), torch.FloatTensor(h_test_X)
    p_test_y, g_test_y, c_test_y, h_test_y = torch.FloatTensor(p_test_y), torch.FloatTensor(g_test_y), torch.FloatTensor(c_test_y), torch.FloatTensor(h_test_y)

    # concatenate into one matrix
    X_train = torch.stack([p_train_X, g_train_X, c_train_X, h_train_X], dim=0)
    y_train = torch.stack([h_train_y.squeeze()], dim=1)

    X_test = torch.stack([p_test_X, g_test_X, c_test_X, h_test_X], dim=0)
    y_test = torch.stack([h_test_y.squeeze()], dim=1)

    return X_train, y_train, X_test, y_test, x_scaler, y_scaler

def get_gold_data(input_gold_dir, seq_length):
    # train, test dataset
    gr_dataset = Data(train_path= input_gold_dir + '/test/revised_gr.csv', test_path= input_gold_dir + '/test/revised_gr.csv', seq_length=seq_length)
    gr_train_X, gr_train_y, gr_test_X, gr_test_y, gr_scaler_x, gr_scaler_y = gr_dataset.get_dataset()

    mh_dataset = Data(train_path= input_gold_dir + '/test/revised_mh.csv', test_path= input_gold_dir + '/test/revised_mh.csv', seq_length=seq_length)
    mh_train_X, mh_train_y, mh_test_X, mh_test_y, mh_scaler_x, mh_scaler_y = mh_dataset.get_dataset()

    beak_dataset = Data(train_path= input_gold_dir + '/test/revised_beak.csv', test_path= input_gold_dir + '/test/revised_beak.csv', seq_length=seq_length)
    beak_train_X, beak_train_y, beak_test_X, beak_test_y, beak_scaler_x, beak_scaler_y = beak_dataset.get_dataset()

    sh_dataset = Data(train_path= input_gold_dir + '/test/revised_sh.csv', test_path= input_gold_dir + '/test/revised_sh.csv', seq_length=seq_length)
    sh_train_X, sh_train_y, sh_test_X, sh_test_y, sh_scaler_x, sh_scaler_y = sh_dataset.get_dataset()
    
    x_gold_scaler = sh_scaler_x
    y_gold_scaler = sh_scaler_y
    
    gr_train_X, mh_train_X, beak_train_X, sh_train_X = torch.FloatTensor(gr_train_X), torch.FloatTensor(mh_train_X), torch.FloatTensor(beak_train_X), torch.FloatTensor(sh_train_X)
    gr_train_y, mh_train_y, beak_train_y, sh_train_y = torch.FloatTensor(gr_train_y), torch.FloatTensor(mh_train_y), torch.FloatTensor(beak_train_y), torch.FloatTensor(sh_train_y)
    gr_test_X, mh_test_X, beak_test_X, sh_test_X = torch.FloatTensor(gr_test_X), torch.FloatTensor(mh_test_X), torch.FloatTensor(beak_test_X), torch.FloatTensor(sh_test_X)
    gr_test_y, mh_test_y, beak_test_y, sh_test_y = torch.FloatTensor(gr_test_y), torch.FloatTensor(mh_test_y), torch.FloatTensor(beak_test_y), torch.FloatTensor(sh_test_y)
    
    X_train = torch.stack([gr_train_X, mh_train_X, beak_train_X, sh_train_X], dim=0)
    y_train = torch.stack([sh_train_y.squeeze()], dim=1)

    X_test = torch.stack([gr_test_X, mh_test_X, beak_test_X, sh_test_X], dim=0)
    y_test = torch.stack([sh_test_y.squeeze()], dim=1)
    
    return X_train, y_train, X_test, y_test, x_gold_scaler, y_gold_scaler
    
def train(model, train_loader, criterion, optimizer, device, num_epochs = None, verbose = 10, patience = 50):
    model.train()
    # epoch마다 loss 저장
    train_hist = np.zeros(num_epochs)

    # Training loop
    best_loss = float('inf')
    counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            model.reset_hidden_state()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_hist[epoch] = avg_loss
        
        if epoch % verbose == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            counter = 0
        else:
            counter += 1

        if counter == patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    return model, train_hist

def nse_metric(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    mse = np.mean((y_true - y_pred) ** 2)
    nse = 1 - mse / np.var(y_true)
    return nse

def rmse_metric(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def buffer_accuracy(y_true, y_pred, buffer):
    assert y_true.shape == y_pred.shape
    within_buffer = np.abs(y_true - y_pred) <= buffer
    accuracy = np.mean(within_buffer)
    return accuracy

def calculate_buffer(y_true, alpha=1):
    std_dev = np.std(y_true)
    buffer = alpha * std_dev
    return buffer

def apply_kmeans(X, y_pred, n_clusters=3):
    """
    Apply k-means clustering to the dataset.

    Args:
        X (numpy array): The input features.
        y_pred (numpy array): The predicted values.
        n_clusters (int): The number of clusters to form.

    Returns:
        kmeans (KMeans): The fitted k-means clustering model.
        y_clustered (numpy array): The cluster labels for each data point.
    """
    # Combine X and y_pred for clustering
    data_for_clustering = np.concatenate([X, y_pred.reshape(-1, 1)], axis=1)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data_for_clustering)

    # Get the cluster labels for each data point
    y_clustered = kmeans.labels_

    return kmeans, y_clustered

def test(model, test_loader, device, x_scaler, y_scaler, output_dir, gold = False):
    model.eval()
    mae_list = []
    all_outputs = []
    all_targets = []
    all_inputs = []  # For storing inputs

    with torch.no_grad():
        for batch_x, batch_y in test_loader:

            model.reset_hidden_state()

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            
            mae = nn.functional.l1_loss(outputs, batch_y, reduction='none').mean(1).cpu().numpy()
            mae_list.extend(mae.tolist())
            
            all_outputs.extend(outputs.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            all_inputs.extend(batch_x.cpu().numpy())  # Store inputs

    all_inputs = np.array(all_inputs)
    last_basin_data = all_inputs[:, 3, :, 1:7]

    # Average over the week
    last_watershed_data_avg = np.mean(last_basin_data, axis=1)
    all_targets = y_scaler.inverse_transform(np.array(all_targets).reshape(-1, 1))
    all_outputs = y_scaler.inverse_transform(np.array(all_outputs).reshape(-1, 1))
    all_inputs = x_scaler.inverse_transform(np.array(last_watershed_data_avg))  # Restore inputs
    buffer = calculate_buffer(all_targets, alpha=1)

    mean_mae = np.mean(mae_list)
    nse = nse_metric(all_targets, all_outputs)
    rmse = rmse_metric(all_targets, all_outputs)
    accuracy = buffer_accuracy(all_targets, all_outputs, buffer)

    print(f"Mean MAE: {mean_mae}")
    print(f"NSE: {nse}")
    print(f"RMSE: {rmse}")
    print(f"Buffer Accuracy: {accuracy * 100}% - Buffer: {buffer}")

    # Time data for x-axis
    time_data = pd.date_range(start='2022-01-01', periods=len(all_targets), freq='D')

    # Compare predicted and actual y values over time
    plt.plot(time_data, all_outputs, label='Predicted')
    plt.plot(time_data, all_targets, label='Actual')
    plt.xlabel('Time')
    plt.ylabel('River Level')
    plt.legend()
    plt.savefig(output_dir+'/Predicted_Han.png')
    plt.show()

    # Feature names for the scatter plot titles
    feature_names = ['flux', 'precipitation', 'temperature', 'humid', 'heat']

    # Scatter plot
    for i in range(all_inputs.shape[1]):  # For each input feature
        sns.scatterplot(x=all_inputs[:, i], y=all_targets.reshape(-1))
        plt.xlabel(feature_names[i])
        plt.ylabel('River Level')
        plt.title(f"Scatter plot of {feature_names[i]} vs River Level")
        if gold:
            plt.savefig(output_dir+ f"/Scatter_plot_{feature_names[i]}_Gold.png")
        else:
            plt.savefig(output_dir+ f"/Scatter_plot_{feature_names[i]}_Han.png")
        plt.show()

    # Convert each numpy array to a pandas DataFrame
    time_df = pd.DataFrame(time_data, columns=['time'])
    x_df = pd.DataFrame(all_inputs, columns=['flux', 'precipitation', 'temperature', 'humid', 'heat'])
    predicted_stage_df = pd.DataFrame(all_targets.reshape(-1), columns=['predicted_stage'])

    # Concatenate all the dataframes along the column axis
    data_df = pd.concat([time_df, x_df, predicted_stage_df], axis=1)

    # Save the dataframe to a csv file
    data_df.to_csv(output_dir + "/data.csv", index=False)


def split_x_data(train_data, num_chunks=5):
    num_basins, num_data, sequence, input_attribute = train_data.shape
    chunk_size = num_data // num_chunks

    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else num_data
        chunk = train_data[:, start_idx:end_idx, :, :]
        chunks.append(chunk)
    
    return chunks

def split_y_data(y_data, num_chunks=5):
    num_data, _ = y_data.shape
    chunk_size = num_data // num_chunks
    
    chunks = []
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < num_chunks - 1 else num_data
        chunk = y_data[start_idx:end_idx, :]
        chunks.append(chunk)
    
    return chunks

def parser():
    parser = argparse.ArgumentParser(description='5년 data input -> 1년 데이터 test')
    parser.add_argument('input', help='데이터 인풋의 경로 지정')
    parser.add_argument('output', help='학습한 모델의 결과를 저장할 폴더 경로')
    parser.add_argument('gold', help= '금강 데이터의 위치' )
    parser.add_argument('at', help = 'attention layer의 적용 여부 y/n')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--ep', default=100, type=int)
    parser.add_argument('--lu', default=7, type=int)
    args = parser.parse_args()
    return args

def main():
    args = parser()
    input_dir = args.input
    output_dir = args.output
    input_gold_dir = args.gold
    n_epochs = args.ep
    learning_rate = args.lr
    seq_length = args.lu
    attention_layer = args.at

    if attention_layer == 'y':
        print("Attention layer used")
        output_dir = output_dir + "/river_level_predictor" + "_ep" + str(n_epochs) + "_lr" + str(learning_rate) + "_lu" + str(seq_length) +'_at_y'
    elif attention_layer == 'n':
        print("Attention layer unused")
        output_dir = output_dir + "/river_level_predictor" + "_ep" + str(n_epochs) + "_lr" + str(learning_rate) + "_lu" + str(seq_length) +'_at_f'

    os.makedirs(output_dir, exist_ok=True)

    # get data
    X_train, y_train, X_test, y_test, x_scaler, y_scaler = get_data(input_dir,seq_length)

    # select device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    # dimension
    input_size = 6  # 강수량, 유량, 수위, 온도, 습도, 열지수
    hidden_size = 64
    num_layers = 1
    output_size = 1  # 예측할 수위
    
    if attention_layer == 'y':
        model = RiverLevelPredictor(4, input_size, hidden_size, num_layers, output_size).to(device)
    elif attention_layer == 'n':
        model = RiverLevelPredictorWithoutAtt(4, input_size, hidden_size, num_layers, output_size).to(device)
   
    # loss function, optimizer, number of epochs
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # split data by year
    X_train_data_chunks = split_x_data(X_train)
    y_train_data_chunks = split_y_data(y_train)

    # loading data from Dataloader
    train_hist = {}
    for i, train_chunk in enumerate(X_train_data_chunks):
        print(f"Training on year {i + 1}")
        train_dataset = BasicDataset(train_chunk, y_train_data_chunks[i])
        train_loader = DataLoader(train_dataset, batch_size=128, drop_last=True)
        model, train_hist[i] = train(model, train_loader, criterion, optimizer, device, num_epochs = n_epochs, verbose = 10, patience = 50)
    
    # plot training loss
    plt.figure(figsize=(10,4))
    plt.plot(train_hist[0], label = 'Training loss- 1year')
    plt.plot(train_hist[1], label = 'Training loss- 2year')
    plt.plot(train_hist[2], label = 'Training loss- 3year')
    plt.plot(train_hist[3], label = 'Training loss- 4year')
    plt.plot(train_hist[4], label = 'Training loss- 5year')
    plt.legend()
    plt.savefig(output_dir + "/TL_Han.png")
    plt.show()

    # save the trained model
    torch.save(model.state_dict(), output_dir + "/model.pth")
    # 나중에 모델 불러올 때 load_state_dict(torch.load(load_path)) 사용

    test_dataset = BasicDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset)
    test(model, test_loader, device, x_scaler, y_scaler, output_dir)

    
    print('-------------Testing gold data with transfer learning------------')
    X_train_gold, y_train_gold, X_test_gold, y_test_gold, x_gold_scaler, y_gold_scaler = get_gold_data(input_gold_dir, seq_length)

    # load trained weights
    for name, param in model.named_parameters():
        if name.count("linear"):
            param.requires_grad = False
        elif name.count("attention"):
            param.requires_grad = False
        elif name.count("lstm"):
            param.requires_grad = True
        elif name.count("fc"):
            param.requires_grad = True

    # model.layer_you_want_to_update.requires_grad = True

    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)  
    # we only pass the parameters that requires gradients (i.e., unfrozen)

    # split data by year
    X_train_gold_data_chunks = split_x_data(X_train_gold)
    y_train_gold_data_chunks = split_y_data(y_train_gold)

    for i, train_chunk in enumerate(X_train_gold_data_chunks):
        test_dataset = BasicDataset(train_chunk, y_train_gold_data_chunks[i])
        train_loader = DataLoader(train_dataset, batch_size=128, drop_last=True)
        model, _ = train(model, train_loader, criterion, optimizer, device, num_epochs = n_epochs, verbose = 10, patience = 50)
   
    # loading data from Dataloader
    test_dataset = BasicDataset(X_test_gold, y_test_gold)
    test_loader = DataLoader(test_dataset)
    test(model, test_loader, device, x_gold_scaler, y_gold_scaler, output_dir, gold=True)
    
if __name__ == '__main__':
    main()

