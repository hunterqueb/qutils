from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from qutils.tictoc import timer
import torch.utils.data as data
from qutils.ml.extras import findDecAcc

try:
    profile  # Check if the decorator is already defined (when running with memory_profiler)
except NameError:
    def profile(func):  # Define a no-op `profile` decorator if it's not defined
        return func


def create_datasets(data,seq_length,train_size,device):
    xs, ys = [], []
    for i in range (len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        y = np.reshape(y,(1,y.shape[0]))
        xs.append(x)
        ys.append(y)
    
    X_train, X_test = xs[:train_size], xs[train_size:]
    Y_train, Y_test = ys[:train_size], ys[train_size:]
    # Convert to PyTorch tensors
    X_train = torch.tensor(np.array(X_train)).double().to(device)
    Y_train = torch.tensor(np.array(Y_train)).double().to(device)
    X_test = torch.tensor(np.array(X_test)).double().to(device)
    Y_test = torch.tensor(np.array(Y_test)).double().to(device)

    return X_train,Y_train,X_test,Y_test

def trainModel(model,n_epochs,datasets,criterion,optimizer,printOutAcc = True,printOutToc = True):
    train_in = datasets[0]
    train_out = datasets[1]
    test_in = datasets[2]
    test_out = datasets[3]
    
    loader = data.DataLoader(data.TensorDataset(train_in, train_out), shuffle=True, batch_size=8)


    if printOutToc: # if printing out toc, use timer class
        trainTime = timer()

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            y_pred_train = model(train_in)
            train_loss = np.sqrt(criterion(y_pred_train, train_out).cpu())
            y_pred_test = model(test_in)
            test_loss = np.sqrt(criterion(y_pred_test, test_out).cpu())

            decAcc, err1 = findDecAcc(train_out,y_pred_train,printOut=False)
            decAcc, err2 = findDecAcc(test_out,y_pred_test,printOut=printOutAcc)
            err = np.concatenate((err1,err2),axis=0)

        if printOutAcc:
            print("Epoch %d: train loss %.4f, test loss %.4f\n" % (epoch, train_loss, test_loss))

    if printOutToc:
        timeToTrain = trainTime.toc()
        return timeToTrain
    else:
        return None


@profile
def genPlotPrediction(model,output_seq,train_in,test_in,train_size,seq_length):
    with torch.no_grad():
        # shift train predictions for plotting
        train_plot = np.ones_like(output_seq) * np.nan
        y_pred = model(train_in)
        y_pred = y_pred[:, -1, :]
        train_plot[seq_length:train_size+seq_length] = model(train_in)[:, -1, :].cpu()
        # shift test predictions for plotting
        test_plot = np.ones_like(output_seq) * np.nan
        test_plot[train_size+seq_length:] = model(test_in)[:, -1, :].cpu()

    return train_plot, test_plot

class SelfAttentionLayer(nn.Module):
   def __init__(self, feature_size):
       super(SelfAttentionLayer, self).__init__()
       self.feature_size = feature_size

       # Linear transformations for Q, K, V from the same source
       self.key = nn.Linear(feature_size, feature_size)
       self.query = nn.Linear(feature_size, feature_size)
       self.value = nn.Linear(feature_size, feature_size)

   def forward(self, x, mask=None):
       # Apply linear transformations
       keys = self.key(x)
       queries = self.query(x)
       values = self.value(x)

       # Scaled dot-product attention
       scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

       # Apply mask (if provided)
       if mask is not None:
           scores = scores.masked_fill(mask == 0, -1e9)

       # Apply softmax
       attention_weights = F.softmax(scores, dim=-1)

       # Multiply weights with values
       output = torch.matmul(attention_weights, values)

       return output, attention_weights


class LSTMSelfAttentionNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_value, heads=1):
        super(LSTMSelfAttentionNetwork, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,dropout=dropout_value)

        # Self-attention layer
        self.self_attention = SelfAttentionLayer(hidden_dim)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass data through LSTM layer
        lstm_out, lstm_hidden = self.lstm(x)

        # Pass data through self-attention layer
        attention_out, attention_weights = self.self_attention(lstm_out,mask=None)

        # Pass data through fully connected layer
        final_out = self.fc(attention_out)

        return final_out

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_value, heads=1):
        super(LSTM, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True,dropout=dropout_value)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Pass data through LSTM layer
        lstm_out, _ = self.lstm(x)

        # Pass data through fully connected layer
        final_out = self.fc(lstm_out)

        return final_out

def transferLSTM(pretrainedModel,newModel):
    '''
    custom function to transfer knowledge of LSTM network from a pretrained model to a new model

    parameters: pretrainedModel - pretrained pytorch model with two LSTM layers
                newModel - untrained pytorch model with two LSTM layers
    '''
    newModel.lstm.load_state_dict(pretrainedModel.lstm.state_dict())

    # Freeze the weights of the LSTM layers
    for param in newModel.lstm.parameters():
        param.requires_grad = True

    return newModel

def transferMamba(pretrainedModel,newModel,trainableLayers = [True,True,True]):
    '''
    custom function to transfer knowledge of a mamba network from a pretrained model to a new model
    the mamba network is from https://github.com/alxndrTL/mamba.py

    parameters: pretrainedModel - pretrained pytorch mamba model with one state space layer
                newModel - untrained pytorch mamba model with one state space layer
    '''
    # deltaBC is calced simultaneously here!
    # model.layers[0].mixer.x_proj.state_dict()

    # load the parameters from the old model to the new, and set all parameters to untrainable
    newModel.load_state_dict(pretrainedModel.state_dict())
    for param in newModel.parameters():
        param.requires_grad = False

    for param in newModel.layers[0].mixer.conv1d.parameters():
        param.requires_grad = True

    # trainanle A matrix
    newModel.layers[0].mixer.A_log.requires_grad = False

    # trainable deltaBC matrix
    for param in newModel.layers[0].mixer.x_proj.parameters():
        param.requires_grad = trainableLayers[0]

    for param in newModel.layers[0].mixer.dt_proj.parameters():
        param.requires_grad = trainableLayers[1]

    # probably not the best to transfer, this is the projection from the latent state space back to the output space
    for param in newModel.layers[0].mixer.out_proj.parameters():
        param.requires_grad = trainableLayers[2]


    return newModel

def transferModelAll(pretrainedModel,newModel):
    newModel.load_state_dict(pretrainedModel.state_dict())
    for param in newModel.parameters():
        param.requires_grad = True
    return newModel
