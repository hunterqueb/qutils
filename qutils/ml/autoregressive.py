import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def getDataLoader(training_data,seq_length=5,batch_size = 8):
    L = 5
    X_list = []
    y_list = []
    for i in range(len(training_data) - L):
        X_list.append(training_data[i : i+L])
        y_list.append(training_data[i+L])

    X = np.array(X_list)  # shape: (num_samples, L,dim)
    y = np.array(y_list)  # shape: (num_samples,dim)

    class TimeSeriesDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(X).float()
            self.y = torch.from_numpy(y).float()
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    dataset = TimeSeriesDataset(X, y)
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=True)

def trainModel(model,data,n_epochs,criterion,optimizer,seq_length=5,batch_size = 8):
    dataset, dataloader = getDataLoader(data,seq_length,batch_size)
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in dataloader:
            # batch_X = batch_X.unsqueeze(-1)  # (batch_size, L) -> (batch_size, L, 1)
            
            # Forward
            predictions = model(batch_X) # .squeeze()  # (batch_size,)
            loss = criterion(predictions[:,-1,:], batch_y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_X.size(0)
        
        epoch_loss = total_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {epoch_loss:.4f}")

def testModel(model,trainingDataset,testingData):
    model.eval()
    predictions = []
    last_window = trainingDataset.X[-1]
    N = testingData.shape[0]
    current_window = torch.tensor(last_window, dtype=torch.float).unsqueeze(0)
    # shape now: (1, L, ndim)

    with torch.no_grad():
        for _ in range(N):
            pred = model(current_window)         # shape: (1, 1)
            next_value = pred[:,-1,:].numpy()
            predictions.append(next_value)
            
            # Shift window
            current_window = torch.cat([current_window[:, 1:, :],
                                        pred[:, -1, :].reshape(1,1,-1)], dim=1)

    return np.array(predictions).squeeze(1)


def genPlotPred(output_seq,train,test,prediction):
    train_size = train.shape[0]
    test_size = test.shape[0]

    train_plot = np.ones_like(output_seq) * np.nan
    test_plot = np.ones_like(output_seq) * np.nan
    pred_plot = np.ones_like(output_seq) * np.nan
    
    train_plot[:train_size,:] = train
    test_plot[train_size:,:] = test
    pred_plot[train_size:,:] = prediction

    return train_plot,test_plot,pred_plot
