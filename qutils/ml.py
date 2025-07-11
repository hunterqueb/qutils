from torchinfo import summary
from datetime import datetime
from torch import nn
import torch.nn.functional as F
import torch
import os
import sys
from torch.optim.optimizer import Optimizer
import math
import torch.distributed as dist
import numpy as np
from qutils.tictoc import timer
import torch.utils.data as data
from qutils.mlExtras import findDecAcc
from qutils.mamba import Mamba
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

try:
    profile  # Check if the decorator is already defined (when running with memory_profiler)
except NameError:
    def profile(func):  # Define a no-op `profile` decorator if it's not defined
        return func

def getDevice():
    is_cuda = torch.cuda.is_available()
    # torch.backends.mps.is_available() checks for metal support, used in nightly build so handled expection incase its run on different version
    
    # right now disable mps - doesnt really work
    try:
        is_mps = torch.backends.mps.is_available()
        is_mps = False
    except:
        is_mps = False
    
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("GPU is available")
    elif is_mps:
        device = torch.device("mps")
        print('Metal GPU is available')
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    
    return device

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

def saveModel(model,input_size=None):
    '''
    Takes a pytorch model and saves the model floats,
    model summary, and a notes file for any notes.

    Parameters:
    model (torch.nn.Module): The architecture of the model into which the state dictionary will be loaded.
    input_size (int in a list): Input size of network for more descriptive summary.

    Returns:
    None

    Created 12/23/23
    Author: Hunter Quebedeaux 
    '''
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a subfolder with the timestamp
    folder_name = f"model_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # Create the filenames with the timestamp
    model_filename = os.path.join(folder_name, "model.pth")
    summary_filename = os.path.join(folder_name, "model_summary.txt")
    notes_filename = os.path.join(folder_name, "notes.txt")  # Filename for the notes file

    # Save the model
    torch.save(model.state_dict(), model_filename)
    print(f"Model saved as '{model_filename}'")

    # Save the summary
    original_stdout = sys.stdout  # Save a reference to the original standard output
    with open(summary_filename, 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        summary(model,input_size=input_size,col_names=["input_size", "output_size", "num_params", "trainable"])
        sys.stdout = original_stdout  # Reset the standard output to its original value
    with open(notes_filename, 'w') as f:
        pass  # This will create an empty file

    print(f"Model summary saved as '{summary_filename}'")
    print('Saved the following model at {}:'.format(timestamp))

    return

def loadModel(modelBase,model_filename,modelMode='eval'):
    """
    Load a PyTorch model's state dictionary from a specified file into the given model architecture.

    Parameters:
    model (torch.nn.Module): The architecture of the model into which the state dictionary will be loaded.
    model_path (str): Path to the file containing the model's state dictionary.

    Returns:
    torch.nn.Module: The model with loaded state dictionary.

    # Created 12/24/23
    # Author: Hunter Quebedeaux
    """

    # Load the state dictionary from the file
    modelBase.load_state_dict(torch.load(model_filename))

    # change to evaluate mode
    if modelMode == 'eval':
        modelBase.eval()
    elif modelMode == 'train':
        modelBase.train()
    else:
        print("Model mode unknown, defaulting to evaluation mode.")

    return modelBase

def printModelParmSize(model):
    total_params = 0
    total_memory = 0

    for param in model.parameters():
        # Number of parameters
        param_count = param.numel()
        total_params += param_count
        # Memory usage in bytes (assuming the parameters are stored as floats, typically 32 bits/4 bytes each)
        param_memory = param_count * param.element_size()
        total_memory += param_memory

    print("\n==========================================================================================")
    print(f"Total parameters: {total_params}")
    print(f"Total memory (bytes): {total_memory}")
    print(f"Total memory (MB): {total_memory / (1024 ** 2)}")
    print("==========================================================================================")

def printModelParameters(model):
    print("\n==========================================================================================")
    for name, param in model.named_parameters():
        print("---")
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")
        print(f"Parameter values:\n{param.data}")
        print("---\n")
    print("==========================================================================================")

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


class Adam_mini(Optimizer):
    '''
    adam mini optimizer from https://github.com/zyushun/Adam-mini \n
    optimizer that achieves on par of better performance than adamw w/ 45% - 50% less memory footprint.
    my custom implementation w/ removal of zero3, nembd, nhead, etc. these are removed b/c we are not training a language model.
    in my testing adam mini works better than adam or adamW, w/ around 1 dec acc increase. \n

    needs testing on cuda hardware

    '''
    def __init__(
        self,
        model=None,
        weight_decay=0.1,
        lr=1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        zero_3=False,
        n_embd = None,
        n_head = None,
        n_query_groups = None
    ):
        '''
        model: the model you are training.

        zero_3: set to True if you are using zero_3 in Deepspeed, or if you are using model parallelism with more than 1 GPU. Set to False if otherwise.
        
        n_embd: number of embedding dimensions. Could be unspecified if you are training non-transformer models.
        
        n_head: number of attention heads. Could be unspecified if you are training non-transformer models.
        
        n_query_groups: number of query groups in Group query Attention. If not specified, it will be equal to n_head. Could be unspecified if you are training non-transformer models.
        '''
       
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.n_embd = n_embd
        self.n_head = n_head
        if n_query_groups is not None:
            self.n_query_groups = n_query_groups
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head

        self.model = model
        self.world_size = torch.cuda.device_count()
        self.zero_optimization_stage = 0
        if zero_3:
            self.zero_optimization_stage = 3
            print("Adam-mini is using zero_3")
        optim_groups = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                dic = {}
                dic["name"] = name
                dic["params"] = param
                if ("norm" in name or "ln_f" in name):
                    dic["weight_decay"] = 0
                else:
                    dic["weight_decay"] = weight_decay
                
                if ("self_attn.k_proj.weight" in name or "self_attn.q_proj.weight" in name):
                    dic["parameter_per_head"] = self.n_embd * self.n_embd // self.n_head
                
                if ("attn.attn.weight" in name or "attn.qkv.weight" in name):
                    dic["n_head"] = self.n_head
                    dic["q_per_kv"] = self.n_head // self.n_query_groups

                optim_groups.append(dic)

        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)

        super(Adam_mini, self).__init__(optim_groups, defaults)


    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                lr = group["lr"]
                name = group["name"]
                epsilon = group["epsilon"]
                
                for p in group["params"]:
                    state = self.state[p]
                    if ("embed_tokens" in name or "wte" in name or "lm_head" in name):
                        if p.grad is None:
                            continue
                        if len(state) == 0:
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["iteration"] = 0
                            state["v"] = torch.zeros_like(p.data).to(torch.float32)

                        grad = p.grad.data.to(torch.float32)
                        state["v"].mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (state["v"].sqrt() / bias_correction_2_sqrt).add_(epsilon)
                        stepsize = lr/ bias_correction_1
                        p.addcdiv_(state["m"], h, value=-stepsize)

                    elif ("self_attn.k_proj.weight" in name or "self_attn.q_proj.weight" in name or "attn.wq.weight" in name or "attn.wk.weight" in name):
                        if p.grad is None:
                            continue
                        dim = group["parameter_per_head"]
                        if (len(state)==0):
                            state["m"]  =  torch.zeros_like(p.data).to(torch.float32)
                            state["m"] = state["m"].view(-1, dim)
                            state['head'] = state['m'].shape[0]
                            state["iteration"] = 0
                            state["vmean"] = torch.zeros(state['head']).to(self.device)

                        grad = p.grad.data.to(torch.float32)
                        head = state['head']
                        grad = grad.view(head, dim)

                        tmp_lr = torch.mean(grad*grad, dim = 1).to(self.device)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        v = state["vmean"]

                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (v.sqrt() / bias_correction_2_sqrt).add_(epsilon)    
                        stepsize = ((1/bias_correction_1) / h).view(head,1)

                        update = state["m"] * (stepsize.to(state['m'].self.device))

                        if p.dim() > 1:
                            d0, d1 = p.size()
                            update = update.view(d0, d1)
                        else: 
                            update = update.view(-1)

                        update.mul_(lr)
                        p.add_(-update)
                        
                    elif ("attn.attn.weight" in name or "attn.qkv.weight" in name): 
                        if p.grad is None:
                            continue
                        if (len(state)==0):
                            state["m"]  =  torch.zeros_like(p.data).to(torch.float32)
                            state["m"] = state["m"].view(group["n_head"], group["q_per_kv"] + 2, -1)
                            state["iteration"] = 0
                            state["vmean"] = torch.zeros(group["n_head"], group["q_per_kv"]+2).to(self.device)
                            

                        grad = p.grad.data.to(torch.float32)
                        grad = grad.view(group["n_head"], group["q_per_kv"] + 2, -1) 

                        tmp_lr = torch.mean(grad*grad, dim = 2).to(self.device)
                        state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                        v = state["vmean"]
                
                        state["iteration"] += 1
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])

                        state["m"].lerp_(grad, 1 - beta1)


                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)

                        h = (v.sqrt() / bias_correction_2_sqrt).add_(epsilon)    
                        stepsize = ((1/bias_correction_1) / h).view(group["n_head"],group["q_per_kv"]+2,1)
                                                    
   
                        update = state["m"] * (stepsize.to(state['m'].self.device))
            
                        if p.dim() > 1:
                            d0, d1 = p.size()
                            update = update.view(d0, d1)
                        else: 
                            update = update.view(-1)
        
                        update.mul_(lr)
                        p.add_(-update)

                        
                    else:        
                        if (len(state)==0):                   
                            dimension = torch.tensor(p.data.numel()).to(self.device).to(torch.float32)
                            reduced = False
                            if (self.world_size > 1) and (self.zero_optimization_stage == 3):
                                tensor_list = [torch.zeros_like(dimension) for _ in range(self.world_size)]
                                dist.all_gather(tensor_list, dimension)
                                s = 0
                                dimension = 0
                                for d in tensor_list:
                                    if (d>0):
                                        s = s + 1
                                    dimension = dimension + d
                                if (s>=2):
                                    reduced = True
                            
                            state["m"] = torch.zeros_like(p.data).to(torch.float32)
                            state["iteration"] = 0
                            state["reduced"] = reduced
                            state["vmean"] = torch.tensor(0.0).to(self.device)                                
                            state["dimension"] = dimension.item()
                        if p.grad is None:
                            tmp_lr = torch.tensor(0.0).to(self.device)
                        else:
                            grad = p.grad.data.to(torch.float32)
                            tmp_lr = torch.sum(grad*grad)                               
                        if (state["reduced"]):
                            dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)
                    
                        tmp_lr = tmp_lr / (state["dimension"])
                        tmp_lr = tmp_lr.to(grad.device)
                        if (p.grad is None):
                            continue
                        if group["weight_decay"] != 0:
                            p.data.mul_(1 - lr * group["weight_decay"])
                        state["iteration"] += 1
                        state["m"].lerp_(grad, 1 - beta1)

                        bias_correction_1 = 1 - beta1 ** state["iteration"]
                        bias_correction_2 = 1 - beta2 ** state["iteration"]
                        bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                        state["vmean"] = (1 - beta2) * tmp_lr + beta2 * state["vmean"]
                        h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(epsilon)    

                        stepsize = (1 / bias_correction_1) / h
                        update = state["m"] * (stepsize.to(state['m'].device))
                        update.mul_(lr)
                        p.add_(-update)    

def trainClassifier(model,optimizer,scheduler,dataloaders,criterion,num_epochs,device,schedulerPatience=5,printReport = False):
    '''
    Trains a classification model using a learning rate scheduler defined by the torch.optim.lr_scheduler module.
    '''
    timeToTrain = timer()
    train_loader,test_loader,val_loader = dataloaders[0],dataloaders[1],dataloaders[2]

    best_loss = float('inf')
    ESpatience = schedulerPatience * 2
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for sequences, labels in train_loader:
            sequences = sequences.to(device,non_blocking=True)
            labels = labels.to(device,non_blocking=True)

            # Forward
            logits = model(sequences)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        avg_val_loss, val_accuracy = validateMultiClassClassifier(model, val_loader, criterion, num_classes=logits.shape[1],device=device,printReport=printReport)

        # Step the scheduler based on validation loss
        scheduler.step(avg_val_loss)

        # Early stopping logic
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            counter = 0
            # Optional: save model checkpoint here
        else:
            counter += 1
            if counter >= ESpatience:
                print("Early stopping triggered.")
                break
    
    return timeToTrain.toc()

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        # h0, c0 default to zero if not provided
        out, (h_n, c_n) = self.lstm(x)
        
        # h_n is shape [num_layers, batch_size, hidden_size].
        # We typically take the last layer's hidden state: h_n[-1]
        last_hidden = h_n[-1]  # [batch_size, hidden_size]
        
        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        return logits
        

class MambaClassifier(nn.Module):
    def __init__(self,config, input_size, hidden_size, num_layers, num_classes):
        super(MambaClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.mamba = Mamba(config)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        
        h_n = self.mamba(x) # [batch_size, seq_length, hidden_size]
        
        # h_n is shape [batch_size, seq_length, hidden_size].
        # We typically take the last layer's hidden state: h_n[:,-1,:]
        last_hidden = h_n[:,-1,:]  # [batch_size, hidden_size]
        
        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        return logits

#tranformer classifier for time series data
class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(TransformerClassifier, self).__init__()
        
        self.d_model = hidden_size  # Output of transformer & input to fc
        self.embedding = nn.Linear(input_size, self.d_model)  # Project input to match d_model

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=8,  # Make sure d_model % nhead == 0
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=64,  # Internal feedforward layer size inside Transformer
            batch_first=True
        )
        
        self.fc = nn.Linear(self.d_model, num_classes)  # Final classification layer

    def forward(self, x):
        """
        x: [batch_size, seq_length, input_size]
        """
        x = self.embedding(x)         # [batch_size, seq_length, d_model]
        out = self.transformer(x, x)  # [batch_size, seq_length, d_model]
        last_output = out[:, -1, :]   # [batch_size, d_model]
        logits = self.fc(last_output) # [batch_size, num_classes]
        return logits


def validateMultiClassClassifier(model, val_loader, criterion, num_classes,device,classlabels=None,printReport=True):

    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    class_correct = torch.zeros(num_classes, dtype=torch.int32)
    class_total = torch.zeros(num_classes, dtype=torch.int32)

    # Collect predictions and labels for scikit-learn metrics
    y_true = []
    y_pred = []

    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(sequences)  # [batch_size, num_classes]
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # Per-class accuracy calculation
            for i in range(labels.size(0)):
                label = labels[i]
                pred = predicted[i]
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100.0 * correct / total

    print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%\n")

    print("Per-Class Validation Accuracy:")
    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i].item() / class_total[i].item()
            if classlabels is not None:
                print(f"  {classlabels[i]}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
            else:
                print(f"  Class {i}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            if classlabels is not None:
                print(f"  {classlabels[i]}: No samples")
            else:
                print(f"  Class {i}: No samples")

    if printReport:
        if classlabels is not None:
            print("\nClassification Report:")
            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=list(range(num_classes)),
                    target_names=classlabels,
                    digits=4,
                    zero_division=0,
                )
            )
        else:
            print("\nClassification Report:")
            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=list(range(num_classes)),
                    digits=4,
                    zero_division=0,
                )
            )
            # Confusion-matrix -----------------------------------------------------
    if printReport:
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        print("\nConfusion Matrix (rows = true, cols = predicted):")
        print(pd.DataFrame(cm,
                            index=[f"T_{cls}" for cls in (classlabels if classlabels else range(num_classes))],
                            columns=[f"P_{cls}" for cls in (classlabels if classlabels else range(num_classes))]))

    return avg_val_loss, val_accuracy

def findClassWeights(train_dataset,device):
    all_labels = np.array([label.item() for _, label in train_dataset])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(all_labels),
        y=all_labels
    )

    print("Class weights:", class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    return class_weights