from torchinfo import summary
from datetime import datetime
from torch import nn
import torch.nn.functional as F
import torch
import os
import sys

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
    newModel.lstm1.load_state_dict(pretrainedModel.lstm1.state_dict())
    newModel.lstm2.load_state_dict(pretrainedModel.lstm2.state_dict())

    # Freeze the weights of the LSTM layers
    for param in newModel.lstm1.parameters():
        param.requires_grad = False
    for param in newModel.lstm2.parameters():
        param.requires_grad = False

    return newModel