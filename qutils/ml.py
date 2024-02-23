from torchinfo import summary
from datetime import datetime
from torch import nn
import torch
import os
import sys

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

class LSTM(nn.Module):
    '''
    simple two layers LSTM network with one fully connected output layer

    parameters: input_size - input size of network
                hidden_size - hidden size of each hidden layer
                output_size - output size of the last layer
    
    returns pytorch model upon instantiation
    '''
    def __init__(self,input_size:int,hidden_size:int,output_size:int):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs

class LSTMNew(nn.Module):
    '''
    simple two layers LSTM network with two fully connected output layers

    parameters: input_size - input size of network
                hidden_size - hidden size of each hidden layer
                output_size - output size of the last layer
    
    returns pytorch model upon instantiation
    '''
    def __init__(self,input_size:int,hidden_size:int,output_size:int):
        super(LSTMNew, self).__init__()
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)  # First fully connected layer
        self.linear2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, input, future = 0):
        outputs = []
        h_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double)
        c_t = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), self.hidden_size, dtype=torch.double)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        for i in range(future):
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear1(h_t2)  # Use the new first fully connected layer
            output = self.linear2(output)  # Use the new second fully connected layer
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


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

if __name__ == "__main__":
    from torchinfo import summary

    model = LSTM(10000,5000,10000)
    summary(model)