from torchinfo import summary
from datetime import datetime
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
        summary(model,input_size=input_size)
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
