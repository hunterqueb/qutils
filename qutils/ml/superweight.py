import torch
from qutils.ml.mamba import Mamba, MambaClassifier
import matplotlib.pyplot as plt

def printoutMaxLayerWeight(model):
    if isinstance(model,Mamba) or isinstance(model,MambaClassifier):
        if isinstance(model,MambaClassifier):
            model = model.mamba
        for param in model.named_parameters():
            print("Weight Tensor Name: ", param[0])  # Name of the parameter
            # Flatten the tensor to find the maximum index
            flat_tensor = param[1].flatten()
            flat_abs_tensor = flat_tensor.abs()
            flat_index = flat_abs_tensor.argmax()   
            max_abs_value = flat_tensor[flat_index]
            print("Maximum Weight Value: ", max_abs_value.item())
            # Calculate the original multi-dimensional index manually
            original_shape_index = (param[1]==torch.max(torch.abs(param[1]))).nonzero()
            print("Maximum Weight Index (original shape): ", original_shape_index)
            print()
    else:
        print("Model is not a mamba model. Returning...")
        return

def getSuperWeight(model):
    if isinstance(model,Mamba) or isinstance(model,MambaClassifier):
        if isinstance(model,MambaClassifier):
            model = model.mamba
        highest = [None,0,None]
        for param in model.named_parameters():
            flat_tensor = param[1].flatten()
            flat_abs_tensor = flat_tensor.abs()
            flat_index = flat_abs_tensor.argmax()   
            max_abs_value = flat_tensor[flat_index]
            if abs(highest[1]) < abs(max_abs_value.item()):
                highest[0] = param[0]
                highest[1] = max_abs_value.item()
                highest[2] = (param[1]==torch.max(torch.abs(param[1]))).nonzero()
                # Calculate the original multi-dimensional index manually
        print("Layer with Superweight",highest)
        return highest[1]
    else:
        print("Model is not a mamba model. Returning...")
        return

def plotSuperWeight(model,newPlot=True):
    if newPlot:
        plt.figure()
    i = 1
    maxVal = []
    t = []
    for param in model.named_parameters():
        flat_tensor = param[1].flatten()
        flat_abs_tensor = flat_tensor.abs()
        flat_index = flat_abs_tensor.argmax()   
        max_abs_value = flat_tensor[flat_index].item()

        maxVal.append(max_abs_value)
        t.append(i)
        i += 1
    plt.plot(t,maxVal)
    plt.xlabel("Learnable Tensor")
    plt.ylabel("Maximum Numerical Value")
    plt.tight_layout()
    plt.grid()

def plotMinWeight(model,newPlot=True):
    if newPlot:
        plt.figure()
    i = 1
    minVal = []
    t = []
    for param in model.named_parameters():
        flat_tensor = param[1].flatten()
        flat_abs_tensor = flat_tensor.abs()
        flat_index = flat_abs_tensor.argmin()   
        min_abs_value = flat_tensor[flat_index].item()

        minVal.append(min_abs_value)
        t.append(i)
        i += 1
    plt.plot(t,minVal)
    plt.xlabel("Learnable Tensor")
    plt.ylabel("Minimum Numerical Value")
    plt.tight_layout()
    plt.grid()
    return


def printoutMaxLayerWeight(model):
    if isinstance(model,Mamba) or isinstance(model,MambaClassifier):
        if isinstance(model,MambaClassifier):
            model = model.mamba
        for param in model.named_parameters():
            print("Weight Tensor Name: ", param[0])  # Name of the parameter
            # Flatten the tensor to find the maximum index
            flat_tensor = param[1].flatten()
            flat_abs_tensor = flat_tensor.abs()
            flat_index = flat_abs_tensor.argmax()   
            max_abs_value = flat_tensor[flat_index]
            print("Maximum Weight Value: ", max_abs_value.item())
            # Calculate the original multi-dimensional index manually
            original_shape_index = (param[1]==torch.max(torch.abs(param[1]))).nonzero()
            print("Maximum Weight Index (original shape): ", original_shape_index)
            print()
    else:
        print("Model is not a mamba model. Returning...")
        return

def getMaxLayerWeight(model):
    if isinstance(model,Mamba) or isinstance(model,MambaClassifier):
        if isinstance(model,MambaClassifier):
            model = model.mamba
        highest = [None,0,None]
        for param in model.named_parameters():
            flat_tensor = param[1].flatten()
            flat_abs_tensor = flat_tensor.abs()
            flat_index = flat_abs_tensor.argmax()   
            max_abs_value = flat_tensor[flat_index]
            if abs(highest[1]) < abs(max_abs_value.item()):
                highest[0] = param[0]
                highest[1] = max_abs_value.item()
                highest[2] = (param[1]==torch.max(torch.abs(param[1]))).nonzero()
                # Calculate the original multi-dimensional index manually
        print("Layer with Superweight",highest)
        return highest[1]
    else:
        print("Model is not a mamba model. Returning...")
        return

def plotMaxLayerWeight(model,newPlot=True):
    if newPlot:
        plt.figure()
    i = 1
    maxVal = []
    t = []
    for param in model.named_parameters():
        flat_tensor = param[1].flatten()
        flat_abs_tensor = flat_tensor.abs()
        flat_index = flat_abs_tensor.argmax()   
        max_abs_value = flat_tensor[flat_index].item()

        maxVal.append(max_abs_value)
        t.append(i)
        i += 1
    plt.plot(t,maxVal)
    plt.xlabel("Learnable Tensor")
    plt.ylabel("Maximum Numerical Value")
    plt.tight_layout()
    plt.grid()

def plotMinLayerWeight(model,newPlot=True):
    if newPlot:
        plt.figure()
    i = 1
    minVal = []
    t = []
    for param in model.named_parameters():
        flat_tensor = param[1].flatten()
        flat_abs_tensor = flat_tensor.abs()
        flat_index = flat_abs_tensor.argmin()   
        min_abs_value = flat_tensor[flat_index].item()

        minVal.append(min_abs_value)
        t.append(i)
        i += 1
    plt.plot(t,minVal)
    plt.xlabel("Learnable Tensor")
    plt.ylabel("Minimum Numerical Value")
    plt.tight_layout()
    plt.grid()
    return

def custom_unravel_index(indices: torch.LongTensor, shape: tuple):
    """
    Mimics torch.unravel_index(indices, shape) without directly calling it.
    
    Args:
        indices (torch.LongTensor): A 1D tensor of flat indices.
        shape (tuple): The target shape to unravel into.
    
    Returns:
        torch.LongTensor: A 2D tensor of shape (len(shape), indices.size(0)),
                          containing the unraveled coordinates.
    """
    # Flatten the N-D indices to a 1D tensor
    flat_indices = indices.view(-1).long()
    
    # Perform the unraveling on the flattened tensor
    coords = []
    for dim in reversed(shape):
        coords.append(flat_indices % dim)
        flat_indices = flat_indices // dim
    
    # Reverse coordinates since we iterated from last dimension to first
    coords.reverse()
    
    # Now reshape each coordinate vector back to the original indices shape
    # so that coords[i] has the same shape as indices (except it gives the
    # i-th dimension's coordinate)
    coords = [c.reshape(indices.shape) for c in coords]
    
    # Finally, stack along dim=0 to get a shape of (len(shape), *indices.shape)
    return torch.stack(coords, dim=0)


def findMambaSuperActivation(model,test_in,input_or_output='output',layer_path="layers"):
    '''
    Custom Find Super Activation function for a mamba layer implemented in qutils
    '''
    if isinstance(model,Mamba) or isinstance(model,MambaClassifier):
        if isinstance(model,MambaClassifier):
            model = model.mamba

        mambaLayerAttributes = ["in_proj","conv1d","x_proj","dt_proj","out_proj"]
        module_name = "mixer"
        all_activations = {}
        all_hooks = []

        def get_activations(layer_index):
            def hook(model, inputs, outputs):
                hidden_states = inputs if input_or_output == "input" else outputs
                all_activations.setdefault(layer_index, {})[f"{module_name}_{input_or_output}_hidden_states"] = hidden_states
            return hook   

        def get_layers(model, layer_path):
            attributes = layer_path.split('.')
            layers = model
            for attr in attributes:
                layers = getattr(layers, attr)
            return layers


        attributes = module_name.split('.') if module_name != "layer" else []
        layers = get_layers(model, layer_path)

        for layer_index, layer in enumerate(layers):
            mixerAttr = layer
            valid = True
            for attr in attributes:
                if hasattr(mixerAttr, attr):
                    mixerAttr = getattr(mixerAttr, attr)
                    layer_index = 0
                    for innerAttr in mambaLayerAttributes:
                        current_attr = getattr(mixerAttr, innerAttr)
                        hook = current_attr.register_forward_hook(get_activations(layer_index))
                        all_hooks.append(hook)
                        layer_index += 1
                else:
                    valid = False
                    break
            


        model.eval()
        with torch.no_grad():
            model(test_in)
        for hook in all_hooks:
            hook.remove()
        top1_values_all_layers = []
        top1_indexes_all_layers = []
        for layer_index, outputs in all_activations.items():
            values = outputs[f'{module_name}_{input_or_output}_hidden_states']
            tensor = values[0] if isinstance(values, tuple) else values
            tensor = tensor.detach().cpu()
            tensor_abs = tensor.abs().float()

            # tensor2d = tensor.abs().reshape((tensor.shape[0],tensor.shape[2]))


            max_value, max_index = torch.max(tensor_abs, 0)
            max_index = custom_unravel_index(max_index, tensor.shape)
            top1_values_all_layers.append(tensor[max_index])
            top1_indexes_all_layers.append(max_index)


        return top1_values_all_layers, top1_indexes_all_layers
    else:
        print("This is not a mamba model, exiting....")
        return None, None
def plotSuperActivation(superWeightMagnitudes,superweightIndices,input_or_output="output",printOutValues=False,mambaLayerAttributes = ["in_proj","conv1d","x_proj","dt_proj","out_proj"]):
    magnitude = superWeightMagnitudes
    
    activationArea = input_or_output.capitalize()
    # Plot input activations
    # plt.figure(figsize=(5,3.5))
    plt.figure()
    for i in range(len(magnitude)):
        plt.plot(i, magnitude[i].norm(), color='blue', marker='o', markersize=5)
        if printOutValues is True:
            print(mambaLayerAttributes[i] + ": ",magnitude[i].norm())
    plt.xlabel('Layer')
    ticks = []
    for i in range(len(mambaLayerAttributes)):
        ticks.append(i)
    plt.xticks(ticks,mambaLayerAttributes)
    plt.ylabel('Max Activation Value')
    plt.grid()
    plt.title(f"{activationArea} Activation")

def zeroModelWeight(model,attributeToZero="x_proj",weightType="weight"):
    mambaLayerAttributes = ["in_proj","conv1d","x_proj","dt_proj","out_proj"]
    if isinstance(model,Mamba) or isinstance(model,MambaClassifier):
        if isinstance(model,MambaClassifier):
            model = model.mamba
        state_dict = model.state_dict()

        # state_dict["layers.0.mixer."+attributeToZero+"."+weightType] = torch.zeros_like(state_dict["layers.0.mixer."+attributeToZero+"."+weightType] )
        for param in model.parameters():
            param.data.zero_()
        
        # model.load_state_dict(state_dict)
        return
    else:
        return