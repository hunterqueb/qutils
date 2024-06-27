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
