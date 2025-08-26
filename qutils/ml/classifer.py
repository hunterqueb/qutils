import torch
from torch import nn
import numpy as np
from qutils.tictoc import timer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def apply_noise(data, pos_noise_std, vel_noise_std):
    mid = data.shape[1] // 2  # Split index
    pos_noise = np.random.normal(0, pos_noise_std, size=data[:, :mid].shape)
    vel_noise = np.random.normal(0, vel_noise_std, size=data[:, mid:].shape)
    noisy_data = data.copy()
    noisy_data[:, :mid] += pos_noise
    noisy_data[:, mid:] += vel_noise
    return noisy_data

def prepareThrustClassificationDatasets(yaml_config,data_config,train_ratio=0.7,val_ratio=0.15,test_ratio=0.15,pos_noise_std=1e-3,vel_noise_std=1e-3,batch_size=16,output_np=False):
    '''
    assumes 4 classes: chemical, electric, impBurn, noThrust
    assumes equal number of ICs for each class
    '''
    useOE = yaml_config['useOE']
    useNorm = yaml_config['useNorm']
    useNoise = yaml_config['useNoise']
    useEnergy = yaml_config['useEnergy']

    numMinProp = yaml_config['prop_time']

    train_set = yaml_config['orbit']
    systems = yaml_config['systems']

    test_set = yaml_config['test_dataset']
    test_systems = yaml_config['test_systems']

    dataLoc = data_config['classification'] + train_set +"/" + str(numMinProp) + "min-" + str(systems)
    dataLoc_test = data_config['classification'] + test_set +"/" + str(numMinProp) + "min-" + str(test_systems)

    print(f"Training data location: {dataLoc}")
    print(f"Test data location: {dataLoc_test}")

    # get npz files in folder and load them into script
    if useOE:
        a = np.load(f"{dataLoc}/OEArrayChemical.npz")
        statesArrayChemical = a['OEArrayChemical'][:,:,0:6]
        a = np.load(f"{dataLoc}/OEArrayElectric.npz")
        statesArrayElectric = a['OEArrayElectric'][:,:,0:6]
        a = np.load(f"{dataLoc}/OEArrayImpBurn.npz")
        statesArrayImpBurn = a['OEArrayImpBurn'][:,:,0:6]
        a = np.load(f"{dataLoc}/OEArrayNoThrust.npz")
        statesArrayNoThrust = a['OEArrayNoThrust'][:,:,0:6]
        n_ic = statesArrayChemical.shape[0]

        if useNorm:
            R = 6378.1363 # km
            statesArrayChemical[:,:,0] = statesArrayChemical[:,:,0] / R
            statesArrayElectric[:,:,0] = statesArrayElectric[:,:,0] / R
            statesArrayImpBurn[:,:,0] = statesArrayImpBurn[:,:,0] / R
            statesArrayNoThrust[:,:,0] = statesArrayNoThrust[:,:,0] / R
        if useNoise:
            statesArrayChemical = apply_noise(statesArrayChemical, pos_noise_std, vel_noise_std)
            statesArrayElectric = apply_noise(statesArrayElectric, pos_noise_std, vel_noise_std)
            statesArrayImpBurn = apply_noise(statesArrayImpBurn, pos_noise_std, vel_noise_std)
            statesArrayNoThrust = apply_noise(statesArrayNoThrust, pos_noise_std, vel_noise_std)

    else:
        a = np.load(f"{dataLoc}/statesArrayChemical.npz")
        statesArrayChemical = a['statesArrayChemical']

        a = np.load(f"{dataLoc}/statesArrayElectric.npz")
        statesArrayElectric = a['statesArrayElectric']

        a = np.load(f"{dataLoc}/statesArrayImpBurn.npz")
        statesArrayImpBurn = a['statesArrayImpBurn']

        a = np.load(f"{dataLoc}/statesArrayNoThrust.npz")
        statesArrayNoThrust = a['statesArrayNoThrust']
        n_ic = statesArrayChemical.shape[0]

        if useNoise:
            statesArrayChemical = apply_noise(statesArrayChemical, pos_noise_std, vel_noise_std)
            statesArrayElectric = apply_noise(statesArrayElectric, pos_noise_std, vel_noise_std)
            statesArrayImpBurn = apply_noise(statesArrayImpBurn, pos_noise_std, vel_noise_std)
            statesArrayNoThrust = apply_noise(statesArrayNoThrust, pos_noise_std, vel_noise_std)
        if useNorm:
            from qutils.orbital import dim2NonDim6
            for i in range(n_ic):
                statesArrayChemical[i,:,:] = dim2NonDim6(statesArrayChemical[i,:,:])
                statesArrayElectric[i,:,:] = dim2NonDim6(statesArrayElectric[i,:,:])
                statesArrayImpBurn[i,:,:] = dim2NonDim6(statesArrayImpBurn[i,:,:])
                statesArrayNoThrust[i,:,:] = dim2NonDim6(statesArrayNoThrust[i,:,:])
    del a

    noThrustLabel = 0
    chemicalLabel = 1
    electricLabel = 2
    impBurnLabel = 3
    n_ic = statesArrayChemical.shape[0]

    # Create labels for each dataset
    labelsChemical = np.full((n_ic,1),chemicalLabel)
    labelsElectric = np.full((n_ic,1),electricLabel)
    labelsImpBurn = np.full((n_ic,1),impBurnLabel)
    labelsNoThrust = np.full((n_ic,1),noThrustLabel)
    # Combine datasets and labels
    dataset = np.concatenate((statesArrayChemical, statesArrayElectric, statesArrayImpBurn, statesArrayNoThrust), axis=0)

    if useEnergy:
        from qutils.orbital import orbitalEnergy
        energyChemical = np.zeros((n_ic,statesArrayChemical.shape[1],1))
        energyElectric= np.zeros((n_ic,statesArrayChemical.shape[1],1))
        energyImpBurn= np.zeros((n_ic,statesArrayChemical.shape[1],1))
        energyNoThrust= np.zeros((n_ic,statesArrayChemical.shape[1],1))
        for i in range(n_ic):
            energyChemical[i,:,0] = orbitalEnergy(statesArrayChemical[i,:,:])
            energyElectric[i,:,0] = orbitalEnergy(statesArrayElectric[i,:,:])
            energyImpBurn[i,:,0] = orbitalEnergy(statesArrayImpBurn[i,:,:])
            energyNoThrust[i,:,0] = orbitalEnergy(statesArrayNoThrust[i,:,:])
        if useNorm:
            normingEnergy = energyNoThrust[0,0,0]
            energyChemical[:,:,0] = energyChemical[:,:,0] / normingEnergy
            energyElectric[:,:,0] = energyElectric[:,:,0] / normingEnergy
            energyImpBurn[:,:,0] = energyImpBurn[:,:,0] / normingEnergy
            energyNoThrust[:,:,0] = energyNoThrust[:,:,0] / normingEnergy

        dataset = np.concatenate((energyChemical, energyElectric, energyImpBurn, energyNoThrust), axis=0)
    if useEnergy and useOE:
        combinedChemical = np.concatenate((statesArrayChemical,energyChemical),axis=2) 
        combinedElectric = np.concatenate((statesArrayElectric,energyElectric),axis=2) 
        combinedImpBurn = np.concatenate((statesArrayImpBurn,energyImpBurn),axis=2) 
        combinedNoThrust = np.concatenate((statesArrayNoThrust,energyNoThrust),axis=2) 
        dataset = np.concatenate((combinedChemical, combinedElectric, combinedImpBurn, combinedNoThrust), axis=0)

    dataset_label = np.concatenate((labelsChemical, labelsElectric, labelsImpBurn, labelsNoThrust), axis=0)

    # shuffle the dataset completely
    groups = np.tile(np.arange(n_ic, dtype=np.int64), 4)   # len == 40000

    # Ratios (must satisfy train+val <= 1.0; test gets the remainder)
    # example:
    # train_ratio, val_ratio = 0.7, 0.15
    n_train_ic = int(np.floor(train_ratio * n_ic))
    n_val_ic   = int(np.floor(val_ratio   * n_ic))
    n_test_ic  = n_ic - n_train_ic - n_val_ic
    assert n_test_ic > 0, "Ratios leave no ICs for test; reduce train/val."

    # Shuffle ICs and partition
    perm_ic = np.random.permutation(n_ic)
    train_ic = perm_ic[:n_train_ic]
    val_ic   = perm_ic[n_train_ic:n_train_ic + n_val_ic]
    test_ic  = perm_ic[n_train_ic + n_val_ic:]

    # Masks select ALL thrust variants for each IC
    train_mask = np.isin(groups, train_ic)
    val_mask   = np.isin(groups, val_ic)
    test_mask  = np.isin(groups, test_ic)

    # Apply masks
    train_data,  train_label  = dataset[train_mask], dataset_label[train_mask]
    val_data,    val_label    = dataset[val_mask],   dataset_label[val_mask]

    if test_set != train_set or test_systems != systems:
        n_ic = statesArrayChemical.shape[0]             

        if useOE:
            a = np.load(f"{dataLoc}/OEArrayChemical.npz")
            statesArrayChemical = a['OEArrayChemical'][:,:,0:6]
            a = np.load(f"{dataLoc}/OEArrayElectric.npz")
            statesArrayElectric = a['OEArrayElectric'][:,:,0:6]
            a = np.load(f"{dataLoc}/OEArrayImpBurn.npz")
            statesArrayImpBurn = a['OEArrayImpBurn'][:,:,0:6]
            a = np.load(f"{dataLoc}/OEArrayNoThrust.npz")
            statesArrayNoThrust = a['OEArrayNoThrust'][:,:,0:6]

            if useNorm:
                R = 6378.1363 # km
                statesArrayChemical[:,:,0] = statesArrayChemical[:,:,0] / R
                statesArrayElectric[:,:,0] = statesArrayElectric[:,:,0] / R
                statesArrayImpBurn[:,:,0] = statesArrayImpBurn[:,:,0] / R
                statesArrayNoThrust[:,:,0] = statesArrayNoThrust[:,:,0] / R
            if useNoise:
                statesArrayChemical = apply_noise(statesArrayChemical, pos_noise_std, vel_noise_std)
                statesArrayElectric = apply_noise(statesArrayElectric, pos_noise_std, vel_noise_std)
                statesArrayImpBurn = apply_noise(statesArrayImpBurn, pos_noise_std, vel_noise_std)
                statesArrayNoThrust = apply_noise(statesArrayNoThrust, pos_noise_std, vel_noise_std)
            dataset_test = np.concatenate((statesArrayChemical, statesArrayElectric, statesArrayImpBurn, statesArrayNoThrust), axis=0)

        else:
            a = np.load(f"{dataLoc}/statesArrayChemical.npz")
            statesArrayChemical = a['statesArrayChemical']
            a = np.load(f"{dataLoc}/statesArrayElectric.npz")
            statesArrayElectric = a['statesArrayElectric']
            a = np.load(f"{dataLoc}/statesArrayImpBurn.npz")
            statesArrayImpBurn = a['statesArrayImpBurn']
            a = np.load(f"{dataLoc}/statesArrayNoThrust.npz")
            statesArrayNoThrust = a['statesArrayNoThrust']

            if useNoise:
                statesArrayChemical = apply_noise(statesArrayChemical, pos_noise_std, vel_noise_std)
                statesArrayElectric = apply_noise(statesArrayElectric, pos_noise_std, vel_noise_std)
                statesArrayImpBurn = apply_noise(statesArrayImpBurn, pos_noise_std, vel_noise_std)
                statesArrayNoThrust = apply_noise(statesArrayNoThrust, pos_noise_std, vel_noise_std)
            if useNorm:
                for i in range(n_ic):
                    statesArrayChemical[i,:,:] = dim2NonDim6(statesArrayChemical[i,:,:])
                    statesArrayElectric[i,:,:] = dim2NonDim6(statesArrayElectric[i,:,:])
                    statesArrayImpBurn[i,:,:] = dim2NonDim6(statesArrayImpBurn[i,:,:])
                    statesArrayNoThrust[i,:,:] = dim2NonDim6(statesArrayNoThrust[i,:,:])
            dataset_test = np.concatenate((statesArrayChemical, statesArrayElectric, statesArrayImpBurn, statesArrayNoThrust), axis=0)

        if useEnergy:
            energyChemical = np.zeros((n_ic,statesArrayChemical.shape[1],1))
            energyElectric= np.zeros((n_ic,statesArrayChemical.shape[1],1))
            energyImpBurn= np.zeros((n_ic,statesArrayChemical.shape[1],1))
            energyNoThrust= np.zeros((n_ic,statesArrayChemical.shape[1],1))
            for i in range(n_ic):
                energyChemical[i,:,0] = orbitalEnergy(statesArrayChemical[i,:,:])
                energyElectric[i,:,0] = orbitalEnergy(statesArrayElectric[i,:,:])
                energyImpBurn[i,:,0] = orbitalEnergy(statesArrayImpBurn[i,:,:])
                energyNoThrust[i,:,0] = orbitalEnergy(statesArrayNoThrust[i,:,:])
            if useNorm:
                normingEnergy = energyNoThrust[0,0,0]
                energyChemical[:,:,0] = energyChemical[:,:,0] / normingEnergy
                energyElectric[:,:,0] = energyElectric[:,:,0] / normingEnergy
                energyImpBurn[:,:,0] = energyImpBurn[:,:,0] / normingEnergy
                energyNoThrust[:,:,0] = energyNoThrust[:,:,0] / normingEnergy
            dataset_test = np.concatenate((energyChemical, energyElectric, energyImpBurn, energyNoThrust), axis=0)
        if useEnergy and useOE:
            combinedChemical = np.concatenate((statesArrayChemical,energyChemical),axis=2) 
            combinedElectric = np.concatenate((statesArrayElectric,energyElectric),axis=2) 
            combinedImpBurn = np.concatenate((statesArrayImpBurn,energyImpBurn),axis=2) 
            combinedNoThrust = np.concatenate((statesArrayNoThrust,energyNoThrust),axis=2) 
            dataset_test = np.concatenate((combinedChemical, combinedElectric, combinedImpBurn, combinedNoThrust), axis=0)

        labelsChemical = np.full((n_ic,1),chemicalLabel)
        labelsElectric = np.full((statesArrayElectric.shape[0],1),electricLabel)
        labelsImpBurn = np.full((statesArrayImpBurn.shape[0],1),impBurnLabel)
        labelsNoThrust = np.full((statesArrayNoThrust.shape[0],1),noThrustLabel)

        dataset_label_test = np.concatenate((labelsChemical, labelsElectric, labelsImpBurn, labelsNoThrust), axis=0)
        
        groups = np.tile(np.arange(n_ic, dtype=np.int64), 4)

        # Ratios (must satisfy train+val <= 1.0; test gets the remainder)
        # example:
        # train_ratio, val_ratio = 0.7, 0.15
        n_train_ic = int(np.floor(train_ratio * n_ic))
        n_val_ic   = int(np.floor(val_ratio   * n_ic))
        n_test_ic  = n_ic - n_train_ic - n_val_ic
        perm_ic = np.random.permutation(n_ic)
        train_ic = perm_ic[:n_train_ic]
        val_ic   = perm_ic[n_train_ic:n_train_ic + n_val_ic]
        test_ic  = perm_ic[n_train_ic + n_val_ic:]

        # Masks select ALL thrust variants for each IC
        train_mask = np.isin(groups, train_ic)
        val_mask   = np.isin(groups, val_ic)
        test_mask  = np.isin(groups, test_ic)

        test_data,   test_label   = dataset_test[test_mask],  dataset_label_test[test_mask]
    else:
        test_data,   test_label   = dataset[test_mask],  dataset_label[test_mask]

    train_loader, val_loader, test_loader = _prepareClassificationDataLoaders((train_data,train_label),(val_data,val_label),(test_data,test_label),batch_size)

    if output_np:
        return train_loader, val_loader, test_loader, train_data,train_label,val_data,val_label,test_data,test_label
    else:
        return train_loader, val_loader, test_loader

def _prepareClassificationDataLoaders(train_dataset,val_dataset,test_dataset,batch_size):
    train_data, train_label = train_dataset[0], train_dataset[1]
    val_data, val_label = val_dataset[0], val_dataset[1]
    test_data, test_label = test_dataset[0], test_dataset[1]

    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader

    train_dataset = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label).squeeze(1).long())
    val_dataset = TensorDataset(torch.from_numpy(val_data), torch.from_numpy(val_label).squeeze(1).long())
    test_dataset = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label).squeeze(1).long())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)

    return train_loader, val_loader, test_loader

def trainClassifier(model,optimizer,scheduler,dataloaders,criterion,num_epochs,device,schedulerPatience=5,printReport = False,classLabels=None):
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
        avg_val_loss, val_accuracy = validateMultiClassClassifier(model, val_loader, criterion, num_classes=logits.shape[1],device=device,printReport=printReport,classlabels=classLabels)

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
    def __init__(self, input_size, hidden_size, num_layers, num_classes,dropout=0.1):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

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
        last_hidden = self.dropout(last_hidden)

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