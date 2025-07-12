import torch
from torch import nn
import numpy as np
from qutils.tictoc import timer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd



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