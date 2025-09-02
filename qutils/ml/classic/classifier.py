import tempfile, pathlib
import numpy as np
from sklearn.metrics import log_loss, classification_report, confusion_matrix
import pandas as pd

# decsion tree based models

def printDTModelSize(model):

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "model.bin"   # any extension is fine
        model.booster_.save_model(str(path))     # binary dump by default
        size_bytes = path.stat().st_size
    print("\n==========================================================================================")
    print(f"Total parameters: NaN")
    print(f"Total memory (bytes): {size_bytes}")
    print(f"Total memory (MB): {size_bytes / (1024 ** 2)}")
    print("==========================================================================================")

def validate_lightgbm(model, val_loader, num_classes, classlabels=None, print_report=True):
    """Evaluate a trained LightGBM multiclass classifier on a PyTorch‑style DataLoader.

    * model          - fitted lightgbm.LGBMClassifier (objective='multiclass')
    * val_loader     - yields (seq_batch, label_batch); seq_batch can be torch.Tensor or np.ndarray
                    Shape per sample must match training: (7, L).  Flatten before predict.
    * num_classes    - integer (4 in your case)
    """
    # --------------------------------------------------------------------- #
    # Aggregate validation data                                             #
    # --------------------------------------------------------------------- #
    X_list, y_list = [], []
    for seq, lab in val_loader:
        # → ndarray, shape (batch, 7*L)
        xb = (seq if isinstance(seq, np.ndarray) else seq.cpu().numpy()).reshape(seq.shape[0], -1)
        yb = (lab if isinstance(lab, np.ndarray) else lab.cpu().numpy())
        X_list.append(xb)
        y_list.append(yb)

    X_val = np.concatenate(X_list, axis=0)
    y_true = np.concatenate(y_list, axis=0)

    # --------------------------------------------------------------------- #
    # Predict                                                               #
    # --------------------------------------------------------------------- #
    proba = model.predict_proba(X_val, num_iteration=model.best_iteration_)
    y_pred = proba.argmax(axis=1)

    # --------------------------------------------------------------------- #
    # Metrics                                                               #
    # --------------------------------------------------------------------- #
    val_loss = log_loss(y_true, proba, labels=np.arange(num_classes))
    accuracy = 100.0 * (y_pred == y_true).mean()

    # Per‑class accuracy
    class_tot = np.bincount(y_true, minlength=num_classes)
    class_corr = np.bincount(y_true[y_true == y_pred], minlength=num_classes)
    per_class_acc = 100.0 * class_corr / np.maximum(class_tot, 1)

    # --------------------------------------------------------------------- #
    # Reporting                                                             #
    # --------------------------------------------------------------------- #
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%\n")

    print("Per-Class Validation Accuracy:")
    for i in range(num_classes):
        label = classlabels[i] if classlabels else f"Class {i}"
        if class_tot[i]:
            print(f"  {label}: {per_class_acc[i]:.2f}% ({class_corr[i]}/{class_tot[i]})")
        else:
            print(f"  {label}: No samples")

    if print_report:
        print("\nClassification Report:")
        print(
            classification_report(
                y_true, y_pred,
                labels=list(range(num_classes)),
                target_names=(classlabels if classlabels else None),
                digits=4,
                zero_division=0,
            )
        )

        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        print("\nConfusion Matrix (rows = true, cols = predicted):")
        print(
            pd.DataFrame(
                cm,
                index=[f"T_{cls}" for cls in (classlabels if classlabels else range(num_classes))],
                columns=[f"P_{cls}" for cls in (classlabels if classlabels else range(num_classes))]
            )
        )

    return val_loss, accuracy


# K-means clustering for pseudo-labeling

def z_normalize(ts, eps=1e-8):
    # ts: [T] or [T,C]
    mean = ts.mean(axis=0, keepdims=True)
    std = ts.std(axis=0, keepdims=True)
    return (ts - mean) / (std + eps)

def train_data_z_normalize(train_data):
    """Z-normalize training data along the time axis."""
    return np.array([z_normalize(ts) for ts in train_data])

def print1_NNModelSize(model):
    import tempfile, pathlib
    import pickle

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)
        size_bytes = path.stat().st_size

    print("\n" + "=" * 90)
    print(f"Total parameters: NaN (non-parametric model)")
    print(f"Total memory (bytes): {size_bytes}")
    print(f"Total memory (MB): {size_bytes / (1024 ** 2):.4f}")
    print("=" * 90)

def validate_1NN(clf, val_loader, num_classes, classlabels=None):
    """Evaluate a 1-NN classifier (e.g., sktime KNeighborsTimeSeriesClassifier) on a PyTorch DataLoader."""
    X_val_list, y_val_list = [], []

    for seq, lab in val_loader:
        xb = seq.cpu().numpy()  # preserve time-series shape
        yb = lab.cpu().numpy()
        X_val_list.append(xb) #z-normalize each time series
        y_val_list.append(yb)

    # Merge batches
    X_val_np = np.concatenate(X_val_list, axis=0)
    y_true = np.concatenate(y_val_list)

    # Adapt shape for sktime: [N,C,T]
    # [N,T,C] → [N,C,T]
    X_val_np = np.transpose(X_val_np, (0, 2, 1))

    # Predict
    y_pred = clf.predict(X_val_np)

    # Accuracy
    correct = (y_pred == y_true).sum()
    total = len(y_true)
    accuracy = 100.0 * correct / total

    print(f"Validation Loss: NaN, Validation Accuracy: {accuracy:.2f}%\n")

    # Per-class accuracy
    class_corr = np.zeros(num_classes, dtype=int)
    class_tot = np.zeros(num_classes, dtype=int)
    for yt, yp in zip(y_true, y_pred):
        class_tot[yt] += 1
        if yt == yp:
            class_corr[yt] += 1
    per_class_acc = 100.0 * class_corr / np.maximum(class_tot, 1)

    print("Per-Class Validation Accuracy:")
    for i in range(num_classes):
        label = classlabels[i] if classlabels else f"Class {i}"
        if class_tot[i]:
            print(f"  {label}: {per_class_acc[i]:.2f}% ({class_corr[i]}/{class_tot[i]})")
        else:
            print(f"  {label}: No samples")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true, y_pred,
            labels=list(range(num_classes)),
            target_names=(classlabels if classlabels else None),
            digits=4,
            zero_division=0,
        )
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print(
        pd.DataFrame(
            cm,
            index=[f"T_{cls}" for cls in (classlabels if classlabels else range(num_classes))],
            columns=[f"P_{cls}" for cls in (classlabels if classlabels else range(num_classes))]
        )
    )
