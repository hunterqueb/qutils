import torch
from torch import nn
import numpy as np
from qutils.tictoc import timer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import torch.nn.functional as F


def apply_noise(data, pos_noise_std, vel_noise_std):
    # assumes data shape is (num_samples, seq_length, 6)
    mid = data.shape[2] // 2  # Split index
    pos_noise = np.random.normal(0, pos_noise_std, size=data[:, :, :mid].shape)
    vel_noise = np.random.normal(0, vel_noise_std, size=data[:, :, mid:].shape)
    noisy_data = data.copy()
    noisy_data[:, :, :mid] += pos_noise
    noisy_data[:, :, mid:] += vel_noise
    return noisy_data

def apply_noise_OE(data, a_noise_std, e_noise_std, i_noise_std, raan_noise_std, argp_noise_std, nu_noise_std):
    # assumes data shape is (num_samples, seq_length, 6)
    noisy_data = data.copy()
    noisy_data[:,:,0] += np.random.normal(0, a_noise_std, size=data[:,:,0].shape)      # semi-major axis
    noisy_data[:,:,1] += np.random.normal(0, e_noise_std, size=data[:,:,1].shape)      # eccentricity
    noisy_data[:,:,2] += np.random.normal(0, i_noise_std, size=data[:,:,2].shape)      # inclination
    noisy_data[:,:,3] += np.random.normal(0, raan_noise_std, size=data[:,:,3].shape)   # right ascension of ascending node
    noisy_data[:,:,4] += np.random.normal(0, argp_noise_std, size=data[:,:,4].shape)   # argument of perigee
    noisy_data[:,:,5] += np.random.normal(0, nu_noise_std, size=data[:,:,5].shape)     # true anomaly
    return noisy_data

def prepareThrustClassificationDatasets(
    yaml_config,
    data_config,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    pos_noise_std=1e-3,
    vel_noise_std=1e-3,
    batch_size=16,
    output_np=False,
    # ---- PCA options ----
    pca_enabled=False,
    pca_mode="per_t",              # "per_t" (default) or "hankel"
    pca_n_components=0.95,         # float in (0,1] for var ratio, or int for fixed dims
    pca_whiten=False,
    pca_standardize=True,
    pca_random_state=0,
    hankel_L=5,                    # only used if pca_mode == "hankel"
    hankel_step=1,
    hankel_pool="none",            # "none" => keep sequence (T -> T-L*step+1); "mean" => (N, d)
    return_pca=False               # if True and pca_enabled, also return fitted PCA state
):
    '''
    assumes 4 classes: chemical, electric, impBurn, noThrust
    assumes equal number of ICs for each class

    PCA behavior:
      - per_t: fits PCA independently at each time index on train only; keeps T; reduces D->d.
      - hankel: builds lag-embedded windows of length L; fits one PCA on concatenated lag features.
                If hankel_pool == "none": returns (N, T-L*step+1, d).
                If hankel_pool == "mean": returns (N, d) (time-averaged), which becomes (N, 1, d) for loader.
    '''
    import numpy as np

    useOE    = yaml_config['useOE']
    useNorm  = yaml_config['useNorm']
    useNoise = yaml_config['useNoise']
    useEnergy= yaml_config['useEnergy']

    numMinProp = yaml_config['prop_time']

    train_set   = yaml_config['orbit']
    systems     = yaml_config['systems']
    test_set    = yaml_config['test_dataset']
    test_systems= yaml_config['test_systems']

    dataLoc      = data_config['classification'] + train_set + "/" + str(numMinProp) + "min-" + str(systems)
    dataLoc_test = data_config['classification'] + test_set  + "/" + str(numMinProp) + "min-" + str(test_systems)

    print(f"Training data location: {dataLoc}")
    print(f"Test data location: {dataLoc_test}")

    a = np.load(f"{dataLoc}/statesArrayChemical.npz")
    statesArrayChemical = a['statesArrayChemical']
    a = np.load(f"{dataLoc}/statesArrayElectric.npz")
    statesArrayElectric = a['statesArrayElectric']
    a = np.load(f"{dataLoc}/statesArrayImpBurn.npz")
    statesArrayImpBurn = a['statesArrayImpBurn']
    a = np.load(f"{dataLoc}/statesArrayNoThrust.npz")
    statesArrayNoThrust = a['statesArrayNoThrust']
    del a
    n_ic = statesArrayChemical.shape[0]

    # ----- optional noise -----
    if useNoise:
        statesArrayChemical = apply_noise(statesArrayChemical, pos_noise_std, vel_noise_std)
        statesArrayElectric = apply_noise(statesArrayElectric, pos_noise_std, vel_noise_std)
        statesArrayImpBurn  = apply_noise(statesArrayImpBurn,  pos_noise_std, vel_noise_std)
        statesArrayNoThrust = apply_noise(statesArrayNoThrust, pos_noise_std, vel_noise_std)

    # ----- optional OE conversion -----
    if useOE:
        from qutils.orbital import ECI2OE
        OEArrayChemical = np.zeros((systems,     numMinProp, 7))
        OEArrayElectric = np.zeros((systems,     numMinProp, 7))
        OEArrayImpBurn  = np.zeros((systems,     numMinProp, 7))
        OEArrayNoThrust = np.zeros((systems,     numMinProp, 7))
        for i in range(systems):
            for j in range(numMinProp):
                OEArrayChemical[i,j,:] = ECI2OE(statesArrayChemical[i,j,0:3], statesArrayChemical[i,j,3:6])
                OEArrayElectric[i,j,:] = ECI2OE(statesArrayElectric[i,j,0:3], statesArrayElectric[i,j,3:6])
                OEArrayImpBurn[i,j,:]  = ECI2OE(statesArrayImpBurn[i,j,0:3],   statesArrayImpBurn[i,j,3:6])
                OEArrayNoThrust[i,j,:] = ECI2OE(statesArrayNoThrust[i,j,0:3],  statesArrayNoThrust[i,j,3:6])
        if useNorm:
            R = 6378.1363
            OEArrayChemical[:,:,0] /= R
            OEArrayElectric[:,:,0] /= R
            OEArrayImpBurn[:,:,0]  /= R
            OEArrayNoThrust[:,:,0] /= R

        statesArrayChemical = OEArrayChemical[:,:,0:6]
        statesArrayElectric = OEArrayElectric[:,:,0:6]
        statesArrayImpBurn  = OEArrayImpBurn[:,:,0:6]
        statesArrayNoThrust = OEArrayNoThrust[:,:,0:6]

    if useNorm and not useOE:
        from qutils.orbital import dim2NonDim6
        for i in range(n_ic):
            statesArrayChemical[i,:,:] = dim2NonDim6(statesArrayChemical[i,:,:])
            statesArrayElectric[i,:,:] = dim2NonDim6(statesArrayElectric[i,:,:])
            statesArrayImpBurn[i,:,:]  = dim2NonDim6(statesArrayImpBurn[i,:,:])
            statesArrayNoThrust[i,:,:] = dim2NonDim6(statesArrayNoThrust[i,:,:])

    # ----- labels -----
    noThrustLabel = 0
    chemicalLabel = 1
    electricLabel = 2
    impBurnLabel  = 3

    labelsChemical = np.full((statesArrayChemical.shape[0],1), chemicalLabel)
    labelsElectric = np.full((statesArrayElectric.shape[0],1), electricLabel)
    labelsImpBurn  = np.full((statesArrayImpBurn.shape[0],1),  impBurnLabel)
    labelsNoThrust = np.full((statesArrayNoThrust.shape[0],1), noThrustLabel)

    # ----- base dataset (N, T, D) -----
    dataset = np.concatenate((statesArrayChemical, statesArrayElectric, statesArrayImpBurn, statesArrayNoThrust), axis=0)

    # ----- optional energy channel -----
    if useEnergy:
        from qutils.orbital import orbitalEnergy
        n_ic = statesArrayChemical.shape[0]
        T = statesArrayChemical.shape[1]
        energyChemical = np.zeros((n_ic, T, 1))
        energyElectric = np.zeros((n_ic, T, 1))
        energyImpBurn  = np.zeros((n_ic, T, 1))
        energyNoThrust = np.zeros((n_ic, T, 1))
        for i in range(n_ic):
            energyChemical[i,:,0] = orbitalEnergy(statesArrayChemical[i,:,:])
            energyElectric[i,:,0] = orbitalEnergy(statesArrayElectric[i,:,:])
            energyImpBurn[i,:,0]  = orbitalEnergy(statesArrayImpBurn[i,:,:])
            energyNoThrust[i,:,0] = orbitalEnergy(statesArrayNoThrust[i,:,:])
        if useNorm:
            normingEnergy = energyNoThrust[0,0,0]
            energyChemical[:,:,0] /= normingEnergy
            energyElectric[:,:,0] /= normingEnergy
            energyImpBurn[:,:,0]  /= normingEnergy
            energyNoThrust[:,:,0] /= normingEnergy

        if useOE:
            combinedChemical = np.concatenate((OEArrayChemical, energyChemical), axis=2)
            combinedElectric = np.concatenate((OEArrayElectric, energyElectric), axis=2)
            combinedImpBurn  = np.concatenate((OEArrayImpBurn,  energyImpBurn ), axis=2)
            combinedNoThrust = np.concatenate((OEArrayNoThrust, energyNoThrust), axis=2)
            dataset = np.concatenate((combinedChemical, combinedElectric, combinedImpBurn, combinedNoThrust), axis=0)
        else:
            dataset = np.concatenate((energyChemical, energyElectric, energyImpBurn, energyNoThrust), axis=0)

    dataset_label = np.concatenate((labelsChemical, labelsElectric, labelsImpBurn, labelsNoThrust), axis=0)

    # ----- split by IC groups (keeps all 4 thrust variants together) -----
    n_ic = statesArrayChemical.shape[0]  # ICs per class for the TRAIN SET
    groups = np.tile(np.arange(n_ic, dtype=np.int64), 4)   # length = 4*n_ic

    n_train_ic = int(np.floor(train_ratio * n_ic))
    n_val_ic   = int(np.floor(val_ratio   * n_ic))
    n_test_ic  = n_ic - n_train_ic - n_val_ic
    assert n_test_ic > 0, "Ratios leave no ICs for test; reduce train/val."

    perm_ic = np.random.permutation(n_ic)
    train_ic = perm_ic[:n_train_ic]
    val_ic   = perm_ic[n_train_ic:n_train_ic + n_val_ic]
    test_ic  = perm_ic[n_train_ic + n_val_ic:]

    train_mask = np.isin(groups, train_ic)
    val_mask   = np.isin(groups, val_ic)
    test_mask  = np.isin(groups, test_ic)

    train_data, train_label = dataset[train_mask], dataset_label[train_mask]
    val_data,   val_label   = dataset[val_mask],   dataset_label[val_mask]

    # ----- test set (OOD path or same dataset) -----
    if test_set != train_set or test_systems != systems:
        a = np.load(f"{dataLoc_test}/statesArrayChemical.npz"); statesArrayChemical = a['statesArrayChemical']
        a = np.load(f"{dataLoc_test}/statesArrayElectric.npz"); statesArrayElectric = a['statesArrayElectric']
        a = np.load(f"{dataLoc_test}/statesArrayImpBurn.npz");  statesArrayImpBurn  = a['statesArrayImpBurn']
        a = np.load(f"{dataLoc_test}/statesArrayNoThrust.npz");statesArrayNoThrust = a['statesArrayNoThrust']
        del a
        n_ic_test = statesArrayChemical.shape[0]

        if useNoise:
            statesArrayChemical = apply_noise(statesArrayChemical, pos_noise_std, vel_noise_std)
            statesArrayElectric = apply_noise(statesArrayElectric, pos_noise_std, vel_noise_std)
            statesArrayImpBurn  = apply_noise(statesArrayImpBurn,  pos_noise_std, vel_noise_std)
            statesArrayNoThrust = apply_noise(statesArrayNoThrust, pos_noise_std, vel_noise_std)
        if useOE:
            from qutils.orbital import ECI2OE
            OEArrayChemical = np.zeros((test_systems, numMinProp, 7))
            OEArrayElectric = np.zeros((test_systems, numMinProp, 7))
            OEArrayImpBurn  = np.zeros((test_systems, numMinProp, 7))
            OEArrayNoThrust = np.zeros((test_systems, numMinProp, 7))
            for i in range(test_systems):
                for j in range(numMinProp):
                    OEArrayChemical[i,j,:] = ECI2OE(statesArrayChemical[i,j,0:3], statesArrayChemical[i,j,3:6])
                    OEArrayElectric[i,j,:] = ECI2OE(statesArrayElectric[i,j,0:3], statesArrayElectric[i,j,3:6])
                    OEArrayImpBurn[i,j,:]  = ECI2OE(statesArrayImpBurn[i,j,0:3],   statesArrayImpBurn[i,j,3:6])
                    OEArrayNoThrust[i,j,:] = ECI2OE(statesArrayNoThrust[i,j,0:3],  statesArrayNoThrust[i,j,3:6])
            if useNorm:
                R = 6378.1363
                OEArrayChemical[:,:,0] /= R
                OEArrayElectric[:,:,0] /= R
                OEArrayImpBurn[:,:,0]  /= R
                OEArrayNoThrust[:,:,0] /= R
            statesArrayChemical = OEArrayChemical[:,:,0:6]
            statesArrayElectric = OEArrayElectric[:,:,0:6]
            statesArrayImpBurn  = OEArrayImpBurn[:,:,0:6]
            statesArrayNoThrust = OEArrayNoThrust[:,:,0:6]

        if useNorm and not useOE:
            from qutils.orbital import dim2NonDim6
            for i in range(n_ic_test):
                statesArrayChemical[i,:,:] = dim2NonDim6(statesArrayChemical[i,:,:])
                statesArrayElectric[i,:,:] = dim2NonDim6(statesArrayElectric[i,:,:])
                statesArrayImpBurn[i,:,:]  = dim2NonDim6(statesArrayImpBurn[i,:,:])
                statesArrayNoThrust[i,:,:] = dim2NonDim6(statesArrayNoThrust[i,:,:])

        dataset_test = np.concatenate((statesArrayChemical, statesArrayElectric, statesArrayImpBurn, statesArrayNoThrust), axis=0)

        if useEnergy:
            from qutils.orbital import orbitalEnergy
            Tt = statesArrayChemical.shape[1]
            energyChemical = np.zeros((n_ic_test, Tt, 1))
            energyElectric = np.zeros((n_ic_test, Tt, 1))
            energyImpBurn  = np.zeros((n_ic_test, Tt, 1))
            energyNoThrust = np.zeros((n_ic_test, Tt, 1))
            for i in range(n_ic_test):
                energyChemical[i,:,0] = orbitalEnergy(statesArrayChemical[i,:,:])
                energyElectric[i,:,0] = orbitalEnergy(statesArrayElectric[i,:,:])
                energyImpBurn[i,:,0]  = orbitalEnergy(statesArrayImpBurn[i,:,:])
                energyNoThrust[i,:,0] = orbitalEnergy(statesArrayNoThrust[i,:,:])
            if useNorm:
                normingEnergy = energyNoThrust[0,0,0]
                energyChemical[:,:,0] /= normingEnergy
                energyElectric[:,:,0] /= normingEnergy
                energyImpBurn[:,:,0]  /= normingEnergy
                energyNoThrust[:,:,0] /= normingEnergy

            if useOE:
                combinedChemical = np.concatenate((OEArrayChemical, energyChemical), axis=2)
                combinedElectric = np.concatenate((OEArrayElectric, energyElectric), axis=2)
                combinedImpBurn  = np.concatenate((OEArrayImpBurn,  energyImpBurn ), axis=2)
                combinedNoThrust = np.concatenate((OEArrayNoThrust, energyNoThrust), axis=2)
                dataset_test = np.concatenate((combinedChemical, combinedElectric, combinedImpBurn, combinedNoThrust), axis=0)
            else:
                dataset_test = np.concatenate((energyChemical, energyElectric, energyImpBurn, energyNoThrust), axis=0)

        labelsChemical = np.full((statesArrayChemical.shape[0],1), chemicalLabel)
        labelsElectric = np.full((statesArrayElectric.shape[0],1), electricLabel)
        labelsImpBurn  = np.full((statesArrayImpBurn.shape[0],1),  impBurnLabel)
        labelsNoThrust = np.full((statesArrayNoThrust.shape[0],1), noThrustLabel)
        dataset_label_test = np.concatenate((labelsChemical, labelsElectric, labelsImpBurn, labelsNoThrust), axis=0)

        groups_test = np.tile(np.arange(n_ic_test, dtype=np.int64), 4)
        n_train_ic_t = int(np.floor(train_ratio * n_ic_test))
        n_val_ic_t   = int(np.floor(val_ratio   * n_ic_test))
        perm_ic_t = np.random.permutation(n_ic_test)
        train_ic_t = perm_ic_t[:n_train_ic_t]
        val_ic_t   = perm_ic_t[n_train_ic_t:n_train_ic_t + n_val_ic_t]
        test_ic_t  = perm_ic_t[n_train_ic_t + n_val_ic_t:]

        train_mask_t = np.isin(groups_test, train_ic_t)
        val_mask_t   = np.isin(groups_test, val_ic_t)
        test_mask_t  = np.isin(groups_test, test_ic_t)
        test_data, test_label = dataset_test[test_mask_t], dataset_label_test[test_mask_t]
    else:
        test_data, test_label = dataset[test_mask], dataset_label[test_mask]

    # =========================
    # PCA (fit on train only)
    # =========================
    pca_state = None
    if pca_enabled:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        def _per_t_fit_transform(X_train, X_splits):
            # X_*: (N, T, D)
            N, T, D = X_train.shape
            scalers = [None] * T
            pcas    = [None] * T
            out_splits = []
            for X in X_splits:
                assert X.ndim == 3 and X.shape[1] == T, "All splits must have same T for per_t PCA."
            # fit
            Xtr = X_train
            Xouts = []
            # determine n_components per time (can be float ratio or int)
            for t in range(T):
                Xt = Xtr[:, t, :]  # (N, D)
                if pca_standardize:
                    scaler = StandardScaler().fit(Xt)
                    Xtn = scaler.transform(Xt)
                else:
                    scaler = None
                    Xtn = Xt
                pca = PCA(n_components=pca_n_components, whiten=pca_whiten, random_state=pca_random_state).fit(Xtn)
                scalers[t] = scaler
                pcas[t] = pca
            # transform all splits
            for X in X_splits:
                Nt = X.shape[0]
                dout_list = []
                for t in range(T):
                    Xt = X[:, t, :]
                    if scalers[t] is not None:
                        Xtn = scalers[t].transform(Xt)
                    else:
                        Xtn = Xt
                    Zt = pcas[t].transform(Xtn)  # (Nt, d_t)
                    dout_list.append(Zt[:, None, :])  # keep time axis
                Z = np.concatenate(dout_list, axis=1)  # (N, T, d_t) with possibly varying d_t across t if ratio used
                # If ratio used and d varies across t, force a common dim = min d_t:
                d_common = min(z.shape[-1] for z in dout_list)
                if any(z.shape[-1] != d_common for z in dout_list):
                    Z = np.concatenate([z[:, :, :d_common] for z in dout_list], axis=1)
                Xouts.append(Z.astype(np.float32))
            state = {"mode":"per_t","scalers":scalers,"pcas":pcas,"d_out":Xouts[0].shape[-1]}
            return Xouts, state

        def _hankel_embed(X, L, step=1):
            # X: (N, T, D) -> (N, T-L*step+1, L*D)
            N, T, D = X.shape
            M = T - L*step + 1
            if M <= 0:
                raise ValueError(f"hankel_L={L} and step={step} invalid for T={T}")
            out = np.empty((N, M, L*D), dtype=X.dtype)
            for i in range(M):
                seg = [X[:, i + k*step, :] for k in range(L)]
                out[:, i, :] = np.concatenate(seg, axis=1)
            return out

        def _hankel_fit_transform(X_train, X_splits):
            # lag embed
            Xe_train = _hankel_embed(X_train, hankel_L, hankel_step)  # (N, M, L*D)
            Ntr, M, LD = Xe_train.shape
            # fit scaler + PCA on pooled time from train
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            Xpool = Xe_train.reshape(-1, LD)  # (Ntr*M, LD)
            if pca_standardize:
                scaler = StandardScaler().fit(Xpool)
                Xpooln = scaler.transform(Xpool)
            else:
                scaler = None
                Xpooln = Xpool
            pca = PCA(n_components=pca_n_components, whiten=pca_whiten, random_state=pca_random_state).fit(Xpooln)
            # transform splits
            outs = []
            for X in X_splits:
                Xe = _hankel_embed(X, hankel_L, hankel_step)  # (N, M, LD)
                Xflat = Xe.reshape(-1, LD)
                if scaler is not None:
                    Xflat = scaler.transform(Xflat)
                Zflat = pca.transform(Xflat)  # (N*M, d)
                Z = Zflat.reshape(Xe.shape[0], Xe.shape[1], -1)  # (N, M, d)
                if hankel_pool == "mean":
                    Z = Z.mean(axis=1, keepdims=True)  # (N, 1, d)
                outs.append(Z.astype(np.float32))
            state = {"mode":"hankel","scaler":scaler,"pca":pca,"L":hankel_L,"step":hankel_step,"pool":hankel_pool,"d_out":outs[0].shape[-1]}
            return outs, state

        if pca_mode not in ("per_t", "hankel"):
            raise ValueError("pca_mode must be 'per_t' or 'hankel'")

        if pca_mode == "per_t":
            (train_data_pca, val_data_pca, test_data_pca), pca_state = _per_t_fit_transform(
                train_data, [train_data, val_data, test_data]
            )
        else:
            (train_data_pca, val_data_pca, test_data_pca), pca_state = _hankel_fit_transform(
                train_data, [train_data, val_data, test_data]
            )

        train_data, val_data, test_data = train_data_pca, val_data_pca, test_data_pca

    # ----- loaders -----
    train_loader, val_loader, test_loader = _prepareClassificationDataLoaders(
        (train_data, train_label),
        (val_data,   val_label),
        (test_data,  test_label),
        batch_size
    )
    if pca_enabled:
        print(f"Using {pca_state['d_out']} features after PCA ({pca_mode} mode).")
    if output_np and return_pca and pca_enabled:
        return train_loader, val_loader, test_loader, train_data, train_label, val_data, val_label, test_data, test_label, pca_state
    if output_np:
        return train_loader, val_loader, test_loader, train_data, train_label, val_data, val_label, test_data, test_label
    if return_pca and pca_enabled:
        return train_loader, val_loader, test_loader, pca_state
    return train_loader, val_loader, test_loader

def _prepareClassificationDataLoaders(train_dataset,val_dataset,test_dataset,batch_size):
    train_data, train_label = train_dataset[0], train_dataset[1]
    val_data, val_label = val_dataset[0], val_dataset[1]
    test_data, test_label = test_dataset[0], test_dataset[1]

    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader

    train_dataset = TensorDataset(torch.from_numpy(train_data).double(), torch.from_numpy(train_label).squeeze(1).long())
    val_dataset = TensorDataset(torch.from_numpy(val_data).double(), torch.from_numpy(val_label).squeeze(1).long())
    test_dataset = TensorDataset(torch.from_numpy(test_data).double(), torch.from_numpy(test_label).squeeze(1).long())
    
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
    def __init__(self, input_size, hidden_size, num_layers, num_classes,dropout=0.1,SA = False):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.SA = SA

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

        # Self-attention layer
        self.self_attention = SelfAttentionLayer(hidden_size)


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
        if self.SA:
            last_hidden, attention_weights = self.self_attention(last_hidden,mask=None)
        else:
            pass
        last_hidden = self.dropout(last_hidden)

        # Pass the last hidden state through a linear layer for classification
        logits = self.fc(last_hidden)  # [batch_size, num_classes]
        
        return logits
        
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