# deps: pip install captum
from captum.attr import GradientShap, IntegratedGradients
import os, gc, numpy as np, pandas as pd,torch, matplotlib.pyplot as plt
import glob

@torch.no_grad()
def _stack_first_n_from_loader(loader, n, device):
    xs, ys, seen = [], [], 0
    for xb, yb in loader:
        b = xb.shape[0]
        take = min(n - seen, b)
        xs.append(xb[:take])
        ys.append(yb[:take])
        seen += take
        if seen >= n:
            break
    X = torch.cat(xs, dim=0).to(device=device, non_blocking=True).float()
    y = torch.cat(ys, dim=0).cpu().numpy()
    return X, y

def run_shap_analysis(
    model: torch.nn.Module,
    train_loader, eval_loader,
    device: str,
    classlabels: list,
    feature_names: list[str] | None = None,
    out_dir: str = "artifacts/shap",
    method: str = "gradshap",          # "gradshap" or "ig"
    baseline_nsamples: int = 32,
    n_eval: int | None = 512,           # None = use all from eval_loader
    ig_steps: int = 32,
    gs_samples: int = 8,
    internal_batch_size: int = 32,      # only used by IG
    use_cpu: bool = False,
    group_by: str = "true",
    seed: int = 0,
):
    os.makedirs(out_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize(); torch.cuda.empty_cache()

    model.eval()
    if use_cpu:
        device = "cpu"
    model = model.to(device)
    model_dtype = next(model.parameters()).dtype  # enforce everywhere

    @torch.no_grad()
    def _stack_first_n(loader, n, dev, dtype):
        xs, ys, seen = [], [], 0
        for xb, yb in loader:
            b = xb.shape[0]
            take = b if (n is None) else min(n - seen, b)
            xs.append(xb[:take])
            ys.append(yb[:take])
            seen += take
            if (n is not None) and (seen >= n):
                break
        X = torch.cat(xs, dim=0).to(device=dev, dtype=dtype, non_blocking=True)
        y = torch.cat(ys, dim=0).to("cpu").numpy()
        return X, y

    # Baselines from TRAIN distribution (no OOD), cast to model dtype
    baselines, _ = _stack_first_n(train_loader, baseline_nsamples, device, model_dtype)

    # Eval slice (None -> all), cast to model dtype
    Xev, yev = _stack_first_n(eval_loader, n_eval, device, model_dtype)

    # Forward must return logits [B, C] requiring grad
    def forward(x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        if not torch.is_tensor(out):
            out = out[0]
        return out

    # Sanity: ensure logits require grad
    k = min(4, Xev.shape[0])
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
        xb_chk = Xev[:k].clone().detach().to(dtype=model_dtype).requires_grad_(True)
        logits_chk = forward(xb_chk)                    # [k, C]
        if logits_chk.ndim != 2:
            raise RuntimeError(f"forward() must return [B, C], got {tuple(logits_chk.shape)}")
        target_chk = torch.from_numpy(yev[:k]).to(xb_chk.device).long()
        score = logits_chk.gather(1, target_chk.view(-1, 1)).sum()
        assert score.requires_grad, "Selected logit does not require grad."
        score.backward(retain_graph=True)
        assert xb_chk.grad is not None, "Input grad is None."
    del xb_chk
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _attrib_batch(xb: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
        xb = xb.clone().detach().to(dtype=model_dtype).requires_grad_(True)
        with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):
            if method.lower() == "ig":
                expl = IntegratedGradients(forward)
                attr_b = expl.attribute(
                    xb,
                    baselines=baselines.mean(dim=0, keepdim=True),
                    n_steps=ig_steps,
                    internal_batch_size=internal_batch_size,
                    target=target_idx,
                )
            else:
                expl = GradientShap(forward)
                attr_b = expl.attribute(
                    xb,
                    baselines=baselines,
                    n_samples=gs_samples,
                    stdevs=0.0,
                    target=target_idx,
                )
        return attr_b

    # Streamed aggregation
    N_seen = 0
    T = None; D = None
    global_feat_sum = None     # (D,)
    global_time_sum = None     # (T,)
    feature_time_sum = None    # (T, D)
    per_class_sum, per_class_count = {}, {}

    bs = max(1, internal_batch_size)
    for i in range(0, Xev.shape[0], bs):
        xb = Xev[i:i+bs]
        yb_true_np = yev[i:i+bs]

        if group_by == "true":
            target_idx = torch.from_numpy(yb_true_np).to(xb.device).long()
            key_np = yb_true_np
        else:
            with torch.no_grad():
                pred = torch.argmax(forward(xb), dim=1)
            target_idx = pred
            key_np = pred.detach().cpu().numpy()

        attr_b = _attrib_batch(xb, target_idx)         # (b,T,D) or (b,D)
        if attr_b.ndim == 2:
            attr_b = attr_b[:, None, :]
        b, Tb, Db = attr_b.shape
        if T is None:
            T, D = Tb, Db
            global_feat_sum = torch.zeros(D, device="cpu")
            global_time_sum = torch.zeros(T, device="cpu")
            feature_time_sum = torch.zeros(T, D, device="cpu")

        abs_b = attr_b.abs().detach().cpu()            # (b,T,D)

        # Global aggregates
        global_feat_sum += abs_b.mean(dim=1).sum(dim=0)  # (D,)
        global_time_sum += abs_b.mean(dim=2).sum(dim=0)  # (T,)
        feature_time_sum += abs_b.sum(dim=0)             # (T,D)

        # Per-class (only classes present this batch)
        present = np.unique(key_np).astype(int)
        for c in present:
            mask = torch.from_numpy(key_np == c).bool()
            Ab = abs_b[mask]
            if c not in per_class_sum:
                per_class_sum[c] = torch.zeros(T, D, device="cpu")
                per_class_count[c] = 0
            per_class_sum[c] += Ab.sum(dim=0)
            per_class_count[c] += Ab.shape[0]

        N_seen += b
        del attr_b, abs_b, xb
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    if feature_names is None:
        feature_names = [f"f{j}" for j in range(D)]

    # Finalize means
    global_feat = (global_feat_sum / max(1, N_seen)).numpy()
    global_time = (global_time_sum / max(1, N_seen)).numpy()
    feature_time_mean = (feature_time_sum / max(1, N_seen)).numpy()  # (T, D)
    per_class = {c: (S / max(1, per_class_count[c])).numpy()
                 for c, S in per_class_sum.items()}

    # Save artifacts
    pd.DataFrame({"feature": feature_names, "mean_abs_attr": global_feat}) \
      .sort_values("mean_abs_attr", ascending=False) \
      .to_csv(os.path.join(out_dir, "global_feature_importance.csv"), index=False)

    pd.DataFrame({"t": np.arange(T), "mean_abs_attr": global_time}) \
      .to_csv(os.path.join(out_dir, "global_time_importance.csv"), index=False)

    for c in sorted(per_class.keys()):
        M = per_class[c]
        df = pd.DataFrame(M, columns=feature_names)
        df.insert(0, "t", np.arange(M.shape[0]))
        name = (classlabels[c] if 0 <= c < len(classlabels) else str(c))
        df.to_csv(os.path.join(out_dir, f"per_class_t_by_feature_c{c}_{name}.csv"), index=False)

    df_ft = pd.DataFrame(feature_time_mean, columns=feature_names)
    df_ft.insert(0, "t", np.arange(feature_time_mean.shape[0]))
    df_ft.to_csv(os.path.join(out_dir, "feature_time_importance.csv"), index=False)

    # Report
    order = np.argsort(global_feat)[::-1][:10]
    for idx in order:
        print(f"[SHAP] {feature_names[idx]:>12s}: {global_feat[idx]:.6g}")
    print(f"[SHAP] N_eval={N_seen}, T={T}, D={D} -> {out_dir}")

    return {"N": int(N_seen), "T": int(T), "D": int(D), "out_dir": out_dir}

def plot_global_feature_importance(shap_dir: str, topk: int = 20, save: bool = True):
    p = os.path.join(shap_dir, "global_feature_importance.csv")
    df = pd.read_csv(p).sort_values("mean_abs_attr", ascending=False).head(topk)
    fig, ax = plt.subplots(figsize=(max(6, 0.5*len(df)), max(4, 0.35*len(df))))
    ax.barh(df["feature"], df["mean_abs_attr"])
    ax.invert_yaxis()
    ax.set_xlabel("mean |attribution|")
    ax.set_title(f"Global Feature Importance (top {len(df)})")
    fig.tight_layout()
    if save:
        out = os.path.join(shap_dir, f"plot_global_feature_importance_top{len(df)}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
    return fig, ax

def plot_global_time_importance(shap_dir: str, save: bool = True):
    p = os.path.join(shap_dir, "global_time_importance.csv")
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["t"].to_numpy(), df["mean_abs_attr"].to_numpy(), marker="o", linewidth=1)
    ax.set_xlabel("t index")
    ax.set_ylabel("mean |attribution|")
    ax.set_title("Global Time Importance")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save:
        out = os.path.join(shap_dir, "plot_global_time_importance.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
    return fig, ax

def _load_per_class_csvs(shap_dir: str):
    return sorted(glob.glob(os.path.join(shap_dir, "per_class_t_by_feature_c*.csv")))

def plot_per_class_heatmap_file(
    csv_path: str,
    topk_features: int | None = None,   # e.g., 20 to focus on most influential features
    vmax: float | None = None,          # set to a constant for comparable color scales
    save: bool = True
):
    df = pd.read_csv(csv_path)
    features = list(df.columns[1:])
    A = df[features].to_numpy()   # shape (T, D); already mean |attr|
    # Optionally keep only top-k features by total importance
    if topk_features is not None and topk_features < A.shape[1]:
        scores = A.sum(axis=0)
        idx = np.argsort(scores)[::-1][:topk_features]
        A = A[:, idx]
        features = [features[i] for i in idx]

    fig_w = max(8, 0.5*len(features))
    fig_h = max(4, 0.4*A.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(A, aspect="auto", vmin=0.0, vmax=vmax)
    ax.set_xlabel("feature")
    ax.set_ylabel("t index")
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_yticks(np.arange(A.shape[0]))
    ax.set_yticklabels(df["t"].to_numpy())
    title = os.path.splitext(os.path.basename(csv_path))[0]
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean |attribution|")
    fig.tight_layout()
    if save:
        out = os.path.join(os.path.dirname(csv_path), f"{title}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
    return fig, ax

def plot_all_per_class_heatmaps(
    shap_dir: str,
    topk_features: int | None = None,
    lock_vmax: bool = False
):
    paths = _load_per_class_csvs(shap_dir)
    if lock_vmax:
        # share a common vmax across classes for fair visual comparison
        vmax = 0.0
        for p in paths:
            df = pd.read_csv(p)
            A = df[df.columns[1:]].to_numpy()
            if topk_features is not None and topk_features < A.shape[1]:
                scores = A.sum(axis=0)
                idx = np.argsort(scores)[::-1][:topk_features]
                A = A[:, idx]
            vmax = max(vmax, float(A.max()))
    else:
        vmax = None
    for p in paths:
        plot_per_class_heatmap_file(p, topk_features=topk_features, vmax=vmax, save=True)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_feature_time_importance_lines(
    shap_dir: str,
    topk: int | None = None,      # None = all features; else keep top-k by mean importance
    normalize: str | None = None, # None | "max" | "sum"
    save: bool = True
):
    p = os.path.join(shap_dir, "feature_time_importance.csv")
    df = pd.read_csv(p)
    t = df["t"].to_numpy()
    F = df.drop(columns=["t"])
    # rank features by overall mean importance
    means = F.mean(axis=0).to_numpy()
    order = np.argsort(means)[::-1]
    cols = F.columns[order]
    if topk is not None:
        cols = cols[:topk]
    F = F[cols]

    # optional normalization per feature
    if normalize == "max":
        F = F / F.max(axis=0).replace(0, np.nan)
    elif normalize == "sum":
        F = F / F.sum(axis=0).replace(0, np.nan)

    # plot
    fig_w = max(8, 0.5 * len(cols))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    for c in cols:
        ax.plot(t, F[c].to_numpy(), label=c, linewidth=1.5)
    ax.set_xlabel("t index")
    ax.set_ylabel("mean |attribution|")
    title = "Feature Importance Over Time"
    if topk: title += f" (top {topk})"
    if normalize: title += f" — normalized by {normalize}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(ncols=min(4, len(cols)), fontsize=8, frameon=False)
    fig.tight_layout()
    if save:
        out = os.path.join(shap_dir, f"plot_feature_time_importance_lines{'_top'+str(topk) if topk else ''}{'_'+normalize if normalize else ''}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
    return fig, ax

def plot_feature_time_importance_heatmap(
    shap_dir: str,
    topk: int | None = None,      # None = all; else top-k by mean importance
    save: bool = True
):
    p = os.path.join(shap_dir, "feature_time_importance.csv")
    df = pd.read_csv(p)
    t = df["t"].to_numpy()
    F = df.drop(columns=["t"])
    means = F.mean(axis=0).to_numpy()
    order = np.argsort(means)[::-1]
    cols = F.columns[order]
    if topk is not None:
        cols = cols[:topk]
    F = F[cols]
    A = F.to_numpy().T  # (D_sel, T)

    fig_w = max(8, 0.5 * A.shape[1])
    fig_h = max(4, 0.35 * A.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(A, aspect="auto")
    ax.set_xlabel("t index")
    ax.set_ylabel("feature")
    ax.set_xticks(np.arange(len(t)))
    ax.set_xticklabels(t)
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols)
    ax.set_title(f"Feature Importance Heatmap (time × feature){f' — top {topk}' if topk else ''}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean |attribution|")
    fig.tight_layout()
    if save:
        out = os.path.join(shap_dir, f"plot_feature_time_importance_heatmap{'_top'+str(topk) if topk else ''}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print("saved:", out)
    return fig, ax
