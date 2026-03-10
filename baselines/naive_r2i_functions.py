import numpy as np
import torch
from datasets import load_adni_data
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset


def load_dataset_numpy(dataset: str):
    if dataset == "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_set = datasets.MNIST('../data/', train=True, download=True, transform=transform)
        test_set  = datasets.MNIST('../data/', train=False, download=True, transform=transform)

    elif dataset == "cifar10":
        MEAN = (0.4914, 0.4822, 0.4465)
        STD  = (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(MEAN, STD)
        ])

        train_set = datasets.CIFAR10('../data/cifar10/', train=True, download=True, transform=transform)
        test_set  = datasets.CIFAR10('../data/cifar10/', train=False, download=True, transform=transform)

    elif dataset == "cifar100":
        MEAN = (0.5071, 0.4867, 0.4408)
        STD  = (0.2675, 0.2565, 0.2761)

        transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(MEAN, STD)
        ])

        train_set = datasets.CIFAR100('../data/cifar100/', train=True, download=True, transform=transform)
        test_set  = datasets.CIFAR100('../data/cifar100/', train=False, download=True, transform=transform)
        
    elif dataset == "adni":
        (X_train, y_train), (X_test, y_test) = load_adni_data(
            rank_worldsize= '1, 1',
            adni_num= 1,
            data_dir= 'path/to/ADNI/data',
            img_dir= 'ADNI1_ALL_T1',
            csv_path= 'path/to/ADNI/csv',
            csv_filename= 'ADNI_ready.csv',
        )
        train_set = TensorDataset(X_train.unsqueeze(1), y_train)
        test_set = TensorDataset(X_test.unsqueeze(1), y_test)
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Stack diretto in tensor unico
    X_train = torch.stack([train_set[i][0] for i in range(len(train_set))])
    y_train = torch.tensor([train_set[i][1] for i in range(len(train_set))])

    X_test = torch.stack([test_set[i][0] for i in range(len(test_set))])
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))])

    # Converti in numpy SENZA cambiare ordine (resta N,C,H,W)
    X_train = X_train.numpy().astype(np.float32)
    y_train = y_train.numpy().astype(np.int64)
    X_test = X_test.numpy().astype(np.float32)
    y_test = y_test.numpy().astype(np.int64)
    
    adni_strategy = "tri_axis_middle"
    adni_axis = -1
    
    if dataset == "adni" and X_train.ndim == 5:
        print(f"[ADNI] Reducing 3D volumes with strategy='{adni_strategy}', axis={adni_axis}")
        print(f"  Before: X_train {X_train.shape}, X_test {X_test.shape}")
        X_train = reduce_adni_3d_to_2d(X_train, strategy=adni_strategy, axis=adni_axis)
        X_test  = reduce_adni_3d_to_2d(X_test,  strategy=adni_strategy, axis=adni_axis)
        print(f"  After:  X_train {X_train.shape}, X_test {X_test.shape}")

    return (X_train, y_train, X_test, y_test)


def data_transform(X: np.ndarray, channel_handling: str, channel_method: str) -> np.ndarray:
    """Prepare images for sktime ROCKET-like transformers.

    Input:
        X: (N, C, H, W)
        channel_handling: separate | concatenate | concatenate_channelwise
        channel_method: zigzag | spiral | hilbert | row_wise | column_wise | none

    Output:
        (N, C, H*W) if separate
        (N, 1, H*W*C) if concatenate / concatenate_channelwise
    """
    if X.ndim != 4:
        raise ValueError(f"Expected X with shape (N,C,H,W), got {X.shape}")

    channel_handling = (channel_handling or "separate").lower()
    channel_method = (channel_method or "none").lower()

    N, C, H, W = X.shape

    if C == 1:
        channel_handling = "separate"

    # Precompute spatial order once (flat indices into H*W)
    flat_idxs = _spatial_flat_indices(H, W, channel_method)

    # ---- Separate: (N, C, L) ----
    if channel_handling == "separate":
        # Base flatten row-wise then reorder spatially via flat_idxs
        X_flat = X.reshape(N, C, H * W)                 # row-wise
        if flat_idxs is not None:
            X_flat = X_flat[..., flat_idxs]            # vectorized reorder
        return X_flat

    # ---- Concatenate (channel-major): (N, 1, C*L) ----
    if channel_handling == "concatenate":
        X_sep = data_transform(X, "separate", channel_method)  # (N,C,L), already reordered if needed
        return X_sep.reshape(N, 1, C * H * W)

    # ---- Concatenate channelwise (pixel-major): (N, 1, L*C) ----
    if channel_handling == "concatenate_channelwise":
        # Work in (N, H, W, C) to make channel-last interleaving easy
        X_nhwc = X.transpose(0, 2, 3, 1)               # (N,H,W,C)
        X_sp = X_nhwc.reshape(N, H * W, C)             # row-wise spatial
        if flat_idxs is not None:
            X_sp = X_sp[:, flat_idxs, :]               # reorder spatial path, vectorized
        return X_sp.reshape(N, 1, H * W * C)            # interleave channels per pixel

    raise ValueError(f"Unsupported channel_handling: {channel_handling}")


def _spatial_flat_indices(H: int, W: int, method: str):
    """Return 1D flat indices of length H*W into row-wise flattened (H*W).
    If method is row-wise/none -> None (no reorder needed).
    """
    method = (method or "none").lower()

    if method in ("none", "row_wise"):
        return None

    if method == "column_wise":
        # Column-major order: indices for (i,j) in col-major, mapped into row-major flatten
        # order: for j in [0..W-1], for i in [0..H-1] -> i*W + j
        idxs = np.fromiter((i * W + j for j in range(W) for i in range(H)), dtype=np.int64, count=H * W)
        return idxs

    if method == "zigzag":
        idxs = np.empty((H * W,), dtype=np.int64)
        k = 0
        for s in range(H + W - 1):
            if s % 2 == 0:
                i = min(s, H - 1)
                j = s - i
                while i >= 0 and j < W:
                    idxs[k] = i * W + j
                    k += 1
                    i -= 1
                    j += 1
            else:
                j = min(s, W - 1)
                i = s - j
                while j >= 0 and i < H:
                    idxs[k] = i * W + j
                    k += 1
                    i += 1
                    j -= 1
        return idxs

    if method == "spiral":
        idxs = np.empty((H * W,), dtype=np.int64)
        k = 0
        top, bottom, left, right = 0, H - 1, 0, W - 1
        while top <= bottom and left <= right:
            for j in range(left, right + 1):
                idxs[k] = top * W + j
                k += 1
            top += 1

            for i in range(top, bottom + 1):
                idxs[k] = i * W + right
                k += 1
            right -= 1

            if top <= bottom:
                for j in range(right, left - 1, -1):
                    idxs[k] = bottom * W + j
                    k += 1
                bottom -= 1

            if left <= right:
                for i in range(bottom, top - 1, -1):
                    idxs[k] = i * W + left
                    k += 1
                left += 1

        # k should be exactly H*W
        return idxs[: H * W]

    if method == "hilbert":
        """
        if H != W or (H & (H - 1)) != 0:
            raise ValueError("Hilbert flatten requires square images with side power of 2")
        try:
            from hilbertcurve.hilbertcurve import HilbertCurve
        except Exception as e:
            raise ImportError(
                "hilbertcurve is required for channel_method='hilbert'. Install it (pip install hilbertcurve)."
            ) from e

        p = int(np.log2(H))
        hc = HilbertCurve(p, 2)

        coords = np.empty((H * W, 2), dtype=np.int64)  # (d, flat_index)
        k = 0
        for i in range(H):
            for j in range(W):
                d = hc.distance_from_point([i, j])
                coords[k, 0] = d
                coords[k, 1] = i * W + j
                k += 1
        coords = coords[np.argsort(coords[:, 0])]
        return coords[:, 1] """

    raise ValueError(f"Unsupported channel_method: {method}")



def reduce_adni_3d_to_2d(
    X: np.ndarray,
    strategy: str = "mean_max_std",
    axis: int = -1,
    n_slices: int = 3,
    resize: tuple[int, int] | None = None,
) -> np.ndarray:
    """Reduce 3D ADNI brain volumes to 2D images.

    Parameters
    ----------
    X : np.ndarray
        Shape (N, 1, D1, D2, D3), e.g. (324, 1, 91, 109, 91).
    strategy : str
        How to collapse the 3D volume along `axis`:
        - "middle"             : single middle slice → (N, 1, H, W)
        - "mean"               : mean projection → (N, 1, H, W)
        - "max"                : max intensity projection → (N, 1, H, W)
        - "multi_slice"        : n_slices equally-spaced slices as channels → (N, n_slices, H, W)
        - "tri_axis_middle"    : middle slice from each of 3 axes → (N, 3, H', W')
                                 (resized to common shape)
        - "mean_max_std"       : [mean, max, std] along axis → (N, 3, H, W) RECOMMENDED
        - "weighted_mean"      : variance-weighted mean → (N, 1, H, W)
    axis : int
        Spatial axis along which to project (2, 3, or 4 in NCHWD order).
        Default -1 = last spatial axis (axial in typical ADNI orientation).
    n_slices : int
        Number of slices for "multi_slice" strategy.
    resize : tuple or None
        If given, resize the output spatial dims to (H, W) using bilinear interpolation.

    Returns
    -------
    np.ndarray with shape (N, C, H, W) — ready for standard 2D pipelines.
    """
    if X.ndim != 5:
        raise ValueError(f"Expected 5D input (N,1,D1,D2,D3), got shape {X.shape}")

    N = X.shape[0]
    # Squeeze the channel dim for easier manipulation: (N, D1, D2, D3)
    V = X[:, 0]

    # Normalize axis to be relative to the volume dims (0, 1, 2 within D1,D2,D3)
    vol_axis = axis if axis >= 0 else V.ndim - 1 + (axis + 1)  # map -1→last spatial
    if vol_axis < 1:
        vol_axis = V.ndim - 1  # default to last
    spatial_axis = vol_axis  # axis index within (N, D1, D2, D3)

    strategy = strategy.lower()

    if strategy == "middle":
        mid = V.shape[spatial_axis] // 2
        slc = _take_slice(V, spatial_axis, mid)  # (N, H, W)
        out = slc[:, np.newaxis]  # (N, 1, H, W)

    elif strategy == "mean":
        out = V.mean(axis=spatial_axis)[:, np.newaxis]

    elif strategy == "max":
        out = V.max(axis=spatial_axis)[:, np.newaxis]

    elif strategy == "weighted_mean":
        # Weight each slice by its variance (more informative slices contribute more)
        # Variance per slice: compute var over spatial dims for each slice
        var_per_slice = _variance_per_slice(V, spatial_axis)  # (N, num_slices)
        weights = var_per_slice / (var_per_slice.sum(axis=-1, keepdims=True) + 1e-8)
        # Weighted sum along the projection axis
        out = _weighted_sum_along_axis(V, spatial_axis, weights)[:, np.newaxis]

    elif strategy == "multi_slice":
        num_total = V.shape[spatial_axis]
        indices = np.linspace(0, num_total - 1, n_slices, dtype=int)
        channels = [_take_slice(V, spatial_axis, idx) for idx in indices]
        out = np.stack(channels, axis=1)  # (N, n_slices, H, W)

    elif strategy == "tri_axis_middle":
        # Middle slice from each spatial axis → 3 channels
        # These have different shapes, so we resize to a common shape
        slices = []
        for ax in [1, 2, 3]:  # D1, D2, D3 axes in (N, D1, D2, D3)
            mid = V.shape[ax] // 2
            s = _take_slice(V, ax, mid)  # (N, H_i, W_i)
            slices.append(s)

        # Determine common shape (max of all dims or user-specified resize)
        if resize is None:
            max_h = max(s.shape[1] for s in slices)
            max_w = max(s.shape[2] for s in slices)
            target = (max_h, max_w)
        else:
            target = resize

        resized = [_resize_2d_batch(s, target) for s in slices]
        out = np.stack(resized, axis=1)  # (N, 3, H, W)

    elif strategy == "mean_max_std":
        ch_mean = V.mean(axis=spatial_axis)
        ch_max = V.max(axis=spatial_axis)
        ch_std = V.std(axis=spatial_axis)
        out = np.stack([ch_mean, ch_max, ch_std], axis=1)  # (N, 3, H, W)

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Choose from: "
            "middle, mean, max, weighted_mean, multi_slice, tri_axis_middle, mean_max_std"
        )

    # Optional resize
    if resize is not None and strategy != "tri_axis_middle":
        C = out.shape[1]
        resized_channels = []
        for c in range(C):
            resized_channels.append(_resize_2d_batch(out[:, c], resize))
        out = np.stack(resized_channels, axis=1)

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _take_slice(V: np.ndarray, axis: int, index: int) -> np.ndarray:
    """Take a single slice along `axis` from array V."""
    return np.take(V, index, axis=axis)


def _variance_per_slice(V: np.ndarray, axis: int) -> np.ndarray:
    """Compute per-slice variance for weighting. Returns (N, num_slices)."""
    num_slices = V.shape[axis]
    # Move projection axis to the end for easy iteration
    V_moved = np.moveaxis(V, axis, -1)  # (N, ..., num_slices)
    # Reshape to (N, pixels, num_slices)
    N = V.shape[0]
    V_flat = V_moved.reshape(N, -1, num_slices)
    return V_flat.var(axis=1)  # (N, num_slices)


def _weighted_sum_along_axis(V: np.ndarray, axis: int, weights: np.ndarray) -> np.ndarray:
    """Weighted sum of slices along `axis`. weights: (N, num_slices)."""
    V_moved = np.moveaxis(V, axis, -1)  # (N, H, W, num_slices)
    # Broadcast weights: (N, 1, 1, num_slices) or similar
    w_shape = [1] * V_moved.ndim
    w_shape[0] = weights.shape[0]
    w_shape[-1] = weights.shape[1]
    w = weights.reshape(w_shape)
    return (V_moved * w).sum(axis=-1)  # (N, H, W)


def _resize_2d_batch(imgs: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Simple bilinear resize for a batch (N, H, W) → (N, H', W').
    Uses numpy-only approach (no extra deps beyond scipy which is usually available).
    """
    try:
        from scipy.ndimage import zoom
    except ImportError:
        # Fallback: nearest-neighbor via simple indexing
        N, H, W = imgs.shape
        th, tw = target_hw
        row_idx = (np.arange(th) * H / th).astype(int).clip(0, H - 1)
        col_idx = (np.arange(tw) * W / tw).astype(int).clip(0, W - 1)
        return imgs[:, row_idx[:, None], col_idx[None, :]]

    N, H, W = imgs.shape
    th, tw = target_hw
    zh, zw = th / H, tw / W
    # Zoom only spatial dims, not batch
    return zoom(imgs, (1, zh, zw), order=1)