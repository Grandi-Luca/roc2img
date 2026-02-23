import numpy as np
import torch
from torchvision import datasets, transforms


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

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    # Stack diretto in tensor unico
    X_train = torch.stack([train_set[i][0] for i in range(len(train_set))])
    y_train = torch.tensor([train_set[i][1] for i in range(len(train_set))])

    X_test = torch.stack([test_set[i][0] for i in range(len(test_set))])
    y_test = torch.tensor([test_set[i][1] for i in range(len(test_set))])

    # Converti in numpy SENZA cambiare ordine (resta N,C,H,W)
    return (
        X_train.numpy().astype(np.float32),
        y_train.numpy().astype(np.int64),
        X_test.numpy().astype(np.float32),
        y_test.numpy().astype(np.int64),
    )


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