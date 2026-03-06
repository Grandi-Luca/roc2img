import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA
from numpy.lib.stride_tricks import sliding_window_view


# ============================================================
# Utilities
# ============================================================

def to_tuple(x):
    return (x, x) if isinstance(x, int) else x


def atleast_4d(images):
    if images.ndim == 3:
        return images[..., None]
    return images


def to_channels_first(images):
    return np.transpose(images, (0, 3, 1, 2))


def extract_patches(image, filter_shape, step_shape):
    """
    Safe, vectorized patch extraction using sliding_window_view.
    Returns mean-centered flattened patches.
    """
    fh, fw = filter_shape
    sh, sw = step_shape

    windows = sliding_window_view(image, (fh, fw))
    windows = windows[::sh, ::sw]  # apply stride

    patches = windows.reshape(-1, fh * fw)
    patches = patches - patches.mean(axis=1, keepdims=True)

    return patches


def components_to_filters(components, in_channels, filter_shape):
    n_filters = components.shape[0]
    return components.reshape(n_filters, in_channels, *filter_shape)


def binarize(x):
    return (x > 0).astype(np.uint8)


def binary_to_decimal(x):
    L2 = x.shape[1]
    dtype = np.uint16 if L2 <= 16 else np.uint32
    weights = (2 ** np.arange(L2 - 1, -1, -1)).astype(dtype)
    return np.tensordot(x.astype(dtype), weights, axes=([1], [0]))


def block_histograms(images, block_shape, step_shape, n_bins):
    fh, fw = block_shape
    sh, sw = step_shape

    features = []

    for img in images:
        windows = sliding_window_view(img, (fh, fw))
        windows = windows[::sh, ::sw]
        blocks = windows.reshape(-1, fh * fw)

        hist = np.apply_along_axis(
            lambda b: np.histogram(b, bins=n_bins, range=(-0.5, n_bins - 0.5))[0],
            1,
            blocks
        )

        features.append(hist.ravel())

    return np.array(features)


# ============================================================
# PCANet
# ============================================================

class PCANet:

    def __init__(self,
                 name,
                 image_shape,
                 filter_shape_l1, step_shape_l1, n_l1_output,
                 filter_shape_l2, step_shape_l2, n_l2_output,
                 block_shape, block_step,
                 device=None):

        self.name = name
        self.image_shape = to_tuple(image_shape)

        self.filter_shape_l1 = to_tuple(filter_shape_l1)
        self.step_shape_l1 = to_tuple(step_shape_l1)
        self.n_l1_output = n_l1_output

        self.filter_shape_l2 = to_tuple(filter_shape_l2)
        self.step_shape_l2 = to_tuple(step_shape_l2)
        self.n_l2_output = n_l2_output

        self.block_shape = to_tuple(block_shape)
        self.block_step = to_tuple(block_step)

        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.pca_l1 = IncrementalPCA(n_components=n_l1_output, batch_size=5000)
        self.pca_l2 = IncrementalPCA(n_components=n_l2_output, batch_size=5000)

        self._filters_l1 = None
        self._filters_l2 = None


    # ========================================================
    # FIT
    # ========================================================

    def fit(self, images, max_patches=200000):

        images = atleast_4d(images)
        images = to_channels_first(images)

        N, C, _, _ = images.shape

        # ---------------------------
        # L1 Incremental PCA
        # ---------------------------

        indices = np.random.permutation(N)
        patch_counter = 0

        for idx in indices:

            img = images[idx]
            channel_patches = []

            for c in range(C):
                patches = extract_patches(
                    img[c],
                    self.filter_shape_l1,
                    self.step_shape_l1
                )
                channel_patches.append(patches)

            patches = np.hstack(channel_patches)

            if max_patches and patch_counter + patches.shape[0] > max_patches:
                patches = patches[:max_patches - patch_counter]

            if patches.shape[0] > 0:
                self.pca_l1.partial_fit(patches)
                patch_counter += patches.shape[0]

            if max_patches and patch_counter >= max_patches:
                break

        self._filters_l1 = components_to_filters(
            self.pca_l1.components_,
            C,
            self.filter_shape_l1
        )

        # ---------------------------
        # L2 Incremental PCA
        # ---------------------------

        w1 = torch.from_numpy(self._filters_l1).float().to(self.device)

        batch_size = max(1, min(256, N))
        patch_counter = 0

        with torch.no_grad():

            for i in range(0, N, batch_size):

                batch = images[i:i + batch_size]
                x = torch.from_numpy(batch).float().to(self.device)

                l1_out = F.conv2d(x, w1, stride=self.step_shape_l1)
                l1_out = l1_out.cpu().numpy()

                for f in range(self.n_l1_output):

                    maps = l1_out[:, f]

                    for img_map in maps:

                        patches = extract_patches(
                            img_map,
                            self.filter_shape_l2,
                            self.step_shape_l2
                        )

                        if max_patches and patch_counter + patches.shape[0] > max_patches:
                            patches = patches[:max_patches - patch_counter]

                        if patches.shape[0] > 0:
                            self.pca_l2.partial_fit(patches)
                            patch_counter += patches.shape[0]

                        if max_patches and patch_counter >= max_patches:
                            break

                    if max_patches and patch_counter >= max_patches:
                        break

                if max_patches and patch_counter >= max_patches:
                    break

        self._filters_l2 = components_to_filters(
            self.pca_l2.components_,
            1,
            self.filter_shape_l2
        )

        return self


    # ========================================================
    # TRANSFORM
    # ========================================================

    def transform(self, images):

        images = atleast_4d(images)
        images = to_channels_first(images)

        N, C, _, _ = images.shape

        w1 = torch.from_numpy(self._filters_l1).float().to(self.device)
        w2 = torch.from_numpy(self._filters_l2).float().to(self.device)

        batch_size = max(1, min(256, N))
        features = []

        with torch.no_grad():

            for i in range(0, N, batch_size):

                batch = images[i:i + batch_size]
                x = torch.from_numpy(batch).float().to(self.device)

                l1_out = F.conv2d(x, w1, stride=self.step_shape_l1)

                batch_features = []

                for f in range(self.n_l1_output):

                    maps = l1_out[:, f:f+1]
                    l2_out = F.conv2d(maps, w2, stride=self.step_shape_l2)

                    l2_out = l2_out.cpu().numpy()
                    l2_out = binarize(l2_out)
                    l2_out = binary_to_decimal(l2_out)

                    hist = block_histograms(
                        l2_out.squeeze(),
                        self.block_shape,
                        self.block_step,
                        n_bins=2 ** self.n_l2_output
                    )

                    batch_features.append(hist)

                batch_features = np.hstack(batch_features)
                features.append(batch_features)

        return np.vstack(features).astype(np.float64)

#########################################
# 3D PCANet
########################################

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA


# ============================================================
# Utilities
# ============================================================

def atleast_5d(volumes):
    # (N,D,H,W) -> (N,D,H,W,1)
    if volumes.ndim == 4:
        volumes = volumes[..., None]
    return volumes


def to_channels_first_3d(volumes):
    # (N,D,H,W,C) -> (N,C,D,H,W)
    return np.transpose(volumes, (0, 4, 1, 2, 3))

def extract_patches_3d(volume, filter_shape, step_shape):

    fd, fh, fw = filter_shape
    sd, sh, sw = step_shape
    windows = sliding_window_view(volume, (fd, fh, fw))
    windows = windows[::sd, ::sh, ::sw]
    patches = windows.reshape(-1, fd * fh * fw)
    patches -= patches.mean(axis=1, keepdims=True)

    return patches


def components_to_filters_3d(components, in_channels, filter_shape):

    n_filters = components.shape[0]

    return components.reshape(
        n_filters,
        in_channels,
        *filter_shape
    )


def block_histograms_3d(volumes, block_shape, step_shape, n_bins):

    fd, fh, fw = block_shape
    sd, sh, sw = step_shape

    features = []

    for vol in volumes:
        windows = sliding_window_view(vol, (fd, fh, fw))
        windows = windows[::sd, ::sh, ::sw]
        blocks = windows.reshape(-1, fd * fh * fw)
        hist = np.apply_along_axis(
            lambda b: np.histogram(b, bins=n_bins, range=(-0.5, n_bins-0.5))[0],
            1,
            blocks
        )

        features.append(hist.ravel())

    return np.array(features)


# ============================================================
# PCANet3D
# ============================================================

class PCANet3D:

    def __init__(self,
                 name, image_shape,
                 filter_shape_l1=(3,3,3),
                 step_shape_l1=(1,1,1),
                 n_l1_output=8,
                 filter_shape_l2=(3,3,3),
                 step_shape_l2=(1,1,1),
                 n_l2_output=8,
                 block_shape=(7,7,7),
                 block_step=(3,3,3),
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        self.name = name
        self.image_shape = image_shape
        self.filter_shape_l1 = filter_shape_l1
        self.step_shape_l1 = step_shape_l1
        self.n_l1_output = n_l1_output

        self.filter_shape_l2 = filter_shape_l2
        self.step_shape_l2 = step_shape_l2
        self.n_l2_output = n_l2_output

        self.block_shape = block_shape
        self.block_step = block_step

        self.device = device

        self.pca_l1 = IncrementalPCA(n_components=n_l1_output)
        self.pca_l2 = IncrementalPCA(n_components=n_l2_output)

        self.filters_l1 = None
        self.filters_l2 = None


    # ============================================================
    # FIT
    # ============================================================

    def fit(self, volumes, max_patches=200000):

        volumes = atleast_5d(volumes)
        volumes = to_channels_first_3d(volumes)

        N, C, D, H, W = volumes.shape

        # ------------------------
        # L1 PCA
        # ------------------------

        patch_counter = 0

        for img in volumes:

            channel_patches = []

            for c in range(C):

                patches = extract_patches_3d(
                    img[c],
                    self.filter_shape_l1,
                    self.step_shape_l1
                )

                channel_patches.append(patches)

            patches = np.hstack(channel_patches)

            if patch_counter + len(patches) > max_patches:
                patches = patches[:max_patches - patch_counter]

            self.pca_l1.partial_fit(patches)

            patch_counter += len(patches)

            if patch_counter >= max_patches:
                break

        self.filters_l1 = components_to_filters_3d(
            self.pca_l1.components_,
            C,
            self.filter_shape_l1
        )


        # ------------------------
        # L2 PCA
        # ------------------------

        w1 = torch.from_numpy(self.filters_l1).float().to(self.device)

        patch_counter = 0
        batch_size = 16

        with torch.no_grad():

            for i in range(0, N, batch_size):

                batch = volumes[i:i+batch_size]
                x = torch.from_numpy(batch).float().to(self.device)

                l1_out = F.conv3d(
                    x,
                    w1,
                    stride=self.step_shape_l1
                )

                l1_out = l1_out.cpu().numpy()

                for f in range(self.n_l1_output):

                    maps = l1_out[:, f]

                    for vol in maps:

                        patches = extract_patches_3d(
                            vol,
                            self.filter_shape_l2,
                            self.step_shape_l2
                        )

                        if patch_counter + len(patches) > max_patches:
                            patches = patches[:max_patches - patch_counter]

                        self.pca_l2.partial_fit(patches)

                        patch_counter += len(patches)

                        if patch_counter >= max_patches:
                            break

                    if patch_counter >= max_patches:
                        break

                if patch_counter >= max_patches:
                    break


        self.filters_l2 = components_to_filters_3d(
            self.pca_l2.components_,
            1,
            self.filter_shape_l2
        )

        return self


    # ============================================================
    # TRANSFORM
    # ============================================================

    def transform(self, volumes):

        volumes = atleast_5d(volumes)
        volumes = to_channels_first_3d(volumes)

        N, C, _, _, _ = volumes.shape

        w1 = torch.from_numpy(self.filters_l1).float().to(self.device)
        w2 = torch.from_numpy(self.filters_l2).float().to(self.device)

        batch_size = 4
        features = []

        with torch.no_grad():

            for i in range(0, N, batch_size):

                batch = volumes[i:i+batch_size]
                x = torch.from_numpy(batch).float().to(self.device)

                l1_out = F.conv3d(x, w1, stride=self.step_shape_l1)

                batch_features = []

                for f in range(self.n_l1_output):

                    maps = l1_out[:, f:f+1]

                    l2_out = F.conv3d(
                        maps,
                        w2,
                        stride=self.step_shape_l2
                    )

                    l2_out = l2_out.cpu().numpy()

                    l2_out = binarize(l2_out)
                    l2_out = binary_to_decimal(l2_out)

                    hist = block_histograms_3d(
                        l2_out,
                        self.block_shape,
                        self.block_step,
                        n_bins=2 ** self.n_l2_output
                    )

                    batch_features.append(hist)

                batch_features = np.hstack(batch_features)

                features.append(batch_features)

        return np.vstack(features).astype(np.float64)


# ============================================================
# Factory
# ============================================================

def build_pcanet_model(dataset_name):

    if dataset_name == "mnist":
        params = dict(
            name="PCAnet",
            image_shape=32,
            filter_shape_l1=5, step_shape_l1=1, n_l1_output=8,
            filter_shape_l2=5, step_shape_l2=1, n_l2_output=8,
            block_shape=7, block_step=3
        )
        return PCANet(**params)

    elif dataset_name in ["cifar10", "cifar100"]:
        params = dict(
            name="PCAnet",
            image_shape=32,
            filter_shape_l1=5, step_shape_l1=1, n_l1_output=16,
            filter_shape_l2=5, step_shape_l2=1, n_l2_output=8,
            block_shape=8, block_step=4
        )
        return PCANet(**params)
    elif dataset_name == "adni":
        params = dict(
            name="PCAnet3D",
            image_shape=(91,109,91),
            filter_shape_l1=(3,3,3), step_shape_l1=(1,1,1), n_l1_output=8,
            filter_shape_l2=(3,3,3), step_shape_l2=(1,1,1), n_l2_output=8,
            block_shape=(7,7,7), block_step=(3,3,3)

        )
        return PCANet3D(**params)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")