import math
import numpy as np
import tensorflow as tf

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset


SEED = 42

def _preprocess_image_tf(image):
    image = tf.image.resize(image, (224, 224))
    image = tf.image.grayscale_to_rgb(image)
    return image

class DatasetControl:

    def __init__(self, path: str):
        self._path_npz = path

        self.imgs            = None
        self.params          = None
        self._origin         = None
        self._cluster_labels = None

        self._idx_train = None
        self._idx_val   = None
        self._idx_test  = None

        self._preprocess_ready = False
        self._y_train = self._y_val = self._y_test = None

        self.scaler = None

        print(f"[DatasetControl] path='{path}'")

    def load(self):
        npz        = np.load(self._path_npz)
        imgs_raw   = npz['imagenes']
        params_raw = npz['parametros'].astype(np.float32)

        if imgs_raw.ndim == 3:
            imgs_raw = imgs_raw[..., np.newaxis]
        self.imgs   = imgs_raw.astype(np.float32)
        self.params = params_raw
        self._origin = np.where(self.params[:, 1] == 0, 0, 1)

        print(f"[load] Loaded {len(self.imgs):,} images  "
              f"| KDM: {(self._origin==0).sum():,}  "
              f"| Jex: {(self._origin==1).sum():,}")
        print(f"       imgs shape : {self.imgs.shape}")
        print(f"       params shape: {self.params.shape}")


    def select(self, dataset: str):

        self._check_loaded()
        dataset = dataset.lower().strip()

        if dataset == 'unified':
            mask = np.ones(len(self.imgs), dtype=bool)
        elif dataset == 'kdm':
            mask = self.params[:, 1] == 0
        elif dataset == 'jex':
            # Jex: KDM==0 AND Jex2!=0  (Jex2==0 always means KDM)
            mask = (self.params[:, 2] == 0) & (self.params[:, 1] != 0)
        else:
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                "Choose from: 'kdm', 'jex', 'unified'."
            )

        self.imgs            = self.imgs[mask]
        self.params          = self.params[mask]
        self._origin         = self._origin[mask]
        self._cluster_labels = None

        print(f"[select] dataset='{dataset}'  -> {len(self.imgs):,} images kept")

    # ------------------------------------------------------------------ #
    #  GROUP 2 — Exploration                                               #
    # ------------------------------------------------------------------ #

    def summary(self):
        """Print a human-readable summary of the current state."""
        self._check_loaded()
        n      = len(self.imgs)
        mem_mb = (self.imgs.nbytes + self.params.nbytes) / 1e6

        print("=" * 55)
        print("  DatasetControl — Summary")
        print("=" * 55)
        print(f"  Total images   : {n:,}")
        print(f"  imgs shape     : {self.imgs.shape}")
        print(f"  params shape   : {self.params.shape}")
        print(f"  Memory (raw)   : {mem_mb:.1f} MB")
        print()
        print("  Origin:")
        print(f"    KDM (Jex2=0) : {(self._origin==0).sum():,}  "
              f"({100*(self._origin==0).mean():.1f}%)")
        print(f"    Jex (KDM=0)  : {(self._origin==1).sum():,}  "
              f"({100*(self._origin==1).mean():.1f}%)")
        print()
        print("  Parameter ranges  [T, Jex2, KDM]:")
        for i, lbl in enumerate(['temperatura', 'Jex2      ', 'KDM       ']):
            col = self.params[:, i]
            print(f"    {lbl}: [{col.min():.4f}, {col.max():.4f}]  "
                  f"mean={col.mean():.4f}  std={col.std():.4f}")
        if self._cluster_labels is not None:
            print(f"\n  Clusters fitted: {len(np.unique(self._cluster_labels))}")
        if self._idx_train is not None:
            print(f"\n  Split sizes:")
            print(f"    train : {len(self._idx_train):,}")
            print(f"    val   : {len(self._idx_val):,}")
            print(f"    test  : {len(self._idx_test):,}")
        print("=" * 55)


    def get_balanced_subset(self, n_samples: int) -> 'DatasetControl':

        self._check_loaded()
        n_total = len(self.imgs)

        if n_samples >= n_total:
            print(f"[get_balanced_subset] n_samples ({n_samples:,}) >= total "
                  f"({n_total:,}). Returning full dataset copy.")
            return self._make_subset(np.arange(n_total))

        _scaler       = MinMaxScaler()
        params_scaled = _scaler.fit_transform(self.params)

        n_clusters = max(5, int(math.sqrt(n_samples)))
        kmeans = MiniBatchKMeans(
            n_clusters  = n_clusters,
            random_state= SEED,
            n_init      = 10,
            batch_size  = min(10_000, n_total)
        )
        cluster_labels = kmeans.fit_predict(params_scaled)
        print(f"[get_balanced_subset] KMeans: {n_clusters} clusters  "
              f"(n_samples={n_samples:,})")

        rng = np.random.default_rng(SEED)
        selected_indices = []

        cluster_ids, cluster_counts = np.unique(
            cluster_labels, return_counts=True
        )
        quotas = np.floor(
            cluster_counts / cluster_counts.sum() * n_samples
        ).astype(int)

        remainder = n_samples - quotas.sum()
        if remainder > 0:
            top_idx = np.argsort(-cluster_counts)[:remainder]
            quotas[top_idx] += 1

        for cid, quota in zip(cluster_ids, quotas):
            idx_in_cluster = np.where(cluster_labels == cid)[0]
            take = min(quota, len(idx_in_cluster))
            if take < quota:
                print(f"  [warn] Cluster {cid}: only {len(idx_in_cluster)} "
                      f"samples available (quota={quota}). Taking all.")
            chosen = rng.choice(idx_in_cluster, take, replace=False)
            selected_indices.extend(chosen.tolist())

        selected_indices = np.array(selected_indices)
        print(f"[get_balanced_subset] Selected {len(selected_indices):,} samples")

        subset = self._make_subset(selected_indices)
        subset._cluster_labels = cluster_labels[selected_indices]
        return subset

    def split(self, val_size: float = 0.15, test_size: float = 0.15):
        self._check_loaded()
        indices = np.arange(len(self.imgs))
        strat   = self._cluster_labels if self._cluster_labels is not None                   else self._origin

        val_test_size = val_size + test_size
        idx_train, idx_valtest = train_test_split(
            indices,
            test_size    = val_test_size,
            random_state = SEED,
            stratify     = strat
        )

        relative_test = test_size / val_test_size
        idx_val, idx_test = train_test_split(
            idx_valtest,
            test_size    = relative_test,
            random_state = SEED,
            stratify     = strat[idx_valtest]
        )

        self._idx_train = idx_train
        self._idx_val   = idx_val
        self._idx_test  = idx_test

        print(f"[split] train={len(idx_train):,}  "
              f"val={len(idx_val):,}  "
              f"test={len(idx_test):,}")

    def preprocess_images(self):
        self._check_split()
        self._preprocess_ready = True
        n_tr = len(self._idx_train)
        n_v  = len(self._idx_val)
        n_te = len(self._idx_test)
        print(f"[preprocess_images] Pipeline registered (lazy).")
        print(f"  train={n_tr:,}  val={n_v:,}  test={n_te:,}")
        print(f"  Output shape per sample: (224, 224, 3)")

    def preprocess_params(self, scaler_type: str = 'minmax'):
        self._check_split()

        if scaler_type != 'minmax':
            raise ValueError("Currently only 'minmax' scaler is supported.")

        self.scaler   = MinMaxScaler()
        self._y_train = self.scaler.fit_transform(
            self.params[self._idx_train]).astype(np.float32)
        self._y_val   = self.scaler.transform(
            self.params[self._idx_val]).astype(np.float32)
        self._y_test  = self.scaler.transform(
            self.params[self._idx_test]).astype(np.float32)

        print(f"[preprocess_params] MinMaxScaler fitted on train "
              f"({len(self._idx_train):,} samples)")
        print(f"  y range after scaling: "
              f"[{self._y_train.min():.3f}, {self._y_train.max():.3f}]")

    def get_loaders_torch(self, batch_size: int = 64,
                          num_workers: int = 0):

        self._check_preprocessed()
        import torch
        from torch.utils.data import Dataset as TorchDataset

        parent = self   

        class _SpinDataset(TorchDataset):
            def __init__(self, indices, y_scaled):
                self.indices  = indices
                self.y_scaled = torch.from_numpy(y_scaled).float()

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, i):
                img = parent.imgs[self.indices[i]]          # (H, W, 1)
                # Resize with TF, then convert to torch (C, H, W)
                img_tf = _preprocess_image_tf(
                    tf.constant(img, dtype=tf.float32)
                ).numpy()                                   # (224, 224, 3)
                img_t  = torch.from_numpy(
                    img_tf.transpose(2, 0, 1)               # (3, 224, 224)
                ).float()
                return img_t, self.y_scaled[i]

        def _make(indices, y_scaled, shuffle):
            ds = _SpinDataset(indices, y_scaled)
            return DataLoader(
                ds, batch_size=batch_size, shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )

        train_loader = _make(self._idx_train, self._y_train, shuffle=True)
        val_loader   = _make(self._idx_val,   self._y_val,   shuffle=False)
        test_loader  = _make(self._idx_test,  self._y_test,  shuffle=False)

        print(f"[get_loaders_torch] batch_size={batch_size}  "
              f"train={len(train_loader.dataset):,}  "
              f"val={len(val_loader.dataset):,}  "
              f"test={len(test_loader.dataset):,}")
        return train_loader, val_loader, test_loader


    def get_loaders_tf(self, batch_size: int = 64):
        self._check_preprocessed()

        H, W, C = self.imgs.shape[1], self.imgs.shape[2], self.imgs.shape[3]

        def _make(indices, y_scaled, shuffle):
            y_const = tf.constant(y_scaled, dtype=tf.float32)  # params are small

            # Generator yields one (img, y) at a time — never the full array
            def _generator():
                for i, idx in enumerate(indices):
                    yield self.imgs[idx], y_const[i]

            ds = tf.data.Dataset.from_generator(
                _generator,
                output_signature=(
                    tf.TensorSpec(shape=(H, W, C), dtype=tf.float32),
                    tf.TensorSpec(shape=(y_scaled.shape[1],), dtype=tf.float32)
                )
            )
            if shuffle:
                ds = ds.shuffle(buffer_size=min(5_000, len(indices)), seed=SEED)
            ds = ds.map(
                lambda x, y: (_preprocess_image_tf(x), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        train_ds = _make(self._idx_train, self._y_train, shuffle=True)
        val_ds   = _make(self._idx_val,   self._y_val,   shuffle=False)
        test_ds  = _make(self._idx_test,  self._y_test,  shuffle=False)

        print(f"[get_loaders_tf] batch_size={batch_size}  "
              f"train={len(self._idx_train):,}  "
              f"val={len(self._idx_val):,}  "
              f"test={len(self._idx_test):,}")
        return train_ds, val_ds, test_ds

    def get_arrays(self):
        self._check_preprocessed()

        def _apply(indices):
            imgs_tf = tf.constant(self.imgs[indices], dtype=tf.float32)
            imgs_tf = tf.map_fn(
                _preprocess_image_tf, imgs_tf,
                fn_output_signature=tf.float32
            )
            return imgs_tf.numpy()

        print("[get_arrays] Materialising images — this may use significant RAM...")
        X_train = _apply(self._idx_train)
        X_val   = _apply(self._idx_val)
        X_test  = _apply(self._idx_test)
        print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
        return (X_train, X_val,   X_test,
                self._y_train, self._y_val, self._y_test)

    def _make_subset(self, indices: np.ndarray) -> 'DatasetControl':
        new = object.__new__(DatasetControl)
        new._path_npz        = self._path_npz
        new.imgs             = self.imgs[indices]
        new.params           = self.params[indices]
        new._origin          = self._origin[indices]
        new._cluster_labels  = None
        new._idx_train       = None
        new._idx_val         = None
        new._idx_test        = None
        new._preprocess_ready = False
        new._y_train         = None
        new._y_val           = None
        new._y_test          = None
        new.scaler           = None
        return new

    def _check_loaded(self):
        if self.imgs is None:
            raise RuntimeError("Dataset not loaded. Call load() first.")

    def _check_split(self):
        self._check_loaded()
        if self._idx_train is None:
            raise RuntimeError("Data not split. Call split() first.")

    def _check_preprocessed(self):
        self._check_split()
        if not getattr(self, '_preprocess_ready', False):
            raise RuntimeError(
                "Images not preprocessed. Call preprocess_images() first.")
        if self._y_train is None:
            raise RuntimeError(
                "Params not preprocessed. Call preprocess_params() first.")

    def __repr__(self):
        state = 'empty'
        if self.imgs is not None:
            state = f'{len(self.imgs):,} images'
        if self._idx_train is not None:
            state += (f' | split: {len(self._idx_train):,}'
                      f'/{len(self._idx_val):,}/{len(self._idx_test):,}')
        return f'DatasetControl({state})'
