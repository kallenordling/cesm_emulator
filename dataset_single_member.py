import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler

# ---------- utilities ----------
def _flatten_tm(T, M):
    # Returns array of (t,m) -> flat idx, and inverse maps
    flat = np.arange(T * M, dtype=np.int64)
    t = flat // M
    m = flat % M
    return flat, t, m

def _reduce_condition_maps(cond_tm_1hw, reduce="mean"):
    """
    cond_tm_1hw: np.ndarray shape (T, M, 1, H, W)
    Returns: 1D array of length T*M with a scalar per (t,m).
    """
    assert cond_tm_1hw.ndim == 5 and cond_tm_1hw.shape[2] == 1, f"Expected (T,M,1,H,W), got {cond_tm_1hw.shape}"
    if reduce == "mean":
        scalars = cond_tm_1hw.mean(axis=(2, 3, 4))  # -> (T,M)
    elif reduce == "median":
        scalars = np.median(cond_tm_1hw, axis=(2, 3, 4))  # -> (T,M)
    else:
        raise ValueError(f"Unknown reduce='{reduce}'")
    return scalars.reshape(-1).astype(np.float64)  # -> (T*M,)

# ---------- Equal-width bins (simple) ----------
def make_equalwidth_bin_sampler_all_members(cond_tm_1hw, n_bins=150, epoch_len=None, replace=True, reduce="mean"):
    """
    Balanced sampling over fixed edges between min..max.
    """
    x = _reduce_condition_maps(cond_tm_1hw, reduce=reduce)   # (T*M,)
    edges = np.linspace(x.min(), x.max(), n_bins + 1, dtype=np.float64)
    # digitize to 0..n_bins-1
    bins = np.clip(np.digitize(x, edges[:-1], right=False) - 1, 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    inv_counts = np.zeros_like(counts, dtype=np.float64)
    nonzero = counts > 0
    inv_counts[nonzero] = 1.0 / counts[nonzero]
    weights = inv_counts[bins]                                         # (T*M,)
    weights = torch.from_numpy(weights).double()
    if epoch_len is None:
        epoch_len = len(x)
    sampler = WeightedRandomSampler(weights=weights, num_samples=epoch_len, replacement=replace)
    return sampler, dict(edges=edges, counts=counts, bins=bins)

# ---------- BSP / KD-style leaves (adaptive) ----------
def _to_2d_features_1d(x1d):
    return np.asarray(x1d)[:, None].astype(np.float64)

def build_bsp_leaves_from_1d(x1d, target_leaves=150, min_leaf=5):
    """
    KD/BSP-like, but for 1D feature: repeatedly split the largest leaf at its median.
    Returns: list of index arrays (leaves).
    """
    X = _to_2d_features_1d(x1d)  # (N,1)
    N = X.shape[0]
    leaves = [np.arange(N)]
    while len(leaves) < target_leaves:
        sizes = np.array([len(ix) for ix in leaves])
        order = np.argsort(-sizes)  # largest first
        split = False
        for j in order:
            idx = leaves[j]
            if len(idx) < 2 * min_leaf:
                continue
            med = np.median(X[idx, 0])
            left = idx[X[idx, 0] <= med]
            right = idx[X[idx, 0] >  med]
            if len(left) >= min_leaf and len(right) >= min_leaf:
                leaves.pop(j)
                leaves.append(left)
                leaves.append(right)
                split = True
                break
        if not split:
            break
    return leaves

def make_bsp_sampler_all_members(cond_tm_1hw, target_leaves=150, min_leaf=5, epoch_len=None, replace=True, reduce="mean"):
    """
    Adaptive BSP: each leaf (region of condition space) gets equal chance; within a leaf, uniform sampling.
    """
    x = _reduce_condition_maps(cond_tm_1hw, reduce=reduce)   # (T*M,)
    leaves = build_bsp_leaves_from_1d(x, target_leaves=target_leaves, min_leaf=min_leaf)

    # weight[i] = 1 / size(leaf(i))
    leaf_sizes = np.zeros_like(x, dtype=np.int64)
    for L in leaves:
        leaf_sizes[L] = len(L)
    weights = (1.0 / leaf_sizes).astype(np.float64)
    weights = torch.from_numpy(weights).double()

    if epoch_len is None:
        epoch_len = len(x)

    sampler = WeightedRandomSampler(weights=weights, num_samples=epoch_len, replacement=replace)
    return sampler, dict(leaves=leaves)

class AllMembersDataset(torch.utils.data.Dataset):
    """
    cond_np: (T, M, 1, H, W)
    tgt_np:  (T, M, 1, H, W)
    time_ids: (T,)
    """
    def __init__(self, cond_np, tgt_np, time_ids=None):
        assert cond_np.shape[:2] == tgt_np.shape[:2], "T and M must match for cond and target"
        self.cond = cond_np
        self.tgt = tgt_np
        self.time_ids = time_ids
        self.T, self.M = cond_np.shape[:2]

    def __len__(self):
        return self.T * self.M

    def __getitem__(self, idx):
        import torch
        t = idx // self.M
        m = idx % self.M
        cond = torch.from_numpy(self.cond[t, m])   # (1,H,W)
        x0   = torch.from_numpy(self.tgt[t, m])    # (1,H,W)
        if self.time_ids is not None:
            year = torch.tensor(self.time_ids[t], dtype=torch.long)
            return cond, x0, year
        return cond, x0
        
class SingleMemberDataset(Dataset):
    """
    Returns:
      cond:  (1,H,W)
      x0:    (1,H,W)  # one randomly chosen member from (M,H,W)
    """
    def __init__(self, cond_arr: np.ndarray, target_arr: np.ndarray,
                 member_mode: str = "random", fixed_member: int = 0):
        assert cond_arr.ndim == 4 and cond_arr.shape[1] == 1, f"cond_arr shape {cond_arr.shape} expected (N,1,H,W)"
        assert target_arr.ndim == 4, f"target_arr shape {target_arr.shape} expected (N,M,H,W)"
        self.cond = cond_arr.astype(np.float32)
        self.tgt  = target_arr.astype(np.float32)
        self.member_mode = member_mode
        self.fixed_member = fixed_member

    def __len__(self):
        return self.cond.shape[0]

    def __getitem__(self, idx):
        cond = torch.from_numpy(self.cond[idx])           # (1,H,W)
        members = torch.from_numpy(self.tgt[idx])         # (M,H,W)
        if self.member_mode == "fixed":
            k = int(self.fixed_member)
        else:
            k = torch.randint(0, members.shape[0], (1,)).item()
        x0 = members[k:k+1, ...]                          # (1,H,W)
        return cond, x0
