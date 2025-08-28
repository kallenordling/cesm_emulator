import numpy as np
import torch
from torch.utils.data import Dataset

class WindowedAllMembersDataset(Dataset):
    """
    cond_np, tgt_np: (T, M, 1, H, W)
    Returns per item:
      cond_win: [1, K, h, w]
      x0     : [1, h, w]
    """
    def __init__(
        self,
        cond_np,
        tgt_np,
        K=5,
        center=True,
        crop_hw=None,          # e.g., (128, 128) or None for no crop
        crop_mode="random",    # "random" or "center"
    ):
        assert cond_np.ndim == 5 and tgt_np.ndim == 5, "Expect (T, M, 1, H, W)"
        assert cond_np.shape == tgt_np.shape, "cond/tgt shapes must match"
        self.cond = cond_np.astype(np.float32)
        self.tgt  = tgt_np.astype(np.float32)
        self.T, self.M, _, self.H, self.W = self.cond.shape
        if K < 2: raise ValueError("K must be >= 2")
        if self.T < K: raise ValueError(f"T={self.T} < K={K}")
        self.K = int(K)
        self.center = bool(center)

        # --- crop config ---
        if crop_hw is None:
            self.crop_h = None
            self.crop_w = None
        else:
            ch, cw = int(crop_hw[0]), int(crop_hw[1])
            # clamp to available size to avoid negative ranges
            self.crop_h = min(ch, self.H)
            self.crop_w = min(cw, self.W)
        if crop_mode not in ("random", "center"):
            raise ValueError("crop_mode must be 'random' or 'center'")
        self.crop_mode = crop_mode

        self.num_windows = self.T - self.K + 1

    def __len__(self):
        return self.num_windows * self.M

    def _index_to_tm(self, idx):
        m = idx % self.M
        w = idx // self.M
        t0 = w
        return t0, m

    def _crop_coords(self, H, W):
        """Choose top-left (i,j) for crop respecting crop_mode."""
        if self.crop_h is None or self.crop_w is None:
            return 0, 0, H, W
        h, w = self.crop_h, self.crop_w
        if self.crop_mode == "center":
            i = max(0, (H - h) // 2)
            j = max(0, (W - w) // 2)
        else:  # random
            i = 0 if H == h else np.random.randint(0, H - h + 1)
            j = 0 if W == w else np.random.randint(0, W - w + 1)
        return i, j, h, w

    def __getitem__(self, idx):
        t0, m = self._index_to_tm(idx)
        t1 = t0 + self.K

        # cond window: (K,1,H,W) -> (1,K,H,W)
        cond_win = torch.from_numpy(self.cond[t0:t1, m])           # (K,1,H,W)
        cond_win = cond_win.permute(1, 0, 2, 3).contiguous()       # (1,K,H,W)

        # target from same member, center or last frame of window
        t_target = t0 + (self.K // 2) if self.center else (t1 - 1)
        x0 = torch.from_numpy(self.tgt[t_target, m])               # (1,H,W)

        # --- apply crop (same i,j,h,w to both) ---
        _, K, H, W = cond_win.shape
        i, j, h, w = self._crop_coords(H, W)
        cond_win = cond_win[:, :, i:i+h, j:j+w].contiguous()       # (1,K,h,w)
        x0       = x0[:, i:i+h, j:j+w].contiguous()                # (1,h,w)

        return cond_win, x0

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
