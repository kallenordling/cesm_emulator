import numpy as np
import torch
from torch.utils.data import Dataset

class WindowedCondDataset(torch.utils.data.Dataset):
    """
    Build temporal windows from (T, M, 1, H, W) arrays.
    Returns:
      cond_win: [1, K, H, W]   # K-frame window for the condition
      x0     : [1, H, W]       # target at center (or last) frame
    """
    def __init__(self, cond_np, tgt_np, K=5, center=True, member_mode="random", fixed_member=0):
        assert cond_np.ndim == 5 and tgt_np.ndim == 5, "Expect (T,M,1,H,W)"
        assert cond_np.shape[:2] == tgt_np.shape[:2], "T and M must match"
        self.cond = cond_np.astype(np.float32)
        self.tgt  = tgt_np.astype(np.float32)
        self.T, self.M = self.cond.shape[:2]
        self.K = int(K)
        self.center = bool(center)
        self.member_mode = member_mode
        self.fixed_member = int(fixed_member)
        if self.K < 2:
            raise ValueError("K must be >= 2")
        if self.T < self.K:
            raise ValueError(f"T={self.T} is smaller than window K={self.K}")

    def __len__(self):
        # number of valid starting indices for a full K window times members
        return (self.T - self.K + 1) * self.M

    def __getitem__(self, idx):
        # map flat idx -> (t0, m)
        m = idx % self.M
        w = idx // self.M
        t0 = w
        t1 = t0 + self.K

        # pick member index for target if requested
        if self.member_mode == "fixed":
            m_tgt = self.fixed_member
        elif self.member_mode == "same":
            m_tgt = m
        else:  # "random"
            m_tgt = np.random.randint(0, self.M)

        # cond window: (K,1,H,W) -> (1,K,H,W)
        cond_win = torch.from_numpy(self.cond[t0:t1, m])    # (K,1,H,W)
        cond_win = cond_win.permute(1, 0, 2, 3).contiguous()  # (1,K,H,W)

        # pick target frame (center or last)
        t_target = t0 + (self.K // 2) if self.center else (t1 - 1)
        x0 = torch.from_numpy(self.tgt[t_target, m_tgt])    # (1,H,W)

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
