import numpy as np
import torch
from torch.utils.data import Dataset

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
