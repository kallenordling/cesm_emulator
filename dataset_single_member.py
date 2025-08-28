import numpy as np
import torch
from torch.utils.data import Dataset

class WindowedAllMembersDataset_random(Dataset):
    """
    cond_np, tgt_np: (T, M, 1, H, W)

    Returns per item:
      cond_win: [1, K, h, w]   # K frames from member m
      x0     : [1, h, w]       # target at 'anchor' time

    Config:
      - K: number of frames in condition
      - center: if True, anchor is placed at the middle index of cond_win;
                if False, anchor is placed at the last index.
      - crop_hw: optional (h, w) random/center crop
      - time_reverse_p: probability to apply temporal reversal augmentation
      - sample_mode: "consecutive" | "random_window" | "random_global"
          * consecutive: frames = [t0, t0+1, ..., t0+K-1]
          * random_window: sample K-1 frames from a window around anchor (size = 2*window_radius+1)
          * random_global: sample K-1 frames from the whole 0..T-1
      - window_radius: for random_window; ignored otherwise
      - keep_chronology: if True, sort sampled frames by time (recommended)
      - causal: if True, sample only from times <= anchor (set center=False for causal)
      - allow_replace: allow duplicated times if pool < K
    """
    def __init__(
        self,
        cond_np, tgt_np,
        K=5, center=True,
        crop_hw=None, crop_mode="random",
        time_reverse_p=0.5,
        sample_mode="consecutive",
        window_radius=5,
        keep_chronology=True,
        causal=False,
        allow_replace=False,
    ):
        assert cond_np.ndim == 5 and tgt_np.ndim == 5, "Expect (T, M, 1, H, W)"
        assert cond_np.shape == tgt_np.shape, "cond/tgt shapes must match"
        self.cond = cond_np.astype(np.float32)
        self.tgt  = tgt_np.astype(np.float32)
        self.T, self.M, _, self.H, self.W = self.cond.shape

        if K < 2: raise ValueError("K must be >= 2")
        if self.T < 2: raise ValueError("Not enough time steps")
        self.K = int(K)
        self.center = bool(center)

        # cropping
        if crop_hw is None:
            self.crop_h = None
            self.crop_w = None
        else:
            ch, cw = int(crop_hw[0]), int(crop_hw[1])
            self.crop_h = min(ch, self.H)
            self.crop_w = min(cw, self.W)
        if crop_mode not in ("random", "center"):
            raise ValueError("crop_mode must be 'random' or 'center'")
        self.crop_mode = crop_mode

        # temporal aug
        self.time_reverse_p = float(time_reverse_p)

        # sampling controls
        assert sample_mode in ("consecutive", "random_window", "random_global")
        self.sample_mode = sample_mode
        self.window_radius = int(window_radius)
        self.keep_chronology = bool(keep_chronology)
        self.causal = bool(causal)
        self.allow_replace = bool(allow_replace)

        if self.causal and self.center:
            # causal sampling conflicts with putting anchor at the center
            # (you don't have future frames). Put anchor at the end.
            self.center = False

        # length: we iterate over all anchor-able starts per member for "consecutive"
        # and over all times per member for random modes.
        if self.sample_mode == "consecutive":
            self.num_units = max(1, self.T - self.K + 1)
            self.use_windows = True
        else:
            self.num_units = self.T  # one item per (anchor time, member)
            self.use_windows = False

    def __len__(self):
        return self.num_units * self.M

    def _index_to_tm(self, idx):
        m = idx % self.M
        u = idx // self.M
        if self.use_windows:
            t0 = u
            anchor = t0 + (self.K // 2) if self.center else (t0 + self.K - 1)
        else:
            anchor = u  # any time may be anchor
            # pick a nominal t0 just for bookkeeping (not used by random modes)
            t0 = max(0, min(anchor - (self.K // 2), self.T - self.K))
        anchor = int(np.clip(anchor, 0, self.T - 1))
        return t0, anchor, m

    def _choose_times(self, t0, anchor):
        K = self.K

        if self.sample_mode == "consecutive":
            times = np.arange(t0, t0 + K, dtype=np.int64)
        else:
            # Build candidate pool
            if self.sample_mode == "random_global":
                pool = np.arange(0, self.T, dtype=np.int64)
            else:  # random_window
                left  = max(0, anchor - self.window_radius)
                right = min(self.T - 1, anchor + self.window_radius)
                pool = np.arange(left, right + 1, dtype=np.int64)

            if self.causal:
                pool = pool[pool <= anchor]

            # ensure anchor is included
            pool_wo_anchor = pool[pool != anchor]
            need = K - 1

            # If not enough unique candidates:
            if (not self.allow_replace) and (pool_wo_anchor.size < need):
                # fallback: allow sampling with replacement from whatever we have
                self.allow_replace = True

            if self.allow_replace:
                if pool_wo_anchor.size == 0:
                    sampled = np.full((need,), anchor, dtype=np.int64)
                else:
                    sampled = np.random.choice(pool_wo_anchor, size=need, replace=True)
            else:
                sampled = np.random.choice(pool_wo_anchor, size=need, replace=False)

            times = np.concatenate([sampled, np.array([anchor], dtype=np.int64)])

            if self.keep_chronology:
                times.sort()

            # place anchor at middle (center=True) or end (center=False)
            if self.center:
                mid = K // 2
                # rotate so that anchor sits at position mid
                idx_anchor = int(np.where(times == anchor)[0][0])
                shift = mid - idx_anchor
                times = np.roll(times, shift)
            else:
                # move anchor to the last position
                times = np.array([t for t in times if t != anchor] + [anchor], dtype=np.int64)

        return times

    def _crop_coords(self, H, W):
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
        t0, anchor, m = self._index_to_tm(idx)
        times = self._choose_times(t0, anchor)     # (K,)

        # gather frames -> (K,1,H,W) then (1,K,H,W)
        cond_k = torch.from_numpy(self.cond[times, m])     # (K,1,H,W)
        cond_win = cond_k.permute(1, 0, 2, 3).contiguous() # (1,K,H,W)

        # target at anchor time from same member
        x0 = torch.from_numpy(self.tgt[anchor, m])         # (1,H,W)

        # temporal augmentation
        if self.time_reverse_p > 0.0 and np.random.rand() < self.time_reverse_p:
            if self.center:
                # reverse context around the center; keep anchor at center
                mid = self.K // 2
                left  = cond_win[:, :mid].flip(dims=(1,))
                right = cond_win[:, mid+1:].flip(dims=(1,))
                cond_win = torch.cat([left, cond_win[:, mid:mid+1], right], dim=1)
            else:
                cond_win = cond_win.flip(dims=(1,))  # full flip

        # spatial crop (same crop for cond and target)
        _, K, H, W = cond_win.shape
        i, j, h, w = self._crop_coords(H, W)
        cond_win = cond_win[:, :, i:i+h, j:j+w].contiguous()
        x0       = x0[:, i:i+h, j:j+w].contiguous()

        return cond_win, x0

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
        time_reverse_p=0.5,      # <--- NEW

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
        self.time_reverse_p = float(time_reverse_p)

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
        
        if self.time_reverse_p > 0.0 and np.random.rand() < self.time_reverse_p:
            cond_win = cond_win.flip(dims=(1,))  # flip along the K dimension
            
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
