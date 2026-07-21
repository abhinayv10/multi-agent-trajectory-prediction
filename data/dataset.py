"""
data/dataset.py

Purpose:
    Provides dataset classes for multi-agent trajectory prediction.
    All classes share an identical interface so they are interchangeable
    via DatasetFactory without modifying any other file.

Inputs (per scene dict):
    past          : (N, T_past, 4)   — agent states (x, y, vx, vy), agent-centric
    future        : (N, T_future, 2) — future positions (x, y), agent-centric
    mask          : (N,)             — bool, True = valid agent
    map_polylines : (P, L, 2)        — P polylines of L points each (x, y)

Outputs:
    Same dict collated into batches by the DataLoader collate function.

Public interface:
    DummyDataset(config)                       -> Dataset
    NuScenesDataset(config)                    -> Dataset
    DatasetFactory(config, split) -> Dataset
"""

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Collation helper
# ---------------------------------------------------------------------------

def collate_scenes(batch):
    """
    Custom collate that pads variable-length agent dimensions to the
    maximum number of agents in the batch.

    Args:
        batch: list of scene dicts with keys past, future, mask, map_polylines

    Returns:
        Batched dict with:
            past          : (B, N_max, T_past, 4)
            future        : (B, N_max, T_future, 2)
            mask          : (B, N_max)
            map_polylines : (B, P, L, 2)
            valid_mask    : (B, N_max, T_future)  bool, True = real future step
            polyline_mask : (B, P_max)            bool, True = real polyline
    """
    max_agents = max(s["past"].shape[0] for s in batch)
    max_polylines = max(s["map_polylines"].shape[0] for s in batch)
    T_past = batch[0]["past"].shape[1]        # 8
    T_future = batch[0]["future"].shape[1]    # 12
    L = batch[0]["map_polylines"].shape[1]    # polyline length

    B = len(batch)
    past_out = torch.zeros(B, max_agents, T_past, 4)          # (B, N, 8, 4)
    future_out = torch.zeros(B, max_agents, T_future, 2)       # (B, N, 12, 2)
    mask_out = torch.zeros(B, max_agents, dtype=torch.bool)    # (B, N)
    poly_out = torch.zeros(B, max_polylines, L, 2)             # (B, P, L, 2)
    # Per-timestep future validity; padded agents stay all-False.
    valid_out = torch.zeros(B, max_agents, T_future, dtype=torch.bool)  # (B, N, T_future)
    # Real vs padded polylines; padded polylines stay False.
    poly_mask_out = torch.zeros(B, max_polylines, dtype=torch.bool)     # (B, P)

    for i, scene in enumerate(batch):
        N = scene["past"].shape[0]
        P = scene["map_polylines"].shape[0]
        past_out[i, :N] = torch.as_tensor(scene["past"], dtype=torch.float32)
        future_out[i, :N] = torch.as_tensor(scene["future"], dtype=torch.float32)
        mask_out[i, :N] = torch.as_tensor(scene["mask"], dtype=torch.bool)
        poly_out[i, :P] = torch.as_tensor(scene["map_polylines"], dtype=torch.float32)
        poly_mask_out[i, :P] = True

        # Backward compatible: if a sample lacks "valid_mask", treat every real
        # agent's whole future as valid (all-True) so downstream code that does
        # batch.get("valid_mask") keeps working.
        vm = scene.get("valid_mask")
        if vm is None:
            valid_out[i, :N] = True
        else:
            valid_out[i, :N] = torch.as_tensor(vm, dtype=torch.bool)

    return {
        "past": past_out,
        "future": future_out,
        "mask": mask_out,
        "map_polylines": poly_out,
        "valid_mask": valid_out,
        "polyline_mask": poly_mask_out,
    }


def build_dataloader(dataset, batch_size, shuffle=False, num_workers=0,
                     pin_memory=False, collate_fn=collate_scenes) -> DataLoader:
    """Wrap a dataset in a DataLoader with the custom collate function."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )


# ---------------------------------------------------------------------------
# DummyDataset
# ---------------------------------------------------------------------------

class DummyDataset(Dataset):
    """
    Generates random but structurally valid driving scenes in memory.
    No external dependencies beyond PyTorch and NumPy.

    Each scene is generated fresh on __getitem__ using a fixed seed derived
    from the index so results are reproducible across runs.

    All coordinates are returned in agent-centric frame (ego agent is the
    first valid agent; its last observed position is the origin and its
    heading aligns with the positive x-axis).
    """

    def __init__(self, config: dict, split: str = "train"):
        """
        Args:
            config : full config dict (reads config["dataset"] sub-dict)
            split  : "train" or "val"  — controls dataset size
        """
        ds_cfg = config["dataset"]
        self.num_agents_max = ds_cfg["num_agents_max"]          # 10
        self.past_steps     = ds_cfg["past_steps"]              # 8
        self.future_steps   = ds_cfg["future_steps"]            # 12
        self.num_polylines  = ds_cfg["num_polylines"]           # 6
        self.polyline_pts   = ds_cfg["polyline_points"]         # 20

        key = "dummy_train_size" if split == "train" else "dummy_val_size"
        self.size = ds_cfg.get(key, 1000 if split == "train" else 200)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(seed=idx)  # reproducible per index

        # --- number of agents in this scene ---
        N = int(rng.integers(2, self.num_agents_max + 1))       # scalar

        # --- past trajectories (already in agent-centric coordinates) ---
        # Start agents near origin with small random offsets
        start_pos = rng.standard_normal((N, 2)) * 5.0           # (N, 2)
        velocities = rng.standard_normal((N, 2)) * 2.0          # (N, 2) m/s

        past = np.zeros((N, self.past_steps, 4), dtype=np.float32)  # (N, 8, 4)
        dt = 0.25  # 4 Hz
        for t in range(self.past_steps):
            past[:, t, :2] = start_pos + velocities * (t - self.past_steps) * dt
            past[:, t, 2:] = velocities + rng.standard_normal((N, 2)) * 0.1

        # Normalize: shift so ego (agent 0) last observed pos is origin
        origin = past[0, -1, :2].copy()                         # (2,)
        past[:, :, :2] -= origin

        # --- future trajectories ---
        future = np.zeros((N, self.future_steps, 2), dtype=np.float32)  # (N, 12, 2)
        for t in range(self.future_steps):
            future[:, t, :] = (past[:, -1, :2]
                                + velocities * (t + 1) * dt
                                + rng.standard_normal((N, 2)) * 0.2)

        # --- valid agent mask ---
        mask = np.ones(N, dtype=bool)                           # (N,)

        # --- map polylines ---
        P, L = self.num_polylines, self.polyline_pts
        polylines = np.zeros((P, L, 2), dtype=np.float32)      # (P, L, 2)
        for p in range(P):
            lane_start = rng.standard_normal(2) * 10.0
            lane_dir   = rng.standard_normal(2)
            lane_dir  /= (np.linalg.norm(lane_dir) + 1e-8)
            for l in range(L):
                polylines[p, l] = lane_start + lane_dir * l * 1.0

        return {
            "past":          past,        # (N, 8, 4)
            "future":        future,      # (N, 12, 2)
            "mask":          mask,        # (N,)
            "map_polylines": polylines,   # (P, L, 2)
        }


# ---------------------------------------------------------------------------
# NuScenesDataset stub
# ---------------------------------------------------------------------------

class NuScenesDataset(Dataset):
    def __init__(self, config: dict, split: str = "train"):
        ds_cfg = config["dataset"]
        self.data_root     = ds_cfg.get("data_root", "./")
        self.past_steps    = ds_cfg["past_steps"]
        self.future_steps  = ds_cfg["future_steps"]
        self.num_polylines = ds_cfg["num_polylines"]
        self.polyline_pts  = ds_cfg["polyline_points"]
        self.num_agents_max = ds_cfg.get("num_agents_max", 10)
        self.split         = split

        from nuscenes.nuscenes import NuScenes

        version = "v1.0-mini" if split in ["train", "val"] else "v1.0-trainval"
        self.nusc = NuScenes(version=version, dataroot=self.data_root, verbose=False)

        # Map polylines require the nuScenes map-expansion pack. Load lazily and
        # defensively so a missing/broken expansion pack degrades to all-zero
        # polylines (the previous behavior) instead of crashing the constructor.
        self.maps = {}
        self._load_maps()

        self.sample_tokens = self._get_sample_tokens()

    def _load_maps(self):
        """Populate self.maps with one NuScenesMap per location found in the
        logs.

        Degrades gracefully: if the map-expansion API/pack is unavailable, warns
        exactly once and leaves self.maps empty so __getitem__ falls back to
        all-zero map_polylines rather than raising.
        """
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
        except Exception as exc:  # ImportError or any expansion-pack issue
            warnings.warn(
                "nuScenes map-expansion API unavailable "
                f"({exc}); map_polylines will be all zeros. Install the maps "
                "expansion pack to enable lane loading.",
                RuntimeWarning,
            )
            return

        known_locations = [
            "singapore-onenorth",
            "singapore-hollandvillage",
            "singapore-queenstown",
            "boston-seaport",
        ]
        # Only load maps for locations actually present in this split's logs.
        try:
            locations = {log["location"] for log in self.nusc.log}
        except Exception:
            locations = set(known_locations)

        warned = False
        for loc in sorted(locations):
            try:
                self.maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            except Exception as exc:
                if not warned:  # warn only once across all locations
                    warnings.warn(
                        f"Failed to load NuScenesMap for '{loc}' ({exc}); "
                        "affected scenes will have all-zero map_polylines. "
                        "Verify the maps expansion pack is installed under "
                        f"'{self.data_root}/maps/expansion'.",
                        RuntimeWarning,
                    )
                    warned = True

    def _get_sample_tokens(self):
        tokens = [s['token'] for s in self.nusc.sample]
        if self.split == 'train':
            return tokens[:-10] if len(tokens) > 10 else tokens
        else:
            return tokens[-10:] if len(tokens) > 10 else tokens

    def __len__(self) -> int:
        return len(self.sample_tokens)

    def __getitem__(self, idx: int) -> dict:
        sample_token = self.sample_tokens[idx]
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        nusc_map = self.maps.get(log['location'])

        sample_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'][:2])

        anns = sample['anns']
        valid_anns = []
        for ann_tok in anns:
            ann = self.nusc.get('sample_annotation', ann_tok)
            # Find history length
            count_prev = 0
            curr = ann
            while curr['prev']:
                count_prev += 1
                curr = self.nusc.get('sample_annotation', curr['prev'])
            # Find future length
            count_next = 0
            curr = ann
            while curr['next']:
                count_next += 1
                curr = self.nusc.get('sample_annotation', curr['next'])
            
            if count_prev > 0 or count_next > 0:
                valid_anns.append(ann)

        valid_anns = valid_anns[:self.num_agents_max]
        N = len(valid_anns)
        if N == 0:
            return {
                "past": np.zeros((1, self.past_steps, 4), dtype=np.float32),
                "future": np.zeros((1, self.future_steps, 2), dtype=np.float32),
                "mask": np.zeros(1, dtype=bool),
                "map_polylines": np.zeros((self.num_polylines, self.polyline_pts, 2), dtype=np.float32),
                "valid_mask": np.zeros((1, self.future_steps), dtype=bool),
            }

        past = np.zeros((N, self.past_steps, 4), dtype=np.float32)
        future = np.zeros((N, self.future_steps, 2), dtype=np.float32)
        mask = np.ones(N, dtype=bool)
        # True where a real future annotation exists; False where padded/repeated.
        valid_mask = np.zeros((N, self.future_steps), dtype=bool)
        
        for i, ann in enumerate(valid_anns):
            curr = ann
            hist = [curr]
            for _ in range(self.past_steps - 1):
                if curr['prev']:
                    curr = self.nusc.get('sample_annotation', curr['prev'])
                hist.append(curr)
            hist = hist[::-1]
            
            for t, h in enumerate(hist):
                vel = self.nusc.box_velocity(h['token'])
                if np.isnan(vel).any(): vel = np.zeros(3)
                past[i, t, 0] = h['translation'][0]
                past[i, t, 1] = h['translation'][1]
                past[i, t, 2] = vel[0]
                past[i, t, 3] = vel[1]

            curr = ann
            fut = []
            fut_valid = []
            for _ in range(self.future_steps):
                if curr['next']:
                    curr = self.nusc.get('sample_annotation', curr['next'])
                    fut_valid.append(True)   # real future annotation
                else:
                    fut_valid.append(False)  # padded: last annotation repeated
                fut.append(curr)

            for t, f in enumerate(fut):
                future[i, t, 0] = f['translation'][0]
                future[i, t, 1] = f['translation'][1]
                valid_mask[i, t] = fut_valid[t]

        map_polylines = np.zeros((self.num_polylines, self.polyline_pts, 2), dtype=np.float32)
        if nusc_map is not None:
            try:
                lanes = nusc_map.get_records_in_radius(ego_translation[0], ego_translation[1], 50.0, ['lane'])
                lanes = lanes['lane']
                lanes = lanes[:self.num_polylines]
                for p_idx, lane_tok in enumerate(lanes):
                    lane_rec = nusc_map.get('lane', lane_tok)
                    poly = nusc_map.extract_polygon(lane_rec['polygon_token'])
                    pts = np.array(poly.exterior.coords)[:self.polyline_pts]
                    L = min(len(pts), self.polyline_pts)
                    map_polylines[p_idx, :L, 0] = pts[:, 0]
                    map_polylines[p_idx, :L, 1] = pts[:, 1]
                    if L > 0 and L < self.polyline_pts:
                        map_polylines[p_idx, L:, :] = pts[-1, :].reshape(1, 2)
            except Exception:
                pass
                    
        from utils.preprocessing import normalize_to_agent_frame
        past_norm, future_norm, poly_norm, _, _ = normalize_to_agent_frame(past, future, map_polylines, ego_idx=0)

        return {
            "past": past_norm,
            "future": future_norm,
            "mask": mask,
            "map_polylines": poly_norm,
            "valid_mask": valid_mask,   # (N, future_steps) bool, True = real
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def DatasetFactory(config: dict, split: str = "train") -> Dataset:
    """
    Returns the appropriate Dataset based on config["dataset"]["type"].

    Args:
        config : full config dict
        split  : "train" or "val"

    Returns:
        Dataset instance with DummyDataset or NuScenesDataset interface.
    """
    dataset_type = config["dataset"]["type"].lower()

    if dataset_type == "dummy":
        return DummyDataset(config, split=split)
    elif dataset_type == "nuscenes":
        return NuScenesDataset(config, split=split)
    else:
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. "
            "Expected 'dummy' or 'nuscenes'."
        )