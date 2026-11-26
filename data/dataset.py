"""
data/dataset.py

Purpose:
    Dataset classes for multi-agent trajectory prediction. All classes return
    the same per-scene dict, so they are interchangeable via DatasetFactory.

Per-scene dict (N agents, P polylines):
    past          : (N, T_past, 4)   float — (x, y, vx, vy) in the scene frame
    past_mask     : (N, T_past)      bool  — True = real observation, False = padded
    future        : (N, T_future, 2) float — (x, y) in the scene frame
    valid_mask    : (N, T_future)    bool  — True = real future annotation
    mask          : (N,)             bool  — True = real agent (social context)
    target_mask   : (N,)             bool  — True = agent is scored (loss + metrics)
    agent_type    : (N,)             int   — 0 pedestrian, 1 cyclist, 2 vehicle/other
    map_polylines : (P, L, 2)        float — lane centerlines (P may be 0)
    origin        : (2,)             float — world position of the scene frame origin
    angle         : float                  — world heading of the scene frame +x axis

Datasets:
    DummyDataset    — random linear motion, no files needed (smoke tests).
    ETHUCYDataset   — ETH/UCY pedestrian benchmark, leave-one-scene-out splits.
    NuScenesDataset — nuScenes prediction samples read straight from the JSON
                      tables (the devkit is optional; only needed for maps).

Public interface:
    DatasetFactory(config, split) -> Dataset
    build_dataloader(dataset, batch_size, shuffle, ...) -> DataLoader
    collate_scenes(batch) -> batched dict (adds polyline_mask)
"""

import glob
import json
import os
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils.preprocessing import normalize_to_agent_frame

AGENT_TYPES = {"pedestrian": 0, "cyclist": 1, "vehicle": 2}
AGENT_TYPE_NAMES = {v: k for k, v in AGENT_TYPES.items()}


# ---------------------------------------------------------------------------
# Collation helper
# ---------------------------------------------------------------------------

def collate_scenes(batch):
    """
    Pads the variable agent / polyline dimensions to the batch maximum.

    Returns:
        past          : (B, N, T_past, 4)
        past_mask     : (B, N, T_past)   padded agents are all-True (avoids
                                         fully-masked attention rows)
        future        : (B, N, T_future, 2)
        valid_mask    : (B, N, T_future) padded agents are all-False
        mask          : (B, N)
        target_mask   : (B, N)
        agent_type    : (B, N)           padded agents are -1
        map_polylines : (B, P, L, 2)
        polyline_mask : (B, P)           True = real polyline
        origin        : (B, 2)
        angle         : (B,)
    """
    max_agents = max(s["past"].shape[0] for s in batch)
    max_polylines = max(s["map_polylines"].shape[0] for s in batch)
    T_past = batch[0]["past"].shape[1]
    T_future = batch[0]["future"].shape[1]
    L = batch[0]["map_polylines"].shape[1]

    B = len(batch)
    past_out = torch.zeros(B, max_agents, T_past, 4)
    past_mask_out = torch.ones(B, max_agents, T_past, dtype=torch.bool)
    future_out = torch.zeros(B, max_agents, T_future, 2)
    valid_out = torch.zeros(B, max_agents, T_future, dtype=torch.bool)
    mask_out = torch.zeros(B, max_agents, dtype=torch.bool)
    target_out = torch.zeros(B, max_agents, dtype=torch.bool)
    type_out = torch.full((B, max_agents), -1, dtype=torch.long)
    poly_out = torch.zeros(B, max_polylines, L, 2)
    poly_mask_out = torch.zeros(B, max_polylines, dtype=torch.bool)
    origin_out = torch.zeros(B, 2)
    angle_out = torch.zeros(B)

    for i, scene in enumerate(batch):
        N = scene["past"].shape[0]
        P = scene["map_polylines"].shape[0]
        mask = torch.as_tensor(scene["mask"], dtype=torch.bool)
        past_out[i, :N] = torch.as_tensor(scene["past"], dtype=torch.float32)
        future_out[i, :N] = torch.as_tensor(scene["future"], dtype=torch.float32)
        mask_out[i, :N] = mask
        if P > 0:
            poly_out[i, :P] = torch.as_tensor(scene["map_polylines"], dtype=torch.float32)
            poly_mask_out[i, :P] = True

        # Optional keys keep older scene dicts (e.g. JSON scenes for
        # inference.py) working: default to "everything real / scored".
        pm = scene.get("past_mask")
        if pm is not None:
            past_mask_out[i, :N] = torch.as_tensor(pm, dtype=torch.bool)
        vm = scene.get("valid_mask")
        valid_out[i, :N] = True if vm is None else torch.as_tensor(vm, dtype=torch.bool)
        tm = scene.get("target_mask")
        target_out[i, :N] = mask if tm is None else torch.as_tensor(tm, dtype=torch.bool) & mask
        at = scene.get("agent_type")
        if at is not None:
            type_out[i, :N] = torch.as_tensor(at, dtype=torch.long)
        if "origin" in scene:
            origin_out[i] = torch.as_tensor(scene["origin"], dtype=torch.float32)
            angle_out[i] = float(scene.get("angle", 0.0))

    # A padded / invalid agent must never have an all-False history: the
    # temporal encoder would produce NaNs for a fully-masked sequence.
    past_mask_out[..., -1] = True

    return {
        "past": past_out,
        "past_mask": past_mask_out,
        "future": future_out,
        "valid_mask": valid_out,
        "mask": mask_out,
        "target_mask": target_out,
        "agent_type": type_out,
        "map_polylines": poly_out,
        "polyline_mask": poly_mask_out,
        "origin": origin_out,
        "angle": angle_out,
    }


class AgentBudgetBatchSampler(torch.utils.data.Sampler):
    """
    Groups scenes of similar agent count into batches holding at most
    `max_agents` padded agent slots (and at most `max_scenes` scenes).

    Collation pads every scene to the largest one in its batch and the social
    attention is O(N^2), so mixing a 57-pedestrian window with 2-pedestrian
    ones wastes most of the compute. Sorting by size inside random chunks
    keeps batches homogeneous while preserving shuffling.
    """

    def __init__(self, sizes, max_agents, max_scenes, shuffle=False, generator=None,
                 chunk=50):
        self.sizes = np.asarray(sizes)
        self.max_agents = max_agents
        self.max_scenes = max_scenes
        self.shuffle = shuffle
        self.generator = generator
        self.chunk = chunk
        self._n = len(list(self._batches(shuffle=False)))

    def _batches(self, shuffle):
        n = len(self.sizes)
        if shuffle:
            order = torch.randperm(n, generator=self.generator).numpy()
            # Sort within chunks of `chunk * max_scenes` for homogeneous batches.
            span = self.chunk * self.max_scenes
            order = np.concatenate([
                sorted(order[i:i + span], key=lambda j: self.sizes[j])
                for i in range(0, n, span)
            ]) if n else order
        else:
            order = np.argsort(self.sizes, kind="stable")
        batches, cur, cur_max = [], [], 0
        for idx in order:
            s = int(self.sizes[idx])
            new_max = max(cur_max, s)
            if cur and (new_max * (len(cur) + 1) > self.max_agents or len(cur) >= self.max_scenes):
                batches.append(cur)
                cur, new_max = [], s
            cur.append(int(idx))
            cur_max = new_max
        if cur:
            batches.append(cur)
        if shuffle:
            perm = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[i] for i in perm]
        return batches

    def __iter__(self):
        return iter(self._batches(self.shuffle))

    def __len__(self):
        return self._n


def scene_sizes(dataset):
    """Agent count per scene, without materialising every sample if possible."""
    if hasattr(dataset, "windows"):
        return [w.shape[0] for w in dataset.windows]
    if hasattr(dataset, "samples"):
        return [s["past"].shape[0] for s in dataset.samples]
    return [dataset[i]["past"].shape[0] for i in range(len(dataset))]


def build_dataloader(dataset, batch_size, shuffle=False, num_workers=0,
                     pin_memory=False, collate_fn=collate_scenes,
                     generator=None, max_agents_per_batch=None) -> DataLoader:
    """Wrap a dataset in a DataLoader with the custom collate function.

    With max_agents_per_batch, batches are formed by AgentBudgetBatchSampler
    (batch_size becomes the maximum number of scenes per batch).
    """
    if max_agents_per_batch:
        sampler = AgentBudgetBatchSampler(scene_sizes(dataset), max_agents_per_batch,
                                          batch_size, shuffle=shuffle, generator=generator)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers,
                          pin_memory=pin_memory, collate_fn=collate_fn)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        generator=generator,
    )


def _finite_diff_velocity(pos: np.ndarray, dt) -> np.ndarray:
    """Backward-difference velocity for (N, T, 2) positions.

    dt is a scalar or an (N, T) array of per-step time gaps. The first step
    reuses the second step's velocity (forward difference) so the history
    never starts with an artificial zero velocity.
    """
    vel = np.zeros_like(pos)
    if pos.shape[1] < 2:
        return vel
    dt = np.asarray(dt, dtype=np.float32)
    step = pos[:, 1:] - pos[:, :-1]
    if dt.ndim == 0:
        vel[:, 1:] = step / dt
    else:
        vel[:, 1:] = step / np.maximum(dt[:, 1:, None], 1e-3)
    vel[:, 0] = vel[:, 1]
    return vel


# ---------------------------------------------------------------------------
# DummyDataset
# ---------------------------------------------------------------------------

class DummyDataset(Dataset):
    """
    Random but structurally valid scenes generated in memory: agents move at a
    constant velocity plus noise. Reproducible per index. Used for smoke tests
    and for running the pipeline with no data on disk.
    """

    def __init__(self, config: dict, split: str = "train"):
        ds_cfg = config["dataset"]
        self.num_agents_max = ds_cfg.get("num_agents_max") or 10
        self.past_steps = ds_cfg["past_steps"]
        self.future_steps = ds_cfg["future_steps"]
        self.num_polylines = ds_cfg.get("num_polylines", 6)
        self.polyline_pts = ds_cfg.get("polyline_points", 20)
        self.dt = ds_cfg.get("dt", 0.25)
        key = "dummy_train_size" if split == "train" else "dummy_val_size"
        self.size = ds_cfg.get(key, 1000 if split == "train" else 200)
        # Different seed ranges per split so val is not a copy of train.
        self.seed_offset = {"train": 0, "val": 10_000_000, "test": 20_000_000}.get(split, 0)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        rng = np.random.default_rng(seed=self.seed_offset + idx)
        N = int(rng.integers(2, self.num_agents_max + 1))
        T_p, T_f, dt = self.past_steps, self.future_steps, self.dt

        start = rng.standard_normal((N, 2)) * 5.0
        vel = rng.standard_normal((N, 2)) * 1.5
        t_all = np.arange(-T_p + 1, T_f + 1) * dt                    # (T_p+T_f,)
        pos = start[:, None] + vel[:, None] * t_all[None, :, None]
        pos = pos + rng.standard_normal(pos.shape) * 0.05
        pos = pos.astype(np.float32)

        past_pos, fut = pos[:, :T_p], pos[:, T_p:]
        past = np.concatenate([past_pos, _finite_diff_velocity(past_pos, dt)], -1)

        P, L = self.num_polylines, self.polyline_pts
        polylines = np.zeros((P, L, 2), dtype=np.float32)
        for p in range(P):
            lane_start = rng.standard_normal(2) * 10.0
            lane_dir = rng.standard_normal(2)
            lane_dir /= (np.linalg.norm(lane_dir) + 1e-8)
            polylines[p] = lane_start + lane_dir * np.arange(L)[:, None]

        origin = past_pos[0, -1].copy()
        past_n, fut_n, poly_n, origin, angle = normalize_to_agent_frame(
            past, fut, polylines, origin=origin, angle=0.0)
        return {
            "past": past_n,
            "past_mask": np.ones((N, T_p), dtype=bool),
            "future": fut_n,
            "valid_mask": np.ones((N, T_f), dtype=bool),
            "mask": np.ones(N, dtype=bool),
            "target_mask": np.ones(N, dtype=bool),
            "agent_type": np.zeros(N, dtype=np.int64),
            "map_polylines": poly_n,
            "origin": origin,
            "angle": angle,
        }


# ---------------------------------------------------------------------------
# ETH / UCY
# ---------------------------------------------------------------------------

class ETHUCYDataset(Dataset):
    """
    ETH/UCY pedestrian benchmark in the Social-GAN leave-one-scene-out layout
    (see scripts/download_eth_ucy.py):

        <data_root>/<test_scene>/{train,val,test}/*.txt   rows: frame ped x y

    A sample is a window of past_steps + future_steps consecutive annotated
    frames (2.5 Hz -> 8 + 12 steps = 3.2 s observed, 4.8 s predicted). As in
    Social-GAN, the agents of a window are the pedestrians present in every
    frame of it, and all of them are scored.

    Coordinates are world metres translated to the agents' mean last-observed
    position. Training windows are randomly rotated (dataset.augment_rotation)
    since the scenes have arbitrary world orientations.
    """

    def __init__(self, config: dict, split: str = "train"):
        ds = config["dataset"]
        self.past_steps = ds["past_steps"]
        self.future_steps = ds["future_steps"]
        self.dt = ds.get("dt", 0.4)
        self.split = split
        self.augment = bool(ds.get("augment_rotation", True)) and split == "train"
        self.polyline_pts = ds.get("polyline_points", 20)
        # Evaluation always uses every window (stride 1, the standard protocol);
        # training may subsample the heavily overlapping windows.
        stride = int(ds.get("train_window_stride", 1)) if split == "train" else 1
        max_agents = ds.get("num_agents_max")

        scene = ds["test_scene"]
        folder = os.path.join(ds["data_root"], scene, split)
        files = sorted(glob.glob(os.path.join(folder, "*.txt")))
        if not files:
            raise FileNotFoundError(
                f"No ETH/UCY files in '{folder}'. Run: python scripts/download_eth_ucy.py")

        T = self.past_steps + self.future_steps
        self.windows = []        # list of (n, T, 2) float32 arrays
        self.window_info = []    # (file stem, first frame id)
        for path in files:
            data = np.loadtxt(path, dtype=np.float64, ndmin=2)
            if data.size == 0:
                continue
            data = data[np.isfinite(data).all(axis=1)]
            frames = np.unique(data[:, 0])
            if len(frames) < T:
                continue
            frame_step = np.median(np.diff(frames))
            by_frame = {f: data[data[:, 0] == f] for f in frames}
            for i in range(0, len(frames) - T + 1, stride):
                win = frames[i:i + T]
                # Skip windows that straddle an annotation gap.
                if not np.allclose(np.diff(win), frame_step):
                    continue
                rows = [by_frame[f] for f in win]
                peds = set(rows[0][:, 1])
                for r in rows[1:]:
                    peds &= set(r[:, 1])
                if not peds:
                    continue
                peds = sorted(peds)
                traj = np.zeros((len(peds), T, 2), dtype=np.float32)
                for t, r in enumerate(rows):
                    lookup = {pid: xy for pid, xy in zip(r[:, 1], r[:, 2:4])}
                    for a, pid in enumerate(peds):
                        traj[a, t] = lookup[pid]
                if max_agents and len(peds) > max_agents:
                    # Keep the agents closest to the window centre.
                    last = traj[:, self.past_steps - 1]
                    d = np.linalg.norm(last - last.mean(0), axis=1)
                    traj = traj[np.argsort(d)[:max_agents]]
                self.windows.append(traj)
                self.window_info.append((os.path.splitext(os.path.basename(path))[0], int(win[0])))

        if not self.windows:
            raise RuntimeError(f"ETH/UCY split '{folder}' produced no windows of length {T}.")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        traj = self.windows[idx]
        n = traj.shape[0]
        past_pos = traj[:, :self.past_steps]
        fut = traj[:, self.past_steps:]
        past = np.concatenate([past_pos, _finite_diff_velocity(past_pos, self.dt)], -1)

        origin = past_pos[:, -1].mean(axis=0)
        angle = float(np.random.uniform(0.0, 2 * np.pi)) if self.augment else 0.0
        no_map = np.zeros((0, self.polyline_pts, 2), dtype=np.float32)
        past_n, fut_n, poly_n, origin, angle = normalize_to_agent_frame(
            past, fut, no_map, origin=origin, angle=angle)
        return {
            "past": past_n,
            "past_mask": np.ones((n, self.past_steps), dtype=bool),
            "future": fut_n,
            "valid_mask": np.ones((n, self.future_steps), dtype=bool),
            "mask": np.ones(n, dtype=bool),
            "target_mask": np.ones(n, dtype=bool),
            "agent_type": np.full(n, AGENT_TYPES["pedestrian"], dtype=np.int64),
            "map_polylines": poly_n,
            "origin": origin,
            "angle": angle,
        }


# ---------------------------------------------------------------------------
# nuScenes
# ---------------------------------------------------------------------------

# Official v1.0-mini scene split (nuscenes.utils.splits.mini_train / mini_val).
NUSCENES_MINI_SPLITS = {
    "train": ["scene-0061", "scene-0553", "scene-0655", "scene-0757",
              "scene-0796", "scene-1077", "scene-1094", "scene-1100"],
    "val": ["scene-0103", "scene-0916"],
}


def _quat_yaw(q) -> float:
    """Yaw (rotation about z) of a (w, x, y, z) quaternion."""
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class NuScenesDataset(Dataset):
    """
    nuScenes prediction samples built from the raw JSON tables.

    Timing: keyframes are annotated at 2 Hz, so every step is 0.5 s. With the
    default past_steps=5 / future_steps=12 the model sees 2 s of history
    (including the current frame) and predicts 6 s, matching the nuScenes
    prediction challenge horizon. There is no interpolation.

    Agents per sample:
      * context agents: every pedestrian / vehicle annotation (static object
        classes such as barriers and cones are dropped),
      * target agents (scored): categories in dataset.target_categories with a
        full future_steps horizon and >= dataset.min_past_obs observations.
        Bicycles only count as targets when annotated with a rider.
    Agents are sorted by distance to the ego vehicle and capped at
    num_agents_max (targets first). Samples without any target are skipped.

    Frame: ego-vehicle pose at the current keyframe (origin = ego position,
    +x = ego heading).

    Splits: scene-level. v1.0-mini uses the official mini_train / mini_val
    split; trainval uses the devkit's create_splits_scenes (devkit required).
    """

    CONTEXT_PREFIXES = ("human.pedestrian", "vehicle.")

    def __init__(self, config: dict, split: str = "train"):
        ds = config["dataset"]
        self.data_root = ds.get("data_root", "./")
        self.version = ds.get("version", "v1.0-mini")
        self.past_steps = ds["past_steps"]
        self.future_steps = ds["future_steps"]
        self.num_polylines = ds.get("num_polylines", 0)
        self.polyline_pts = ds.get("polyline_points", 20)
        self.num_agents_max = ds.get("num_agents_max") or 32
        self.min_past_obs = ds.get("min_past_obs", 2)
        self.target_prefixes = tuple(ds.get("target_categories",
                                            ["human.pedestrian", "vehicle.bicycle"]))
        self.split = split

        self._load_tables()
        scene_names = self._split_scene_names(split)
        self.maps = self._load_maps() if self.num_polylines > 0 else {}
        self.samples = []
        for scene in self.scenes:
            if scene["name"] not in scene_names:
                continue
            tok = scene["first_sample_token"]
            while tok:
                item = self._build_sample(tok)
                if item is not None:
                    self.samples.append(item)
                tok = self.sample[tok]["next"]
        if not self.samples:
            raise RuntimeError(f"nuScenes split '{split}' produced no samples with target agents.")

    # --- table loading -------------------------------------------------------
    def _load_tables(self):
        folder = os.path.join(self.data_root, self.version)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"nuScenes tables not found in '{folder}'.")

        def table(name):
            with open(os.path.join(folder, f"{name}.json")) as fh:
                return json.load(fh)

        self.scenes = table("scene")
        self.sample = {s["token"]: s for s in table("sample")}
        self.ann = {a["token"]: a for a in table("sample_annotation")}
        categories = {c["token"]: c["name"] for c in table("category")}
        self.inst_cat = {i["token"]: categories[i["category_token"]] for i in table("instance")}
        self.attr_name = {a["token"]: a["name"] for a in table("attribute")}
        self.logs = {l["token"]: l for l in table("log")}
        ego_pose = {e["token"]: e for e in table("ego_pose")}
        # Ego pose of each sample via its LIDAR_TOP keyframe.
        self.sample_ego = {}
        for sd in table("sample_data"):
            if sd["is_key_frame"] and "LIDAR_TOP" in sd["filename"]:
                self.sample_ego[sd["sample_token"]] = ego_pose[sd["ego_pose_token"]]
        self.anns_by_sample = {}
        for a in self.ann.values():
            self.anns_by_sample.setdefault(a["sample_token"], []).append(a)

    def _split_scene_names(self, split):
        if self.version == "v1.0-mini":
            if split not in NUSCENES_MINI_SPLITS:
                raise ValueError(f"v1.0-mini has splits {list(NUSCENES_MINI_SPLITS)}, got '{split}'")
            return set(NUSCENES_MINI_SPLITS[split])
        try:
            from nuscenes.utils.splits import create_splits_scenes
        except ImportError as exc:
            raise ImportError("Splits for full nuScenes need nuscenes-devkit.") from exc
        return set(create_splits_scenes()[split])

    def _load_maps(self):
        """Optional lane centerlines; needs the devkit and the map expansion."""
        try:
            from nuscenes.map_expansion.map_api import NuScenesMap
        except Exception as exc:
            warnings.warn(f"nuScenes map API unavailable ({exc}); training without maps.",
                          RuntimeWarning)
            return {}
        maps = {}
        for loc in sorted({l["location"] for l in self.logs.values()}):
            try:
                maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            except Exception as exc:
                warnings.warn(f"Map '{loc}' unavailable ({exc}); its scenes have no polylines.",
                              RuntimeWarning)
        return maps

    # --- sample construction -------------------------------------------------
    def _category_type(self, cat):
        if cat.startswith("human.pedestrian"):
            return AGENT_TYPES["pedestrian"]
        if cat.startswith("vehicle.bicycle"):
            return AGENT_TYPES["cyclist"]
        return AGENT_TYPES["vehicle"]

    def _chain(self, ann, key, n):
        out, cur = [], ann
        for _ in range(n):
            if not cur[key]:
                break
            cur = self.ann[cur[key]]
            out.append(cur)
        return out

    def _ts(self, ann):
        return self.sample[ann["sample_token"]]["timestamp"] * 1e-6

    def _build_sample(self, sample_token):
        ego = self.sample_ego.get(sample_token)
        if ego is None:
            return None
        ego_xy = np.asarray(ego["translation"][:2], dtype=np.float32)
        ego_yaw = _quat_yaw(ego["rotation"])
        T_p, T_f = self.past_steps, self.future_steps

        agents = []
        for ann in self.anns_by_sample.get(sample_token, []):
            cat = self.inst_cat[ann["instance_token"]]
            if not cat.startswith(self.CONTEXT_PREFIXES):
                continue
            hist = self._chain(ann, "prev", T_p - 1)[::-1] + [ann]     # oldest..current
            fut = self._chain(ann, "next", T_f)
            attrs = {self.attr_name[t] for t in ann["attribute_tokens"]}
            is_target = (cat.startswith(self.target_prefixes)
                         and len(fut) == T_f and len(hist) >= self.min_past_obs
                         and not (cat.startswith("vehicle.bicycle")
                                  and "cycle.with_rider" not in attrs))
            dist = float(np.linalg.norm(np.asarray(ann["translation"][:2]) - ego_xy))
            agents.append((not is_target, dist, ann, hist, fut, cat, is_target))
        if not any(a[-1] for a in agents):
            return None
        agents.sort(key=lambda a: (a[0], a[1]))      # targets first, then nearest
        agents = agents[:self.num_agents_max]

        N = len(agents)
        past_pos = np.zeros((N, T_p, 2), dtype=np.float32)
        past_dt = np.full((N, T_p), 0.5, dtype=np.float32)
        past_mask = np.zeros((N, T_p), dtype=bool)
        future = np.zeros((N, T_f, 2), dtype=np.float32)
        valid = np.zeros((N, T_f), dtype=bool)
        target = np.zeros(N, dtype=bool)
        types = np.zeros(N, dtype=np.int64)
        for i, (_, _, ann, hist, fut, cat, is_target) in enumerate(agents):
            pad = T_p - len(hist)
            xy = np.array([h["translation"][:2] for h in hist], dtype=np.float32)
            ts = np.array([self._ts(h) for h in hist])
            past_pos[i, pad:] = xy
            past_pos[i, :pad] = xy[0]                 # repeat oldest; masked out
            past_mask[i, pad:] = True
            if len(ts) > 1:
                past_dt[i, pad + 1:] = np.diff(ts)
            for t, f in enumerate(fut):
                future[i, t] = f["translation"][:2]
                valid[i, t] = True
            if fut:
                future[i, len(fut):] = future[i, len(fut) - 1]
            target[i] = is_target
            types[i] = self._category_type(cat)

        vel = _finite_diff_velocity(past_pos, past_dt)
        # The first real step after padding differences against a padded copy
        # of itself; use the forward difference instead (as for t = 0).
        n_obs = past_mask.sum(1)
        for i in np.where((n_obs > 1) & (n_obs < T_p))[0]:
            first = T_p - n_obs[i]
            vel[i, first] = vel[i, first + 1]
        # Velocity is undefined on padded steps and for single-observation agents.
        vel[~past_mask] = 0.0
        if T_p > 1:
            single = past_mask.sum(1) == 1
            vel[single] = 0.0
        past = np.concatenate([past_pos, vel], -1)

        polylines = self._map_polylines(sample_token, ego_xy)
        past_n, fut_n, poly_n, origin, angle = normalize_to_agent_frame(
            past, future, polylines, origin=ego_xy, angle=ego_yaw)
        return {
            "past": past_n,
            "past_mask": past_mask,
            "future": fut_n,
            "valid_mask": valid,
            "mask": np.ones(N, dtype=bool),
            "target_mask": target,
            "agent_type": types,
            "map_polylines": poly_n,
            "origin": origin,
            "angle": angle,
            "sample_token": sample_token,
        }

    def _map_polylines(self, sample_token, ego_xy):
        """Up to num_polylines lane centerlines near the ego, each resampled
        to polyline_points points. Empty (0, L, 2) when maps are unavailable."""
        L = self.polyline_pts
        empty = np.zeros((0, L, 2), dtype=np.float32)
        if not self.maps:
            return empty
        scene = next(s for s in self.scenes if s["token"] == self.sample[sample_token]["scene_token"])
        nusc_map = self.maps.get(self.logs[scene["log_token"]]["location"])
        if nusc_map is None:
            return empty
        try:
            recs = nusc_map.get_records_in_radius(float(ego_xy[0]), float(ego_xy[1]), 50.0,
                                                  ["lane", "lane_connector"])
            tokens = recs["lane"] + recs["lane_connector"]
            lanes = nusc_map.discretize_lanes(tokens, 0.5)
        except Exception:
            return empty
        polys = []
        for pts in lanes.values():
            pts = np.asarray(pts, dtype=np.float32)[:, :2]
            if len(pts) < 2:
                continue
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            q = np.linspace(0.0, s[-1], L)
            polys.append(np.stack([np.interp(q, s, pts[:, 0]), np.interp(q, s, pts[:, 1])], -1))
        if not polys:
            return empty
        polys.sort(key=lambda p: np.linalg.norm(p - ego_xy, axis=1).min())
        return np.stack(polys[:self.num_polylines]).astype(np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

DATASETS = {
    "dummy": DummyDataset,
    "eth_ucy": ETHUCYDataset,
    "nuscenes": NuScenesDataset,
}


def DatasetFactory(config: dict, split: str = "train") -> Dataset:
    """Return the dataset selected by config["dataset"]["type"]."""
    dataset_type = config["dataset"]["type"].lower()
    if dataset_type not in DATASETS:
        raise ValueError(f"Unknown dataset type '{dataset_type}'. Expected one of {list(DATASETS)}.")
    return DATASETS[dataset_type](config, split=split)
