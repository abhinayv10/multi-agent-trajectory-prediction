"""
scripts/download_eth_ucy.py

Download the ETH/UCY pedestrian benchmark in the standard leave-one-scene-out
layout used by Social-GAN, Trajectron++ and most follow-up work:

    data/eth_ucy/<test_scene>/{train,val,test}/*.txt

Each .txt file has tab-separated rows: frame_id  ped_id  x  y  (metres, world
frame). Annotations are at 2.5 Hz, so 8 observed + 12 predicted steps is
3.2 s of history and 4.8 s of future.

Source: the raw split files mirrored in the Trajectron++ repository
(StanfordASL/Trajectron-plus-plus, experiments/pedestrians/raw), which are the
Social-GAN split files.

Usage:
    python scripts/download_eth_ucy.py [--out data/eth_ucy]
"""

import argparse
import os
import sys
import urllib.request

BASE = ("https://raw.githubusercontent.com/StanfordASL/Trajectron-plus-plus/"
        "master/experiments/pedestrians/raw")

# Test files per held-out scene; every other source is used for train/val.
TEST_FILES = {
    "eth":   ["biwi_eth"],
    "hotel": ["biwi_hotel"],
    "univ":  ["students001", "students003"],
    "zara1": ["crowds_zara01"],
    "zara2": ["crowds_zara02"],
}
ALL_FILES = [
    "biwi_eth", "biwi_hotel", "crowds_zara01", "crowds_zara02", "crowds_zara03",
    "students001", "students003", "uni_examples",
]


def file_list(scene: str):
    """(phase, filename) pairs for one leave-one-scene-out split."""
    held_out = set(TEST_FILES[scene])
    files = [("test", f"{name}.txt") for name in TEST_FILES[scene]]
    for name in ALL_FILES:
        if name in held_out:
            continue
        files.append(("train", f"{name}_train.txt"))
        files.append(("val", f"{name}_val.txt"))
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/eth_ucy")
    args = ap.parse_args()

    n_new = 0
    for scene in TEST_FILES:
        for phase, fname in file_list(scene):
            dest = os.path.join(args.out, scene, phase, fname)
            if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            url = f"{BASE}/{scene}/{phase}/{fname}"
            try:
                urllib.request.urlretrieve(url, dest + ".part")
            except Exception as exc:
                print(f"[download] FAILED {url}: {exc}", file=sys.stderr)
                sys.exit(1)
            os.replace(dest + ".part", dest)
            n_new += 1
            print(f"[download] {scene}/{phase}/{fname}")
    print(f"[download] done ({n_new} new files) -> {args.out}")


if __name__ == "__main__":
    main()
