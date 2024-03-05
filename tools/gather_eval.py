import os
import sys
import glob
import shutil
import os.path as osp
import numpy as np

inp_path = "/home/guandao/tava/pretrained/"
out_path = "/home/guandao/tava/data/actorhq_dataset/evaluations/tava"
meta_data_path = "/home/guandao/tava/data/actorhq_dataset/Actor%02d/Sequence1/4x/tava/meta_data.npy"

for actor_id in [1, 2, 3, 6]:
    imgs_from = osp.join(inp_path, f"actor{actor_id:02d}", "eval_imgs", "test")
    imgs_to = osp.join(out_path, f"actor{actor_id:02d}")
    os.makedirs(imgs_to, exist_ok=True)
    meta_data = np.load(meta_data_path % actor_id, allow_pickle=True).item()
    fids = meta_data["fids"]
    cnt = 0 
    for f in glob.glob(f"{imgs_from}/*pred.png"):
        base_path = osp.dirname(f)
        fname_items = osp.basename(f)[:-len(".png")].split("_")
        _, sid, fid_idx, cid, _ = fname_items
        fid = fids[int(fid_idx)]
        out_fname = osp.join(imgs_to, f"image_{fid:06d}.png")
        # print(f, out_fname)
        shutil.copy(f, out_fname)
        cnt += 1
    print("Actor%02d" % actor_id, cnt)
