import glob
import os
import os.path as osp
import argparse
from dpscan import DPScan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="./data/in/", help='Img folder')
    parser.add_argument("--out_dir", type=str, default="./data/out/", help='Out folder')
    parser.add_argument("--ckpt", type=str, default="./models/dpscan_saved_weights.pth.tar", help='Checkpoint path')
    parser.add_argument("--size", type=str, default="1072x720", help='Image size. Default: 1072x720')
    args = parser.parse_args()

    if not osp.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    dpscan = DPScan(args)
    img_names = [osp.basename(k) for k in glob.glob(args.in_dir + "/*.png")]
    img_names.sort()

    for i, img_name in enumerate(img_names):
        print("{}/{}: {}".format(i+1, len(img_names), img_name))
        img_dir = osp.join(args.in_dir, img_name)
        out_path = osp.join(args.out_dir, img_name)
        dpscan(img_dir, out_path)
    print("Done !")
