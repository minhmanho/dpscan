import numpy as np
import torch
from PIL import Image
from networks.networks import get_model
from utils import *

class DPScan(object):
    def __init__(self, args):
        ckpt = torch.load(args.ckpt)

        self.G = get_model(ckpt['opts'].g_net)(ckpt['opts']).cuda()
        self.G.load_state_dict(ckpt['G'])
        self.G.eval()
        self.img_size = size_str2tuple(args.size)

    def __call__(self, img_path, out_path):
        with torch.no_grad():
            pil_img = Image.open(img_path).convert('RGB')
            if self.img_size[0] != -1:
                pil_img = pil_img.resize(self.img_size, resample=Image.BICUBIC)

            tensor_img = self.totensor(np.array(pil_img)).cuda()
            tensor_out = self.G(tensor_img)

            tensor_img = tensor_img.cpu()
            tensor_out   = tensor_out.cpu()

            tensor_out = (tensor_out + 1)/2
            tensor_out = np.array(tensor_out[0,:,:,:].clamp(0,1).numpy().transpose(1,2,0) * 255.0, dtype=np.uint8)

            if out_path is not None:
                Image.fromarray(tensor_out).save(out_path)

        return tensor_out

    @staticmethod
    def totensor(tmp):
        tmp = tmp / 255.0
        tmp = (tmp - 0.5)/0.5
        tmp = tmp.transpose((2, 0, 1))
        return torch.from_numpy(tmp).unsqueeze(0).float()
