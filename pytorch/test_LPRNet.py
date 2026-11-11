# -*- coding: utf-8 -*-
# Nhan dang bien so tren 1 anh bat ky
# Vi du: python predict_LPRNet.py --image anh.jpg --lpr_max_len 9

from data.load_data import CHARS
from LPRNet import build_lprnet
import numpy as np
import argparse
import torch
import cv2
import os


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image', required=True, help='duong dan anh (hoac nhieu anh, cach nhau dau phay)')
    p.add_argument('--pretrained_model', default='./weights/Best_LPRNet_model.pth')
    p.add_argument('--img_size', default=[94, 24], type=int, nargs=2, help='width height')
    p.add_argument('--lpr_max_len', default=9, type=int, help='phai trung voi luc train')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def preprocess(path, img_size):
    # dung imdecode de doc duoc duong dan co dau/khoang trang tren Windows
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Khong doc duoc anh: " + path)
    img = cv2.resize(img, tuple(img_size)).astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img).unsqueeze(0)


def greedy_decode(logits):
    blank = len(CHARS) - 1
    seq = logits[0].argmax(0).cpu().numpy()  # (T,)
    out, prev = [], blank
    for c in seq:
        if c != prev and c != blank:
            out.append(int(c))
        prev = c
    return ''.join(CHARS[i] for i in out)


def main():
    args = get_args()
    device = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')

    net = build_lprnet(lpr_max_len=args.lpr_max_len, phase="test", class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    net.to(device).eval()

    for path in args.image.split(','):
        path = os.path.expanduser(path.strip())
        x = preprocess(path, args.img_size).to(device)
        with torch.no_grad():
            logits = net(x)
        print("%s -> %s" % (path, greedy_decode(logits)))


if __name__ == "__main__":
    main()