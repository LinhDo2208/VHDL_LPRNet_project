# -*- coding: utf-8 -*-
# Kiem tra do chinh xac LPRNet tren mot thu muc anh da duoc gan nhan (nhan = ten file).
#
# Dat file nay cung thu muc voi predict_LPRNet.py (can co LPRNet.py va data/load_data.py).
#
# Vi du:
#   python evaluate_LPRNet.py --img_dir data/lpr_dataset/test
#   python evaluate_LPRNet.py --img_dir data/lpr_dataset/test --pretrained_model weights/Best_LPRNet_model.pth --lpr_max_len 9
#   python evaluate_LPRNet.py --img_dir data/lpr_dataset/test --copy_wrong wrong_images

from data.load_data_elec import CHARS, CHARS_DICT, merge_two_lines, parse_label, read_image
from LPRNet import build_lprnet
from collections import Counter, defaultdict
import numpy as np
import argparse
import torch
import shutil
import time
import cv2
import csv
import sys
import os

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
BLANK = len(CHARS) - 1  # ky tu '-' cuoi bang CHARS la blank cua CTC

try:  # in duoc tieng Viet tren console Windows va khi chuyen huong ra file
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


def get_args():
    p = argparse.ArgumentParser(description='Kiem tra do chinh xac LPRNet tren thu muc anh co nhan')
    p.add_argument('--img_dir', required=True, help='thu muc anh (quet ca thu muc con); nhan nam trong ten file')
    p.add_argument('--pretrained_model', default='./weights/Best_LPRNet_model.pth')
    p.add_argument('--img_size', default=[94, 24], type=int, nargs=2, help='width height')
    p.add_argument('--lpr_max_len', default=10, type=int, help='phai trung voi luc train')
    p.add_argument('--batch_size', default=64, type=int,
                   help='nen dung cung batch size nhu luc danh gia khi train (xem ghi chu ve global context)')
    p.add_argument('--no_merge', action='store_true', help='khong ghep bien 2 dong (mac dinh CO ghep, giong luc train)')
    p.add_argument('--report', default='eval_report.csv', help='file csv chi tiet tung anh ("" = khong ghi)')
    p.add_argument('--copy_wrong', default='', help='thu muc de copy cac anh doan sai vao (de xem bang mat)')
    p.add_argument('--show_wrong', default=30, type=int, help='so anh sai in ra man hinh')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def list_images(root):
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                out.append(os.path.join(d, f))
    return sorted(out)


def label_from_path(path):
    # giong LPRDataLoader (co doi 'd' thuong thanh 'Đ')
    return parse_label(path)


def label_is_valid(label):
    return len(label) > 0 and all(c in CHARS_DICT and c != '-' for c in label)


def preprocess(path, img_size, merge):
    """Giong het tien xu ly luc train (LPRDataLoader): ghep 2 dong -> resize -> chuan hoa."""
    try:
        img = read_image(path)  # doc duoc duong dan co dau/khoang trang tren Windows
    except Exception:
        return None
    if img is None:
        return None
    if merge:
        img = merge_two_lines(img)
    h, w = img.shape[:2]
    if h != img_size[1] or w != img_size[0]:
        img = cv2.resize(img, tuple(img_size))
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    return np.transpose(img, (2, 0, 1))  # HWC -> CHW


def greedy_decode(seq):
    """seq: day chi so lop (T,). Gop ky tu lap lien tiep roi bo blank."""
    out, prev = [], BLANK
    for c in seq:
        if c != prev and c != BLANK:
            out.append(int(c))
        prev = c
    return ''.join(CHARS[i] for i in out)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    args = get_args()
    device = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')

    if not os.path.isdir(args.img_dir):
        raise SystemExit('Khong tim thay thu muc: ' + args.img_dir)
    if not os.path.isfile(args.pretrained_model):
        raise SystemExit('Khong tim thay file trong so: ' + args.pretrained_model)

    net = build_lprnet(lpr_max_len=args.lpr_max_len, phase='test', class_num=len(CHARS), dropout_rate=0)
    net.load_state_dict(torch.load(args.pretrained_model, map_location=device))
    net.to(device).eval()  # bat buoc eval(): tat Dropout, BatchNorm dung thong ke da hoc

    # ---- doc danh sach anh va nhan ----
    all_paths = list_images(args.img_dir)
    items, bad_label = [], []
    for p in all_paths:
        lab = label_from_path(p)
        if label_is_valid(lab):
            items.append((p, lab))
        else:
            bad_label.append(p)

    print('=' * 64)
    print('Thu muc anh      :', args.img_dir)
    print('Trong so         :', args.pretrained_model)
    print('Thiet bi         :', device, '| batch size:', args.batch_size,
          '| ghep bien 2 dong:', 'khong' if args.no_merge else 'co')
    print('Tong so anh tim thay        : %d' % len(all_paths))
    print('Anh co nhan hop le          : %d' % len(items))
    if bad_label:
        print('Anh bi bo qua (nhan khong hop le, vi du co dau cach/chu thuong/ky tu la): %d' % len(bad_label))
        for p in bad_label[:10]:
            print('   -', p)
        if len(bad_label) > 10:
            print('   ... va %d file khac' % (len(bad_label) - 10))
    if not items:
        raise SystemExit('Khong co anh nao co nhan hop le de kiem tra.')

    # ---- du doan theo batch ----
    results, unreadable = [], []
    t0 = time.time()
    for s in range(0, len(items), args.batch_size):
        chunk = items[s:s + args.batch_size]
        imgs, ok_items = [], []
        for p, lab in chunk:
            x = preprocess(p, args.img_size, not args.no_merge)
            if x is None:
                unreadable.append(p)
            else:
                imgs.append(x)
                ok_items.append((p, lab))
        if not imgs:
            continue
        batch = torch.from_numpy(np.stack(imgs, 0)).to(device)
        with torch.no_grad():
            logits = net(batch)                                  # N x classes x T
        preds = logits.argmax(dim=1).cpu().numpy()               # N x T
        for (p, lab), seq in zip(ok_items, preds):
            pred = greedy_decode(seq)
            results.append({'path': p, 'label': lab, 'pred': pred, 'ok': pred == lab,
                            'dist': edit_distance(lab, pred)})
    elapsed = time.time() - t0

    if unreadable:
        print('Anh khong doc duoc (bi bo qua): %d' % len(unreadable))
        for p in unreadable[:10]:
            print('   -', p)
    if not results:
        raise SystemExit('Khong co anh nao duoc du doan.')

    # ---- thong ke ----
    n = len(results)
    n_ok = sum(r['ok'] for r in results)
    wrong = [r for r in results if not r['ok']]
    n_wrong_len = sum(len(r['pred']) != len(r['label']) for r in wrong)
    n_wrong_chr = len(wrong) - n_wrong_len
    chars_total = sum(len(r['label']) for r in results)
    char_acc = max(0.0, 1.0 - sum(r['dist'] for r in results) / max(chars_total, 1))

    print('=' * 64)
    print('KET QUA (danh gia ca bien so: dung het cac ky tu moi tinh la dung)')
    print('  Tong so anh da kiem tra : %d' % n)
    print('  Dung                    : %d' % n_ok)
    print('  Sai                     : %d' % len(wrong))
    print('  Do chinh xac            : %.2f%%' % (100.0 * n_ok / n))
    print('  Do chinh xac ky tu      : %.2f%%  (tinh theo khoang cach chinh sua)' % (100.0 * char_acc))
    print('  Trong cac anh sai:')
    print('     sai do dai chuoi (thieu/thua ky tu)   : %d' % n_wrong_len)
    print('     dung do dai nhung sai ky tu            : %d' % n_wrong_chr)
    print('  Toc do: %.1f anh/giay (%.1fs)' % (n / max(elapsed, 1e-9), elapsed))

    by_len = defaultdict(lambda: [0, 0])
    for r in results:
        by_len[len(r['label'])][0] += 1
        by_len[len(r['label'])][1] += r['ok']
    print('-' * 64)
    print('Theo do dai nhan:   do dai | so anh | dung | do chinh xac')
    for L in sorted(by_len):
        tot, ok = by_len[L]
        print('                    %6d | %6d | %4d | %6.2f%%' % (L, tot, ok, 100.0 * ok / tot))

    conf = Counter()
    for r in wrong:
        if len(r['pred']) == len(r['label']):
            for a, b in zip(r['label'], r['pred']):
                if a != b:
                    conf[(a, b)] += 1
    if conf:
        print('-' * 64)
        print('Cac cap ky tu hay nham nhat (that -> doan), chi tinh anh dung do dai:')
        for (a, b), c in conf.most_common(10):
            print('   %s -> %s : %d lan' % (a, b, c))

    if wrong and args.show_wrong > 0:
        print('-' * 64)
        print('Cac anh sai (toi da %d):' % args.show_wrong)
        for r in wrong[:args.show_wrong]:
            print('   that: %-10s doan: %-10s %s' % (r['label'], r['pred'] or '(rong)', r['path']))
        if len(wrong) > args.show_wrong:
            print('   ... va %d anh khac (xem file csv)' % (len(wrong) - args.show_wrong))

    # ---- ghi file ----
    if args.report:
        with open(args.report, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['path', 'label', 'prediction', 'correct', 'edit_distance'])
            for r in results:
                w.writerow([r['path'], r['label'], r['pred'], int(r['ok']), r['dist']])
        print('=' * 64)
        print('Da ghi chi tiet tung anh vao:', args.report)
    if args.copy_wrong and wrong:
        os.makedirs(args.copy_wrong, exist_ok=True)
        for i, r in enumerate(wrong):
            ext = os.path.splitext(r['path'])[1]
            name = '%s__doan-%s__%05d%s' % (r['label'].replace('Đ', 'd'), (r['pred'] or 'RONG').replace('Đ', 'd'), i, ext)
            shutil.copy2(r['path'], os.path.join(args.copy_wrong, name))
        print('Da copy %d anh sai vao: %s' % (len(wrong), args.copy_wrong))

    # ---- tom tat cuoi cung (in sau cung de luon nam o cuoi man hinh terminal) ----
    print('=' * 64)
    print('TOM TAT: tong %d bien' % n)
    print('   Doan DUNG  : %d bien' % n_ok)
    print('   Doan SAI   : %d bien' % len(wrong))
    print('   Ty le dung : %.2f%%' % (100.0 * n_ok / n))
    print('=' * 64)


if __name__ == '__main__':
    main()
