# -*- coding: utf-8 -*-
# /usr/bin/env/python3

'''
Pytorch implementation for LPRNet.
Author: aiboy.wei@outlook.com .

Bản đã sửa để chạy được với PyTorch / NumPy mới, gõ tham số từ command line,
và xuất biểu đồ train loss + test accuracy sau khi train xong.
Các chỗ sửa đều có chú thích "# FIX" hoặc "# NEW".
'''

from data.load_data_elec import CHARS, CHARS_DICT, LPRDataLoader
from LPRNet import build_lprnet
# import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import *
from torch import optim
import torch.nn as nn
import numpy as np
import argparse
import torch
import time
import math
import csv
import os

def sparse_tuple_for_ctc(T_length, lengths):
    input_lengths = []
    target_lengths = []

    for ch in lengths:
        input_lengths.append(T_length)
        target_lengths.append(ch)

    return tuple(input_lengths), tuple(target_lengths)

def adjust_learning_rate(optimizer, cur_epoch, base_lr, lr_schedule):
    """
    Sets the learning rate
    """
    lr = 0
    for i, e in enumerate(lr_schedule):
        if cur_epoch < e:
            lr = base_lr * (0.1 ** i)
            break
    if lr == 0:
        # FIX: bản gốc quay lại base_lr khi qua mốc cuối; giữ mức đã giảm
        lr = base_lr * (0.1 ** len(lr_schedule))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def str2bool(s):
    # FIX: type=bool của argparse coi "False" cũng là True
    return str(s).lower() in ('1', 'true', 'yes', 'y')

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    # FIX: thêm type=... để giá trị gõ từ command line không bị hiểu là chuỗi
    parser.add_argument('--max_epoch', default=15, type=int, help='epoch to train the network')
    parser.add_argument('--img_size', default=[94, 24], type=int, nargs=2, help='the image size: width height')
    parser.add_argument('--train_img_dirs', default="~/workspace/trainMixLPR", help='the train images path')
    parser.add_argument('--val_img_dirs', default="", help='NEW: thu muc val: dung de tinh val loss/accuracy moi epoch va chon model tot nhat. De trong = dung tap test thay the')
    parser.add_argument('--test_img_dirs', default="~/workspace/testMixLPR", help='NEW: thu muc test: chi danh gia 1 lan o cuoi bang model tot nhat')
    parser.add_argument('--dropout_rate', default=0.5, type=float, help='dropout rate.')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='base value of learning rate.')
    # NEW: bien so co the dai toi 10 ky tu (vi du 60MĐ1 01658) nen mac dinh 10
    parser.add_argument('--lpr_max_len', default=10, type=int, help='license plate number max length.')
    parser.add_argument('--train_batch_size', default=128, type=int, help='training batch size.')
    parser.add_argument('--test_batch_size', default=120, type=int, help='testing batch size.')
    parser.add_argument('--phase_train', default=True, type=str2bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
    parser.add_argument('--cpu', action='store_true', help='Ép chạy bằng CPU')  # FIX
    parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
    # NEW: mặc định đánh giá sau mỗi epoch (điểm trên biểu đồ accuracy); --test_interval theo iteration, 0 = tắt
    parser.add_argument('--eval_epoch', default=1, type=int, help='đánh giá test sau mỗi N epoch (0 = tắt)')
    parser.add_argument('--test_interval', default=0, type=int, help='đánh giá thêm sau mỗi N iteration (0 = tắt)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=2e-5, type=float, help='Weight decay for SGD')
    parser.add_argument('--lr_schedule', default=[4, 8, 12, 14, 16], type=int, nargs='+', help='schedule for learning rate.')
    parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
    # parser.add_argument('--pretrained_model', default='./weights/Final_LPRNet_model.pth', help='pretrained base model')
    parser.add_argument('--pretrained_model', default='', help='pretrained base model')

    args = parser.parse_args()
    # FIX: tự chuyển sang CPU nếu máy không có CUDA
    args.cuda = args.cuda and (not args.cpu) and torch.cuda.is_available()

    return args

def collate_fn(batch):
    imgs = []
    labels = []
    lengths = []
    for _, sample in enumerate(batch):
        img, label, length = sample
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    labels = np.asarray(labels).flatten().astype(np.int64)  # FIX: np.int đã bị xóa trong NumPy mới

    return (torch.stack(imgs, 0), torch.from_numpy(labels), lengths)

def save_training_curves(history, epoch_size, save_folder, test_result=None):
    """NEW: lưu 2 file csv và ảnh biểu đồ train loss + test accuracy (trục x tính theo epoch)."""
    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(save_folder, 'train_loss.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration', 'epoch', 'loss'])
        for it, loss in zip(history['iter'], history['loss']):
            w.writerow([it, '%.4f' % (it / epoch_size), '%.6f' % loss])
    # NEW: them cot val_loss va train_loss_epoch (loss train trung binh trong epoch vua xong) de so sanh
    with open(os.path.join(save_folder, 'test_accuracy.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['iteration', 'epoch', 'accuracy', 'val_loss', 'train_loss_epoch'])
        for it, acc, vl, tl in zip(history['eval_iter'], history['eval_acc'], history['eval_loss'], history['eval_train_loss']):
            w.writerow([it, '%.4f' % (it / epoch_size), '%.6f' % acc, '%.6f' % vl, '%.6f' % tl])

    try:
        import matplotlib
        matplotlib.use('Agg')  # không cần cửa sổ, chỉ lưu ảnh
        import matplotlib.pyplot as plt
    except ImportError:
        print("Chưa cài matplotlib nên chưa vẽ được biểu đồ (pip install matplotlib). "
              "Dữ liệu đã lưu trong train_loss.csv và test_accuracy.csv")
        return None

    fig, (ax1, ax3, ax2) = plt.subplots(1, 3, figsize=(19, 4.8))  # NEW: them o giua cho train/val loss

    # --- Train loss ---
    if history['loss']:
        x = np.asarray(history['iter'], dtype=np.float64) / epoch_size
        y = np.asarray(history['loss'], dtype=np.float64)
        ax1.plot(x, y, color='tab:blue', alpha=0.25, linewidth=1, label='Mỗi iteration')
        win = int(min(max(epoch_size, 5), len(y)))
        if win > 1:
            smooth = np.convolve(y, np.ones(win) / win, mode='valid')
            ax1.plot(x[win - 1:], smooth, color='tab:blue', linewidth=2, label='Trung bình trượt (%d iter)' % win)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Chưa có dữ liệu', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('Train loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('CTC loss')
    ax1.grid(True, alpha=0.3)

    # --- Test accuracy ---
    if history['eval_acc']:
        xe = np.asarray(history['eval_iter'], dtype=np.float64) / epoch_size
        ye = np.asarray(history['eval_acc'], dtype=np.float64) * 100
        ax2.plot(xe, ye, color='tab:green', marker='o', markersize=3, linewidth=1.8)
        best = int(np.argmax(ye))
        ax2.scatter([xe[best]], [ye[best]], color='tab:red', zorder=5)
        ax2.text(0.97, 0.05, 'Tốt nhất: %.2f%% (epoch %.1f)\nCuối cùng: %.2f%%' % (ye[best], xe[best], ye[-1]),
                 transform=ax2.transAxes, ha='right', va='bottom', color='tab:red',
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='tab:red', alpha=0.9))
        ax2.set_ylim(0, 100)
    else:
        ax2.text(0.5, 0.5, 'Chưa có dữ liệu', ha='center', va='center', transform=ax2.transAxes)
    # --- NEW: Train loss vs Val loss (cung cach tinh, cung che do eval) -> nhan biet overfit ---
    if history['eval_loss']:
        xv = np.asarray(history['eval_iter'], dtype=np.float64) / epoch_size
        yv = np.asarray(history['eval_loss'], dtype=np.float64)
        yt = np.asarray(history['eval_train_loss'], dtype=np.float64)
        ax3.plot(xv, yt, color='tab:blue', marker='o', markersize=3, linewidth=1.8, label='Train loss (TB moi epoch)')
        ax3.plot(xv, yv, color='tab:orange', marker='o', markersize=3, linewidth=1.8, label='Val loss')
        bv = int(np.argmin(yv))
        ax3.scatter([xv[bv]], [yv[bv]], color='tab:red', zorder=5)
        ax3.text(0.97, 0.95, 'Val loss thap nhat: %.4f (epoch %.1f)\nCuoi: train %.4f | val %.4f' % (yv[bv], xv[bv], yt[-1], yv[-1]),
                 transform=ax3.transAxes, ha='right', va='top', color='tab:red', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', edgecolor='tab:red', alpha=0.9))
        ax3.legend(loc='center right')
        ax3.set_ylim(0, min(max(yv.max(), yt.max()) * 1.05, 10 * max(np.median(yv), 1e-3)))
    else:
        ax3.text(0.5, 0.5, 'Chua co du lieu', ha='center', va='center', transform=ax3.transAxes)
    ax3.set_title('Train loss vs Val loss (val tang len = overfit)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('CTC loss')
    ax3.grid(True, alpha=0.3)

    ax2.set_title('Val accuracy (đúng cả biển số)' + ('' if not test_result else '\nTest cuối: %.2f%%' % (100 * test_result[0])))
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(save_folder, 'training_curves.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Đã lưu biểu đồ:", out_path)
    return out_path

def train():
    args = get_parser()

    T_length = 18 # args.lpr_max_len  (so buoc thoi gian cua dau ra model; CTC can T >= do dai nhan + so cap ky tu lap lien tiep)
    # NEW: bang ky tu co them chu Đ => 38 lop, blank (ky tu '-') la lop cuoi (index 37)
    print("So lop: %d (blank CTC = %d) | do dai nhan toi da: %d" % (len(CHARS), len(CHARS) - 1, args.lpr_max_len))
    if args.lpr_max_len > T_length:
        raise SystemExit("--lpr_max_len (%d) khong duoc lon hon so buoc thoi gian T=%d" % (args.lpr_max_len, T_length))
    epoch = 0 + args.resume_epoch
    loss_val = 0

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # FIX: bản gốc truyền phase=True nên build_lprnet trả về model ở chế độ eval khi đang train
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase="train", class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    lprnet.train()
    print("Device:", device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        try:
            lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        except RuntimeError as e:
            # NEW: trong so cu (37 lop, khong co Đ) khong khop model moi (38 lop) -> phai train lai tu dau
            raise SystemExit("Khong load duoc --pretrained_model: so lop da doi (them chu Đ: 37 -> 38). "
                             "Hay bo --pretrained_model va train lai tu dau.\nChi tiet: %s" % str(e).splitlines()[0])
        print("load pretrained model successful!")
    else:
        def xavier(param):
            nn.init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        nn.init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = xavier(1)
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0.01

        lprnet.backbone.apply(weights_init)
        lprnet.container.apply(weights_init)
        print("initial net weights successful!")

    # define optimizer
    # optimizer = optim.SGD(lprnet.parameters(), lr=args.learning_rate,
    #                       momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.RMSprop(lprnet.parameters(), lr=args.learning_rate, alpha = 0.9, eps=1e-08,
                         momentum=args.momentum, weight_decay=args.weight_decay)
    train_img_dirs = os.path.expanduser(args.train_img_dirs)
    test_img_dirs = os.path.expanduser(args.test_img_dirs)
    train_dataset = LPRDataLoader(train_img_dirs.split(','), args.img_size, args.lpr_max_len)
    test_dataset = LPRDataLoader(test_img_dirs.split(','), args.img_size, args.lpr_max_len)
    # NEW: tap val rieng (neu khong chi dinh thi dung test nhu truoc)
    if args.val_img_dirs:
        val_dataset = LPRDataLoader(os.path.expanduser(args.val_img_dirs).split(','), args.img_size, args.lpr_max_len)
    else:
        val_dataset = test_dataset
    print("Số ảnh -> train: %d | val: %d | test: %d" % (len(train_dataset), len(val_dataset), len(test_dataset)))
    # NEW: bao loi ro rang neu duong dan sai / thu muc rong
    for _n, _d, _p in (('train', train_dataset, train_img_dirs), ('val', val_dataset, args.val_img_dirs or test_img_dirs)):
        if len(_d) == 0:
            raise SystemExit("Tap %s khong co anh nao. Kiem tra duong dan: %s" % (_n, _p))
    if len(test_dataset) == 0:
        print("CANH BAO: tap test rong (duong dan: %s). Se bo qua buoc danh gia test cuoi. Hay truyen --test_img_dirs dung." % test_img_dirs)

    epoch_size = len(train_dataset) // args.train_batch_size
    max_iter = args.max_epoch * epoch_size
    if epoch_size == 0:  # FIX: báo lỗi rõ thay vì chia cho 0
        raise SystemExit("Số ảnh train (%d) ít hơn batch size (%d) hoặc sai đường dẫn: %s"
                         % (len(train_dataset), args.train_batch_size, train_img_dirs))

    ctc_loss = nn.CTCLoss(blank=len(CHARS)-1, reduction='mean') # reduction: 'none' | 'mean' | 'sum'

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    # NEW: lưu lịch sử để vẽ biểu đồ
    history = {'iter': [], 'loss': [], 'eval_iter': [], 'eval_acc': [], 'eval_loss': [], 'eval_train_loss': []}
    state = {'last_eval': -1, 'best_acc': -1.0, 'best_val_loss': float('inf'), 'best_val_loss_epoch': 0.0}  # NEW: them best_val_loss

    def run_eval(it_done):
        acc, val_loss = Greedy_Decode_Eval(lprnet, val_dataset, args, ctc_loss, T_length)
        history['eval_iter'].append(it_done)
        history['eval_acc'].append(acc)
        history['eval_loss'].append(val_loss)
        # NEW: loss train trung binh cua cac iteration thuoc epoch vua xong (hoac 20 iter gan nhat neu eval theo iteration)
        n_recent = max(1, min(epoch_size, len(history['loss'])))
        history['eval_train_loss'].append(float(np.mean(history['loss'][-n_recent:])) if history['loss'] else float('nan'))
        print("[Info] Val loss: %.4f | Train loss (TB epoch): %.4f" % (val_loss, history['eval_train_loss'][-1]))
        state['last_eval'] = it_done
        # NEW: chi luu model khi test accuracy tot nhat tu truoc den gio
        if acc > state['best_acc']:
            state['best_acc'] = acc
            best_path = os.path.join(args.save_folder, 'Best_LPRNet_model.pth')
            torch.save(lprnet.state_dict(), best_path)
            print("Da luu model tot nhat (acc=%.4f): %s" % (acc, best_path))
        # NEW: luu them model co VAL LOSS thap nhat (doc lap voi model co accuracy cao nhat)
        if math.isfinite(val_loss) and val_loss < state['best_val_loss']:
            state['best_val_loss'] = val_loss
            state['best_val_loss_epoch'] = it_done / epoch_size
            bv_path = os.path.join(args.save_folder, 'Best_valloss_LPRNet_model.pth')
            torch.save(lprnet.state_dict(), bv_path)
            print("Da luu model val loss thap nhat (val_loss=%.4f, epoch %.1f): %s" % (val_loss, it_done / epoch_size, bv_path))

    it_done = start_iter
    interrupted = False
    try:
        for iteration in range(start_iter, max_iter):
            if iteration % epoch_size == 0:
                # create batch iterator
                batch_iterator = iter(DataLoader(train_dataset, args.train_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))
                loss_val = 0
                epoch += 1

            if args.test_interval > 0 and (iteration + 1) % args.test_interval == 0:
                run_eval(iteration)

            start_time = time.time()
            # load train data
            images, labels, lengths = next(batch_iterator)
            # labels = np.array([el.numpy() for el in labels]).T
            # print(labels)
            # get ctc parameters
            input_lengths, target_lengths = sparse_tuple_for_ctc(T_length, lengths)
            # update lr
            lr = adjust_learning_rate(optimizer, epoch, args.learning_rate, args.lr_schedule)

            if args.cuda:
                images = Variable(images, requires_grad=False).cuda()
                labels = Variable(labels, requires_grad=False).cuda()
            else:
                images = Variable(images, requires_grad=False)
                labels = Variable(labels, requires_grad=False)

            # forward
            logits = lprnet(images)
            log_probs = logits.permute(2, 0, 1) # for ctc loss: T x N x C
            # print(labels.shape)
            log_probs = log_probs.log_softmax(2).requires_grad_()
            # log_probs = log_probs.detach().requires_grad_()
            # print(log_probs.shape)
            # backprop
            optimizer.zero_grad()
            loss = ctc_loss(log_probs, labels, input_lengths=input_lengths, target_lengths=target_lengths)
            if math.isfinite(loss.item()):  # FIX: bỏ qua cả nan lẫn inf
                loss.backward()
                optimizer.step()
                loss_val += loss.item()
                history['iter'].append(iteration + 1)  # NEW
                history['loss'].append(loss.item())
            else:
                print("Bỏ qua batch có loss không hợp lệ (nan/inf)")
            end_time = time.time()
            it_done = iteration + 1
            if iteration % 20 == 0:
                print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                      + '|| Totel iter ' + repr(iteration) + ' || Loss: %.4f||' % (loss.item()) +
                      'Batch time: %.4f sec. ||' % (end_time - start_time) + 'LR: %.8f' % (lr))

            # NEW: đánh giá cuối mỗi epoch -> một điểm trên biểu đồ accuracy
            if args.eval_epoch > 0 and it_done % epoch_size == 0 and (it_done // epoch_size) % args.eval_epoch == 0:
                run_eval(it_done)
    except KeyboardInterrupt:
        interrupted = True
        print("\nĐã dừng sớm bằng Ctrl+C, đang lưu model và biểu đồ...")

    # final test
    if not interrupted and state['last_eval'] != it_done:
        print("Final test Accuracy:")
        run_eval(it_done)

    # save final parameters
    torch.save(lprnet.state_dict(), os.path.join(args.save_folder, 'Final_LPRNet_model.pth'))

    # NEW: ve bieu do NGAY (truoc buoc test) de loi o buoc test khong lam mat bieu do
    save_training_curves(history, epoch_size, args.save_folder)

    # NEW: danh gia TAP TEST mot lan bang model tot nhat (chon theo val)
    test_result = None
    best_path = os.path.join(args.save_folder, 'Best_LPRNet_model.pth')
    if args.val_img_dirs and len(test_dataset) > 0 and os.path.isfile(best_path):
        lprnet.load_state_dict(torch.load(best_path, map_location=device))
        print("=== KET QUA TREN TAP TEST (model tot nhat theo val) ===")
        try:
            test_result = Greedy_Decode_Eval(lprnet, test_dataset, args, ctc_loss, T_length)
        except Exception as e:
            print("Loi khi danh gia test:", e)
            test_result = None
    if test_result is not None:
        print("TEST accuracy: %.2f%% | test loss: %.4f" % (100 * test_result[0], test_result[1]))
    print("Đã lưu model:", os.path.join(args.save_folder, 'Final_LPRNet_model.pth'))
    # NEW: tom tat cac model da luu
    print("Best_LPRNet_model.pth          : accuracy cao nhat = %.2f%%" % (100 * max(state['best_acc'], 0)))
    if math.isfinite(state['best_val_loss']):
        print("Best_valloss_LPRNet_model.pth  : val loss thap nhat = %.4f (epoch %.1f)" % (state['best_val_loss'], state['best_val_loss_epoch']))

    # NEW: xuất biểu đồ
    save_training_curves(history, epoch_size, args.save_folder, test_result)

def Greedy_Decode_Eval(Net, datasets, args, ctc_loss=None, T_length=18):
    Net.eval()  # FIX: đánh giá ở chế độ eval (tắt, BN dùng thống kê đã học)
    # FIX: dùng ceil để tập nhỏ hơn 1 batch vẫn được đánh giá (bản gốc chia cho 0)
    epoch_size = math.ceil(len(datasets) / args.test_batch_size)
    batch_iterator = iter(DataLoader(datasets, args.test_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn))

    Tp = 0
    Tn_1 = 0
    Tn_2 = 0
    loss_sum, n_loss = 0.0, 0  # NEW: cong don val loss (co trong so theo so mau cua batch)
    t1 = time.time()
    with torch.no_grad():
        for i in range(epoch_size):
            # load train data
            images, labels, lengths = next(batch_iterator)
            start = 0
            targets = []
            for length in lengths:
                label = labels[start:start+length]
                targets.append(label)
                start += length
            targets = [el.numpy() for el in targets]  # FIX: np.array() trên danh sách độ dài khác nhau bị lỗi ở NumPy mới

            if args.cuda:
                images = Variable(images.cuda())
            else:
                images = Variable(images)

            # forward
            prebs = Net(images)
            # NEW: val loss tren tap val (cung cong thuc CTC voi luc train, che do eval, khong backward)
            if ctc_loss is not None:
                lp = prebs.permute(2, 0, 1).log_softmax(2)
                tgt = torch.from_numpy(np.concatenate(targets).astype(np.int64)).to(lp.device)
                il, tl = sparse_tuple_for_ctc(T_length, lengths)
                bl = ctc_loss(lp, tgt, input_lengths=il, target_lengths=tl).item()
                if math.isfinite(bl):
                    loss_sum += bl * len(lengths)
                    n_loss += len(lengths)
            # greedy decode
            prebs = prebs.cpu().detach().numpy()
            preb_labels = list()
            for i in range(prebs.shape[0]):
                preb = prebs[i, :, :]
                preb_label = list()
                for j in range(preb.shape[1]):
                    preb_label.append(np.argmax(preb[:, j], axis=0))
                no_repeat_blank_label = list()
                pre_c = preb_label[0]
                if pre_c != len(CHARS) - 1:
                    no_repeat_blank_label.append(pre_c)
                for c in preb_label: # dropout repeate label and blank label
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(c)
                    pre_c = c
                preb_labels.append(no_repeat_blank_label)
            for i, label in enumerate(preb_labels):
                if len(label) != len(targets[i]):
                    Tn_1 += 1
                    continue
                if (np.asarray(targets[i]) == np.asarray(label)).all():
                    Tp += 1
                else:
                    Tn_2 += 1

    Net.train()  # FIX: quay lại chế độ train
    total = max(Tp + Tn_1 + Tn_2, 1)
    Acc = Tp * 1.0 / total
    print("[Info] Test Accuracy: {} [{}:{}:{}:{}]".format(Acc, Tp, Tn_1, Tn_2, (Tp+Tn_1+Tn_2)))
    t2 = time.time()
    print("[Info] Test Speed: {}s 1/{}]".format((t2 - t1) / max(len(datasets), 1), len(datasets)))
    val_loss = loss_sum / n_loss if n_loss > 0 else float('nan')
    return Acc, val_loss  # NEW: tra them val loss


if __name__ == "__main__":
    train()