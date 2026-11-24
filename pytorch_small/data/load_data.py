from torch.utils.data import Dataset
from imutils import paths
import numpy as np
import random
import cv2
import os
import torch

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {char: i for i, char in enumerate(CHARS)}

def merge_two_lines(img):
    """Biển 2 dòng (xe máy, ô tô vuông): cắt đôi rồi ghép thành 1 dải ngang.
    Dòng trên nằm bên trái, dòng dưới nằm bên phải.
    Biển 1 dòng (tỉ lệ rộng/cao > 2) giữ nguyên.
    Khi inference phải dùng đúng hàm này."""
    h, w = img.shape[:2]
    if w / h > 2:
        return img
    h2 = h // 2
    top, bottom = img[:h2], img[h2:2 * h2]
    return np.hstack([top, bottom])


class LPRDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, lpr_max_len, PreprocFun=None):
        # img_dir phải là LIST các thư mục, ví dụ ["data/train"]
        if isinstance(img_dir, str):
            img_dir = [img_dir]
        self.img_dir = img_dir
        self.img_paths = []
        for d in img_dir:
            self.img_paths += [el for el in paths.list_images(d)]
        random.shuffle(self.img_paths)
        self.img_size = imgSize  # (rộng, cao) = (94, 24)
        self.lpr_max_len = lpr_max_len
        self.PreprocFun = PreprocFun if PreprocFun is not None else self.transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        Image = cv2.imread(filename)
        if Image is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {filename}")

        # Ghép biển 2 dòng thành 1 dải ngang trước khi resize
        Image = merge_two_lines(Image)

        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        Image = self.PreprocFun(Image)

        # Nhãn nằm trong tên file: 30A12345.jpg, 30A12345_0002.jpg
        basename = os.path.basename(filename)
        imgname, _ = os.path.splitext(basename)
        imgname = imgname.split("-")[0].split("_")[0]

        label = []
        for c in imgname:
            if c not in CHARS_DICT or c == '-':
                raise ValueError(f"Ký tự '{c}' không hợp lệ trong nhãn của file: {filename}")
            label.append(CHARS_DICT[c])

        if len(label) == 0 or len(label) > self.lpr_max_len:
            raise ValueError(f"Độ dài nhãn {len(label)} không hợp lệ (tối đa {self.lpr_max_len}): {filename}")

        return Image, label, len(label)

    def transform(self, img):
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img


def collate_fn(batch):
    imgs, labels, lengths = [], [], []
    for img, label, length in batch:
        imgs.append(torch.from_numpy(img))
        labels.extend(label)
        lengths.append(length)
    return (torch.stack(imgs, 0),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long))


if __name__ == "__main__":
    # Ví dụ sử dụng
    from torch.utils.data import DataLoader

    train_set = LPRDataLoader(["data/train"], (94, 24), 9)
    loader = DataLoader(train_set, batch_size=64, shuffle=True,
                        num_workers=2, collate_fn=collate_fn)

    imgs, labels, lengths = next(iter(loader))
    print(imgs.shape, labels.shape, lengths.shape)