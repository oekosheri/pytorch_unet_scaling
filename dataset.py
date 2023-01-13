from torch.utils.data.dataset import random_split
from torchvision.transforms.transforms import ColorJitter
import torchvision.transforms as trans
import torchvision.transforms.functional as F
import numpy as np
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import cv2

# from PIL import Image


class Resize_with_pad:
    def __init__(self, w=1024, h=768):
        self.w = w
        self.h = h

    def __call__(self, image):

        w_1, h_1 = image.size
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1

        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            wp = int(ratio_f * h_1 - w_1)
            hp = int(w_1 / ratio_f - h_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                image = F.pad(image, (0, hp, 0, hp), 0, "constant")
                return F.resize(image, [self.h, self.w])

            elif hp < 0 and wp > 0:
                wp = wp // 2
                image = F.pad(image, (wp, 0, wp, 0), 0, "constant")
                return F.resize(image, [self.h, self.w])

        else:
            return F.resize(image, [self.h, self.w])


class Segmentation_dataset(Dataset):
    def __init__(self, image_dir, mask_dir, images, masks, augment=False):
        self.images = images
        self.masks = masks
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

    def transfrom(self, image, mask):

        # Resize

        pil_image = trans.ToPILImage()
        resize = Resize_with_pad(w=1024, h=768)
        image = resize(pil_image(image))
        mask = resize(pil_image(mask))

        # augment
        if self.augment:
            jitter = trans.ColorJitter(brightness=0.3, contrast=0.3)
            image = jitter(image)

        # # Random horizontal flipping

        # if random.random() > 0.5:
        # image = F.hflip(image)
        # mask = F.hflip(mask)

        # # Random vertical flipping

        # if random.random() > 0.5:
        #   image = F.vflip(image)
        #   mask = F.vflip(mask)

        # to_tensor
        image = F.to_tensor(image)
        mask = F.to_tensor(mask)
        mask[mask > 0.8] = 1
        mask[mask < 0.2] = 0
        # mask[mask > 0.003] = 1

        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        img_path = self.image_dir + "/" + self.images[index]
        msk_path = self.mask_dir + "/" + self.masks[index]

        # check if cv2 is faster
        img = cv2.imread(img_path).astype(np.float32) / 255.0
        mask = cv2.imread(msk_path).astype(np.float32) / 255.0

        # make sure channel n is 1
        img = img[:, :, 0:1]
        mask = mask[:, :, 0:1]

        # transfrom
        img, mask = self.transfrom(img, mask)

        return img, mask
