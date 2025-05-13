import torch
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import os
import random
import math
import json 

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """change color hue, saturation, value"""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
def random_perspective(combination,  degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    img, gray, line = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
            line = cv2.warpPerspective(line, M, dsize=(width, height), borderValue=0)
        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
            gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)
            line = cv2.warpAffine(line, M[:2], dsize=(width, height), borderValue=0)



    combination = (img, gray, line)
    return combination

def random_perspective2(combination,  degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    img = combination
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))

        else:  # affine
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))




    combination = img
    return combination


class MyDataset(torch.utils.data.Dataset):
    '''
    Class to load the dataset
    '''
    def __init__(self, transform=None, valid=False, engin='kaggle', data='bdd', task='multi', data_path=None, iadd_oversample_ratio=10):
        '''
        :param transform: Type of transformation
        :param valid: Whether this is validation set
        :param engin: 'kaggle' or other (e.g., 'colab')
        :param data: Dataset choice: 'bdd', 'IADD', or 'combined'
        :param task: Task type
        :param data_path: Custom data path if provided
        :param iadd_oversample_ratio: How many times to oversample IADD dataset when in combined mode
        '''

        self.transform = transform
        self.Tensor = transforms.ToTensor()
        self.valid = valid
        self.engin = engin
        self.data = data
        self.task = task
        self.data_path = data_path
        self.iadd_oversample_ratio = iadd_oversample_ratio

        # Initialize empty lists
        self.names = []
        self.roots = []
        self.dataset_indicators = []  # 0 for BDD, 1 for IADD
        
        # Handle paths for BDD dataset
        if self.data == 'bdd' or self.data == 'combined':
            if self.engin == 'kaggle':
                if valid:
                    bdd_root = '/kaggle/input/bdd100k-dataset/bdd100k/bdd100k/images/100k/val'
                else:
                    bdd_root = '/kaggle/input/bdd100k-dataset/bdd100k/bdd100k/images/100k/train'
            else:
                if valid:
                    bdd_root = '/content/data/bdd100k/bdd100k/images/100k/val'
                else:
                    bdd_root = '/content/data/bdd100k/bdd100k/images/100k/train'
            
            bdd_names = os.listdir(bdd_root)
            self.names.extend(bdd_names)
            self.roots.extend([bdd_root] * len(bdd_names))
            self.dataset_indicators.extend([0] * len(bdd_names))
        
        # Handle paths for IADD dataset
        if self.data == 'IADD' or self.data == 'combined':
            if self.engin == 'kaggle':
                if valid:
                    iadd_root = '/kaggle/working/IADD/IADD.v7i.coco-segmentation/valid/img'
                else:
                    iadd_root = '/kaggle/working/IADD/IADD.v7i.coco-segmentation/train/img'
            else:
                if valid:
                    iadd_root = '/content/IADD/IADD.v7i.coco-segmentation/valid/img'
                else:
                    iadd_root = '/content/IADD/IADD.v7i.coco-segmentation/train/img'
            
            iadd_names = os.listdir(iadd_root)
            
            # For combined mode, oversample IADD dataset to balance with BDD
            if self.data == 'combined' and not valid:
                # Create multiple copies of IADD data to balance the datasets
                self.names.extend(iadd_names * self.iadd_oversample_ratio)
                self.roots.extend([iadd_root] * len(iadd_names) * self.iadd_oversample_ratio)
                self.dataset_indicators.extend([1] * len(iadd_names) * self.iadd_oversample_ratio)
            else:
                self.names.extend(iadd_names)
                self.roots.extend([iadd_root] * len(iadd_names))
                self.dataset_indicators.extend([1] * len(iadd_names))
    
        # Print dataset statistics
        if self.data == 'combined':
            bdd_count = sum(1 for indicator in self.dataset_indicators if indicator == 0)
            iadd_count = sum(1 for indicator in self.dataset_indicators if indicator == 1)
            print(f"BDD samples: {bdd_count}, IADD samples: {iadd_count}, Ratio: {iadd_count/bdd_count:.3f}")

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        '''
        :param idx: Index of the image file
        :return: returns the image and corresponding label file.
        '''
        W_ = 512
        H_ = 512
        
        # Get image path based on dataset type
        image_name = os.path.join(self.roots[idx], self.names[idx])
        dataset_indicator = self.dataset_indicators[idx]  # 0 for BDD, 1 for IADD

        image = cv2.imread(image_name)
        
        # Handle label paths based on dataset type
        if dataset_indicator == 0:  # BDD
            if self.engin == 'kaggle':
                label1 = cv2.imread(image_name.replace("input/bdd100k-dataset/bdd100k/bdd100k/images/100k", "working/labels/bdd_seg_gt").replace("jpg", "png"), 0)
                label2 = cv2.imread(image_name.replace("input/bdd100k-dataset/bdd100k/bdd100k/images/100k", "working/labels/bdd_lane_gt").replace("jpg", "png"), 0)
            else:
                label1 = cv2.imread(image_name.replace("bdd100k/bdd100k/images/100k", "labels/bdd_seg_gt").replace("jpg", "png"), 0)
                label2 = cv2.imread(image_name.replace("bdd100k/bdd100k/images/100k", "labels/bdd_lane_gt").replace("jpg", "png"), 0)
        else:  # IADD
            label1 = cv2.imread(image_name.replace("img", "drivable").replace(".jpg", ".png"), 0)
            label2 = cv2.imread(image_name.replace("img", "lane").replace(".jpg", ".png"), 0)

        # Data augmentation for training
        if not self.valid:
            if random.random() < 0.5:
                combination = (image, label1, label2)
                (image, label1, label2) = random_perspective(
                    combination=combination,
                    degrees=10,
                    translate=0.1,
                    scale=0.25,
                    shear=0.0
                )
            if random.random() < 0.5:
                augment_hsv(image)
            if random.random() < 0.5:
                image = np.fliplr(image)
                label1 = np.fliplr(label1)
                label2 = np.fliplr(label2)

        # Resize all images and labels
        label1 = cv2.resize(label1, (W_, H_))
        label2 = cv2.resize(label2, (W_, H_))
        image = cv2.resize(image, (W_, H_))

        # Prepare segmentation masks
        _, seg_b1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg_b2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY_INV)
        _, seg1 = cv2.threshold(label1, 1, 255, cv2.THRESH_BINARY)
        _, seg2 = cv2.threshold(label2, 1, 255, cv2.THRESH_BINARY)

        # Convert to tensors
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_b1 = self.Tensor(seg_b1)
        seg_b2 = self.Tensor(seg_b2)
        seg_da = torch.stack((seg_b1[0], seg1[0]), 0)
        seg_ll = torch.stack((seg_b2[0], seg2[0]), 0)
        
        image = np.ascontiguousarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image_name, image, (seg_da, seg_ll)

# Keep the LaneDataset class unchanged
class LaneDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path="/kaggle/input/tusimple/TUSimple/train_set", train=True, size=(512, 256)):
        self._dataset_path = dataset_path
        self._mode = "train" if train else "eval"
        self._image_size = size # w, h
        self.Tensor = transforms.ToTensor()

        if self._mode == "train":
            label_files = [
                os.path.join(self._dataset_path, f"label_data_{suffix}.json")
                for suffix in ("0313", "0531")
            ]
        elif self._mode == "eval":
            label_files = [
                os.path.join(self._dataset_path, f"label_data_{suffix}.json")
                for suffix in ("0601",)
            ]

        self._data = []

        for label_file in label_files:
            self._process_label_file(label_file)

    def __getitem__(self, idx):
        W_=512
        H_=512
        image_path = os.path.join(self._dataset_path, self._data[idx][0])
        image = cv2.imread(image_path)
        h, w, c = image.shape
        image = cv2.resize(image, (W_, H_))
        lanes = self._data[idx][1]

        segmentation_image = self._draw(h, w, lanes, "segmentation")
        segmentation_image = cv2.resize(segmentation_image, (W_, H_))

        image = torch.from_numpy(image).float().permute((2, 0, 1))

        return image_path, image, (segmentation_image, segmentation_image)
    
    def __len__(self):
        return len(self._data)

    def _draw(self, h, w, lanes, image_type):
        image = np.zeros((h, w), dtype=np.uint8)
        for i, lane in enumerate(lanes):
            color = 1 if image_type == "segmentation" else i + 1
            cv2.polylines(image, [lane], False, color, 10)

        image = cv2.resize(image, self._image_size, interpolation=cv2.INTER_NEAREST)

        return image

    def _process_label_file(self, file_path):
        with open(file_path) as f:
            for line in f:
                info = json.loads(line)
                image = info["raw_file"]
                lanes = info["lanes"]
                h_samples = info["h_samples"]
                lanes_coords = []
                for lane in lanes:
                    x = np.array([lane]).T
                    y = np.array([h_samples]).T
                    xy = np.hstack((x, y))
                    idx = np.where(xy[:, 0] > 0)
                    lane_coords = xy[idx]
                    lanes_coords.append(lane_coords)
                self._data.append((image, lanes_coords))