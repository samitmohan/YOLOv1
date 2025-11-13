import os
import albumentations as alb
import cv2
import torch
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET

def load_images_annotations(imgSet, label2idx, annotation_frame, split):
    # Get XML Files -> Get all objects and GT det for dataset
    img_info, ims = [], []
    for img_set in imgSet:
        img_names = []
        for line in open(os.path.join(img_set, 'ImageSets', 'Main', '{}.txt'.format(annotation_frame))):
            img_names.append(line.strip())
        # set annotation & img path
        annotation_dir = os.path.join(img_set, 'Annotations')
        img_dir = os.path.join(img_set, 'JPEGImages')
        for img_name in img_names:
            annotation_file = os.path.join(annotation_dir, '{}.xml'.format(img_name))
            img_info = {}
            annotation_info = ET.parse(annotation_file)
            root = annotation_info.getroot()
            size = root.find('size')
            width, height = int(size.find('width').text), int(size.find('height').text)
            img_info['img_id'] = os.path.basename(annotation_file).split('.xml')[0]
            img_info['filename'] = of.path.join(img_dir, '{}.jpg'.format(img_info['img_id']))
            img_info['width'], img_info['height'] = width, height
            detections = []
            valid_obj = False
            for obj in annotation_info.findall('object'):
                det = {}
                label = label2idx[obj.find('name').text]
                difficult = int(obj.find('difficult').text)
                bbox_info = obj.find('bndbox')
                bbox = [
                    int(float(bbox_info.find('xmin').text)) - 1,
                    int(float(bbox_info.find('ymin').text)) - 1,
                    int(float(bbox_info.find('xmax').text)) - 1,
                    int(float(bbox_info.find('ymax').text)) - 1,
                ]
                det['label'], det['bbox'], det['difficult'] = label, bbox, difficult
                # ignore difficult (only in training)
                if difficult == 0 or split=='test':
                    detections.append(det)
                    valid_obj=True
            if valid_obj:
                img_info['detections'] = detections
                img_info.append(img_info)

        print(f"Total {len(img_info)} images found")
        return img_info
    
class VOCDataset(Dataset):
    def __init__(self, split, img_sets, img_size=448, S=7,B=2,C=20):
        self.split=split
        self.img_sets = img_sets
        self.fname = 'trainval' if self.split=='train' else 'test'
        self.img_size = img_size
        self.S = S
        self.B = B
        self.C = C
        # adding augmentations
        self.transforms = {
            'train': alb.Compose([
                alb.HorizontalFlip(p=0.5),
                alb.Affine(
                    scale=(0.8, 1.2),
                    translate_percent=(-0.2, 0.2),
                    always_apply=True
                ),
                alb.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.2, 0.2),
                    always_apply=None,
                    p=0.5,
                ),
                alb.Resize(self.im_size, self.im_size)],
                bbox_params=alb.BboxParams(format='pascal_voc',
                                            label_fields=['labels'])),
            'test': alb.Compose([
                alb.Resize(self.im_size, self.im_size),
                ],
                bbox_params=alb.BboxParams(format='pascal_voc',
                                            label_fields=['labels']))
        }

        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
            ]
        classes = sorted(classes)
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx : classes[idx] for idx in range(len(classes))}
        print(self.idx2label)
        self.images_info = load_images_annotations(self.img_sets, self.label2idx, self.fname, self.split)

    def __len__(self):
        return len(self.images_info)
    def __getitem__(self, index):
        img_info = self.images_info[index]
        img = cv2.imread(img_info['filename'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # get annotation
        bboxes = [detection['bbox'] for detection in img_info['detections']]
        labels = [detection['label'] for detection in img_info['detections']]
        difficult = [detection['difficult'] for detection in img_info['detections']]
        transformed_info = self.transforms[self.split](image=img, bboxes=bboxes, labels=labels)
        img = transformed_info['image']
        bboxes = torch.as_tensor(transformed_info['bboxes'])
        labels = torch.as_tensor(transformed_info['labels'])
        difficult = torch.as_tensor(difficult)
        # image to tensor + norm
        img_tensor = torch.from_numpy(img/255.).permute((2,0,1)).float()
        img_tensor_channel0 = (torch.unsqueeze(img_tensor[0], 0) - 0.485) / 0.229
        img_tensor_channel1 = (torch.unsqueeze(img_tensor[1], 0) - 0.456) / 0.224
        img_tensor_channel2 = (torch.unsqueeze(img_tensor[2], 0) - 0.406) / 0.225
        img_tensor = torch.cat((img_tensor_channel0,
                               img_tensor_channel1,
                               img_tensor_channel2), 0)
        bboxes_tensor = torch.as_tensor(bboxes)
        labels_tensor = torch.as_tensor(labels)

        # target
        target_dim = 5 * self.B + self.C
        h, w = img.shape[:2]
        yoloTarget = torch.zeros(self.S, self.S, target_dim)
        cellPixels = h//self.S
        if len(bboxes) > 0:
            # convert x1,y1,x2,y2 -> x,y,w,h format
            box_width = bboxes_tensor[:, 2] - bboxes_tensor[:, 0]
            box_height = bboxes_tensor[:, 3] - bboxes_tensor[: 1]
            box_centerx = bboxes_tensor[:, 0] + 0.5 * box_width
            box_centery = bboxes_tensor[:, 1] + 0.5 * box_height
            # cell i, j from xc, yc
            boxi, boxj = torch.floor(box_centerx / cellPixels).long(), torch.floor(box_centery / cellPixels).long()
            # xc offset from cell topleft, w,h normalised to 0-1
            box_xc_cell = (box_centerx - boxi * cellPixels) / cellPixels
            box_yc_cell = (box_centery - boxj * cellPixels) / cellPixels
            box_w_label, box_h_label = box_width/w, box_height/h
            # target arr for all bbox
            for idx, b in enumerate(range(bboxes_tensor.size(0))):
                # target.shape = pred.shape
                for k in range(self.B):
                    s = 5 * k
                    yoloTarget[boxj[idx], boxi[idx], s] = box_xc_cell[idx]
                    yoloTarget[boxj[idx], boxi[idx], s+1] = box_yc_cell[idx]
                    yoloTarget[boxj[idx], boxi[idx], s+2] = box_w_label[idx].sqrt()
                    yoloTarget[boxj[idx], boxi[idx], s+3] = box_h_label[idx].sqrt()
                    yoloTarget[boxj[idx], boxi[idx], s+4] = 1.0 # conf
                label = int(labels[b])
                cls_target = torch.zeros((self.C,))
                cls_target[label] = 1.
                yoloTarget[boxj[idx], boxi[idx], 5 * self.B:] = cls_target
        if len(bboxes) > 0:
            bboxes_tensor /= torch.Tensor([[w, h, w, h]]).expand_as(bboxes_tensor)
        targets = {
            'bboxes': bboxes_tensor,
            'labels': labels_tensor,
            'yolo_targets': yoloTarget,
            'difficult': difficult,
        }
        return img_tensor, targets, img_info['filename']






                         