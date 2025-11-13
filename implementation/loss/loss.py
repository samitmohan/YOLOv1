import torch
import torch.nn as nn

def iou(b1, b2):
    # Box : (x1,y1,x2,y2)
    # Area = (x2 - x1) * (y2 - y1)
    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    # top left x1, y1 coordinates
    x_topleft = torch.max(b1[..., 0], b2[..., 0])
    y_topleft = torch.max(b1[..., 1], b2[..., 1])
    x_bottomright = torch.min(b1[..., 2], b2[..., 2])
    y_bottomright = torch.min(b1[..., 3], b2[..., 3])
    intersection_area = (x_bottomright - x_topleft).clamp(min=0)
    union_area = area1.clamp(min=0) + area2.clamp(min=0) - intersection_area
    iou = intersection_area / (union_area + 1E-6) # for error
    return iou

class YOLOLoss(nn.Module):
    '''Localisation loss, Objectness loss, Classification loss'''
    def __init__(self, S=7, B=2, C=20):
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5
        self.lambda_no_obj = 0.5 # as given in paper

    def forward(self, preds, target, use_sigmoid=False):
        '''
        preds: (Batch, S*S*(5B+C)) tensor
        targets: (Batch, S, S, (5B+C)) tensor.
        '''
        batch_size = preds.size(0)
        # shape: batch, S, S, 5B+c
        preds = preds.reshape(batch_size, self.S, self.S, 5 * self.B + self.C)
        if use_sigmoid:
            preds[..., :5 * self.B] = torch.nn.functional.sigmoid(preds[..., :5 * self.B])        
        
        # Shifting from xcenter, ycenter -> x1, y1 , x2, y2 (normalised 0-1)
        xshift = torch.arange(0, self.S) * 1 / float(self.S) # S * (1 / S)
        yshift = torch.arange(0, self.S) * 1 / float(self.S) # S * (1 / S)
        # create grid 
        shifty, shiftx = torch.meshgrid(xshift, yshift, indexing='ij')
        # shifts -> [1, S, S, B]
        xshift = xshift.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)
        yshift = yshift.reshape((1, self.S, self.S, 1)).repeat(1, 1, 1, self.B)

        # pred = batchsize, S, S, B, 5
        pred_boxes = preds[..., :5 * self.B].reshape(batch_size, self.S, self.S, self.B, -1)

        # xcenter, ycenter, w, h -> x1 y1 x2 y2 (normalized 0-1)
        # x_center = (xcenter / S + xshift)
        # x1 = x_center - 0.5 * w
        # x2 = x_center + 0.5 * w
        pred_boxes_x1 = ((pred_boxes[..., 0]/self.S + xshift)
                         - 0.5*torch.square(pred_boxes[..., 2]))
        pred_boxes_x1 = pred_boxes_x1[..., None]
        pred_boxes_y1 = ((pred_boxes[..., 1]/self.S + yshift)
                         - 0.5*torch.square(pred_boxes[..., 3]))
        pred_boxes_y1 = pred_boxes_y1[..., None]
        pred_boxes_x2 = ((pred_boxes[..., 0]/self.S + xshift)
                         + 0.5*torch.square(pred_boxes[..., 2]))
        pred_boxes_x2 = pred_boxes_x2[..., None]
        pred_boxes_y2 = ((pred_boxes[..., 1]/self.S + yshift)
                         + 0.5*torch.square(pred_boxes[..., 3]))
        pred_boxes_y2 = pred_boxes_y2[..., None]
        pred_boxes_x1y1x2y2 = torch.cat([ pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=-1)

        # target = (bs, S, S, B, 5) same
        target_boxes = target[..., :5*self.B].reshape(batch_size, self.S, self.S, self.B, -1)
        target_boxes_x1 = ((target_boxes[..., 0] / self.S + xshift)
                           - 0.5 * torch.square(target_boxes[..., 2]))
        target_boxes_x1 = target_boxes_x1[..., None]
        target_boxes_y1 = ((target_boxes[..., 1] / self.S + yshift)
                           - 0.5 * torch.square(target_boxes[..., 3]))
        target_boxes_y1 = target_boxes_y1[..., None]
        target_boxes_x2 = ((target_boxes[..., 0] / self.S + xshift)
                           + 0.5 * torch.square(target_boxes[..., 2]))
        target_boxes_x2 = target_boxes_x2[..., None]
        target_boxes_y2 = ((target_boxes[..., 1] / self.S + yshift)
                           + 0.5 * torch.square(target_boxes[..., 3]))
        target_boxes_y2 = target_boxes_y2[..., None]
        target_boxes_x1y1x2y2 = torch.cat([ target_boxes_x1, target_boxes_y1, target_boxes_x2, target_boxes_y2 ], dim=-1)

        # iou between pred and target boxes
        iou_pred_target = iou(pred_boxes_x1y1x2y2, target_boxes_x1y1x2y2)
        max_iou_val, max_iou_idx = iou.max(dim=-1, keepdim=True) # -1 means outer

        # need to do this again since paper calculates 2 bounding box for 1 grid cell

        
        max_iou_idx = max_iou_idx.repeat(1, 1, 1, self.B)
        # bb_idxs -> (Batch_size, S, S, B)
        #  Eg. [[0, 1], [0, 1], [0, 1], [0, 1]] assuming B = 2
        bb_idxs = (torch.arange(self.B).reshape(1, 1, 1, self.B).expand_as(max_iou_idx)
                   .to(preds.device))
        # is_max_iou_box -> (Batch_size, S, S, B)
        # Eg. [[True, False], [False, True], [True, False], [True, False]]
        # only the index which is max iou boxes index will be 1 rest all 0
        is_max_iou_box = (max_iou_idx == bb_idxs).long()

        # obj_indicator -> (Batch_size, S, S, 1)
        obj_indicator = target[..., 4:5]

        # CL
        cls_target = target[..., 5 * self.B:]
        cls_preds = preds[..., 5 * self.B:]
        cls_loss = (cls_preds - cls_target) ** 2
        # only keep loss from cells which have object in them
        cls_loss = (obj_indicator * cls_loss).sum()

        # Objectness Loss (For responsible predictor boxes )
        # indicator is now object_cells * is_best_box
        is_max_box_obj_indicator = is_max_iou_box * obj_indicator
        obj_mse = (pred_boxes[..., 4] - max_iou_val) ** 2
        # Only keep losses from boxes of cells with object assigned
        obj_mse = (is_max_box_obj_indicator * obj_mse).sum()

        # LL
        x_mse = (pred_boxes[..., 0] - target_boxes[..., 0]) ** 2
        # only keep losses from boxes which have obj 
        x_mse = (is_max_box_obj_indicator * x_mse).sum()
        y_mse = (pred_boxes[..., 1] - target_boxes[..., 1]) ** 2
        y_mse = (is_max_box_obj_indicator * y_mse).sum()
        w_sqrt_mse = (pred_boxes[..., 2] - target_boxes[..., 2]) ** 2
        w_sqrt_mse = (is_max_box_obj_indicator * w_sqrt_mse).sum()
        h_sqrt_mse = (pred_boxes[..., 3] - target_boxes[..., 3]) ** 2
        h_sqrt_mse = (is_max_box_obj_indicator * h_sqrt_mse).sum()

        # OL : for boxes of cells which have object but arent responsbile for predictor boxes and for boxes of cells which dont have obj

        no_object_indicator = 1 - is_max_box_obj_indicator
        no_obj_mse = (pred_boxes[..., 4] - torch.zeros_like(pred_boxes[..., 4])) ** 2
        no_obj_mse = (no_object_indicator * no_obj_mse).sum()


        # total
        loss = self.lambda_coord*(x_mse + y_mse + w_sqrt_mse + h_sqrt_mse)
        loss += cls_loss + obj_mse
        loss += self.lambda_noobj*no_obj_mse
        loss = loss / batch_size
        return loss


