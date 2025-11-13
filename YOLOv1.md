I use yolo everyday at my job. Wanted to figure out how it internally works by implementing the first and latest paper.

Earlier: We picked up patterns based on math from images in form of matrix and figured out where the hue is more, contrast is more, pixel density is more and used that to figure out what the image is (using ResNet models)

But this doesn't work for dozens of images which have complex features (like CCTV footage of cars where there is so much occlusion, complex features and Resnet takes huge amount of time even in inference)

Then came:

RCNN, Faster-RCNN : They used 2 stage networks (generate bbox first then classify boxes in second stage)

Image -> Backbone (CNN / Resnet) -> Convolutional Feature Map (Contain high level semantic info about the images) -> First Stage (Bounding Boxes) -> Then we use same feature map to get class and bounding box info (Classifiation and Regression Head) (Boxes gets refined + labelled)

Multi stage pipeline, each component trained seperately, complex (not good for real-time apps) ~5fps on GPUso


# Single stage regression problem
Hence YOLO was formed by ultralytics  (Single network to box detection and category prediction)

# Steps
- Take input image (1920 * 1080 for our CCTV) -> gets mapped to 448 * 448 (depends on YOLO architecture but it's fixed) -> Divide this into S * S grid cells (S = 7 in paper) therefore each grid has 64 pixels (448/7)
  - Each cell is responsbile for predicting 1 bounding box / 1 object.
  **Insert Picture Here of Horse and Person** :: What do we mean by 'responsble' ? -> center point of person (blue dot) so that cell is responsbile for detecting person class, similarily center point of horse (red dot) so that cell is responsbile for detecting horse class
  - So we are saying each cell is responsbile for getting 1 prediction (every cell has it's own targets (value which we compare with network predictions for loss function)) ideally predictions = targets.

    - So let's say person class ground truth values (100,200,130,202) (for entire person bbox) These are very big values are not easy to predict by the network directly... So we make predictions smaller somehow and decode them later it should do the task. So we predict values not of the entire class but values that are relative to grid cell that the object center falls to (yellow box)

Targets:
Center Pt(x,y): Relative to anchor that (x,y) falls into (here x,y = 100,200, w,h = 130, 202)
x' = (x - x_a) / 64 # dist between obj center and top left coordinate
y' = (y - y_a) / 64

What does these ' value mean -> If I am standing at the top left corner of the grid -> how much do I have to move to reach the center? **Insert Image of delta_x, delta_y along with w,h and normalised GT values -> Relative grid cell values normalised**
Here (x_a, y_a) are coordinate of left-top point of grid
Width Height (w,h) relative to whole image :: w' = w/448, h' = h/448

From ground truth value, calculate the targets with respect to the grid position

## Label Encoding
So Now I have my targets (Relative to grid cells x,y,w,h points) for this grid cell, we do this for all grid cells.
  - Some grid cells might not have any object (targets = all zeros)
  - Example : Boxes: (x,y,w,h,conf) -> (0.9, 0.7, 0.1, 1.0), All objects have class assigned to it, Classes: (1.0, 0, 0, 0...n) - One hot encoding  (n is the number of classes here)



# Prediction Vector
Model Output comparision with Target to get loss
Each grid cell predicts:
- 2 bounding boxes (b=2), for each bounding box you have (x', y', w', h', c) c here is class objectness score (indicates whether that particular box has object or not (confidence))
x',y' = offset relative to top left corner, w',h' = offset relative to width and height of img, c = prob that box contains an object (doesn't know class yet)
Class prob: (p1,p2,,,p20) between 0-1
Number of parameters: 5 values per each bbox (2 bbox in one cell) = 10 + 20 (class scores) = 30 {For each grid cell we get a vector of 30 values}
FOr actual model, 7x7 gridcells (49) hence output layer = 7*7*30

Even though we are getting 2 boxes, we will only keep one of them, each grid only outputs 1 box (final) :: which one to keep? one with highest conf score (postprocessing)

So now we get (x1', y1', w1', h1', c1) and  (x2', y2', w2', h2', c2) .. How do we get the person bbox from this (these are just relative to the grid between 0-1) ::
Converting back (scaling back to big box)
xa,ya = top left corner
- x1 = x1' * 64 + xa 
- y1 = y1' * 64 + ya 
- w1 = w' * 448
- h1 = h' * 448
**Insert Image here of this same visualisation**

Now we are left with confidence score
c1' = c1 * p where p = max(p1,p2,,,,p20) (FINAL CLASS PROB : 14th posn -> highest prob -> 0.8) 0.8 * conf score of both boxes 
c2' = c2 * p

Keep the box which has highest confidence score. (Discard red box, keep blue box) **Insert image here**

All parsing is done.

# Architecture
Add a gif / demo for CNN and how it works (visualiser website)
CNN : CONV + MAX POOL layers
Network: 24 conv layers (also known as backbone to generate feature maps), 2 fully connected layers
Flatten last conv map 7 * 7 * 1024 = 50176 feature vector -> pass through 2 fully connected layers -> output = 1470 feature vector -> reshape to 7*7*30 feature map (output layer)
irst of all 24 convolutional layers that "stretch" an image with size (448x448x3: w,h, RGB channels) into a tensor with the size of (SxSx1024, where S=7). 
Then, two fully connected layers at the end (4096, 7x7x30). The first FC layer gets a flattened result of the last convolutional layer (7x7x1023 - 50176).
Each layer has Leaky ReLU as an activation function under the hood, except the final one, which uses the Linear activation function. (Explain Leaky RELU and Linear Activation)

main reason for stacking convolutional layers is always to extract features from some spatial structure of a given object (image, for most computer vision tasks).

And the main problem is that the “convolution queue” doesn't increase the degree of nonlinearity in the data structure. If you simply try to add some fully connected layers in between the convolutional blocks, it will absolutely destroy the spatial structure. And moreover, it will take more memory for calculation.

he simple solution is to add some nonlinearities by using nonlinear activation functions between convolutional layers. And YOLO algorithms use ReLU for it 
And 3x3 convolution "stretches" and "reduces" a spatial representation of the input image after the 1x1 convolutional stage. So, it decreases the width and height of tensors (each side lost 2 "pixels") and increases the count of channels two times.

And, of course, YOLO algorithms use some regularization techniques, such as data augmentation and dropout, to prevent overfitting.

So the original YOLO model output for the Pascal VOC dataset has a 7x7x30 dimension, so it's (x,y,w,h,c) two times responsible for predicting objects and 20 class probabilities for each 49 "detectors."
**Insert Arch photos here**


# Training process
dataset: pascal voc -> 20 classes, pretrained on imagenet (for classification task) at 224 * 224 (performed to make model learn features that are useful later (reduces training time of detector))), actual training on 448*448 (increased resolution for localisation) on VOC dataset


# Loss Computation

3 Componenets:
First is localisation loss -> Ensure that model learns to predict the x,y,w,h pred val of responsbile predicted box of cell closer to the target assigned to that cell
Conf loss -> Trained to Predict correct scores which is going to capture two -> presence of object, how good fit the predicted box is. Conf score of high val when pred box has high overlap with GT val, lower val when box contains an object but does not fit it well, 0 when box with no object **Insert Image here**
Classification Loss -> Conditional class prob for grid cells (conditioned on fact that there is an object) right prob val for cells that contain object (model should pred 1 for that class, 0 for remaining class)
For all losses regression (MSE) is used

**Insert image of loss functions formulas for 1 grid cell**
We want to compute this for all grid cells hence sum
Indictator function: used to filter
IT will be 1 if i is assigned a target and box j is respoinsbile for that target

For conf loss: we want tot snesure conf of these boxes are closer to target val + train model to predict the conf score of boxes which are not assigned to any obj as 0  (1 for those objets that are not responsbile for any box)

Classificationloss: only for boxes where some object is present
**Updated 3 losses + final loss is just sum of these**

For most of images we would only have some cells that have some obj in them and a lot fo cells which dont have object, to fix that we have lambda val (increase of weight of localisation error + reduce the weight of conf loss of no object from background objects (5 and 0.5 resp))

Now loss is minimsed thru backprop (explain in one line)




Loss is calculated for each grid cell and then sum of losses over all grid cells S*S = Loss ,, but these grid cells might have objects and some might not have objects.
We only care about the ones where objects are there (more imp) hence to put more importance on grid cells that contain objects we decrease importance of grid cells having no objects .. Example: 2 object cells, 47 no object cellsA **insert image** 
**Insert image of lambda .5 in paper**

#### Loss function for object cells
Since we have two types of grid cell, the first one which contains object and second one which does not contain any object, so loss function is the sum of two types of losses
Loss = objectness loss + classification loss + box regression loss
Put more weightage on box parametrs (in paper)

##### Box loss in detail
Sum of squared errors : standard stuff
x',y', w', h' = ground truth box.
x*', y*', w*', h*' = predicted box that has largest iou (explain and write code for iou also) with GT box.

Lbox = (x*' - x')^2 + (y*' - y')^2 + (sqrt(w*') - sqrt(w'))^2 + (sqrt(h*') - sqrt(h'))^2 # Why sqrt? bcs values can be really high for img w and h hence sqrt to downscale/normalise (low values will still be less affected, high will be downscaled)

#### Object Conf Loss

L = (c* - c')^2 = (0.9 - 1.0) :: squared error between predicted confidence and label confidence (90% ocnf that it is an object)

Everything is squared difference here hence they have named it 'regression' Regression Loss is used
we may face a class imbalance problem, because most of the boxes don't contain any objects. The cure is that we can split the loss function into two parts: for a cell with and without a box. And then somehow decrease the weight of the cells without boxes.

#### Classification loss

Sum of squared errors over all class probabilities
L = sum(pi - pi')^2 for 20 values (all classes) [predicted-actual]^2

**Insert image for loss functions for all (formula)**

What if no object? Only loss will be objectness score loss (no box or class)


# Lighter version of the model
FastYOLO (instead of 24 layers it has only 9 layers) to do faster training Jk:


# Performance
REAL TIME
45 fps map accuracy of 63.4
Better than FasterRCNN (accuracy is a little better but fps is too slow for real time), DPM Deformable Part Model for Object Detection)

Key ideas: to use the window sliding approach or move on the picture with some smaller region, classify this region, and use high-scoring regions for object localization.

Key difference: YOLO performs object detection in a single stage (again) for all tasks, extracting features and detecting multiple objects and class probabilities directly while using the entire input image.etc...

# Limitation: Each grid cell can only do one object and if 7*7 cells = only 49 objects can be detected from a particular image (crowded scenarios dont work) but newer versions of yolo are much better than this.

During inference yolo removes low prob bbox by applying NMS suppression (this was reduced in yolov10)



# More
input: image
output: {b1, b2, ... bn} bounding boxes of n detected objects along with {c1,c2,...cn} class labels of all detected objects

So overall:
Algo:
- Split image into grid cells
- For each grid cell predict bounding box, compute loss function, minimise it using backprop/CNN
- Choose best option using prob

# On IOU, NMS, Metrics
## IOU

Intersection over Union : measure of overlap between two boxes and tells you how well the two boxes align with each other (here the two boxes are ground truth boxes and prediction made by model boxes) any IOU > 0.8 -> We have predicted object correctly.


**Insert image of x1,y1,x2,y2 of bounding box of man running along with centered x,y**
Easy to convert from one to the other
x1 = x - w/2
y1 = y - h/2
x2 = x + w/2
y2 = y + h/2

We need width and height of the overlap rectangle.
x1,y1 for top left of overlap rectangle is simply max(x1_box1,x1_box2) , max(y1_box1, y1_box2)
x1,y1 for bottom right of overlap rectangle is min(x1_box1, x1_box2), min(y1_box1, y1_box2)

Once coordinates are computed for overlap rectangle, area = product of difference of y and x coordinates. (numerator), denominator = Area(box1) + Area(box2) - intersection
union(A,B) = A + B - intersection(A,B)
**Insert image here**

```python
def iou(pred, gt):
    pred_x1, pred_y1, pred_x2, pred_y2 = pred
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_topleft = max(pred_x1, gt_x1)
    y_topleft = max(pred_y1, gt_y1)
    x_bottomright = min(pred_x1, gt_x1)
    y_bottomright = min(pred_y1, gt_y1)
    # boxes dont overlap at all
    if x_bottomright < x_topleft or y_bottomright < y_topleft: return 0.0
    
    intersection = (x_bottomright - x_topleft) * (y_bottomright - y_topleft)
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    union = pred_area + gt_area - intersection
    iou = union/intersection
    return iou
```

## NMS 
Non maximum supression is a post processing step to filter out all the redundant and duplicate boxes that the model has predicted and calculate the best box that matches with GT.

All predictions of the model -> Apply NMS seperately for each class (We also have confidence score for each {classification prob computed earlier}) -> sort in decreasing order

Pick the top conf (put from prediction list pile to output detection list) -> remove instances having high overlap with max score instead (repeat until no prediction of class remains)

This removes duplicate boxes and leave us with filtered set of predcictions. (Suppress the ones that are not max)

**Insert algo**
```python
# nms can also be class agonistic (later models of YOLO)
def nms(detections, nms_threshold=.5):
    # detections = [[x1,y1,x2,y2,score], [x1,y1,x2,...], []]
    # sort by dec
    sorted_det = sorted(detections, key=lambda k : -k[-1])
    keep_detections = []
    while len(sorted_det) > 0:
        keep_detections(sorted_det[0])
        # remove this and all other boxes that have high overlap with this box
        # every it keeps removing
        sorted_det = [
                box for box in sorted_det[1:] if iou(sorted_det[0][:-1], box[:-1] <  nms_threshold
                ]
    return keep_detections
```

#### MAP : The main metric to look for while doing Object Detection
TP : Boxes where GT and Pred match (correct guess)
FP: Boxes where Pred is different from GT (incorrect guess)
FN: GT instances which model fails to detect

Precision: out of all predictions our model make, how many were actually correct?
How many had high iou of GT and Pred (TP / TP + FP)


Recall
Out of all GT boxes, how many were actually detected by our model (had high overlap with our predicted model) : (TP / TP + FN) or TP / number of GT boxes

We want both Precision(boxes are detected correctly)  and Recall(boxes are detected) to be high 

Precision Recall curve: draws value of precision and recall for different confidence threshold

So bascially plot precision recall curve, area under the curve = average precision for that class. Doing this for all classes -> mean average precision (MAP)
Map50-90 or map.50:.05:.95 mean we are computing map at different iou values (for prediction and ground truth boxes), map50-90 means we measures the average precision across different classes at various Intersection over Union (IoU) thresholds between (0.50) and (0.90)

Both precisely localized (high IoU) and correctly classified across a range of overlap requirements

# Implementation of map using area under curve
```python
def map(pred_boxes, gt_boxes, iou_threshold=0.5):
    # calculate map of two sets of boxes
    '''
    pred_boxes = [
                    {
                        'person' : [[x1, y1, x2, y2, score], ...],
                        'car' : [[x1, y1, x2, y2, score], ...]
                        'class_with_no_detections' : [],
                        ...,
                        'class_K':[[x1, y1, x2, y2, score], ...]
                    },
                    {det_boxes_img_2},
                     ...
                    {det_boxes_img_N},
                ]
     gt_boxes = [
                    {
                        'person' : [[x1, y1, x2, y2], ...],
                        'car' : [[x1, y1, x2, y2], ...]
                        'class_with_no_ground_truth_objects' : [],
                        ...,
                        'class_K':[[x1, y1, x2, y2], ...]
                    },
                    {gt_boxes_img_2},
                     ...
                    {gt_boxes_img_N},
                ]
    '''
    # gt label
    for im_gt in gt_boxes:
        for cls_key in im_get.keys():
            gt_labels = cls_key

    all_aps = {}
    # average precision for all classes
    aps = []
    for idx, label in enumerate(gt_labels):
        cls_preds = [
            [im_idx, im_pred_labels] for im_idx, im_pred_labels in enumerate(pred_boxes) if label in im_preds for im_pred_labels in im_preds[label]
        ]

        # cls_preds = [
        #   (0, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (0, [x1_M, y1_M, x2_M, y2_M, score_M]),
        #   (1, [x1_0, y1_0, x2_0, y2_0, score_0]),
        #   ...
        #   (1, [x1_N, y1_N, x2_N, y2_N, score_N]),
        #   ...
        # ]
        cls_preds = sorted(cls_preds, key = lambda k: -k[1][-1]) # by conf
        # tracking which gt boxes of this class have already been matched
        gt_match = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # number of gt boxes for this class (for recall)
        num_gt = sum([len(im_gts[label]) for im_gts in gt_boxes])
        tp, fp = [0] * len(cls_preds), [0] * len(cls_preds)
        # for each pred
        for pred_idx, (im_idx, pred_det) in enumerate(cls_preds):
            # get gt box for this image and this label
            im_gts= gt_boxes[im_idx][label]
            max_iou_found, max_iou_gt_idx = -1,-1
            # get best matching gt box
            for gt_box_idx, gt_box in enumerate(im_gts):
                gt_box_iou = get_iou(pred_det[:-1], gt_box)
                if gt_box_iou > max_iou_found:
                    max_iou_found = gt_box_iou
                    max_iou_gt_idx = gt_box_idx
            # tp only if iou>=threshold AND this gt has not been matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[pred_idx] = 1
            else:
                tp[pred_idx] = 1
                gt_matched[im_idx][max_iou_gt_idx] = True # matched
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recalls = tp / num_gts
        precisions = tp / (tp + fp)
        # calculating area under curve by setting boundary and converting zig zag nature of curve -> pure rectangles.

        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        # max precision val with recall r (converting zig zag -> stable)
        for i in range(precisions.size-1,0,-1):
            precisions[i-1] = np.maximum(precisions[i-1], precisions[i])
        # area: get points where recall changes value
        i = np.where(recalls[1:] != recalls[:-1][0])
        ap = np.sum((recalls[i+1] - recalls[i]) * precisions[i+1])
        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan

        mean_aps= sum(aps) / len(aps)
        return mean_aps, all_aps
```

# More on YOLO

In the paper they have used S = 7*7 grid, B = 2 (two bounding box generated by each grid, later we take the one with higher conf), n = 20 (classes) of VOC dataset

Yolov3 onwards support multi class predictions (dog and animal in an image) but yolov1 just predicts one class (dog)
Yolov1 is just a regression problem.

# Limitations in V1
Each cell can only contain one class, If one dog and one bicycle in same cell -> problem. 
Each cell contains finite amount of bbox (busy environments) -> can't
Bbox predictions struggles with unusual aspect ratios (If YOLO was mostly trained on pictures of automobiles that were about twice as wide as they were tall, it might have trouble finding very long, thin cars or things that were fashioned in a strange way.)
Same error in small box is worse than in large box (no error scaling wrt bounding boxes -> regression as a loss function doesn't perform great)
Many cells don't contain image (training conf scores pushed to 0, coordinate loss contribution is low) -> scale down the no obj conf score and scale up the obj conf score with lambda
YOLOv1 had trouble finding little things and those that were in groups. Because of the 7×7 grid design, each grid cell could only see a small number of items (at most 2 different classes

# From Blogs (on YOLO)

When labeling objects in an image for object detection, we need to find the center point coordinates, (center_x, center_y), using the provided coordinates of the top-left corner (x_min, y_min) and the bottom-right corner (x_max, y_max) of a bounding box. This involves calculating the average of x_min and x_max to determine center_x and the average of y_min and y_max to determine center_y. These center coordinates help locate the central position of the labeled object within the bounding box. After this these coordinates are normalised for consistency (clamped to values b'w 0-1 by dividing by width and height of image) and also normalise w, h of bbox by dividing the image width and height
center_x = (xmin+xmax/2) / w_img, w = xmax-xmin/w_img

In the prediction model for object detection, the output is represented as (Δx, Δy, Δw, Δh, c) (p1, p2, …, p20). Here, Δx corresponds to the normalized center x-coordinate of the object, Δy represents the normalized center y-coordinate, Δw indicates the normalized width of the bounding box, and Δh signifies the normalized height of the bounding box. The ‘c’ value stands for the confidence or objectness score, with a value of 1 assigned to every grid cell containing an object center and 0 for grid cells without an object.

What makes YOLOv1 so good is it's simplicity (both in terms of network (single network only 1 forward pass) and loss calculation) since regression loss is used, scaling factor is used smartly to give more score to grid cells that have objects in them.

First, it guesses bounding boxes, which are rectangular areas that should hold things. Each cell can predict more than one bounding box (usually two in YOLOv1). For each bounding box, the network predicts five values: the box’s center’s x and y coordinates, its width and height, and a confidence score that shows how sure the network is that there is an object in that box.

Each grid cell does more than just find things; it also predicts class probabilities. The network gives each possible object class (like “car,” “person,” “dog,” etc.) a chance of being in that cell

 best thing about YOLOv1 was how fast it was. Previous state-of-the-art approaches, such R-CNN, could take 47 seconds to process one image. The speed stemmed from the basic design choice to employ a single network pass instead of the thousands that region-proposal approaches need

 How they understand context when it comes to CNN -> sliding window with stride (only see little parts of the picture at a time)


The loss function put a lot of weight on mistakes in bounding box predictions when objects were present and less weight on confidence forecasts for grid cells that didn’t have objects
task of finding objects on an image can be defined as the regression task of predicting the (x,y) coordinates of the center and the (width, height) of object bounding boxes

For each cell:
1. Predicts bounding box coordinates, sizes for objects (two per cell for the original YOLO algorithm according to the original article), and their confidence scores — a measure of how likely it is for a cell to contain an object and how accurately we detect it.

2. Chooses only one predicted bounding box using confidence scores. 

3. Predicts conditional probabilities for each class (probabilities of whether the cell relates to a certain class when there is an object in it).

# first step of filtering -> iou
 training step for the confidence level prediction task, for each cell, we put 1.0 if there is some object in the cell and the cell contains the center of the bounding box, and 0.0 if not.

This setup helps us pre-filter bounding boxes before the next step and will cost us O(Count of predictions). That helps us decrease the time of YOLO operation because the next step of filtering will cost us O(Count of prediction ^ 2).

# second step of filtering -> nms
NMS algorithm uses B - list of bounding boxes candidates, C - list of their confidence values, and τ - the threshold for filtering overlapped bounding boxes. Let R be the result list:

1. Sort B according to their confidence scores in ascending order. 

2. Put the predicted box for B with the highest score (let it be a candidate) into R and remove it from B. 

3. Find all boxes from B with IoU(candidate, box) > τ and remove them from B. 

4. Repeat steps 1-3 until B is not empty.
```python3
def nms(boxes, confidences, overlapping_treshold):
    indexes = [i_s[0] for i_s in sorted(list(enumerate(confidences)), key=lambda v: v[1], reverse=True))
    boxes = [boxes[index_score[0]] for index_score in indexes]
    result = []
    while len(boxes) > 0:
        result.append(boxes.pop(0))
        removing_indexes = [i for i, box in enumerate(boxes) if IOU(box, result[-1]) >= overlapping_treshold]
        for index in removing_indexes:
            del boxes[index]

    return result
```

# Paper Reading

    Training and validation data sets are Pascal VOC 2007 and 2012.
    About 135 epochs.
    Batch size: 64.
    Momentum: 0.9 | Decay: 0.0005.
    Learning rate: for the first epochs, slowly raise the learning rate from 1e−3 to 1e−2, continue training with 1e−2 for 75 epochs, then 1e−3 for 30 epochs, and finally 1e−4 for 30 epochs.
    The main insight is: if you start training a model on a high learning rate, the original YOLO will diverge due to unstable gradients.

Data augmentation:

    Random scaling and translations of up to 20% of the original image size.
    Random adjustment of the exposure and saturation of the image by up to a factor of 1.5 in the HSV color space.


## YOLO Annotation Format
Resize the image
Split the resized image into a grid
And as we know, every cell is responsible for predicting a tensor of 30 values (for Pascal VOC object recognition task): (x,y,w,h,c) ×2 for the predicted bounding box, and 20 values of class probabilities.

**Insert bicycle image here**
    The value is 1 if there is some object of the class in a cell, and this cell contains the center of the bounding box.
    In other cases, it's 0.
    we have only two bounding boxes per cell, and it's completely up to you how two fill target data for each predictor. The main options and questions are:

    What is the priority value of using bounding boxes that have intersections?
    Which bounding box should be set as data for the first predictor? And which for the second?

Yolo predicts 5 parameters for each box from each cell (w,h (relative to img width and height normalised b/w 0to1), xcenter, ycenter (offset of center from top left corner of grid cell)) also conf score for each box (how conf is the model that there indeed is an object within this box, how good a fit does the model think that the predicted model is for the object that the box contains)
Yolo also predicts one set of class conditional probabilities (these val are prob that a class exist in that grid cell condition on the fact that object is inside the grid cell) 5B + C values for each grid cell, VOC dataset = 20 classes, 2 Bbox per cell -> (5*2) + 20 = 30 values for each grid cell

Each grid cell will only predict one type of object,
How do we choose which box? since 2 boxes are predicted per grid -> YOLO will predict multiple box (b=2) per grid cell but only one predictor box is resposible for that target, the one with highest IOU with GT box

## add v10 info, metrics (from odn.zip)
# Before YOLO
# Why YOLO
# YOLO High Level Steps
# YOLO BBox Format + Predictions
# Grid Cell predictions, IOU, NMS
# Architecture + Training + Annotation Format and conversion
# Loss fn
# Code
# Inference + MAP (Metrics)
# Limitations
# Improvements (v10 now) + Metrics of V10 on our model.one

## Summary of YOLO + training


## Paper Reading and imp points
- Yolo creates S*S grid cells convering entire image and predicts B bbox for each opf those cells
- For each of these B bboxes we predict 5 values (x,y,w,h) of box center from top left corner of grid cell that box belongs to, we also predict confidence score for each of the boxes which represent the prob that box contains an object and as well how good a fit is that box for the object that it contains, additionally for each grid cell we predict class condtional prob which encodes how likely that grid cell contains a particular class (condtion on fact that object center exists in that grid cell)
- For training each GT obj is assigned to one grid cell (the one that contains target box center) and for each grid cell YOLO predicts 5B parameters + C prob (%B+C val per cell) 
- Given target object, out of B pred boxes we only choose one predictor responsible for predicting the GT object (suing IOU)
- Final output will be S*S feature map with %B+C channels, during implementation:
-- Pretrain backbone net with image size 224*224 (resnet pytorch pretrained), get rid of classification FC layer and add detection layer
- Convert taget box coordinates to yolo format and normalise all targets b/w 0 and 1
- Implement yolo loss (localisation, confi, classiciation loss)
- Train detection network on resized image of 448*448 (number chosen by the paper)

-- More Paper Read --

# Underlinings from the Paper

we frame object detection as a re-
gression problem to spatially separated bounding boxes and
associated class probabilities. A single neural network pre-
dicts bounding boxes and class probabilities directly from
full images in one evaluation. Since the whole detection
pipeline is a single network, it can be optimized end-to-end
directly on detection performance
45 frames
per second.
Systems like deformable parts
models (DPM) use a sliding window approach where the
classifier is run at evenly spaced locations over the entire
image
predicts multi-
ple bounding boxes and class probabilities for those boxes
no batch processing 
one set of class probabilities per grid cell, regardless of the
number of boxes B.
At test time we multiply the conditional class probabili-
ties and the individual box confidence predictions,
Pr(Classi |Object) ∗Pr(Object) ∗IOUtruth
pred = Pr(Classi ) ∗IOUtruth
pred 
class-specific confidence scores for each
box.

network reasons glob-
ally about the full image and all the objects in the image.

S×S grid.
If the center of an object falls into a grid cell, that grid cell
is responsible for detecting that object.

Each grid cell predicts Bbounding boxes and confidence
scores for those boxes. These confidence scores reflect how
confident the model is that the box contains an object and
also how accurate it thinks the box is that it predicts. For-
mally we define confidence as Pr(Object) ∗IOUtruth
pred . If no
object exists in that cell, the confidence scores should be
zero. Otherwise we want the confidence score to equal the
intersection over union (IOU) between the predicted box
and the ground truth


The (x,y) coordinates represent the center
of the box relative to the bounds of the grid cell. The width
and height are predicted relative to the whole image. Finally
the confidence prediction represents the IOU between the
predicted box and any ground truth box
predicts C conditional class proba-
bilities, Pr(Classi|Object). These probabilities are condi-
tioned on the grid cell containing an object

We only predict one set of class probabilities per grid cell, regardless of the
number of boxes B


network extract
features from the image while the fully connected layers
predict the output probabilities and coordinates.

We pretrain the convolutional layers on the ImageNet classification
task at half the resolution (224 × 224 input image) and then double the resolution for detection.
Detection often requires fine-grained visual infor-
mation so we increase the input resolution of the network
from 224 ×224 to 448 ×448
both class probabilities and
bounding box coordinates. We normalize the bounding box
width and height by the image width and height so that they
fall between 0 and 1. We parametrize the bounding box x
and ycoordinates to be offsets of a particular grid cell loca-
tion so they are also bounded between 0 and 1.

 linear activation function for the final layer and
all other layers use the following leaky rectified linear acti-
vation

in every image many grid cells do not contain any
object. This pushes the “confidence” scores of those cells
towards zero, often overpowering the gradient from cells
that do contain objects. 

increase the loss from bounding box
coordinate predictions and decrease the loss from confi-
dence predictions for boxes that don’t contain objects.

Sum-squared error also equally weights errors in large
boxes and small boxes. Our error metric should reflect that
small deviations in large boxes matter less than in small
boxes. To partially address this we predict the square root
of the bounding box width and height instead of the width
and height directly.

loss function only penalizes classification
error if an object is present in that grid cell (hence the con-
ditional class probability discussed earlier). It also only pe-
nalizes bounding box coordinate error if that predictor is
“responsible” for the ground truth box (i.e. has the highest
IOU of any predictor in that grid cell).

To avoid overfitting we use dropout and extensive data
augmentation. A dropout layer with rate = .5 after the first
connected layer prevents co-adaptation between layers [18].
For data augmentation we introduce random scaling and
translations of up to 20% of the original image size. We
also randomly adjust the exposure and saturation of the im-
age by up to a factor of 1.5 in the HSV color space.

each grid cell only predicts two boxes
and can only have one class. This spatial constraint lim-
its the number of nearby objects that our model can pre-
dict. Our model struggles with small objects that appear in
groups, such as flocks of birds

struggles to generalize to objects in new or unusual
aspect ratios or configurations.

Instead of trying to optimize individual components of
a large detection pipeline, YOLO throws out the pipeline
entirely and is fast by design.

While
YOLO processes images individually, when attached to a
webcam it functions like a tracking system, detecting ob-
jects as they move around and change in appearance.

Unlike classifier-based approaches,
YOLO is trained on a loss function that directly corresponds
to detection performance and the entire model is trained
jointly.


# About Yolo

Object Detection + Tracking (Done by DEEPSORT)

Architecture: it has 24 convolutional layers, four max-pooling layers, and two fully connected layers.

Evolution
• YOLOv1: grid, fixed boxes
• YOLOv2: anchor-boxes, batch-norm, high-res classifier
• YOLOv3: Darknet-53 backbone, multi-scale heads
• YOLOv4/5/6/7: bag-of-freebies/boosters, PyTorch refactor
• YOLOv8+: anchor-free, PyTorch-native, auto-batching

End to end single pass + real time
YOLO reformulates detection as a single regression problem:
– Input: image of size H * W
– Divide into an S * S grid
– Each cell predicts B boxes, each with
(x,y,w,h, conf) and C class-probs
– Output tensor shape:
Each box carries 5 numbers
S * S * (B * 5 + C)

- Loss = MSE for box coords + confidence + classification

### Input Processing
- **Input**: Image of size H × W
- **Grid Division**: Divide image into S × S grid
- **Prediction per Cell**: 
  - B bounding boxes
  - Each box contains: 
    - Coordinates (x, y, w, h)
    - Confidence score
    - Class probabilities

### Output Tensor Structure
\[S × S × (B × 5 + C)\]
- S: Grid size
- B: Number of boxes per cell
- 5: (x, y, w, h, confidence)
- C: Number of classes

- 24 convolutional layers
- 4 max-pooling layers
- 2 fully-connected layers
- Final output reshape to \((S,S,B\cdot5+C)\)

YOLO’s loss is a sum of:

1. MSE on box coordinates \((x,y,w,h)\)
2. MSE on objectness confidence
3. MSE on class probabilities

After prediction, many boxes overlap. NMS keeps only the highest-confidence boxes by:

1. Sorting boxes by confidence.
2. Picking the top box and removing any with
\(\mathrm{IOU} \le T\).
3. Repeating until no boxes remain.



Y = [pc, bx, by, bh, bw, c1, c2]
probclass, xy of box, classes

Most of the time, a single object in an image can have multiple grid box candidates for prediction, even though not all are relevant. 

The goal of the IOU (a value between 0 and 1) is to discard such grid boxes to only keep those that are relevant.

IOU between Ground Truth and Prediction:

The user defines its IOU selection threshold, which can be, for instance, 0.5. 
Then, YOLO computes the IOU of each grid cell, which is the Intersection area divided by the Union Area. 
Finally, it ignores the prediction of the grid cells having an IOU ≤ threshold and considers those with an IOU > threshold

The intersection divided by the Union gives us the ratio of the overlap to the total area, providing a good estimate of how close the prediction bounding box is to the original bounding box.

# History
Two stage detectors: Fast RCNN
The first pass is used to generate a set of proposals or potential object locations, and the second pass is used to refine these proposals and make final predictions. This approach is more accurate than single-shot object detection but is also more computationally expensive (not good for real time apps)


One stage: YOLO.

One key technique used in the YOLO models is non-maximum suppression (NMS). NMS is a post-processing step that is used to improve the accuracy and efficiency of object detection. In object detection, it is common for multiple bounding boxes to be generated for a single object in an image. These bounding boxes may overlap or be located at different positions, but they all represent the same object. NMS is used to identify and remove redundant or incorrect bounding boxes and to output a single bounding box for each object in the image.

Disadv:
Struggling with small objects and the inability to perform fine-grained object classification

For each vehicle:
state = [x, y, width, height, vx, vy, vw, vh]
#        position + size  + velocities
```

## About V10
- add v10, metrics etc..
- add website blog here along with implementation and show inference examples on one of our images + add yolov10 and general yolo info in the blog.