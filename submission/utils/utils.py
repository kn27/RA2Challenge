import numpy as np
import torch

def filter_prediction(prediction, threshold = 0, filter_labels = None):
    if filter_labels is None:
        filter_labels = prediction['labels'].cpu()
    labels = prediction['labels'].cpu()
    scores = prediction['scores'].cpu()
    boxes = prediction['boxes'].cpu()
    filtered_prediction = {'labels':[], 'scores':[], 'boxes':[]}
    for label in set(filter_labels):
        loc = np.where(labels == label)[0]
        if len(loc) > 0: 
            label_scores = scores[loc]
            if max(label_scores) > threshold:
                box = boxes[loc[np.where(label_scores == max(label_scores))]]
                filtered_prediction['labels'].append(label)
                filtered_prediction['scores'].append(max(label_scores))
                filtered_prediction['boxes'].append(box[0])
    return filtered_prediction

def filter_predictions(predictions, img_names, NAME_TO_ID_MAP, threshold = 0):
    filtered_predictions = []
    false_negative = {}
    limbs = ['LH', 'RH', 'LF', 'RF']
    print(f'Apply objectness threshold = {threshold}')
    for i,prediction in enumerate(predictions):
        limb = [limb for limb in limbs if limb in img_names[i]][0]
        LABELS = [NAME_TO_ID_MAP[x] for x in NAME_TO_ID_MAP.keys() if limb in x]
        filtered_prediction = filter_prediction(prediction, 0, LABELS)
        filtered_predictions.append(filtered_prediction)
        if len(filtered_prediction['labels']) < len(LABELS):
            false_negative[i] = set(LABELS) - set(filtered_prediction['labels'])
    print(f'Number of images that has at least one False Negative: {len(false_negative)}')
    print(f'Number of False Negative: {len([y for x in false_negative.values() for y in x])}')
    #print(f'All False Negatives dict(img_index: labels) : {false_negative}')
    return filtered_predictions, false_negative

def get_statistics(predictions, targets, iou_threshold):
    metrics = []
    for i, (prediction, target) in enumerate(zip(predictions, targets)):
        pred_boxes = prediction['boxes']
        pred_scores = prediction['scores']
        pred_labels = prediction['labels']

        target_boxes = target['boxes']
        target_labels = target['labels']
        
        true_positives = np.zeros(len(pred_boxes))
        img_idx = np.ones(len(pred_boxes)) * i
        ious = np.zeros(len(pred_boxes))
        if len(target_labels):
            for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                if pred_label not in target_labels:
                    print('Look into this!')
                    print(i)
                    print(pred_label)
                    print(target_labels)
                else:
                    iou = bbox_iou(pred_box, target_boxes[np.where(target_labels == pred_label)][0])
                    ious[i] = iou                    
                    if iou >= iou_threshold:
                        true_positives[i] = 1
                      
        metrics.append([true_positives, pred_scores, pred_labels, target_labels, img_idx, ious])
    return metrics

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
