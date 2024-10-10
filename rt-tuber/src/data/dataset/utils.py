import random
import numpy as np
import cv2

'''data augmentation functions'''
def random_brightness(imglist, brightness_prob, brightness_delta):
    if random.random() < brightness_prob:
        brig = random.uniform(-brightness_delta, brightness_delta)
        for i in range(len(imglist)):
            imglist[i] += brig

    return imglist


def random_contrast(imglist, contrast_prob, contrast_lower, contrast_upper):
    if random.random() < contrast_prob:
        cont = random.uniform(contrast_lower, contrast_upper)
        for i in range(len(imglist)):
            imglist[i] *= cont

    return imglist


def random_saturation(imglist, saturation_prob, saturation_lower, saturation_upper):
    if random.random() < saturation_prob:
        satu = random.uniform(saturation_lower, saturation_upper)
        for i in range(len(imglist)):
            hsv = cv2.cvtColor(imglist[i], cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] *= satu
            imglist[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return imglist


def random_hue(imglist, hue_prob, hue_delta):
    if random.random() < hue_prob:
        hue = random.uniform(-hue_delta, hue_delta)
        for i in range(len(imglist)):
            hsv = cv2.cvtColor(imglist[i], cv2.COLOR_BGR2HSV)
            hsv[:, :, 0] += hue
            imglist[i] = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return imglist


def apply_distort(imglist, distort_param):
    out_imglist = imglist

    if distort_param['random_order_prob'] != 0:
        raise NotImplementedError

    if random.random() > 0.5:
        out_imglist = random_brightness(out_imglist, distort_param['brightness_prob'], distort_param['brightness_delta'])
        out_imglist = random_contrast(out_imglist, distort_param['contrast_prob'], distort_param['contrast_lower'], distort_param['contrast_upper'])
        out_imglist = random_saturation(out_imglist, distort_param['saturation_prob'], distort_param['saturation_lower'], distort_param['saturation_upper'])
        out_imglist = random_hue(out_imglist, distort_param['hue_prob'], distort_param['hue_delta'])
    else:
        out_imglist = random_brightness(out_imglist, distort_param['brightness_prob'], distort_param['brightness_delta'])
        out_imglist = random_saturation(out_imglist, distort_param['saturation_prob'], distort_param['saturation_lower'], distort_param['saturation_upper'])
        out_imglist = random_hue(out_imglist, distort_param['hue_prob'], distort_param['hue_delta'])
        out_imglist = random_contrast(out_imglist, distort_param['contrast_prob'], distort_param['contrast_lower'], distort_param['contrast_upper'])

    return out_imglist


def apply_expand(imglist, tubes, expand_param, mean_values=None):
    # Tubes: dict of label -> list of tubes with tubes being <x1> <y1> <x2> <y2>
    out_imglist = imglist
    out_tubes = tubes

    if random.random() < expand_param['expand_prob']:
        expand_ratio = random.uniform(1, expand_param['max_expand_ratio'])
        oh, ow = imglist[0].shape[:2]
        h = int(oh * expand_ratio)
        w = int(ow * expand_ratio)
        out_imglist = [np.zeros((h, w, 3), dtype=np.float32) for i in range(len(imglist))]
        h_off = int(np.floor(h - oh))
        w_off = int(np.floor(w - ow))
        if mean_values is not None:
            for i in range(len(imglist)):
                out_imglist[i] += np.array(mean_values).reshape(1, 1, 3)
        for i in range(len(imglist)):
            out_imglist[i][h_off:h_off + oh, w_off:w_off + ow, :] = imglist[i]
        # project boxes
        for ilabel in tubes:
            for itube in range(len(tubes[ilabel])):
                out_tubes[ilabel][itube] += np.array([[w_off, h_off, w_off, h_off]], dtype=np.float32)

    return out_imglist, out_tubes


def sample_cuboids(tubes, batch_samplers, imheight, imwidth):
    sampled_cuboids = []
    for batch_sampler in batch_samplers:
        max_trials = batch_sampler['max_trials']
        max_sample = batch_sampler['max_sample']
        itrial = 0
        isample = 0
        sampler = batch_sampler['sampler']

        min_scale = sampler['min_scale'] if 'min_scale' in sampler else 1
        max_scale = sampler['max_scale'] if 'max_scale' in sampler else 1
        min_aspect = sampler['min_aspect_ratio'] if 'min_aspect_ratio' in sampler else 1
        max_aspect = sampler['max_aspect_ratio'] if 'max_aspect_ratio' in sampler else 1

        while itrial < max_trials and isample < max_sample:
            # sample a normalized box
            scale = random.uniform(min_scale, max_scale)
            aspect = random.uniform(min_aspect, max_aspect)
            width = scale * np.sqrt(aspect)
            height = scale / np.sqrt(aspect)
            if width > 1 or height > 1:
                continue
            x = random.uniform(0, 1 - width)
            y = random.uniform(0, 1 - height)

            # rescale the box
            sampled_cuboid = np.array([x * imwidth, y * imheight, (x + width) * imwidth, (y + height) * imheight], dtype=np.float32)
            # check constraint
            itrial += 1
            if not 'sample_constraint' in batch_sampler:
                sampled_cuboids.append(sampled_cuboid)
                isample += 1
                continue

            constraints = batch_sampler['sample_constraint']
            ious = np.array([np.mean(iou2d(t, sampled_cuboid)) for t in sum(tubes.values(), [])])
            if ious.size == 0:  # empty gt
                isample += 1
                continue

            if 'min_jaccard_overlap' in constraints and ious.max() >= constraints['min_jaccard_overlap']:
                sampled_cuboids.append(sampled_cuboid)
                isample += 1
                continue

            if 'max_jaccard_overlap' in constraints and ious.min() >= constraints['max_jaccard_overlap']:
                sampled_cuboids.append(sampled_cuboid)
                isample += 1
                continue

    return sampled_cuboids


def crop_image(imglist, tubes, batch_samplers):
    candidate_cuboids = sample_cuboids(tubes, batch_samplers, imglist[0].shape[0], imglist[0].shape[1])

    if not candidate_cuboids:
        return imglist, tubes
    h = imglist[0].shape[0]
    w = imglist[0].shape[1]
    crop_cuboid = random.choice(candidate_cuboids)
    x1, y1, x2, y2 = map(int, crop_cuboid.tolist())

    for i in range(len(imglist)):
        imglist[i] = imglist[i][y1:y2 + 1, x1:x2 + 1, :]
    out_tubes = {}
    wi = x2 - x1
    hi = y2 - y1
    # if x1<0 or x2>w or y1<0 or y2>h:
    #    print('error:')
    #    print(x1,y1,x2,y2)
    #    print(imglist[0].shape)

    for ilabel in tubes:
        for itube in range(len(tubes[ilabel])):
            t = tubes[ilabel][itube]
            t -= np.array([[x1, y1, x1, y1]], dtype=np.float32)

            # check if valid
            cx = 0.5 * (t[:, 0] + t[:, 2])
            cy = 0.5 * (t[:, 1] + t[:, 3])

            if np.any(cx < 0) or np.any(cy < 0) or np.any(cx > wi) or np.any(cy > hi):
                continue

            if not ilabel in out_tubes:
                out_tubes[ilabel] = []

            # clip box
            t[:, 0] = np.maximum(0, t[:, 0])
            t[:, 1] = np.maximum(0, t[:, 1])
            t[:, 2] = np.minimum(wi, t[:, 2])
            t[:, 3] = np.minimum(hi, t[:, 3])

            out_tubes[ilabel].append(t)

    return imglist, out_tubes

'''data loader functions'''
# 封装resize函数
def resize_img_keep_ratio(img,target_size):
    old_size= img.shape[0:2] # 原始图像大小
    ratio = min(float(target_size[i])/(old_size[i]) for i in range(len(old_size))) # 计算原始图像宽高与目标图像大小的比例，并取其中的较小值
    new_size = tuple([int(i*ratio) for i in old_size]) # 根据上边求得的比例计算在保持比例前提下得到的图像大小
    img = cv2.resize(img,(new_size[1], new_size[0])) # 根据上边的大小进行放缩
    pad_w = target_size[1] - new_size[1] # 计算需要填充的像素数目（图像的宽这一维度上）
    pad_h = target_size[0] - new_size[0] # 计算需要填充的像素数目（图像的高这一维度上）
    top,bottom = pad_h//2, pad_h-(pad_h//2)
    left,right = pad_w//2, pad_w -(pad_w//2)
    img_new = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,None,(0,0,0)) 
    return img_new, left, top, ratio

def adjust_bboxes(bboxes, left, top, ratio):
    """
    Adjust an array of bounding boxes based on the resizing and padding information.

    :param bboxes: a NumPy array of shape (N, 4) where each row is [x_min, y_min, x_max, y_max]
    :param left: number of pixels padded on the left side.
    :param top: number of pixels padded on the top side.
    :param ratio: the scaling ratio used during resizing.
    
    :return: a NumPy array of shape (N, 4) with adjusted bounding boxes.
    """
    adjusted_bboxes = np.zeros_like(bboxes)

    for i in range(bboxes.shape[0]):
        x_min, y_min, x_max, y_max = bboxes[i]
        
        # Scale the bounding box coordinates
        new_x_min = int(x_min * ratio)
        new_y_min = int(y_min * ratio)
        new_x_max = int(x_max * ratio)
        new_y_max = int(y_max * ratio)
        
        # Add the padding offsets
        new_x_min += left
        new_y_min += top
        new_x_max += left
        new_y_max += top
        
        adjusted_bboxes[i] = [new_x_min, new_y_min, new_x_max, new_y_max]

    return adjusted_bboxes

def expand_and_crop(images, gt_bbox, _mean_values):
    """
    将图像宽高随机扩展0~0.1，然后裁剪原始图像大小
    images: list[7] -> array(H, W, 3) float32
    gt_bbox: dict[1] -> list[1] -> array(7, 4) float32
    """
    if random.random() < 0.5:
        w_h_expand_ratio = 0.1

        ori_h, ori_w, _ = images[0].shape
        pad_h = int(w_h_expand_ratio * ori_h)
        pad_w = int(w_h_expand_ratio * ori_w)

        # 上下左右扩展
        for i in range(len(images)):
            new_img = np.zeros((pad_h * 2 + ori_h, pad_w * 2 + ori_w, 3), dtype=np.float32) + np.array(_mean_values).reshape(1, 1, 3)
            new_img[pad_h: pad_h + ori_h, pad_w: pad_w + ori_w, :] = images[i]
            images[i] = new_img

        for cls in gt_bbox.keys():
            for i in range(len(gt_bbox[cls])):
                gt_bbox[cls][i] = gt_bbox[cls][i] + np.array([[pad_w, pad_h, pad_w, pad_h]])  # x1, y1, x2, y2

        # 随机裁剪(ori_h, ori_w)大小的图像
        crop_x = np.random.randint(2 * pad_w)
        crop_y = np.random.randint(2 * pad_h)
        for i in range(len(images)):
            images[i] = images[i][crop_y: crop_y + ori_h, crop_x: crop_x + ori_w, :]

        for cls in gt_bbox.keys():
            for i in range(len(gt_bbox[cls])):
                gt_bbox[cls][i] = gt_bbox[cls][i] - np.array([[crop_x, crop_y, crop_x, crop_y]])
                gt_bbox[cls][i] = np.maximum(gt_bbox[cls][i], 0)
                gt_bbox[cls][i][:, 0::2] = np.minimum(gt_bbox[cls][i][:, 0::2], ori_w - 1)
                gt_bbox[cls][i][:, 1::2] = np.minimum(gt_bbox[cls][i][:, 1::2], ori_h - 1)

    return images, gt_bbox

def tubelet_in_tube(tube, i, K):
    # True if all frames from i to (i + K - 1) are inside tube
    # it's sufficient to just check the first and last frame.
    # return (i in tube[: ,0] and i + K - 1 in tube[:, 0])
    return all([j in tube[:, 0] for j in range(i, i + K)])


def tubelet_out_tube(tube, i, K):
    # True if all frames between i and (i + K - 1) are outside of tube
    return all([not j in tube[:, 0] for j in range(i, i + K)])


def tubelet_in_out_tubes(tube_list, i, K):
    # Given a list of tubes: tube_list, return True if
    # all frames from i to (i + K - 1) are either inside (tubelet_in_tube)
    # or outside (tubelet_out_tube) the tubes.
    return all([tubelet_in_tube(tube, i, K) or tubelet_out_tube(tube, i, K) for tube in tube_list])


def tubelet_has_gt(tube_list, i, K):
    # Given a list of tubes: tube_list, return True if
    # the tubelet starting spanning from [i to (i + K - 1)]
    # is inside (tubelet_in_tube) at least a tube in tube_list.
    return any([tubelet_in_tube(tube, i, K) for tube in tube_list])

## 判定在不在关键帧附近
def tubelet_near_key_frame(i, K, key_frame):
    if key_frame == -1:
        return True
    else:
        if abs(i-key_frame) < K:
            return True
        else:
            return False

# BOXES
""" boxes are represented as a numpy array with 4 columns corresponding to the coordinates (x1, y1, x2, y2)"""


def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)


def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""

    xmin = np.maximum(b1[:, 0], b2[:, 0])
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    xmax = np.minimum(b1[:, 2] + 1, b2[:, 2] + 1)
    ymax = np.minimum(b1[:, 3] + 1, b2[:, 3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height


def iou2d(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2"""

    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d(b1, b2)

    return ov / (area2d(b1) + area2d(b2) - ov)


def nms2d(boxes, overlap=0.6):
    """Compute the soft nms given a set of scored boxes,
    as numpy array with 5 columns <x1> <y1> <x2> <y2> <score>
    return the indices of the tubelets to keep
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + 1

    while order.size > 0:
        i = order[0]

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        index = np.where(iou > overlap)[0]
        weight[order[index + 1]] = 1 - iou[index]

        index2 = np.where(iou <= overlap)[0]
        order = order[index2 + 1]

    boxes[:, 4] = boxes[:, 4] * weight

    return boxes


# TUBELETS
""" tubelets of length K are represented using numpy array with 4K columns """


def nms_tubelets(dets, overlapThresh=0.3, top_k=None):
    """Compute the NMS for a set of scored tubelets
    scored tubelets are numpy array with 4K+1 columns, last one being the score
    return the indices of the tubelets to keep
    """

    # If there are no detections, return an empty list
    if len(dets) == 0:
        dets
    if top_k is None:
        top_k = len(dets)

    K = int((dets.shape[1] - 1) / 4)

    # Coordinates of bounding boxes
    x1 = [dets[:, 4 * k] for k in range(K)]
    y1 = [dets[:, 4 * k + 1] for k in range(K)]
    x2 = [dets[:, 4 * k + 2] for k in range(K)]
    y2 = [dets[:, 4 * k + 3] for k in range(K)]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    # area = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = dets[:, -1]
    area = [(x2[k] - x1[k] + 1) * (y2[k] - y1[k] + 1) for k in range(K)]
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + 1
    counter = 0

    while order.size > 0:
        i = order[0]
        counter += 1

        # Compute overlap
        xx1 = [np.maximum(x1[k][i], x1[k][order[1:]]) for k in range(K)]
        yy1 = [np.maximum(y1[k][i], y1[k][order[1:]]) for k in range(K)]
        xx2 = [np.minimum(x2[k][i], x2[k][order[1:]]) for k in range(K)]
        yy2 = [np.minimum(y2[k][i], y2[k][order[1:]]) for k in range(K)]

        w = [np.maximum(0, xx2[k] - xx1[k] + 1) for k in range(K)]
        h = [np.maximum(0, yy2[k] - yy1[k] + 1) for k in range(K)]

        inter_area = [w[k] * h[k] for k in range(K)]
        ious = sum([inter_area[k] / (area[k][order[1:]] + area[k][i] - inter_area[k]) for k in range(K)])
        index = np.where(ious > overlapThresh * K)[0]
        weight[order[index + 1]] = 1 - ious[index]

        index2 = np.where(ious <= overlapThresh * K)[0]
        order = order[index2 + 1]

    dets[:, -1] = dets[:, -1] * weight

    new_scores = dets[:, -1]
    new_order = np.argsort(new_scores)[::-1]
    dets = dets[new_order, :]

    return dets[:top_k, :]


# TUBES
""" tubes are represented as a numpy array with nframes rows and 5 columns (frame, x1, y1, x2, y2). frame number are 1-indexed, coordinates are 0-indexed """


def iou3d(b1, b2):
    """Compute the IoU between two tubes with same temporal extent"""

    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d(b1[:, 1:5], b2[:, 1:5])

    return np.mean(ov / (area2d(b1[:, 1:5]) + area2d(b2[:, 1:5]) - ov))


def iou3dt(b1, b2, spatialonly=False):
    """Compute the spatio-temporal IoU between two tubes"""

    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])

    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0]) + 1

    tube1 = b1[int(np.where(b1[:, 0] == tmin)[0]): int(np.where(b1[:, 0] == tmax)[0]) + 1, :]
    tube2 = b2[int(np.where(b2[:, 0] == tmin)[0]): int(np.where(b2[:, 0] == tmax)[0]) + 1, :]

    return iou3d(tube1, tube2) * (1. if spatialonly else temporal_inter / temporal_union)


def nms3dt(tubes, overlap=0.5):
    """Compute NMS of scored tubes. Tubes are given as list of (tube, score)
    return the list of indices to keep
    """

    if not tubes:
        return np.array([], dtype=np.int32)

    I = np.argsort([t[1] for t in tubes])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0

    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([iou3dt(tubes[ii][0], tubes[i][0]) for ii in I[:-1]])
        I = I[np.where(ious <= overlap)[0]]

    return indices[:counter]

