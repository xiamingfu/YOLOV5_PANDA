import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages_panda
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import os
from torchvision.ops import nms
import json
import os.path as osp
from nms_pytorch import soft_nms_pytorch,cluster_nms,cluster_SPM_nms,cluster_diounms,cluster_SPM_dist_nms
import numpy as np
from cython_bbox import bbox_overlaps as bbox_ious
from mpi4py import MPI
import sys
import pandas as pd
from utils.metrics import ConfusionMatrix_PANDA
import glob
from ensemble_boxes_diou import weighted_boxes_fusion

def confusionMatrix_panda(det_list, gt,im0s):
    h,w,_ = im0s.shape
    gt[:,2] = (gt[:,2]-gt[:,4]/2)*w
    gt[:,3] = (gt[:,3]-gt[:,5]/2)*h
    gt[:,4] = gt[:,2]+gt[:,4]*w
    gt[:,5] = gt[:,3]+gt[:,5]*h
    gt = torch.from_numpy(gt)
    confusionMatrix_ret = ConfusionMatrix_PANDA(1,im0s,conf = 0.25, iou_thres = 0.5) 
    confusionMatrix_ret.process_batch(det_list,gt)
    return confusionMatrix_ret.matrix

def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious

def box_iou_self(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None])  # iou = inter / (area1 + area2 - inter)


def WBF_process(im0,detScale):
    h,w,_ = im0.shape
    prediction_box = detScale.cpu()
    prediction_box[:,0] /= w
    prediction_box[:,1] /= h
    prediction_box[:,2] /= w
    prediction_box[:,3] /= h
    boxes = prediction_box[:,:4].tolist() # Format and coordinates conversion
    scores = prediction_box[:,4].tolist()
    labels = prediction_box[:,5].tolist()
    # print(labels)
    return boxes,scores,labels

def WBF_fuse(im0, detScale_list, weights=[], iou_thres=0.5, conf_thres=0.5):
    h,w,_ = im0.shape
    boxes_list,scores_list,labels_list = [],[],[]
    for detScale in detScale_list:
        boxes,scores,labels = WBF_process(im0,detScale)
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thres, skip_box_thr=conf_thres)
    boxes = np.array(boxes)
    scores = np.array(scores)[:,np.newaxis]# increase dimension
    labels = np.array(labels)[:,np.newaxis]
    boxes[:,0] *= w
    boxes[:,1] *= h
    boxes[:,2] *= w
    boxes[:,3] *= h
    output = np.hstack((boxes,scores))
    output = np.hstack((output,labels))
    return torch.from_numpy(output)


def fuse_all_det(prediction,im0,conf_thres=0.5, nms_thres=0.4, method='standard',merge=False):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    Args:
        prediction,
        conf_thres,
        nms_thres,
        method = 'standard' or 'fast'
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate([prediction]):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        #pred[:, :4] = xywh2xyxy(pred[:, :4])

        # Non-maximum suppression
        if method == 'standard':
            nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == 'soft':
            nms_indices = soft_nms_pytorch(pred[:, :4], pred[:, 4], sigma=0.5, thresh=0.2, cuda=1)
        elif method == "cluster":
            nms_indices = cluster_nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == "cluster_SPM":
            nms_indices = cluster_SPM_nms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == "cluster_diou":
            nms_indices = cluster_diounms(pred[:, :4], pred[:, 4], nms_thres)
        elif method == "cluster_SPM_dist":
            nms_indices = cluster_SPM_dist_nms(pred[:, :4], pred[:, 4], nms_thres)
        else:
            raise ValueError('Invalid NMS type!')

        if merge and (1 < nP < 3E3):  # Merge NMS (boxes merged using weighted mean)
            iou = box_iou(pred[:, :4], pred[:, :4]) > nms_thres  # iou matrix
            weights = iou * pred[:, 4][None]  # box weights
            pred[:, :4] = torch.mm(weights, pred[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            #nms_indices = nms_indices[iou.sum(1) > 1]  # require redundancy

        det_max = pred[nms_indices]

        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    #nms_indices = del_include(output[0][:, :4],output[0][:, 4],0.5)
    #output[0] = output[0][nms_indices]
  
    return output[0].cpu()




def write_results(filename, results, data_type):
    num = 0
    id = []
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
                num += 1
                if track_id not in id:
                    id += [track_id]
    print('save results to {}'.format(filename))
    return num,len(id)

def del_more(dets,img_size,boundary,big_thres = 0.3):
    w = (dets[:,2] - dets[:,0]) / img_size[0]
    h = (dets[:,3] - dets[:,1]) / img_size[1]
    size_state = (w < big_thres) * (h < big_thres*2)
    x1_state = (0 <= dets[:, 0]) * (dets[:, 0] <= img_size[0])
    x2_state = (0 <= dets[:, 2]) * (dets[:, 2] <= img_size[0])
    y1_state = (0 <= dets[:, 1]) * (dets[:, 1] <= img_size[1])
    y2_state = (0 <= dets[:, 3]) * (dets[:, 3] <= img_size[1])
    boundary_state = [x1_state,x2_state,y1_state,y2_state]
    index = size_state
    for b_i in range(len(boundary)):
        if boundary[b_i] == 0:
            index *= boundary_state[b_i]
    return dets[index]


def detect(save_img=False):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
    source, weights, save_img, save_txt, imgsz = opt.source, opt.weights, opt.save_img, opt.save_txt, opt.img_size

    #save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    print(save_dir)
    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # dataset = LoadImages_panda(source, img_size=imgsz, stride=stride, split_size=[[1000, 1000],[1500, 1500], [3000, 3000],[6000, 6000],[10000,10000]], over_lap=0.3)
    dataset = LoadImages_panda(source, img_size=imgsz, stride=stride, split_size=[[1500, 1500], [3000, 3000],[6000, 6000],[10000,10000]], over_lap=0.3)
    #dataset = LoadImages_panda(source, img_size=imgsz, stride=stride,split_size=[], over_lap=0.3)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    ret_matrix = np.zeros((2, 2))

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    jdict = []
    results = []
    for path, img_list, start_list, split_size_list,boundary_list, img0_list, im0s, vid_cap in dataset:
        det_list = []
        det_scale_dict = {}
        # Inference
        t1 = time_synchronized()
        for img_i in range(len(img_list)):
            img = torch.from_numpy(img_list[img_i]).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            pred = model(img, augment=opt.augment)[0]

            # Apply
            pred_list = []

            pred_output = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            if pred_list == []:
                pred_list = pred_output[0]
            if len(pred_output[0]):
                pred_list= torch.cat((pred_list, pred_output[0]), dim=0)
            pred_out = [pred_list]

            # Apply Classifier
            if classify:
                pred_out = apply_classifier(pred_out, modelc, img, img0_list[img_i])

            # Process detections
            for i, det in enumerate(pred_out):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, img0_list[img_i][i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', img0_list[img_i], getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                det = del_more(det, img.shape[2:], big_thres=0.3, boundary=boundary_list[img_i])
                
                # del_small
                if len(det) != 0:
                    small_thres = 15
                    dets_wh_thres = det[:, 2:4] - det[:, :2]
                    det_thres = torch.minimum(dets_wh_thres[:, 0], dets_wh_thres[:, 1])
                    if split_size_list[img_i] > 2000:
                        det = det[det_thres > small_thres]
                
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    det[..., 0] += start_list[img_i][0]
                    det[..., 2] += start_list[img_i][0]
                    det[..., 1] += start_list[img_i][1]
                    det[..., 3] += start_list[img_i][1]
                    if det_list == []:
                        det_list = det
                    else:
                        det_list = torch.cat((det_list, det), dim=0)
                    
                    if split_size_list[img_i] not in det_scale_dict.keys():
                        det_scale_dict[split_size_list[img_i]] = det
                    else:
                        det_scale_dict[split_size_list[img_i]] = torch.cat((det_scale_dict[split_size_list[img_i]], det), dim=0)

        det_list = fuse_all_det(det_list[:, :6],im0,conf_thres=opt.conf_thres, nms_thres=opt.iou_thres, method='standard',merge=False)
        
        scale_key = []
        for key in det_scale_dict.keys():
            scale_key.append(key)
        det_list = WBF_fuse(im0, [det_list,det_scale_dict[scale_key[0]]], weights=[1,1], iou_thres=0.5, conf_thres=0.5)
        

        # Print results
        for c in det_list[:, -1].unique():
            n = (det_list[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        frame_id = int(path.split(".")[-2].split("_")[-1])
        id_list = []
        online_tlwhs = []
        for *xyxy, conf, cls in reversed(det_list):
            online_tlwhs.append((float(xyxy[0]),float(xyxy[1]),float(xyxy[2])-float(xyxy[0]),float(xyxy[3])-float(xyxy[1])))
            id_list.append(conf)
            if save_img:  # Add bbox to image
                plot_one_box(xyxy, im0s, label=0, color=colors[int(cls)], line_thickness=3)
        results.append((frame_id, online_tlwhs, id_list))
        # Print time (per image)
        t2 = time_synchronized()
        print(f'{s}Done. ({t2 - t1:.3f}s)')
        sys.stdout.flush()

        # Save results (image with detections)
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0s)
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    result_detection, id_num = write_results(opt.source.split("/")[-1]+".txt", results, "mot")
    print("detection_num",result_detection)
    print("id_num",id_num)
    print(f'Done. ({time.time() - t0:.3f}s)')
    print(f'object num. ({len(jdict)})')
    sys.stdout.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5_panda.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='../tcdata/panda_round1_train_202104_part1/03_Train_Station Square', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_img', action='store_true', help='display results')
    parser.add_argument('--save_txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_json', action='store_true', help='save results to *.txt')
    parser.add_argument('--save_conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')


    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    if rank == 0:
        opt.device = "0"
        opt.gpus = opt.device
        opt.source = '../tcdata/panda_round2_train_20210331_part10/10_Huaqiangbei'
        # opt.source = '../tcdata/panda_round2_test_20210331_B_part1/14_Ceremony'

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                detect()
    elif rank == 1:
        opt.device = "1"
        opt.gpus = opt.device
        #opt.source = '../tcdata/panda_round2_test_20210331_A_part2/12_Nanshan_i_Park'
        opt.source = '../tcdata/panda_round2_test_20210331_B_part2/15_Dongmen_Street'
        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                detect()
    elif rank == 2:
        opt.device = "2"
        opt.gpus = opt.device
        opt.source = '../tcdata/panda_round2_test_20210331_A_part3/13_University_Playground'
        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    detect()
                    strip_optimizer(opt.weights)
            else:
                detect()


