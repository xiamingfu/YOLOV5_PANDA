import argparse
import time
from pathlib import Path

import cv2
from numpy import random

import os
import json
import os.path as osp
from utils.plots import plot_one_box

def detect(save_img=False):
    source, weights, save_img, save_txt, imgsz = opt.source, opt.weights, opt.save_img, opt.save_txt, opt.img_size
    with open(opt.source_json) as sj:
        sj_img = json.load(sj)

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
    dataset = LoadImages_panda(source, img_size=imgsz, stride=stride, split_size=[3000, 3000], over_lap=1000)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    jdict = []
    for path, img_list, start_list, img0_list, im0s, vid_cap in dataset:
        det_list = {0:[],1:[],2:[],3:[]}
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
            pred_list = {0:[],1:[],2:[],3:[]}
            for p_i in range(4):
                pred_input = torch.cat((pred[:,:,:5],pred[:,:,5+p_i:5+p_i+1]),dim=2)
                pred_output = non_max_suppression(pred_input, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                if pred_list[p_i] == []:
                    pred_list[p_i] = pred_output[0]
                if len(pred_output[0]):
                    pred_output[0][:,5] = p_i
                    pred_list[p_i] = torch.cat((pred_list[p_i], pred_output[0]), dim=0)
                pred_out = [pred_list[p_i]]

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
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        det[..., 0] += start_list[img_i][0]
                        det[..., 2] += start_list[img_i][0]
                        det[..., 1] += start_list[img_i][1]
                        det[..., 3] += start_list[img_i][1]
                        if det_list[p_i] == []:
                            det_list[p_i] = det
                        else:
                            det_list[p_i] = torch.cat((det_list[p_i], det), dim=0)

        det_list_sum = []
        for p_i in range(4):
            if p_i == 0:
                det_list_sum = fuse_all_det(det_list[p_i][:, :6], conf_thres=opt.conf_thres, nms_thres=opt.iou_thres, method='standard')
            else:
                det_list_sum = torch.cat((det_list_sum,fuse_all_det(det_list[p_i][:, :6], conf_thres=opt.conf_thres, nms_thres=opt.iou_thres, method='standard')),dim=0)

        # Print results
        for c in det_list_sum[:, -1].unique():
            n = (det_list_sum[:, -1] == c).sum()  # detections per class
            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        # Write results
        path_sub = path.split("/")
        path_key = path_sub[-2]+"/"+path_sub[-1]

        for *xyxy, conf, cls in reversed(det_list_sum):
            name = ["visable","global","head","car"]
            jdict.append({
                'image_id': sj_img[path_key]["image id"],
                'category_id': int(cls),
                'score': float(conf),
                'bbox_left': float(xyxy[0]),
                'bbox_top': float(xyxy[1]),
                'bbox_width': float(xyxy[2])-float(xyxy[0]),
                'bbox_height': float(xyxy[3])-float(xyxy[1])
                })

            if save_txt:  # Write to file
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                with open(txt_path + '.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if save_img:  # Add bbox to image
                label = f'{name[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=3)

        # Print time (per image)
        t2 = time_synchronized()
        print(f'{s}Done. ({t2 - t1:.3f}s)')

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

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    if opt.save_json:
        with open('det_results.json', 'w') as json_file:
            json.dump(jdict, json_file, indent=4)


    print(f'Done. ({time.time() - t0:.3f}s)')

#历遍文件夹
def findfile(path, ret,file_state):
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(file_state):
                ret.append(de_path)
        else:
            findfile(de_path, ret,file_state)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/data/lc/panda/panda_round1_test_20210222_A/panda_round1_test_202104_A', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source_json', type=str,
                        default='/data/lc/panda/panda_round1_test_A_annos_20210222/panda_round1_test_A_annos_202104/person_bbox_test_A.json',
                        help='source_json')
    parser.add_argument('--source_result', type=str,
                        default='merge.json',
                        help='source_json')
    parser.add_argument('--save_path', type=str,
                        default='vis/',
                        help='source_json')


    opt = parser.parse_args()
    print(opt)
    ret = []
    name = ["visable", "global", "head", "car"]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in name]
    findfile(opt.source, ret, ".jpg")
    with open(opt.source_result) as sj:
        sj_result = json.load(sj)
    img_dict = {}
    for object in sj_result:
        if object["image_id"] not in img_dict.keys():
            img_dict[object["image_id"]] = []
        img_dict[object["image_id"]].append([object["bbox_left"],object["bbox_top"],object["bbox_left"]+object["bbox_width"],object["bbox_top"]+object["bbox_height"],object["score"],object["category_id"]])

    with open(opt.source_json) as sj:
        sj_img = json.load(sj)

    img = cv2.imread(ret[0])
    path_sub = ret[0].split("/")
    path_key = path_sub[-2] + "/" + path_sub[-1]
    for *xyxy, conf, cls in reversed(img_dict[sj_img[path_key]["image id"]]):
        cls = int(cls)-1
        label = f'{name[int(cls)]} {conf:.2f}'
        plot_one_box(xyxy, img, label=label, color=colors[int(cls)], line_thickness=3)
    save_path_img = opt.save_path+str(sj_img[path_key]["image id"])+".jpg"
    cv2.imwrite(save_path_img, img)

