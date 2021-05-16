import os
import cv2
import random
import math
import numpy as np
import shutil
import json
import argparse

# Traverse folders
def findfile(path, ret,file_state):
    filelist = os.listdir(path)
    for filename in filelist:
        de_path = os.path.join(path, filename)
        if os.path.isfile(de_path):
            if de_path.endswith(file_state):
                ret.append(de_path)
        else:
            findfile(de_path, ret,file_state)


def split_img(img, p_label, v_label, output_size_list=[], over_lap=0.3,save_path_img="",save_path_label="",save_path_label_list=[]):

    zero_list = ["00000","0000","000","00","0"]
    h = len(img)
    w = len(img[0])

    label_dict = {0:[],1:[],2:[],3:[]} # 0 vis_part; 1 global; 2 head; 3 car
    img_id = zero_list[len(str(p_label["image id"]))-1]+str(p_label["image id"])
    p_label = p_label["objects list"]
    for per_label in p_label:
        if per_label["category"] in ["person"]:
            label_dict[0].append([per_label["rects"]["visible body"]["tl"]["x"]*w,per_label["rects"]["visible body"]["tl"]["y"]*h,
                                  per_label["rects"]["visible body"]["br"]["x"]*w,per_label["rects"]["visible body"]["br"]["y"]*h])
            label_dict[1].append([per_label["rects"]["full body"]["tl"]["x"]*w,per_label["rects"]["full body"]["tl"]["y"]*h,
                                  per_label["rects"]["full body"]["br"]["x"]*w,per_label["rects"]["full body"]["br"]["y"]*h])
            label_dict[2].append([per_label["rects"]["head"]["tl"]["x"]*w,per_label["rects"]["head"]["tl"]["y"]*h,
                                  per_label["rects"]["head"]["br"]["x"]*w,per_label["rects"]["head"]["br"]["y"]*h])
    v_label = v_label["objects list"]
    for per_label in v_label:
        if per_label["category"] != "vehicles":
            label_dict[3].append([per_label["rect"]["tl"]["x"]*w,per_label["rect"]["tl"]["y"]*h,
                                  per_label["rect"]["br"]["x"]*w,per_label["rect"]["br"]["y"]*h])

    split_num = 0
    for output_size in output_size_list:
        num_i = int((w // (output_size[0] - output_size[0] * over_lap)) + 1)
        num_j = int((h // (output_size[1] - output_size[1] * over_lap)) + 1)
        for i in range(num_i):
            if i == 0:
                x_now = 0
            else:
                x_now += (output_size[0] - output_size[0]*over_lap)
            if x_now > w - output_size[0]:
                x_now = w - output_size[0]
            for j in range(num_j):
                if j == 0:
                    y_now = 0
                else:
                    y_now += (output_size[1] - output_size[1]*over_lap)
                if y_now > h - output_size[1]:
                    y_now = h - output_size[1]

                label_list = []
                label_list_single = {0:[],1:[],2:[],3:[]}
                for key in label_dict.keys():
                    for label in label_dict[key]:
                        if (x_now < label[0] < x_now + output_size[0] and y_now < label[1] < y_now + output_size[1]) or (x_now < label[2] < x_now + output_size[0] and y_now < label[3] < y_now + output_size[1]):
                            x_c = ((label[0]+label[2])/2-x_now)/output_size[0]
                            y_c = ((label[1]+label[3])/2-y_now)/output_size[1]
                            w_c = (label[2]-label[0])/output_size[0]
                            h_c = (label[3]-label[1])/output_size[1]
                            if not (x_c < 0 or x_c > output_size[0] or y_c < 0 or y_c > output_size[1]):
                                if 0.01 <= w_c <= 0.5 and 0.01 <= h_c <= 0.5:
                                    label_list.append([key,x_c,y_c,w_c,h_c])
                                    label_list_single[key].append([0,x_c,y_c,w_c,h_c])

                #save train dataset
                if len(label_list) != 0:
                    img_save = img[int(y_now):int(y_now + output_size[1]), int(x_now):int(x_now + output_size[0])]
                    txt_path = save_path_label + "/" + img_id + "_" + zero_list[len(str(split_num))-1]+str(split_num)+".txt"
                    txt_path_v = save_path_label_list[0] + "/" + img_id + "_" + zero_list[len(str(split_num)) - 1] + str(
                        split_num) + ".txt"
                    txt_path_g = save_path_label_list[1] + "/" + img_id + "_" + zero_list[len(str(split_num)) - 1] + str(
                        split_num) + ".txt"
                    txt_path_h = save_path_label_list[2] + "/" + img_id + "_" + zero_list[len(str(split_num)) - 1] + str(
                        split_num) + ".txt"
                    txt_path_c = save_path_label_list[3] + "/" + img_id + "_" + zero_list[len(str(split_num)) - 1] + str(
                        split_num) + ".txt"
                    txt_path_list = [txt_path_v,txt_path_g,txt_path_h,txt_path_c]
                    with open(txt_path, "w") as w_txt:
                        for label in label_list:
                            w_txt.write(str(label[0]) + "," + str(label[1]) + "," + str(label[2]) + "," + str(label[3]) + "," + str(label[4]) + "\n")
                            #cv2.rectangle(img_save, (int(label[1]*output_size[0]), int(label[2]*output_size[1])), (int(label[3]*output_size[0]), int(label[4]*output_size[1])), (0,0,255), 4)
                    for key in label_list_single.keys():
                        with open(txt_path_list[key], "w") as w_txt:
                            for label in label_list_single[key]:
                                w_txt.write(str(label[0]) + "," + str(label[1]) + "," + str(label[2]) + "," + str(
                                    label[3]) + "," + str(label[4]) + "\n")
                    img_save = cv2.resize(img_save, (1080, 1080), interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(save_path_img + "/" + img_id + "_" + zero_list[len(str(split_num))-1]+str(split_num)+".jpg",img_save)
                    split_num += 1

                if y_now == h - output_size[1]:
                    break

            if x_now == w - output_size[0]:
                break



# create new directory
def dir_make(path):
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

def catalogue_make(save_path_img):
    ret = []
    findfile(save_path_img, ret, ".jpg")
    with open(save_path+"/panda_train.txt", "w") as w_txt:
        for name in ret:
            w_txt.write(name + "\n")
    with open(save_path+"/panda_val.txt", "w") as w_txt:
        for name in ret[int(len(ret)*0.9):]:
            w_txt.write(name + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_p_json', type=str,
                        default='../tcdata/panda_round1_train_annos_20210222/panda_round1_train_annos_202104/person_bbox_train.json',
                        help='source_json')
    parser.add_argument('--source_v_json', type=str,
                        default='../tcdata/panda_round1_train_annos_20210222/panda_round1_train_annos_202104/vehicle_bbox_train.json',
                        help='source_json')
    parser.add_argument('--img_path', type=str,
                        default='../tcdata/panda_round1_train_20210222',
                        help='source_json')
    opt = parser.parse_args()
    print(opt)

    save_path = "../user_data/cdata"
    save_path_img = save_path + "/images"
    save_path_label = save_path + "/labels"
    save_path_label_v = save_path + "/labels_0"
    save_path_label_g = save_path + "/labels_1"
    save_path_label_h = save_path + "/labels_2"
    save_path_label_c = save_path + "/labels_3"

    dir_make(save_path)
    dir_make(save_path_img)
    dir_make(save_path_label)
    dir_make(save_path_label_v)
    dir_make(save_path_label_g)
    dir_make(save_path_label_h)
    dir_make(save_path_label_c)
    source_p_json = opt.source_p_json
    source_v_json = opt.source_v_json
    with open(source_p_json) as sj:
        sj_p = json.load(sj)
    with open(source_v_json) as sj:
        sj_v = json.load(sj)
    ret = []
    img_path = opt.img_path
    findfile(img_path, ret,".jpg")
    leg = len(ret)
    for i in range(leg):
        print(i/leg)
        img = cv2.imread(ret[i])
        path_sub = ret[i].split("/")
        path_key = path_sub[-2] + "/" + path_sub[-1]
        p_label = sj_p[path_key]
        v_label = sj_v[path_key]
        h = len(img)
        w = len(img[0])
        split_img(img, p_label, v_label, output_size_list=[[1500, 1500],[3000, 3000],[6000, 6000],[10000,10000],[w,h]], over_lap=0.3, save_path_img=save_path_img, save_path_label=save_path_label,save_path_label_list=[save_path_label_v,save_path_label_g,save_path_label_h,save_path_label_c])
    catalogue_make(save_path_img)


