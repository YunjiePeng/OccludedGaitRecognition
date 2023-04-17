import os
import os.path as osp
import cv2
import csv
import numpy as np
import pickle
import argparse
import pdb

from collections import OrderedDict

def load_seq(seq_dir):
    frame_name_list = sorted(os.listdir(seq_dir))
    frame_data_list = []
    for frame_name in frame_name_list:
        frame_data = cv2.imread(osp.join(seq_dir, frame_name), cv2.IMREAD_GRAYSCALE)
        frame_data_list.append(frame_data)
    frame_num = len(frame_name_list)
    seq_data = {'frame_data_list': frame_data_list, 'frame_name_list': frame_name_list, 'org_frame_num': frame_num}
    return seq_data

def save_seq(save_dir, save_info_list):
    if osp.exists(save_dir) is False:
        os.makedirs(save_dir)

    for save_name, save_frame in save_info_list:
        cv2.imwrite(osp.join(save_dir, save_name), save_frame)

def calc_box_for_seq(seq_data):
    frame_data_list = seq_data['frame_data_list']
    frame_name_list = seq_data['frame_name_list']

    new_frame_data_list = []
    new_frame_name_list = []
    box_list = []
    for i, (frame_data, frame_name) in enumerate(zip(frame_data_list, frame_name_list)):
        y = frame_data.sum(axis=1)
        y_top = (y != 0).argmax(axis=0)
        y_btm = (y != 0).cumsum(axis=0).argmax(axis=0)

        x = frame_data.sum(axis=0)
        x_top = (x != 0).argmax(axis=0)
        x_btm = (x != 0).cumsum(axis=0).argmax(axis=0)

        box = [x_top, y_top, x_btm, y_btm]
        if np.array(box).sum() == 0:
            continue
        else:
            new_frame_data_list.append(frame_data)
            new_frame_name_list.append(frame_name)
            box_list.append(box)
    seq_data['frame_data_list'] = new_frame_data_list
    seq_data['frame_name_list'] = new_frame_name_list
    return box_list

def calc_occlusion_ratio(sil_img, occ_img):
    occ_mask = np.zeros(sil_img.shape, dtype=np.uint8)
    sil_mask = np.zeros(sil_img.shape, dtype=np.uint8)
    overlap_mask = np.zeros(sil_img.shape, dtype=np.uint8)

    sil_mask[sil_img > 0] = 1
    occ_mask[occ_img > 0] = 1

    overlap_mask = sil_mask + occ_mask
    occ_area = overlap_mask==2
    occ_area = occ_area.sum()
    sil_area = sil_mask.sum()

    occ_ratio = occ_area / sil_area    
    return occ_ratio

def have_iou(box1, box2):
    x1, y1, x2, y2 = box1
    _x1, _y1, _x2, _y2 = box2
    if (min(x2, _x2) >= max(x1, _x1)) and (min(y2, _y2) >= max(y1, _y1)):
        return True
    else:
        return False

def horizontal_relevant(x1, x2, _x1, _x2):
    if min(x2, _x2) >= max(x1, _x1):
        return True
    else:
        return False

def distortion_check(box1, box2, bottom_gap, img_h):
    if box2[-1] >= img_h:
        return True
    elif have_iou(box1, box2) and (box1[-1] > (box2[-1] - bottom_gap)):
        return False
    else:
        return True

def random_choose_seq_for_crowdocc(exclude_seq_info, input_silhouette_path, pid_num, ideal_frame_num):
    label, seq_type, view = exclude_seq_info

    label_list = sorted(list(os.listdir(input_silhouette_path)))

    if int(label) <= pid_num:
        # Train 
        label_list = label_list[:pid_num]
    else:
        # Test
        label_list = label_list[pid_num:]

    label_list.remove(label)

    while True:
        chosen_label = np.random.choice(label_list)
        seq_type_list = sorted(list(os.listdir(osp.join(input_silhouette_path, chosen_label))))
        chosen_type = np.random.choice(seq_type_list)
        chosen_seq_info = [chosen_label, chosen_type, view]
        chosen_seq_dir = osp.join(input_silhouette_path, *chosen_seq_info)
        if osp.exists(chosen_seq_dir) is True:
            occ_seq_data = load_seq(chosen_seq_dir)
            if len(occ_seq_data['frame_name_list']) > ideal_frame_num:
                break

    occ_seq_data['seq_info'] = chosen_seq_info
    occ_seq_data['frame_box_list'] = calc_box_for_seq(occ_seq_data)
    
    return occ_seq_data

def get_skip_param_for_crowdocc(view, occ_direction, frame_num, occ_frame_num, occ_param):
    ideal_frame_num = occ_param['ideal_frame_num']
    skip_num_max = min(frame_num, occ_frame_num, occ_param['skip_num_max'])

    view = int(view)
    if ((view > 90) and (view < 270)) or ((view == 90) and (occ_direction == 'right')) or ((view == 270) and (occ_direction == 'left')):
        # seq goes first
        go_first = 'seq'
        skip_num_max = min(skip_num_max, max(frame_num - ideal_frame_num, 0))

    elif (view < 90) or (view > 270) or ((view == 90) and (occ_direction == 'left')) or ((view == 270) and (occ_direction == 'right')):
        # occ_seq goes first
        go_first = 'occ_seq'
        skip_num_max = min(skip_num_max, max(occ_frame_num - ideal_frame_num, 0))
    
    return go_first, skip_num_max

def non_occlusion(seq_data):
    org_frame_num = seq_data['org_frame_num']
    frame_data_list = seq_data['frame_data_list']
    frame_name_list = seq_data['frame_name_list']
    seq_info = seq_data['seq_info']
    save_dir = seq_data['save_dir']

    save_list = []
    for frame_data, frame_name in zip(frame_data_list, frame_name_list):
        frame_index = frame_name.split('-')[-1].split('.')[0]
        save_frame_name = seq_data['save_name'][0] + '-' + frame_index + '_' + seq_data['save_name'][1] + '#0.png'
        save_list.append([save_frame_name, frame_data])

    save_seq(save_dir, save_list)
    frame_count = len(save_list)

    save_info = ['-'.join(seq_info), 'nonOcc', '-', '-', org_frame_num, frame_count]
    print(save_info)
    return save_info
    
def generate_detect_occlusion(seq_data, occ_param):
    detect_occ_type = np.random.choice(['top', 'bottom', 'left', 'right'])
    occ_ratio_min = occ_param[detect_occ_type]['occ_ratio_min']
    occ_ratio_max = occ_param[detect_occ_type]['occ_ratio_max']

    org_frame_num = seq_data['org_frame_num']
    frame_data_list = seq_data['frame_data_list']
    frame_name_list = seq_data['frame_name_list']
    frame_box_list = seq_data['frame_box_list']
    seq_info = seq_data['seq_info']
    save_dir = seq_data['save_dir']

    save_list = []
    for frame_data, frame_name, frame_box in zip(frame_data_list, frame_name_list, frame_box_list):
        frame_index = frame_name.split('-')[-1].split('.')[0]

        x1, y1, x2, y2 = frame_box
        box_h = y2 - y1
        box_w = x2 - x1

        sil_area = (frame_data > 0).sum()
        if detect_occ_type == 'top':
            while True:
                occwh_ratio = np.random.uniform(0, 1)
                occ_h = int(box_h * occwh_ratio)
                occ_crop = frame_data[:y1+occ_h, :]
                occ_area = (occ_crop > 0).sum()
                occ_ratio = occ_area / sil_area
                if (occ_ratio > occ_ratio_min) and (occ_ratio < occ_ratio_max):
                    break
            frame_data[:y1+occ_h, :] = 0

        elif detect_occ_type == 'bottom':
            while True:
                occwh_ratio = np.random.uniform(0, 1)
                occ_h = int(box_h * occwh_ratio)
                occ_crop = frame_data[y2-occ_h+1:, :]
                occ_area = (occ_crop > 0).sum()
                occ_ratio = occ_area / sil_area
                if (occ_ratio > occ_ratio_min) and (occ_ratio < occ_ratio_max):
                    break
            frame_data[y2-occ_h+1:, :] = 0

        elif detect_occ_type == 'left':
            while True:
                occwh_ratio = np.random.uniform(0, 1)
                occ_w = int(box_w * occwh_ratio)
                occ_crop = frame_data[:, :x1+occ_w]
                occ_area = (occ_crop > 0).sum()
                occ_ratio = occ_area / sil_area
                if (occ_ratio > occ_ratio_min) and (occ_ratio < occ_ratio_max):
                    break
            frame_data[:, :x1+occ_w] = 0

        elif detect_occ_type == 'right':
            while True:
                occwh_ratio = np.random.uniform(0, 1)
                occ_w = int(box_w * occwh_ratio)
                occ_crop = frame_data[:, x2-occ_w+1:]
                occ_area = (occ_crop > 0).sum()
                occ_ratio = occ_area / sil_area
                if (occ_ratio > occ_ratio_min) and (occ_ratio < occ_ratio_max):
                    break
            frame_data[:, x2-occ_w+1:] = 0

        save_frame_name = seq_data['save_name'][0] + '-' + frame_index + '_' + seq_data['save_name'][1] + '#1.png'
        save_list.append([save_frame_name, frame_data])

    save_seq(save_dir, save_list)
    frame_count = len(save_list)
    save_info = ['-'.join(seq_info), 'detectOcc', detect_occ_type, '-', org_frame_num, frame_count]
    print(save_info)
    return save_info

def generate_static_occlusion(seq_data, occ_param):
    static_occlusion = OrderedDict({
              'Bench': {'min_h_ratio': 0.40, 'max_h_ratio': 0.55}, 
            'Bicycle': {'min_h_ratio': 0.60, 'max_h_ratio': 0.70}, 
        'FireHydrant': {'min_h_ratio': 0.30, 'max_h_ratio': 0.40}, 
         'Motorcycle': {'min_h_ratio': 0.65, 'max_h_ratio': 0.80}, 
               'Pole': {'min_h_ratio': 2.50, 'max_h_ratio': 3.50}, 
           'TrashCan': {'min_h_ratio': 0.45, 'max_h_ratio': 0.60},
    })

    input_staticOcclusion_path = occ_param['input_staticOcclusion_path']
    ideal_frame_num = occ_param['ideal_frame_num']
    occ_ratio_min = occ_param['occ_ratio_min']
    occ_ratio_max = occ_param['occ_ratio_max']
    occ_ratio_ignored = occ_param['occ_ratio_ignored']

    org_frame_num = seq_data['org_frame_num']
    frame_data_list = seq_data['frame_data_list'] 
    frame_name_list = seq_data['frame_name_list']
    frame_box_list = seq_data['frame_box_list']
    seq_info = seq_data['seq_info']
    save_dir = seq_data['save_dir']
    label, seq_type, view = seq_info

    seq_len = len(frame_name_list)
    img_h, img_w = frame_data_list[0].shape

    loop_count = 0
    cur_frame_num_max = 0
    while True:
        if loop_count % 1024 == 0:
            static_occ_type = np.random.choice(list(static_occlusion.keys()))
            static_occ_img_name = np.random.choice(sorted(os.listdir(osp.join(input_staticOcclusion_path, static_occ_type))))
            static_occ_img = cv2.imread(osp.join(input_staticOcclusion_path, static_occ_type, static_occ_img_name), cv2.IMREAD_GRAYSCALE)

        if loop_count % 128 == 0:
            # choose a ref_frame
            ref_index = np.random.choice(seq_len)
            ref_frame = frame_data_list[ref_index]
            x1, y1, x2, y2 = frame_box_list[ref_index]
            box_h = y2 - y1
            box_w = x2 - x1

            # Resize the static occlusion to a proper size.
            h, w = static_occ_img.shape
            static_occ_h_ratio = np.random.uniform(static_occlusion[static_occ_type]['min_h_ratio'], static_occlusion[static_occ_type]['max_h_ratio'])
            static_occ_h = int(box_h * static_occ_h_ratio)
            static_occ_w = int(w * static_occ_h / h)
            static_occ_img_resize = cv2.resize(static_occ_img, (static_occ_w, static_occ_h), interpolation=cv2.INTER_NEAREST)
            occ_bottom_gap = occ_param['occ_bottom_gap'] + 0.01 * static_occ_h
            print("loop_count:{}, static_occ_type:{}, cur_frame_num_max:{}".format(loop_count, static_occ_type, cur_frame_num_max))
        
        loop_count += 1

        # Place the static occlusion at a proper position.
        # X-Coordinate
        static_occ_x1 = np.random.randint(x1 - static_occ_w, x2)
        cur_static_occ_img = static_occ_img_resize.copy()
        static_occ_h, static_occ_w = cur_static_occ_img.shape
        if static_occ_x1 < 0:
            cur_static_occ_img = cur_static_occ_img[:, abs(static_occ_x1):]
            static_occ_x1 = 0
            static_occ_w = cur_static_occ_img.shape[1]
            
        static_occ_x2 = static_occ_x1 + static_occ_w
        if static_occ_x2 > img_w:
            cur_static_occ_img = cur_static_occ_img[:, :-(static_occ_x2 - img_w)]
            static_occ_x2 = img_w
            static_occ_w = cur_static_occ_img.shape[1]

        # Y-Coordinate
        static_occ_y2_max = min(img_h - 1, y2 + static_occ_h)

        if int(view) % 180 == 0:
            static_occ_y2_min = y2 + occ_bottom_gap
        else:
            relevant_frames = []
            for frame_box in frame_box_list:
                relevant_frames.append(horizontal_relevant(frame_box[0], frame_box[2], static_occ_x1, static_occ_x2))

            relevant_frames = np.array(relevant_frames)
            box_frames = np.array(frame_box_list)
            y2_frames = box_frames[:, -1]
            static_occ_y2_min = (y2_frames * relevant_frames).max() + occ_bottom_gap

        static_occ_y2_min = min(img_h - 1, static_occ_y2_min)
        if static_occ_y2_max < static_occ_y2_min:
            continue
        elif static_occ_y2_max == static_occ_y2_min:
            static_occ_y2 = static_occ_y2_max
        else:
            static_occ_y2 = np.random.randint(static_occ_y2_min, static_occ_y2_max)

        static_occ_y1 = static_occ_y2 - static_occ_h
        if static_occ_y1 < 0:
            cur_static_occ_img = cur_static_occ_img[abs(static_occ_y1):, :]
            static_occ_y1 = 0
            static_occ_h = cur_static_occ_img.shape[0]

        static_occ_frame = np.zeros((img_h, img_w), dtype=np.uint8)
        static_occ_frame[static_occ_y1:static_occ_y2, static_occ_x1:static_occ_x2] = cur_static_occ_img

        occ_ratio = calc_occlusion_ratio(ref_frame, static_occ_frame)
        if (occ_ratio > occ_ratio_min) and (occ_ratio < occ_ratio_max):
            static_occ_box = [static_occ_x1, static_occ_y1, static_occ_x2, static_occ_y2]

            save_list = []
            for frame_data, frame_name, frame_box in zip(frame_data_list, frame_name_list, frame_box_list):
                if distortion_check(frame_box, static_occ_box, occ_bottom_gap, img_h):
                    occ_ratio = calc_occlusion_ratio(frame_data, static_occ_frame)
                    if occ_ratio > occ_ratio_ignored:
                        # Large Occlusion, regard as not segmented.
                        continue

                    occ_frame_data = frame_data.copy()
                    occ_frame_data[static_occ_frame > 0] = 128 # for visualization
                    frame_index = frame_name.split('-')[-1].split('.')[0]
                    if occ_ratio > 0:
                        save_frame_name = seq_data['save_name'][0] + '-' + frame_index + '_' + seq_data['save_name'][1] + '#1.png'
                    else:
                        save_frame_name = seq_data['save_name'][0] + '-' + frame_index + '_' + seq_data['save_name'][1] + '#0.png'
                    save_list.append([save_frame_name, occ_frame_data])
                    
            frame_count = len(save_list)
            if frame_count > cur_frame_num_max:
                cur_frame_num_max = frame_count
                temporary_storage = save_list

            if (frame_count < min(ideal_frame_num, seq_len)) and (loop_count < 4096):
                continue
            else:
                save_list = temporary_storage
                break

    save_seq(save_dir, save_list)
    
    save_info = ['-'.join(seq_info), 'staticOcc', static_occ_type, static_occ_img_name, org_frame_num, frame_count]
    print(save_info)
    return save_info

def generate_crowd_occlusion(seq_data, occ_param):
    occ_direction = np.random.choice(['left', 'right'])

    pid_num = occ_param['pid_num']
    input_silhouette_path = occ_param['input_silhouette_path']
    ideal_frame_num = occ_param['ideal_frame_num']
    occ_ratio_min = occ_param['occ_ratio_min']
    occ_ratio_max = occ_param['occ_ratio_max']
    occ_ratio_ignored = occ_param['occ_ratio_ignored']
    occ_bottom_gap = occ_param['occ_bottom_gap']

    org_frame_num = seq_data['org_frame_num']
    frame_data_list = seq_data['frame_data_list']
    frame_name_list = seq_data['frame_name_list']
    frame_box_list = seq_data['frame_box_list']
    seq_info = seq_data['seq_info']
    save_dir = seq_data['save_dir']

    label, seq_type, view = seq_info

    seq_len = len(frame_name_list)
    img_h, img_w = frame_data_list[0].shape
    
    loop_count = 0
    cur_frame_num_max = 0
    while True:
        if loop_count % 1024 == 0:
            occ_seq_data = random_choose_seq_for_crowdocc(seq_info, input_silhouette_path, pid_num, ideal_frame_num)
            occ_seq_info = occ_seq_data['seq_info']
            occ_seq_name = '-'.join(occ_seq_info)
            occ_frame_data_list = occ_seq_data['frame_data_list']
            occ_frame_name_list = occ_seq_data['frame_name_list']
            occ_frame_box_list = occ_seq_data['frame_box_list']

            go_first, skip_num_max = get_skip_param_for_crowdocc(view, occ_direction, len(frame_name_list), len(occ_frame_name_list), occ_param)

        if (loop_count % 128) == 0:
            print("loop_count:{}, cur_frame_num_max:{}".format(loop_count, cur_frame_num_max))

        skip_num = np.random.choice(skip_num_max+1)
        loop_count += 1

        if go_first == 'seq':
            seq_start_index = skip_num
            occ_start_index = 0
        elif go_first == 'occ_seq':
            seq_start_index = 0
            occ_start_index = skip_num

        x1, y1, x2, y2 = frame_box_list[seq_start_index]
        occ_x1, occ_y1, occ_x2, occ_y2 = occ_frame_box_list[occ_start_index]
        occ_w = occ_x2 - occ_x1

        random_vertical_gap = np.random.choice(range(occ_bottom_gap, int(occ_bottom_gap * 2)))

        occ_on_seq_frame_box = np.array(occ_frame_box_list[occ_start_index:])
        if y2 > (occ_y2 - random_vertical_gap):
            vertical_translation = random_vertical_gap + y2 - occ_y2
            occ_on_seq_frame_box[:, 1] += vertical_translation
            occ_on_seq_frame_box[:, 3] += vertical_translation
        
        if occ_direction == 'left':
            occ_x1_max = x1
            occ_x1_min = x1 - occ_w
        elif occ_direction == 'right':
            occ_x1_min = x1
            occ_x1_max = x2

        loop_random_x1 = 0
        while True:
            if loop_random_x1 > 1024:
                break
            random_occ_x1 = np.random.choice(range(occ_x1_min, occ_x1_max))
            horizontal_translation = occ_x1 - random_occ_x1

            cur_occ_x1, cur_occ_y1, cur_occ_x2, cur_occ_y2 = occ_on_seq_frame_box[0]
            
            cur_occ_x1 -= horizontal_translation
            cur_occ_x2 -= horizontal_translation

            cur_occ_frame = np.zeros((img_h, img_w), dtype=np.uint8)

            left_crop = 0 if cur_occ_x1 > 0 else abs(cur_occ_x1)
            right_crop = 0 if cur_occ_x2 < img_w else (cur_occ_x2 - img_w + 1) 

            top_crop = 0 if cur_occ_y1 > 0 else abs(cur_occ_y1)
            bottom_crop = 0 if cur_occ_y2 < img_h else (cur_occ_y2 - img_h + 1)

            cur_occ_x1 = max(0, cur_occ_x1)
            cur_occ_y1 = max(0, cur_occ_y1)

            cur_occ_frame[cur_occ_y1:cur_occ_y2+1, cur_occ_x1:cur_occ_x2+1] = occ_frame_data_list[occ_start_index][occ_y1+top_crop:occ_y2+1-bottom_crop, occ_x1+left_crop:occ_x2+1-right_crop]
            
            occ_ratio = calc_occlusion_ratio(frame_data_list[seq_start_index], cur_occ_frame)
            if (occ_ratio > occ_ratio_min) and (occ_ratio < occ_ratio_max):
                break
            loop_random_x1 += 1

        if loop_random_x1 > 1024:
            continue
        
        occ_on_seq_frame_box[:, 0] -= horizontal_translation
        occ_on_seq_frame_box[:, 2] -= horizontal_translation

        save_list = []
        for i, (frame_data, frame_box, frame_name) in enumerate(zip(frame_data_list[seq_start_index:], frame_box_list[seq_start_index:], frame_name_list[seq_start_index:])):
            frame_index = frame_name.split('-')[-1].split('.')[0]
            if occ_start_index + i < len(occ_frame_name_list):
                occ_x1, occ_y1, occ_x2, occ_y2 = occ_frame_box_list[occ_start_index+i]
                cur_occ_x1, cur_occ_y1, cur_occ_x2, cur_occ_y2 = occ_on_seq_frame_box[i]
                if cur_occ_x2 < 0:
                    save_frame_name = seq_data['save_name'][0] + '-' + frame_index + '_' + seq_data['save_name'][1] + '#0.png'
                    save_list.append([save_frame_name, frame_data])

                elif distortion_check(frame_box, occ_on_seq_frame_box[i], occ_bottom_gap, img_h):
                    cur_occ_frame = np.zeros((img_h, img_w), dtype=np.uint8)
                    left_crop = 0 if cur_occ_x1 > 0 else abs(cur_occ_x1)
                    right_crop = 0 if cur_occ_x2 < img_w else (cur_occ_x2 - img_w + 1) 
                    top_crop = 0 if cur_occ_y1 > 0 else abs(cur_occ_y1)
                    bottom_crop = 0 if cur_occ_y2 < img_h else (cur_occ_y2 - img_h + 1)

                    cur_occ_x1 = max(0, cur_occ_x1)
                    cur_occ_y1 = max(0, cur_occ_y1)

                    cur_occ_frame[cur_occ_y1:cur_occ_y2+1, cur_occ_x1:cur_occ_x2+1] = occ_frame_data_list[occ_start_index+i][occ_y1+top_crop:occ_y2+1-bottom_crop, occ_x1+left_crop:occ_x2+1-right_crop]

                    occ_ratio = calc_occlusion_ratio(frame_data, cur_occ_frame)
                    if occ_ratio > occ_ratio_ignored:
                        # Large Occlusion, regard as not segmented.
                        continue

                    if occ_ratio > 0:
                        save_frame_name = seq_data['save_name'][0] + '-' + frame_index + '_' + seq_data['save_name'][1] + '#1.png'
                    else:
                        save_frame_name = seq_data['save_name'][0] + '-' + frame_index + '_' + seq_data['save_name'][1] + '#0.png'

                    occ_frame_data = frame_data.copy()
                    occ_frame_data[cur_occ_frame > 0] = 128
                    save_list.append([save_frame_name, occ_frame_data])
                else:
                    break
            else:
                break

        frame_count = len(save_list)
        if frame_count > cur_frame_num_max:
            cur_frame_num_max = frame_count
            temporary_storage = save_list
            

        if (frame_count < min(ideal_frame_num, seq_len)) and (loop_count < 4096):
            continue
        else:
            save_list = temporary_storage
            break

    frame_count = len(save_list)
    save_seq(save_dir, save_list)
    
    save_info = ['-'.join(seq_info), 'crowdOcc', occ_direction, occ_seq_name, org_frame_num, frame_count]
    print(save_info)
    return save_info

def get_occ_type_name(occ_type, occ_count_dict):
    occ_count_dict[occ_type] += 1
    occ_type_name = occ_type + '-%02d'%(occ_count_dict[occ_type])
    return occ_type_name

def generate_testset(test_list, start_label, occ_type, occ_param):
    _file = open(opt.csv_testset_path, "w")
    writer = csv.writer(_file)
    writer.writerow(header)
        
    for index, _label in enumerate(test_list):
        _occ_label = "%0{}d".format(len(_label))%(start_label+index+1)
        type_list = sorted(os.listdir(osp.join(opt.input_silhouette_path, _label)))
        for _type in type_list:
            view_list = sorted(os.listdir(osp.join(opt.input_silhouette_path, _label, _type)))
            for _view in view_list:
                seq_info = [_label, _type, _view]
                seq_dir = osp.join(opt.input_silhouette_path, *seq_info)
                seq_data = load_seq(seq_dir)

                if len(seq_data['frame_name_list']) == 0:
                    print("Empty Seq:", seq_info)
                    continue

                if _type in opt.probe_seq:
                    _occ_type = occ_type
                else:
                    _occ_type = 'nonOcc'
                
                save_seq_info = [_occ_label, _type, _view]
                save_dir = osp.join(opt.output_path, *save_seq_info)
                seq_name = '-'.join([_occ_label, _type, _view])

                seq_data['seq_info'] = seq_info
                seq_data['save_dir'] = save_dir
                seq_data['save_name'] = [seq_name, _occ_type]
                if _occ_type != 'nonOcc':
                    seq_data['frame_box_list'] = calc_box_for_seq(seq_data)
                    if len(seq_data['frame_name_list']) == 0:
                        print("Empty Seq:", seq_info)
                        continue

                save_info = generate_occlusion(seq_data, _occ_type, occ_param)
                writer.writerow(save_info)

def generate_occlusion(seq_data, occ_type, occ_param):
    print("occ_type:{}, seq_name:{}".format(occ_type, seq_data['save_name'][0]))
    if occ_type == "nonOcc":
        save_info = non_occlusion(seq_data)
    elif occ_type == "detectOcc":
        save_info = generate_detect_occlusion(seq_data, occ_param['detectOcc'])
    elif occ_type == "staticOcc":
        save_info = generate_static_occlusion(seq_data, occ_param['staticOcc'])
    elif occ_type == "crowdOcc":
        save_info = generate_crowd_occlusion(seq_data, occ_param['crowdOcc'])
    else:
        print("Occlusion type: {} not defined".format(occ_type))
    return save_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Occlusion...")
    #---Dataset
    parser.add_argument("--dataset", default="casia-b", choices=["casia-b"])
    parser.add_argument("--pid_num", default=74, type=int)
    parser.add_argument("--flag", default="train", choices=["train", "test1", "test2", "test3"])
    # test1(detectOcc); test2(staticOcc); test3(crowdOcc)

    #---Input/Output Path---
    parser.add_argument("--input_silhouette_path", default="/data_1/pengyunjie/dataset/casia-bn2/silhouettes/", help="Path to silhouettes.")
    parser.add_argument("--input_staticOcclusion_path", default="/data_1/pengyunjie/dataset/mapillary-vistas/mapillary-vistas/static_occlusions/", help="Path to static occlusions.")
    parser.add_argument("--output_path", default="/data_1/pengyunjie/dataset/casia-bn2/occ/occCASIA-BN_occlab", help="Path to save the generated occluded data.")
    parser.add_argument("--csv_path", default="/data_1/pengyunjie/dataset/casia-bn2/occ/OccCASIA-BN_occlab_train_test1.csv", help="Path to save the information during generation.")
    parser.add_argument("--csv_testset_path", default="/data_1/pengyunjie/dataset/casia-bn2/occ/OccCASIA-BN_occlab_test4.csv", help="Path to save the information during generation.")

    #---Parameters---
    parser.add_argument("--ideal_frame_num_min", default=30, type=int, help="The ideal number of frames for generated occlusion sequence.")
    parser.add_argument("--occ_probability", default=0.6, help="The occlusion probability for training set.")
    parser.add_argument("--occ_type", default=['detectOcc', 'staticOcc', 'crowdOcc'], help="Occlution types.")
    parser.add_argument("--occ_bottom_gap", default=2, type=int, help="The number of pixels between objects for avoiding distortion.")
    parser.add_argument("--detectOcc_param", default={'top':{'occ_ratio_min':0.04, 'occ_ratio_max':0.06}, 'bottom':{'occ_ratio_min':0.04, 'occ_ratio_max':0.12}, 'left':{'occ_ratio_min':0.05, 'occ_ratio_max':0.15}, 'right':{'occ_ratio_min':0.05, 'occ_ratio_max':0.15}}, help="Detect Occlusion parameters.")
    parser.add_argument("--staticOcc_param", default={'occ_ratio_min':0.10, 'occ_ratio_max':0.40, 'occ_ratio_ignored':0.40}, help="Static Occlusion parameters.")
    parser.add_argument("--crowdOcc_param", default={'occ_ratio_min':0.10, 'occ_ratio_max':0.40, 'occ_ratio_ignored':0.40, 'skip_num_max':25}, help="Crowd Occlusion parameters.")
    parser.add_argument("--occ_type_num", default=[4, 2, 2, 2], help="The number of different occlusion types when generating testing set.")
    parser.add_argument("--probe_seq", default=['00'], help="The sequences in Probe for the testing set.")
    
    opt = parser.parse_args()
    
    SEED = 2021
    np.random.seed(SEED)

    opt.staticOcc_param['occ_bottom_gap'] = opt.occ_bottom_gap
    opt.staticOcc_param['ideal_frame_num'] = opt.ideal_frame_num_min
    opt.staticOcc_param['input_staticOcclusion_path'] = opt.input_staticOcclusion_path
    
    opt.crowdOcc_param['pid_num'] = opt.pid_num
    opt.crowdOcc_param['occ_bottom_gap'] = opt.occ_bottom_gap
    opt.crowdOcc_param['ideal_frame_num'] = opt.ideal_frame_num_min
    opt.crowdOcc_param['input_silhouette_path'] = opt.input_silhouette_path

    occ_param = {
        'detectOcc': opt.detectOcc_param,
        'staticOcc': opt.staticOcc_param,
        'crowdOcc': opt.crowdOcc_param,
    }

    label_list = sorted(os.listdir(opt.input_silhouette_path))
    train_list = label_list[:opt.pid_num]
    test_list = label_list[opt.pid_num:]
    header = ["seq_name", 'occ_type', 'occ_subtype', 'occ_param', 'org_frame_num', 'frame_num']

    if opt.flag == "train_test1":
        #---Train, Random0.6---
        _file = open(opt.csv_path, "w")
        writer = csv.writer(_file)
        writer.writerow(header)
    
        for _label in train_list:
            type_list = sorted(os.listdir(osp.join(opt.input_silhouette_path, _label)))
            for _type in type_list:
                view_list = sorted(os.listdir(osp.join(opt.input_silhouette_path, _label, _type)))
                for _view in view_list:
                    seq_info = [_label, _type, _view]
                    seq_dir = osp.join(opt.input_silhouette_path, *seq_info)
                    seq_data = load_seq(seq_dir)
                    if len(seq_data['frame_name_list']) == 0:
                        print("Empty Seq:", seq_info)
                        continue

                    seq_data['seq_info'] = seq_info
                    save_dir = osp.join(opt.output_path, *seq_info)
                    seq_name = '-'.join(seq_info)

                    occ_type = 'nonOcc'
                    occ_random = np.random.uniform(0,1)
                    if occ_random <= opt.occ_probability:
                        # Detect/Static/Crowd Occlusion
                        occ_type = np.random.choice(opt.occ_type)
                        seq_data['frame_box_list'] = calc_box_for_seq(seq_data)
                        if len(seq_data['frame_name_list']) == 0:
                            print("Empty Seq:", seq_info)
                            continue
                        
                    seq_data['save_dir'] = save_dir
                    seq_data['save_name'] = [seq_name, occ_type]
                        
                    save_info = generate_occlusion(seq_data, occ_type, occ_param)
                    writer.writerow(save_info)
                    
    elif opt.flag == "test1":
        #---Test1, crowdOcc---
        start_label = int(label_list[-1]) + len(test_list) * 2
        generate_testset(test_list, start_label, 'crowdOcc', occ_param)

    elif opt.flag == "test2":
        #---Test2, detectOcc---
        start_label = int(label_list[-1])
        generate_testset(test_list, start_label, 'detectOcc', occ_param)
    
    elif opt.flag == "test3":
        #---Test3, staticOcc---
        start_label = int(label_list[-1]) + len(test_list)
        generate_testset(test_list, start_label, 'staticOcc', occ_param)

    
