from icecream import ic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.globals import CurrentConfig, NotebookType
import os
import cv2
import pylab as pl
from PIL import Image
import ujson as json

CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
CurrentConfig.ONLINE_HOST


def visualization_of_trajectories(open_path):
    path_save_list = adjust_image_size(open_path)
    h = np.array([[1.1048200e-02, 6.6958900e-04, -3.3295300e+00],
                  [-1.5966000e-03, 1.1632400e-02, -5.3951400e+00],
                  [1.1190700e-04, 1.3617400e-05, 5.4276600e-01]])
    numpy_list_128_true = np.array([[1.863, 0.305, 1],
                                    [1.862, -0.388, 1],
                                    [1.969, -1.117, 1],
                                    [2.004, -1.836, 1],
                                    [2.004, -2.492, 1],
                                    [2.129, -3.241, 1],
                                    [2.179, -4.001, 1],
                                    [2.299, -4.828, 1]])
    numpy_list_128_pred = np.array([[1.863, 0.305, 1],
                                    [1.862, 0.305, 1],
                                    [1.804, - 0.389, 1],
                                    [2.032, - 1.101, 1],
                                    [2.043, - 1.654, 1],
                                    [2.135, - 2.409, 1],
                                    [2.210, - 3.003, 1],
                                    [2.278, - 3.650, 1]])
    seq_hotel_analyze_id(path_save_list[0], h, numpy_list_128_true, numpy_list_128_pred)
    numpy_list_129_true = np.array([[3.687, -2.853, 1],
                                    [3.541, -2.581, 1],
                                    [3.488, -2.242, 1],
                                    [3.393, -2.015, 1],
                                    [3.243, -1.799, 1],
                                    [3.181, -1.615, 1],
                                    [3.118, -1.45, 1],
                                    [3.101, -1.428, 1]])
    numpy_list_129_pred = np.array([[3.687, - 2.853, 1],
                                    [3.688, - 2.853, 1],
                                    [3.533, - 2.548, 1],
                                    [3.506, - 2.117, 1],
                                    [3.401, - 1.929, 1],
                                    [3.159, - 1.664, 1],
                                    [3.131, - 1.529, 1],
                                    [3.108, - 1.406, 1]])
    seq_hotel_analyze_id(path_save_list[1], h, numpy_list_129_true, numpy_list_129_pred)
    numpy_list_345_true = np.array([[2.072, - 9.998, 1],
                                    [2.106, - 9.379, 1],
                                    [2.197, - 8.724, 1],
                                    [2.288, - 8.051, 1],
                                    [2.363, - 7.359, 1],
                                    [2.472, - 6.691, 1],
                                    [2.546, - 6.002, 1],
                                    [2.602, - 5.293, 1],
                                    [2.731, - 4.594, 1],
                                    [2.804, - 3.91, 1],
                                    [2.967, - 3.238, 1],
                                    [3.074, - 2.602, 1],
                                    [3.166, - 1.887, 1],
                                    [3.307, - 1.279, 1],
                                    [3.379, - 0.605, 1],
                                    [3.414, 0.053, 1],
                                    [3.484, 0.705, 1],
                                    [3.575, 1.391, 1],
                                    [3.557, 2.055, 1],
                                    [3.552, 2.639, 1]])
    numpy_list_345_pred = np.array([[2.072, - 9.998, 1],
                                    [2.072, - 9.998, 1],
                                    [2.083, - 9.355, 1],
                                    [2.204, - 8.742, 1],
                                    [2.404, - 8.145, 1],
                                    [2.433, - 7.324, 1],
                                    [2.567, - 6.958, 1],
                                    [2.605, - 6.230, 1],
                                    [2.604, - 5.751, 1],
                                    [2.652, - 5.377, 1],
                                    [2.661, - 4.761, 1],
                                    [2.643, - 4.094, 1],
                                    [2.767, - 3.614, 1],
                                    [2.919, - 3.275, 1],
                                    [3.253, - 2.510, 1],
                                    [3.332, - 2.210, 1],
                                    [3.360, - 1.932, 1],
                                    [3.247, - 1.873, 1],
                                    [3.424, - 1.918, 1],
                                    [3.544, - 1.799, 1]])
    seq_hotel_analyze_id(path_save_list[2], h, numpy_list_345_true, numpy_list_345_pred)
    numpy_list_392_true = np.array([[3.887, - 4.105, 1],
                                    [3.523, - 4.262, 1],
                                    [3.115, - 4.18, 1],
                                    [2.789, - 4.185, 1],
                                    [2.455, - 4.308, 1],
                                    [2.163, - 4.298, 1],
                                    [1.872, - 4.228, 1],
                                    [1.63, - 4.243, 1],
                                    [1.342, - 4.394, 1],
                                    [1.022, - 4.421, 1],
                                    [0.686, - 4.347, 1],
                                    [0.363, - 4.355, 1],
                                    [- 0.142, - 4.487, 1],
                                    [- 0.384, - 4.32, 1],
                                    [- 0.678, - 4.373, 1],
                                    [- 0.975, - 4.446, 1],
                                    [- 1.23, - 4.441, 1],
                                    [- 1.365, - 4.387, 1],
                                    [- 1.365, - 4.387, 1],
                                    [- 1.365, - 4.387, 1]])
    numpy_list_392_pred = np.array([[3.887, - 4.105, 1],
                                    [3.888, - 4.106, 1],
                                    [3.558, - 4.237, 1],
                                    [3.098, - 4.109, 1],
                                    [2.662, - 4.137, 1],
                                    [2.322, - 4.266, 1],
                                    [1.951, - 4.248, 1],
                                    [1.752, - 4.188, 1],
                                    [1.491, - 4.222, 1],
                                    [1.153, - 4.234, 1],
                                    [0.755, - 4.301, 1],
                                    [0.652, - 4.377, 1],
                                    [0.397, - 4.502, 1],
                                    [0.184, - 4.591, 1],
                                    [- 0.125, - 4.683, 1],
                                    [- 0.398, - 4.736, 1],
                                    [- 0.445, - 4.726, 1],
                                    [- 0.644, - 4.853, 1],
                                    [- 0.977, - 4.937, 1],
                                    [- 1.377, - 5.060, 1]])
    seq_hotel_analyze_id(path_save_list[3], h, numpy_list_392_true, numpy_list_392_pred)


# if you want to add some new pictures, please add some code here following by this.
def adjust_image_size(open_path):
    path_save_list = []
    dirs = os.listdir(open_path)
    for i in range(len(dirs)):
        if dirs[i] == 'eth':
            eth_path = open_path + '\\' + dirs[i]
            dirs_eth = os.listdir(eth_path)
        if dirs[i] == 'hotel':
            hotel_path = open_path + '\\' + dirs[i]
            dirs_hotel = os.listdir(hotel_path)
    for i in range(len(dirs_eth)):
        if dirs_eth[i] == 'map_eth':
            eth_source_image_path = eth_path + '\\' + dirs_eth[i]
    for i in range(len(dirs_hotel)):
        if dirs_hotel[i] == 'map_hotel.jpg':
            hotel_source_image_path = hotel_path + '\\' + dirs_hotel[i]
        if dirs_hotel[i] == 'img1.jpg':
            hotel_revise1_image_path = hotel_path + '\\' + dirs_hotel[i]
            hotel_revise1_image_path_save = hotel_path + '\\' + 'img1_rev.png'
        if dirs_hotel[i] == 'img2.jpg':
            hotel_revise2_image_path = hotel_path + '\\' + dirs_hotel[i]
            hotel_revise2_image_path_save = hotel_path + '\\' + 'img2_rev.png'
        if dirs_hotel[i] == 'img3.jpg':
            hotel_revise3_image_path = hotel_path + '\\' + dirs_hotel[i]
            hotel_revise3_image_path_save = hotel_path + '\\' + 'img3_rev.png'
        if dirs_hotel[i] == 'img4.jpg':
            hotel_revise4_image_path = hotel_path + '\\' + dirs_hotel[i]
            hotel_revise4_image_path_save = hotel_path + '\\' + 'img4_rev.png'
    hotel_image = Image.open(hotel_source_image_path)
    hotel_image_revise1 = Image.open(hotel_revise1_image_path)
    hotel_image_revise2 = Image.open(hotel_revise2_image_path)
    hotel_image_revise3 = Image.open(hotel_revise3_image_path)
    hotel_image_revise4 = Image.open(hotel_revise4_image_path)
    hotel_image = np.array(hotel_image)
    hotel_image_rev1 = hotel_image_revise1.resize((hotel_image.shape[1], hotel_image.shape[0]),
                                                  resample=Image.Resampling.BILINEAR)
    hotel_image_rev2 = hotel_image_revise2.resize((hotel_image.shape[1], hotel_image.shape[0]),
                                                  resample=Image.Resampling.BILINEAR)
    hotel_image_rev3 = hotel_image_revise3.resize((hotel_image.shape[1], hotel_image.shape[0]),
                                                  resample=Image.Resampling.BILINEAR)
    hotel_image_rev4 = hotel_image_revise4.resize((hotel_image.shape[1], hotel_image.shape[0]),
                                                  resample=Image.Resampling.BILINEAR)
    hotel_image_rev1.save(hotel_revise1_image_path_save)
    hotel_image_rev2.save(hotel_revise2_image_path_save)
    hotel_image_rev3.save(hotel_revise3_image_path_save)
    hotel_image_rev4.save(hotel_revise4_image_path_save)
    path_save_list.append(hotel_revise1_image_path_save)
    path_save_list.append(hotel_revise2_image_path_save)
    path_save_list.append(hotel_revise3_image_path_save)
    path_save_list.append(hotel_revise4_image_path_save)
    return path_save_list


def seq_hotel_analyze_id(im_src_path, h, numpy_list_id_true, numpy_list_id_pred):
    im_src = cv2.imread(im_src_path)
    finally_coordinate_id_true = []
    finally_coordinate_id_pred = []
    finally_coordinate_id_true_new = []
    finally_coordinate_id_pred_new = []
    h_inv = np.linalg.inv(h)
    for i in range(len(numpy_list_id_true)):
        res_temp = h_inv @ numpy_list_id_true[i]
        finally_coordinate_id_true.append(res_temp)
    for i in range(len(numpy_list_id_pred)):
        res_temp = h_inv @ numpy_list_id_pred[i]
        finally_coordinate_id_pred.append(res_temp)
    for i in range(len(finally_coordinate_id_true)):
        temp = []
        temp.append(finally_coordinate_id_true[i][0] / finally_coordinate_id_true[i][2])
        temp.append(finally_coordinate_id_true[i][1] / finally_coordinate_id_true[i][2])
        temp.append(1.0)
        finally_coordinate_id_true_new.append(temp)
    for i in range(len(finally_coordinate_id_pred)):
        temp = []
        temp.append(finally_coordinate_id_pred[i][0] / finally_coordinate_id_pred[i][2])
        temp.append(finally_coordinate_id_pred[i][1] / finally_coordinate_id_pred[i][2])
        temp.append(1.0)
        finally_coordinate_id_pred_new.append(temp)
    pl.figure(), \
    pl.imshow(im_src[:, :, ::-1]), \
    pl.scatter([finally_coordinate_id_true_new[i][1] for i in range(len(finally_coordinate_id_true_new))],
               [finally_coordinate_id_true_new[i][0] for i in range(len(finally_coordinate_id_true_new))],
               color='black',
               s=5), \
    pl.scatter([finally_coordinate_id_pred_new[i][1] for i in range(len(finally_coordinate_id_pred_new))],
               [finally_coordinate_id_pred_new[i][0] for i in range(len(finally_coordinate_id_pred_new))],
               color='green',
               s=5), \
    pl.show()


def seq_hotel_destination():
    file_path = './ETH-dataset/seq_hotel/destinations.txt'
    list_small, list_big = [], []
    list_pos_x, list_pos_y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            list_small = []
            temp = line.replace('\n', '').split(' ')
            for i in range(len(temp)):
                if temp[i] == '':
                    continue
                list_small.append(float(temp[i]))
            list_big.append(list_small)
    f.close()
    ic(list_big)
    for i in range(len(list_big)):
        list_pos_x.append(list_big[i][0])
        list_pos_y.append(list_big[i][1])
    ic(max(list_pos_x), min(list_pos_x))
    ic(max(list_pos_y), min(list_pos_y))


def seq_hotel_max_min_position():
    file_path = './ETH-dataset/seq_hotel/obsmat.txt'
    list_pos_x, list_pos_y = [], []
    with open(file_path, 'r') as f:
        for line in f:
            temp = line.replace('\n', '').split(' ')
            list_pos_x.append(float(temp[2]))
            list_pos_y.append(float(temp[3]))
    f.close()
    ic(max(list_pos_x), min(list_pos_x))
    ic(max(list_pos_y), min(list_pos_y))


def seq_hotel_analyze2(pred_length):
    file_path = './ETH-dataset/seq_hotel/obsmat2.txt'
    dict_key, dict_key_eight = {}, {}
    list_frame_id, list_ped_id, list_pos_x, list_pos_y = [], [], [], []
    cnt_8 = 0
    with open(file_path, 'r') as f:
        for line in f:
            temp = line.replace('\n', '').split(' ')
            frame_number = temp[1]
            if frame_number not in dict_key:
                dict_key[frame_number] = 1
            else:
                dict_key[frame_number] += 1
    f.close()
    for key in dict_key:
        if dict_key[key] == pred_length:
            if key not in dict_key_eight:
                dict_key_eight[key] = 1
            else:
                dict_key_eight[key] += 1
    with open(file_path, 'r') as f:
        for line in f:
            temp = line.replace('\n', '').split(' ')
            frame_number = temp[1]
            if frame_number not in dict_key_eight:
                continue
            list_frame_id.append(float(temp[0]))
            list_ped_id.append(float(temp[1]))
            list_pos_x.append(float(temp[2]))
            list_pos_y.append(float(temp[3]))
    f.close()
    file_path_save = './ETH-dataset/seq_hotel/obsmat' + str(pred_length) + '.txt'
    with open(file_path_save, 'w') as f:
        for i in range(len(list_frame_id)):
            f.write(str(list_frame_id[i]) + ' ' + str(list_ped_id[i]) + ' ' + str(list_pos_x[i]) + ' ' + str(
                list_pos_y[i]) + '\n')
    f.close()


def seq_hotel_analyze():
    file_path = './ETH-dataset/seq_hotel/obsmat.txt'
    list_big = []
    list_frame, list_ped_id, list_pos_x, list_pos_y = [], [], [], []
    with open(file_path, 'r') as f:
        for line in f:
            list_small = []
            temp = line.replace('\n', '').split(' ')
            for i in range(len(temp)):
                if temp[i] == '':
                    continue
                list_small.append(float(temp[i]))
            list_big.append(list_small)
    f.close()
    for i in range(len(list_big)):
        list_frame.append(int(list_big[i][0] * 1000) / 1000)
        list_ped_id.append(int(list_big[i][1] * 1000) / 1000)
        list_pos_x.append(int(list_big[i][2] * 1000) / 1000)
        list_pos_y.append(int(list_big[i][4] * 1000) / 1000)
    path_save = './ETH-dataset/seq_hotel/obsmat2.txt'
    with open(path_save, 'w') as f:
        for i in range(len(list_frame)):
            f.write(str(list_frame[i]) + ' ' + str(list_ped_id[i]) + ' ' + str(list_pos_x[i]) + ' ' + str(
                list_pos_y[i]) + '\n')
    f.close()
    # ic(max(list_pos_x), min(list_pos_x))
    # ic(max(list_pos_y), min(list_pos_y))


def biwi_dataset_analyze():
    BIWI_ADE_wo = [0.9642, 0.9716, 0.974, 0.9641, 0.983, 0.9753, 0.9731, 0.9643, 0.9659, 0.9592, 0.9678, 0.9579, 0.963,
                   0.9675, 0.9523, 0.9588, 0.9661, 0.9615, 0.9729, 0.9532, 0.9712, 0.9737, 0.9569, 0.969, 0.9797,
                   0.9632, 0.9556, 0.9679, 0.9485, 0.976]
    BIWI_FDE_wo = [1.2744, 1.3073, 1.3123, 1.285, 1.3267, 1.3167, 1.3192, 1.2981, 1.2891, 1.2687, 1.2887, 1.2654,
                   1.2968, 1.3077, 1.277, 1.2656, 1.3005, 1.2804, 1.325, 1.2743, 1.3046, 1.3052, 1.2929, 1.3085, 1.3272,
                   1.2978, 1.2775, 1.3021, 1.2816, 1.3219]
    BIWI_ADE_w = [0.8521, 0.8414, 0.8308, 0.8392, 0.8294, 0.8381, 0.8232, 0.8456, 0.8359, 0.8509, 0.8408, 0.8313,
                  0.8479, 0.8323, 0.8381, 0.8466, 0.8405, 0.8412, 0.8296, 0.8367, 0.8368, 0.8418, 0.8454, 0.8372,
                  0.8372, 0.8365, 0.8296, 0.8296, 0.8379, 0.8314]
    BIWI_FDE_w = [1.1324, 1.0596, 1.0294, 1.0528, 1.0273, 1.0468, 0.9922, 1.0829, 1.0762, 1.1068, 1.073, 1.0334, 1.0528,
                  1.0402, 1.0507, 1.0789, 1.0561, 1.0661, 1.0598, 1.0964, 1.0717, 1.0771, 1.0688, 1.0613, 1.0535,
                  1.0702, 1.0329, 1.0329, 1.0757, 1.0481]
    BIWI_ADE_S = [1.0958, 1.0775, 1.0967, 1.1054, 1.0798, 1.0765, 1.0981, 1.0876, 1.1104, 1.1049, 1.0994, 1.1042,
                  1.1055, 1.0948, 1.1087, 1.1138, 1.0854, 1.0866, 1.0823, 1.0916, 1.0908, 1.0918, 1.0889, 1.1186,
                  1.0875, 1.1124, 1.1136, 1.1016, 1.0834, 1.0885]
    BIWI_FDE_S = [1.5391, 1.5529, 1.5673, 1.6006, 1.4819, 1.5212, 1.5596, 1.5006, 1.5624, 1.6413, 1.5776, 1.554, 1.6365,
                  1.5809, 1.6243, 1.5749, 1.5605, 1.5369, 1.5596, 1.5245, 1.5363, 1.58, 1.5525, 1.5787, 1.5493, 1.5484,
                  1.58, 1.5544, 1.5652, 1.5496]
    BIWI_ADE_test_wo = [1.0121, 1.0285, 1.0157, 1.0294, 1.0308, 1.023, 1.0309, 1.0246, 1.0332, 1.0323, 1.0223, 1.0303,
                        1.0225, 1.0315, 1.025, 1.0319, 1.0255, 1.015, 1.0312, 1.015, 1.0081, 1.025, 1.0349, 1.0149,
                        1.0224, 1.0199, 1.0303, 1.0276, 1.0279, 1.0188]
    BIWI_FDE_test_wo = [1.3807, 1.4312, 1.4042, 1.4449, 1.446, 1.4055, 1.4217, 1.4116, 1.4274, 1.4363, 1.4057, 1.4518,
                        1.4207, 1.4406, 1.4317, 1.4316, 1.3811, 1.3831, 1.4162, 1.3985, 1.3997, 1.4023, 1.4422, 1.4066,
                        1.411, 1.4188, 1.4284, 1.4192, 1.4435, 1.3834]
    BIWI_ADE_test_w = [0.91, 0.8872, 0.8922, 0.91, 0.9104, 0.8843, 0.9064, 0.9005, 0.9203, 0.8836, 0.8962, 0.9099,
                       0.8939, 0.8904, 0.8988, 0.9098, 0.8936, 0.8947, 0.8918, 0.906, 0.8949, 0.9737, 0.8973, 0.9045,
                       0.897, 0.9113, 0.9096, 0.9019, 0.9081, 0.8817]
    BIWI_FDE_test_w = [1.2448, 1.1703, 1.2012, 1.2046, 1.2305, 1.1596, 1.2098, 1.1557, 1.2229, 1.1481, 1.1816, 1.2246,
                       1.1911, 1.2208, 1.2014, 1.2287, 1.1951, 1.2033, 1.1958, 1.1716, 1.1888, 1.1018, 1.1849, 1.2107,
                       1.2505, 1.2417, 1.2507, 1.1992, 1.2217, 1.1762]
    # plt.plot([i for i in range(len(BIWI_ADE_wo))], BIWI_ADE_wo, '--.', color='blue', linewidth=1,
    #         label='BIWI_ADE_wo')
    # plt.plot([i for i in range(len(BIWI_FDE_wo))], BIWI_FDE_wo, '-v', color='#d725de',
    #         linewidth=1, label='BIWI_FDE_wo')

    # plt.plot([i for i in range(len(BIWI_ADE_w))], BIWI_ADE_w, '--.', color='blue', linewidth=1,
    #         label='BIWI_ADE_w')
    # plt.plot([i for i in range(len(BIWI_FDE_w))], BIWI_FDE_w, '-v', color='#d725de',
    #         linewidth=1, label='BIWI_FDE_w')

    # plt.plot([i for i in range(len(BIWI_ADE_S))], BIWI_ADE_S, '--.', color='blue', linewidth=1,
    #        label='BIWI_ADE_S')
    # plt.plot([i for i in range(len(BIWI_FDE_S))], BIWI_FDE_S, '-v', color='#d725de',
    #        linewidth=1, label='BIWI_FDE_S')

    # plt.plot([i for i in range(len(BIWI_ADE_test_wo))], BIWI_ADE_test_wo, '--.', color='blue', linewidth=1,
    #        label='BIWI_ADE_test_wo')
    # plt.plot([i for i in range(len(BIWI_FDE_test_wo))], BIWI_FDE_test_wo, '-v', color='#d725de',
    #        linewidth=1, label='BIWI_FDE_test_wo')

    plt.plot([i for i in range(len(BIWI_ADE_test_w))], BIWI_ADE_test_w, '--.', color='blue', linewidth=1,
             label='BIWI_ADE_test_w')
    plt.plot([i for i in range(len(BIWI_FDE_test_w))], BIWI_FDE_test_w, '-v', color='#d725de',
             linewidth=1, label='BIWI_FDE_test_w')

    plt.legend()
    plt.show()


def sdd_dataset_analyze():
    sdd_ade_source = [0.4302, 0.4306, 0.4303, 0.4297, 0.4316, 0.4309, 0.428, 0.4305, 0.4319, 0.4311, 0.4299, 0.4294,
                      0.4293, 0.4303, 0.4306, 0.4312, 0.4307, 0.4303, 0.4288, 0.4301, 0.4305, 0.4297, 0.4293, 0.4299,
                      0.4301, 0.4297, 0.4296, 0.4293, 0.4309, 0.4304]
    sdd_fde_source = [0.5033, 0.4991, 0.4974, 0.4988, 0.5034, 0.5004, 0.4869, 0.5006, 0.5041, 0.5008, 0.5006, 0.5028,
                      0.5004, 0.5007, 0.5005, 0.5, 0.5053, 0.5014, 0.4965, 0.5017, 0.5038, 0.5023, 0.497, 0.5, 0.4949,
                      0.4982, 0.5022, 0.5013, 0.4992, 0.5019]
    sdd_ade_revise_noparticle = [0.3914, 0.3926, 0.3924, 0.393, 0.392, 0.3916, 0.3916, 0.3935, 0.3913, 0.3919, 0.3927,
                                 0.3922, 0.3925, 0.3927, 0.3907, 0.3912, 0.3921, 0.3928, 0.3935, 0.3907, 0.3925, 0.3918,
                                 0.3921, 0.3915, 0.393, 0.3924, 0.3919, 0.3924, 0.3921, 0.3917]
    sdd_fde_revise_noparticle = [0.434, 0.434, 0.4351, 0.4346, 0.4336, 0.4354, 0.4298, 0.4368, 0.4326, 0.4354, 0.4348,
                                 0.4346, 0.4345, 0.433, 0.4318, 0.4317, 0.4342, 0.4363, 0.4344, 0.4346, 0.4335, 0.4361,
                                 0.4348, 0.4348, 0.4381, 0.4341, 0.4348, 0.4329, 0.4307, 0.4326]
    sdd_ade_revise_particle_observe = [0.3826, 0.3828, 0.3833, 0.3828, 0.3832, 0.3828, 0.3828, 0.3833, 0.3825, 0.3824,
                                       0.3824, 0.3826, 0.3832, 0.3828, 0.383, 0.3825, 0.3821, 0.3822, 0.3821, 0.3832,
                                       0.3831, 0.3825, 0.3826, 0.383, 0.3826, 0.3827, 0.3823, 0.3836, 0.3826, 0.3833]
    sdd_fde_revise_particle_observe = [0.4165, 0.4187, 0.4185, 0.415, 0.4185, 0.4178, 0.4177, 0.4171, 0.4152, 0.4159,
                                       0.417, 0.417, 0.4179, 0.4179, 0.4171, 0.4175, 0.4171, 0.4163, 0.4156, 0.4175,
                                       0.4181, 0.4166, 0.4159, 0.4172, 0.4173, 0.4177, 0.417, 0.4185, 0.4165, 0.4174]
    sdd_ade_revise_particle_no_observe = [0.3867, 0.3872, 0.3882, 0.387, 0.3876, 0.3867, 0.3871, 0.3871, 0.3876, 0.3867,
                                          0.3875, 0.3881, 0.388, 0.3879, 0.3876, 0.3871, 0.3878, 0.3874, 0.3873,
                                          0.3887, 0.3873, 0.3872, 0.3872, 0.3872, 0.3873, 0.3863, 0.3872, 0.3876,
                                          0.3869, 0.3874]
    sdd_fde_revise_particle_no_observe = [0.4253, 0.4269, 0.4288, 0.4254, 0.4274, 0.4269, 0.4249, 0.4278, 0.4275,
                                          0.4242, 0.4267, 0.4288, 0.4273, 0.4259, 0.4274, 0.424, 0.4257, 0.4258, 0.4263,
                                          0.4297, 0.4253, 0.4255, 0.4275, 0.4261, 0.4263, 0.4251, 0.4274, 0.4272,
                                          0.4259, 0.4252]
    biwi_ade_source = [1.0958, 1.0775, 1.0967, 1.1054, 1.0798, 1.0765, 1.0981, 1.0876, 1.1104, 1.1049, 1.0994, 1.1042,
                       1.1055, 1.0948, 1.1087, 1.1138, 1.0854, 1.0866, 1.0823, 1.0916, 1.0908, 1.0918, 1.0889, 1.1186,
                       1.0875, 1.1124, 1.1136, 1.1016, 1.0834, 1.0885]
    biwi_fde_source = [1.5391, 1.5529, 1.5673, 1.6006, 1.4819, 1.5212, 1.5596, 1.5006, 1.5624, 1.6413, 1.5776, 1.554,
                       1.6365, 1.5809, 1.6243, 1.5749, 1.5605, 1.5369, 1.5596, 1.5245, 1.5363, 1.58, 1.5525, 1.5787,
                       1.5493, 1.5484, 1.58, 1.5544, 1.5652, 1.5496]
    biwi_ade_revise_noparticle = [1.0332, 1.0331, 1.0383, 1.034, 1.0333, 1.0262, 1.0381, 1.0386, 1.0269, 1.0379, 1.0467,
                                  1.0353, 1.0374, 1.021, 1.034, 1.0258, 1.0393, 1.0331, 1.027, 1.0366, 1.0182, 1.0294,
                                  1.0247, 1.0299, 1.0089, 1.0089, 1.0391, 1.02, 1.0332, 1.0378]
    biwi_fde_revise_noparticle = [1.4586, 1.4482, 1.4075, 1.4441, 1.3965, 1.4208, 1.44, 1.5056, 1.4203, 1.4728, 1.4827,
                                  1.4778, 1.4551, 1.4763, 1.4325, 1.3947, 1.4308, 1.4707, 1.4489, 1.4226, 1.3584,
                                  1.3984, 1.448, 1.4203, 1.398, 1.398, 1.4306, 1.4215, 1.4156, 1.4106]
    biwi_ade_revise_particle_observe = [0.91, 0.8872, 0.8922, 0.91, 0.9104, 0.8843, 0.9064, 0.9005, 0.9203, 0.8836,
                                        0.8962, 0.9099, 0.8939, 0.8904, 0.8988, 0.9098, 0.8936, 0.8947, 0.8918, 0.906,
                                        0.8949, 0.8737, 0.8973, 0.9045, 0.897, 0.9113, 0.9096, 0.9019, 0.9081, 0.8817]
    biwi_fde_revise_particle_observe = [1.2448, 1.1703, 1.2012, 1.2046, 1.2305, 1.1596, 1.2098, 1.1557, 1.2229, 1.1481,
                                        1.1816, 1.2246, 1.1911, 1.2208, 1.2014, 1.2287, 1.1951, 1.2033, 1.1958,
                                        1.1716, 1.1888, 1.1018, 1.1849, 1.2107, 1.2505, 1.2417, 1.2507, 1.1992, 1.2217,
                                        1.1762]
    biwi_ade_revise_particle_no_observe = [1.0121, 1.0285, 1.0157, 1.0294, 1.0308, 1.023, 1.0309, 1.0246, 1.0332,
                                           1.0323, 1.0223, 1.0303, 1.0225, 1.0315, 1.025, 1.0319, 1.0255, 1.015, 1.0312,
                                           1.015, 1.0081, 1.025, 1.0349, 1.0149, 1.0224, 1.0199, 1.0303, 1.0276, 1.0279,
                                           1.0188]
    biwi_fde_revise_particle_no_observe = [1.3807, 1.4312, 1.4042, 1.4449, 1.446, 1.4055, 1.4217, 1.4116, 1.4274,
                                           1.4363, 1.4057, 1.4518, 1.4207, 1.4406, 1.4317, 1.4316, 1.3811, 1.3831,
                                           1.4162, 1.3985, 1.3997, 1.4023, 1.4422, 1.4066, 1.411, 1.4188, 1.4284,
                                           1.4192, 1.4435, 1.3834]
    biwi_train_p_ade_NAN_p = [0.9732, 0.9664, 0.997, 0.9885, 0.995, 0.9667, 0.982, 0.9816, 0.9812, 0.977, 0.9839,
                              0.9802, 0.9666, 0.9774, 0.974, 0.9753, 0.9807, 0.9765, 0.9712, 0.9728, 0.9747, 0.9724,
                              0.9599, 0.9738, 0.9858, 0.9822, 0.9729, 0.9731, 0.9679, 0.9735]
    biwi_train_p_fde_NAN_p = [1.3245, 1.285, 1.3575, 1.3361, 1.333, 1.3213, 1.3341, 1.3417, 1.3098, 1.2991, 1.3376,
                              1.3741, 1.2993, 1.3219, 1.3176, 1.2937, 1.3211, 1.3171, 1.299, 1.3156, 1.3432, 1.3029,
                              1.3043, 1.3497, 1.3197, 1.3006, 1.3143, 1.3161, 1.272, 1.3251]
    biwi_train_p_ade_p_no_obs = [0.9748, 0.9792, 0.9669, 0.9667, 0.9687, 0.9672, 0.9651, 0.9798, 0.9577, 0.9701, 0.9736,
                                 0.9733, 0.9855, 0.9808, 0.9619, 0.9605, 0.9652, 0.966, 0.9572, 0.9653, 0.9664, 0.9712,
                                 0.9752, 0.9691, 0.9758, 0.9699, 0.9821, 0.9715, 0.9586, 0.9682]
    biwi_train_p_fde_p_no_obs = [1.3286, 1.3171, 1.3101, 1.3161, 1.305, 1.2856, 1.2753, 1.2963, 1.306, 1.2793, 1.3183,
                                 1.3248, 1.3338, 1.3301, 1.3163, 1.2734, 1.2831, 1.3123, 1.2828, 1.3211, 1.2795, 1.3238,
                                 1.3153, 1.3412, 1.3262, 1.3417, 1.3072, 1.3044, 1.268, 1.2943]
    biwi_train_p_ade_p_obs = [0.8384, 0.8485, 0.8457, 0.8538, 0.8341, 0.8407, 0.8622, 0.8546, 0.8566, 0.8481, 0.8391,
                              0.8406, 0.8497, 0.8432, 0.8506, 0.8449, 0.8487, 0.8393, 0.8425, 0.8415, 0.8468, 0.8439,
                              0.844, 0.8505, 0.8475, 0.8598, 0.853, 0.8409, 0.8297, 0.8461]
    biwi_train_p_fde_p_obs = [1.0612, 1.1108, 1.1067, 1.0994, 1.0808, 1.0621, 1.1214, 1.1367, 1.1148, 1.077, 1.0764,
                              1.0733, 1.0744, 1.0948, 1.1167, 1.0687, 1.0966, 1.1038, 1.0741, 1.0817, 1.0922, 1.0797,
                              1.0561, 1.1421, 1.073, 1.0993, 1.0978, 1.0804, 1.0331, 1.0937]
    # plt.plot([i for i in range(len(sdd_ade_source))], sdd_ade_source, '--.', color='blue', linewidth=1,
    #         label='SDD_ADE_source')
    # plt.plot([i for i in range(len(sdd_fde_source))], sdd_fde_source, '-v', color='#d725de', linewidth=1,
    #         label='SDD_FDE_source')

    # plt.plot([i for i in range(len(sdd_ade_revise_noparticle))], sdd_ade_revise_noparticle, '--.', color='blue',
    #        linewidth=1, label='SDD_ADE_rev_NAN_p')
    # plt.plot([i for i in range(len(sdd_fde_revise_noparticle))], sdd_fde_revise_noparticle, '-v', color='#d725de',
    #        linewidth=1, label='SDD_FDE_rev_NAN_p')

    # plt.plot([i for i in range(len(sdd_ade_revise_particle_no_observe))], sdd_ade_revise_particle_no_observe, '--.',
    #         color='blue', linewidth=1, label='SDD_ADE_rev_p_NAN_obs')
    # plt.plot([i for i in range(len(sdd_fde_revise_particle_no_observe))], sdd_fde_revise_particle_no_observe, '-v',
    #         color='#d725de', linewidth=1, label='SDD_FDE_rev_p_NAN_obs')

    # plt.plot([i for i in range(len(sdd_ade_revise_particle_observe))], sdd_ade_revise_particle_observe, '--.',
    #         color='blue', linewidth=1, label='SDD_ADE_rev_p_obs')
    # plt.plot([i for i in range(len(sdd_fde_revise_particle_observe))], sdd_fde_revise_particle_observe, '-v',
    #         color='#d725de', linewidth=1, label='SDD_FDE_rev_p_obs')

    # plt.plot([i for i in range(len(biwi_ade_source))], biwi_ade_source, '--.', color='blue', linewidth=1,
    #         label='BIWI_ADE_source')
    # plt.plot([i for i in range(len(biwi_fde_source))], biwi_fde_source, '-v', color='#d725de', linewidth=1,
    #         label='BIWI_FDE_source')

    # plt.plot([i for i in range(len(biwi_ade_revise_noparticle))], biwi_ade_revise_noparticle, '--.', color='blue',
    #        linewidth=1, label='BIWI_ADE_revise_NAN_p')
    # plt.plot([i for i in range(len(biwi_fde_revise_noparticle))], biwi_fde_revise_noparticle, '-v', color='#d725de',
    #        linewidth=1, label='BIWI_FDE_revise_NAN_p')

    # plt.plot([i for i in range(len(biwi_ade_revise_particle_observe))], biwi_ade_revise_particle_observe,
    #        '--.', color='blue', linewidth=1, label='BIWI_ADE_revise_p_obs')
    # plt.plot([i for i in range(len(biwi_fde_revise_particle_observe))], biwi_fde_revise_particle_observe,
    #        '-v', color='#d725de', linewidth=1, label='BIWI_FDE_revise_p_obs')

    # plt.plot([i for i in range(len(biwi_ade_revise_particle_no_observe))], biwi_ade_revise_particle_no_observe,
    #         '--.', color='blue', linewidth=1, label='BIWI_ADE_revise_p_NAN_obs')
    # plt.plot([i for i in range(len(biwi_fde_revise_particle_no_observe))], biwi_fde_revise_particle_no_observe,
    #         '-v', color='#d725de', linewidth=1, label='BIWI_FDE_revise_p_NAN_obs')

    # plt.plot([i for i in range(len(biwi_train_p_ade_NAN_p))], biwi_train_p_ade_NAN_p, '--.', color='blue', linewidth=1,
    #        label='biwi_train_p_ade_NAN_p')
    # plt.plot([i for i in range(len(biwi_train_p_fde_NAN_p))], biwi_train_p_fde_NAN_p, '-v', color='#d725de',
    #        linewidth=1, label='biwi_train_p_fde_NAN_p')

    # plt.plot([i for i in range(len(biwi_train_p_ade_p_no_obs))], biwi_train_p_ade_p_no_obs, '--.', color='blue',
    #         linewidth=1, label='biwi_train_p_ade_p_no_obs')
    # plt.plot([i for i in range(len(biwi_train_p_fde_p_no_obs))], biwi_train_p_fde_p_no_obs, '-v', color='#d725de',
    #         linewidth=1, label='biwi_train_p_fde_p_no_obs')

    plt.plot([i for i in range(len(biwi_train_p_ade_p_obs))], biwi_train_p_ade_p_obs, '--.', color='blue', linewidth=1,
             label='biwi_train_p_ade_p_obs')
    plt.plot([i for i in range(len(biwi_train_p_fde_p_obs))], biwi_train_p_fde_p_obs, '-v', color='#d725de',
             linewidth=1, label='biwi_train_p_fde_p_obs')
    plt.legend()
    plt.show()


def ETH_Avg_analyze():
    ETH_AVG_ADE_FDE_models = [1.329, 1.278, 1.214, 1.149, 1.139]
    plt.xlabel("Five different models")
    plt.ylabel("AVG_error: [(ADE+FDE)/2]")
    plt.plot([i for i in range(len(ETH_AVG_ADE_FDE_models))], ETH_AVG_ADE_FDE_models, label='ETH_Avg', color='#d725de',
             linewidth=1)
    plt.legend()
    plt.show()


def UCY_Avg_analyze():
    ETH_AVG_ADE_FDE_models = [0.539, 0.468, 0.464, 0.469, 0.464]
    plt.xlabel("Five different models")
    plt.ylabel("AVG_error: [(ADE+FDE)/2]")
    plt.plot([i for i in range(len(ETH_AVG_ADE_FDE_models))], ETH_AVG_ADE_FDE_models, label='UCY_Avg', color='#d725de',
             linewidth=1)
    plt.legend()
    plt.show()


def UCY_ADE_FDE_analyze():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("Five different models on ETH dataset")
    plt.ylabel("Errors")
    plt.title("The effect of different models to ADE and FDE errors")
    UCY_ADE = [0.513, 0.462, 0.459, 0.461, 0.458]
    UCY_FDE = [0.564, 0.473, 0.469, 0.476, 0.469]
    fake_x = [1, 2, 3, 4, 5]
    x_index = ['S-LSTM', 'SR-LSTM', 'SR-LSTM+SIRF(test)', 'SL-SIRF', 'SL-SIRF+SIRF(test)']
    plt.plot(fake_x, UCY_ADE, '-v', color='#d725de', linewidth=1, label='ADE')
    plt.plot(fake_x, UCY_FDE, '--.', color='blue', linewidth=1, label='FDE')
    _ = plt.xticks(fake_x, x_index)
    plt.legend()
    plt.show()


def ETH_ADE_FDE_analyze():
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("Five different models on ETH dataset")
    plt.ylabel("Errors")
    plt.title("The effect of different models to ADE and FDE errors")
    ETH_ADE = [1.096, 1.053, 1.012, 0.977, 0.97]
    ETH_FDE = [1.561, 1.502, 1.416, 1.32, 1.307]
    fake_x = [1, 2, 3, 4, 5]
    x_index = ['S-LSTM', 'SR-LSTM', 'SR-LSTM+SIRF(test)', 'SL-SIRF', 'SL-SIRF+SIRF(test)']
    plt.plot(fake_x, ETH_ADE, '-v', color='#d725de', linewidth=1, label='ADE')
    plt.plot(fake_x, ETH_FDE, '--.', color='blue', linewidth=1, label='FDE')
    _ = plt.xticks(fake_x, x_index)
    plt.legend()
    plt.show()


def datasets_analyze():
    list_x, list_y = [], []  # storage x and y coordinates
    min_list_x, max_list_x = [], []  # storage the maximum and minimum of x-coordinates
    min_list_y, max_list_y = [], []  # storage the maximum and minimum of y-coordinates
    list_path = []  # storage path
    path = './data/train/stanford'
    # path = './data/train/mot'
    # path = './data/train/crowds'
    # path = './data/train/biwi'
    for root, dirs, files in os.walk(path):
        for file in files:
            list_path.append(os.path.join(root, file))
    number_of_pedestrian = 0  # record the number of pedestrians
    for file_path in list_path:
        list_not_20 = []  # traverse every file and initialize list_not_20, dict_people and frame_distance
        dict_people = {}
        frame_distance = []
        frame_distance_not_10 = []  # initialize the frame distance list
        start_frame, end_frame = 0.0, 0.0
        with open(file_path, 'r') as f:
            for line in f:
                temp = line.replace('\n', '').split(' ')
                list_x.append(float(temp[2]))
                list_y.append(float(temp[3]))
                if temp[1] not in dict_people:
                    dict_people[temp[1]] = 1
                    start_frame = float(temp[0])
                else:
                    dict_people[temp[1]] += 1
                    end_frame = float(temp[0])
                    if dict_people[temp[1]] == 20:
                        frame_diff = end_frame - start_frame
                        frame_diff_divide = frame_diff / (dict_people[temp[1]] - 1)
                        frame_distance.append(frame_diff_divide)
        f.close()
        for key in dict_people:
            if dict_people[key] != 20:
                list_not_20.append(float(key))
        for ele in frame_distance:
            if ele == 0:
                continue
            if ele != 12:
                frame_distance_not_10.append(ele)
        min_list_x.append(min(list_x))
        max_list_x.append(max(list_x))
        min_list_y.append(min(list_y))
        max_list_y.append(max(list_y))
        number_of_pedestrian += len(dict_people)
        # ic(len(list_not_20), len(frame_distance_not_10), len(dict_people), len(frame_distance))
        ic(len(list_not_20), len(frame_distance_not_10))
    ic(min(min_list_x), min(min_list_y))
    ic(max(max_list_x), max(max_list_y))
    ic(number_of_pedestrian)


def particle_filter_distribution_FDE():
    # plot non-equally spaced data(FDE)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("Different value of σ*σ")
    plt.ylabel("Average FDE error")
    plt.title("The effect of different values of σ*σ on FDE")
    Avg_FDE = [1.206, 1.229, 1.267, 1.296, 1.332, 1.376]
    fake_x = [1, 2, 3, 4, 5, 6]
    x_index = ['1/50', '1/100', '1/200', '1/500', '1/1000', '1/10000']
    plt.plot(fake_x, Avg_FDE, '-v', color='#d725de', linewidth=1, label='Avg_FDE_error')
    _ = plt.xticks(fake_x, x_index)
    plt.legend()
    plt.show()


def particle_filter_distribution_ADE():
    # plot non-equally spaced data(ADE)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("Different value of σ*σ")
    plt.ylabel("Average ADE error")
    plt.title("The effect of different values of σ*σ on ADE")
    Avg_ADE = [0.9008, 0.9353, 0.9736]
    fake_x_2 = [1, 2, 3]
    x_index_2 = ['1/100', '1/1000', '1/10000']
    plt.plot(fake_x_2, Avg_ADE, '--.', color='blue', linewidth=1, label='Avg_ADE_error')
    _ = plt.xticks(fake_x_2, x_index_2)
    plt.legend()
    plt.show()


def source_code_crowds_ADE_FDE():
    source_code_crowds_ade_minimum = [0.506, 0.506, 0.509, 0.505, 0.511, 0.506, 0.508, 0.509, 0.509, 0.509, 0.505,
                                      0.504, 0.502, 0.509, 0.509, 0.507, 0.509, 0.503, 0.507, 0.509]
    source_code_crowds_ade_maximum = [0.516, 0.52, 0.518, 0.52, 0.518, 0.516, 0.515, 0.517, 0.515, 0.522, 0.523,
                                      0.521, 0.523, 0.518, 0.523, 0.517, 0.519, 0.524, 0.522, 0.515]
    source_code_crowds_ade_mean = [0.512, 0.513, 0.513, 0.512, 0.515, 0.511, 0.512, 0.513, 0.513, 0.513, 0.515, 0.514,
                                   0.513, 0.512, 0.514, 0.512, 0.512, 0.512, 0.512, 0.511]
    source_code_crowds_fde_minimum = [0.552, 0.539, 0.55, 0.54, 0.555, 0.548, 0.544, 0.555, 0.549, 0.548, 0.529,
                                      0.544, 0.53, 0.554, 0.543, 0.536, 0.552, 0.551, 0.549, 0.549]
    source_code_crowds_fde_maximum = [0.576, 0.572, 0.582, 0.572, 0.596, 0.577, 0.572, 0.577, 0.579, 0.593, 0.588,
                                      0.6, 0.59, 0.589, 0.583, 0.584, 0.577, 0.582, 0.58, 0.57]
    source_code_crowds_fde_mean = [0.562, 0.557, 0.563, 0.558, 0.573, 0.561, 0.559, 0.563, 0.562, 0.564, 0.564, 0.568,
                                   0.564, 0.564, 0.568, 0.563, 0.568, 0.566, 0.566, 0.559]
    x = [i for i in range(len(source_code_crowds_fde_mean))]
    plt.rcParams['font.family'] = 'simhei'
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1.fill_between(x, source_code_crowds_ade_minimum, source_code_crowds_ade_maximum, color='#F0F8FF')
    ax1.plot(x, source_code_crowds_ade_mean, '--.', color='#d725de', linewidth=2, label='Ave_ADE')
    # ax1.set_xlabel("The number of groups")
    ax1.set_ylabel("ADE Error")
    ax2.fill_between(x, source_code_crowds_fde_minimum, source_code_crowds_fde_maximum, color='#F0F8FF')
    ax2.plot(x, source_code_crowds_fde_mean, '-v', color='#d725de', linewidth=2, label='Avg_FDE')
    ax2.set_xlabel("The number of groups")
    ax2.set_ylabel("FDE Error")
    plt.show()


def source_code_sdd_ADE_FDE():
    source_code_sdd_ade_maximum = [0.401, 0.401, 0.399, 0.4, 0.4, 0.4, 0.4, 0.401, 0.401, 0.4, 0.4, 0.4, 0.401, 0.4,
                                   0.401, 0.401, 0.4, 0.401, 0.4, 0.399]
    source_code_sdd_ade_minimum = [0.398, 0.398, 0.399, 0.398, 0.398, 0.398, 0.399, 0.398, 0.398, 0.398, 0.398, 0.399,
                                   0.399, 0.398, 0.399, 0.399, 0.398, 0.398, 0.399, 0.398]
    source_code_sdd_ade_mean = [0.399, 0.399, 0.399, 0.399, 0.399, 0.399, 0.399, 0.4, 0.4, 0.399, 0.399, 0.4, 0.4,
                                0.399, 0.4, 0.4, 0.399, 0.4, 0.399, 0.399]
    source_code_sdd_fde_maximum = [0.445, 0.45, 0.443, 0.446, 0.446, 0.446, 0.445, 0.449, 0.449, 0.448, 0.445, 0.447,
                                   0.447, 0.447, 0.447, 0.449, 0.446, 0.446, 0.446, 0.443]
    source_code_sdd_fde_minimum = [0.438, 0.441, 0.438, 0.441, 0.44, 0.441, 0.439, 0.441, 0.4, 0.438, 0.44, 0.437,
                                   0.442, 0.439, 0.437, 0.445, 0.44, 0.441, 0.44, 0.435]
    source_code_sdd_fde_mean = [0.442, 0.445, 0.441, 0.443, 0.443, 0.443, 0.442, 0.444, 0.446, 0.443, 0.442, 0.442,
                                0.444, 0.443, 0.443, 0.446, 0.443, 0.443, 0.443, 0.441]
    x = [i for i in range(len(source_code_sdd_fde_mean))]
    plt.rcParams['font.family'] = 'simhei'
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1.fill_between(x, source_code_sdd_ade_minimum, source_code_sdd_ade_maximum, color='#F0F8FF')
    ax1.plot(x, source_code_sdd_ade_mean, '--.', color='#d725de', linewidth=2, label='Ave_ADE')
    ax1.set_xlabel("The number of groups")
    ax1.set_ylabel("ADE Error")
    ax2.fill_between(x, source_code_sdd_fde_minimum, source_code_sdd_fde_maximum, color='#F0F8FF')
    ax2.plot(x, source_code_sdd_fde_mean, '-v', color='#d725de', linewidth=2, label='Avg_FDE')
    ax2.set_xlabel("The number of groups")
    ax2.set_ylabel("FDE Error")
    plt.show()


def rev_p_crowds_ADE_FDE():
    rev_p_crowds_ade_minimum = [0.453, 0.458, 0.453, 0.456, 0.448, 0.455, 0.456, 0.453, 0.451, 0.453, 0.452, 0.453,
                                0.453, 0.454, 0.453, 0.454, 0.451, 0.455, 0.454, 0.453]
    rev_p_crowds_ade_maximum = [0.465, 0.462, 0.467, 0.461, 0.462, 0.462, 0.463, 0.461, 0.461, 0.462, 0.461, 0.461,
                                0.459, 0.464, 0.46, 0.46, 0.464, 0.464, 0.462, 0.462]
    rev_p_crowds_ade_mean = [0.458, 0.46, 0.459, 0.459, 0.456, 0.458, 0.458, 0.457, 0.457, 0.457, 0.457, 0.458, 0.456,
                             0.458, 0.457, 0.457, 0.459, 0.459, 0.459, 0.457]
    rev_p_crowds_fde_minimum = [0.461, 0.469, 0.461, 0.466, 0.448, 0.457, 0.463, 0.458, 0.456, 0.457, 0.463, 0.453,
                                0.462, 0.459, 0.46, 0.461, 0.464, 0.462, 0.466, 0.459]
    rev_p_crowds_fde_maximum = [0.48, 0.48, 0.487, 0.48, 0.476, 0.477, 0.48, 0.476, 0.478, 0.475, 0.479, 0.473, 0.47,
                                0.483, 0.477, 0.475, 0.484, 0.481, 0.481, 0.48]
    rev_p_crowds_fde_mean = [0.47, 0.475, 0.471, 0.472, 0.466, 0.466, 0.47, 0.467, 0.467, 0.466, 0.469, 0.468, 0.465,
                             0.473, 0.468, 0.466, 0.473, 0.469, 0.472, 0.47]
    x = [i for i in range(len(rev_p_crowds_ade_minimum))]
    plt.rcParams['font.family'] = 'simhei'
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1.fill_between(x, rev_p_crowds_ade_minimum, rev_p_crowds_ade_maximum, color='#F0F8FF')
    ax1.plot(x, rev_p_crowds_ade_mean, '--.', color='#d725de', linewidth=2, label='Ave_ADE')
    # ax1.set_xlabel("The number of groups")
    ax1.set_ylabel("ADE Error")
    ax2.fill_between(x, rev_p_crowds_fde_minimum, rev_p_crowds_fde_maximum, color='#F0F8FF')
    ax2.plot(x, rev_p_crowds_fde_mean, '-v', color='#d725de', linewidth=2, label='Avg_FDE')
    ax2.set_xlabel("The number of groups")
    ax2.set_ylabel("FDE Error")
    plt.show()


def rev_NAN_p_crowds_ADE_FDE():
    rev_NAN_p_crowds_ADE_maximum = [0.461, 0.464, 0.463, 0.463, 0.461, 0.463, 0.463, 0.466, 0.464, 0.463, 0.465,
                                    0.466, 0.463, 0.465, 0.461, 0.462, 0.464, 0.461, 0.464, 0.463]
    rev_NAN_p_crowds_ADE_minimum = [0.458, 0.458, 0.458, 0.458, 0.457, 0.456, 0.457, 0.46, 0.459, 0.456, 0.454, 0.46,
                                    0.456, 0.457, 0.459, 0.458, 0.456, 0.453, 0.458, 0.458]
    rev_NAN_p_crowds_FDE_maximum = [0.485, 0.482, 0.485, 0.49, 0.483, 0.493, 0.485, 0.491, 0.491, 0.481, 0.486, 0.488,
                                    0.487, 0.492, 0.48, 0.481, 0.484, 0.484, 0.495, 0.478]
    rev_NAN_p_crowds_FDE_minimum = [0.469, 0.463, 0.469, 0.462, 0.46, 0.466, 0.465, 0.471, 0.467, 0.466, 0.473, 0.471,
                                    0.464, 0.461, 0.462, 0.464, 0.461, 0.46, 0.467, 0.459]


def rev_p_sdd_ADE_FDE():
    rev_p_sdd_ADE_maximum = [0.371, 0.37, 0.37, 0.37, 0.371, 0.371, 0.372, 0.37, 0.37, 0.37, 0.37, 0.37, 0.371, 0.37,
                             0.37, 0.371, 0.371, 0.371, 0.371, 0.371]
    rev_p_sdd_ADE_minimum = [0.369, 0.369, 0.369, 0.368, 0.369, 0.369, 0.368, 0.369, 0.368, 0.369, 0.368, 0.369,
                             0.369, 0.369, 0.369, 0.368, 0.368, 0.369, 0.369, 0.368]
    rev_p_sdd_ADE_mean = [0.37, 0.37, 0.37, 0.369, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.369, 0.37, 0.37, 0.37, 0.369,
                          0.37, 0.369, 0.37, 0.37, 0.37]
    rev_p_sdd_FDE_maximum = [0.406, 0.399, 0.398, 0.398, 0.402, 0.4, 0.404, 0.399, 0.404, 0.399, 0.406, 0.404, 0.404,
                             0.398, 0.402, 0.402, 0.4, 0.401, 0.404, 0.402]
    rev_p_sdd_FDE_minimum = [0.395, 0.394, 0.39, 0.394, 0.394, 0.396, 0.394, 0.392, 0.389, 0.393, 0.392, 0.394, 0.393,
                             0.394, 0.391, 0.394, 0.389, 0.395, 0.393, 0.393]
    rev_p_sdd_FDE_mean = [0.398, 0.396, 0.394, 0.395, 0.399, 0.398, 0.397, 0.395, 0.398, 0.396, 0.397, 0.399, 0.398,
                          0.397, 0.397, 0.398, 0.395, 0.397, 0.398, 0.397]
    x = [i for i in range(len(rev_p_sdd_FDE_minimum))]
    plt.rcParams['font.family'] = 'simhei'
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    ax1.fill_between(x, rev_p_sdd_ADE_minimum, rev_p_sdd_ADE_maximum, color='#F0F8FF')
    ax1.plot(x, rev_p_sdd_ADE_mean, '--.', color='#d725de', linewidth=2, label='Ave_ADE')
    ax1.set_xlabel("Times")
    ax1.set_ylabel("ADE Error")
    ax2.fill_between(x, rev_p_sdd_FDE_minimum, rev_p_sdd_FDE_maximum, color='#F0F8FF')
    ax2.plot(x, rev_p_sdd_FDE_mean, '-v', color='#d725de', linewidth=2, label='Avg_FDE')
    ax2.set_xlabel("Times")
    ax2.set_ylabel("FDE Error")
    plt.show()


def rev_NAN_p_sdd_ADE_FDE():
    rev_NAN_p_sdd_ADE_maximum = [0.375, 0.376, 0.375, 0.375, 0.376, 0.375, 0.375, 0.375, 0.374, 0.374, 0.375, 0.374,
                                 0.375, 0.376, 0.375, 0.374, 0.375, 0.376, 0.375, 0.374]
    rev_NAN_p_sdd_ADE_minimum = [0.372, 0.371, 0.372, 0.372, 0.373, 0.373, 0.373, 0.372, 0.372, 0.372, 0.372, 0.373,
                                 0.373, 0.373, 0.373, 0.373, 0.373, 0.372, 0.373, 0.372]
    rev_NAN_p_sdd_FDE_maximum = [0.413, 0.419, 0.409, 0.41, 0.423, 0.412, 0.413, 0.412, 0.408, 0.405, 0.408, 0.404,
                                 0.412, 0.414, 0.408, 0.411, 0.41, 0.414, 0.41, 0.407]
    rev_NAN_p_sdd_FDE_minimum = [0.404, 0.399, 0.401, 0.403, 0.403, 0.402, 0.399, 0.398, 0.401, 0.399, 0.4, 0.398,
                                 0.405, 0.403, 0.403, 0.4, 0.402, 0.402, 0.399, 0.396]


if __name__ == "__main__":
    source_code_sdd_ADE_FDE()
    # rev_p_sdd_ADE_FDE()
    # datasets_analyze()
    # ETH_ADE_FDE_analyze()
    # particle_filter_distribution_FDE()
    # particle_filter_distribution_ADE()
    # source_code_crowds_ADE_FDE()
    # rev_p_crowds_ADE_FDE()
    # seq_hotel_analyze()
    # seq_hotel_analyze2(21)
    # seq_hotel_max_min_position()
    # seq_hotel_destination()

    # input the path which will be visualized in the future!
    # open_path = './LSTM-analyze'
    # visualization_of_trajectories(open_path)
