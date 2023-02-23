import os
import pickle
import numpy as np
import pandas as pd
import random
import torch
import math
from torch.autograd import Variable
from helper import *
from icecream import ic


class DataLoader():

    def __init__(self, f_prefix, batch_size=5, seq_length=20, num_of_validation=0, forcePreProcess=False, infer=False,
                 generate=False):
        '''
        Initialiser function for the DataLoader class
        params:
        batch_size : Size of the mini-batch
        seq_length : Sequence length to be considered
        num_of_validation : number of validation dataset will be used
        infer : flag for test mode
        generate : flag for data generation mode
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # base test files
        base_test_dataset = [
            # './data/test/biwi/obsmat3.txt',
            './data/test/biwi/obsmat20.txt',
            # '/data/test/biwi/biwi_eth.txt',
            # '/data/test/crowds/crowds_zara01.txt',
            # '/data/test/crowds/uni_examples.txt',
            # '/data/test/stanford/coupa_0.txt',
            # '/data/test/stanford/coupa_1.txt', '/data/test/stanford/gates_2.txt',
            # '/data/test/stanford/hyang_0.txt', '/data/test/stanford/hyang_1.txt',
            # '/data/test/stanford/hyang_3.txt', '/data/test/stanford/hyang_8.txt',
            # '/data/test/stanford/little_0.txt', '/data/test/stanford/little_1.txt',
            # '/data/test/stanford/little_2.txt', '/data/test/stanford/little_3.txt',
            # '/data/test/stanford/nexus_5.txt', '/data/test/stanford/nexus_6.txt',
            # '/data/test/stanford/quad_0.txt', '/data/test/stanford/quad_1.txt',
            # '/data/test/stanford/quad_2.txt', '/data/test/stanford/quad_3.txt'
        ]
        # base train files
        base_train_dataset = ['/data/train/biwi/biwi_hotel.txt',
                              # '/data/train/crowds/crowds_zara02.txt',
                              # '/data/train/crowds/arxiepiskopi1.txt', '/data/train/crowds/crowds_zara02.txt',
                              # '/data/train/crowds/crowds_zara03.txt', '/data/train/crowds/students001.txt',
                              # '/data/train/crowds/students003.txt',
                              # '/data/train/mot/PETS09-S2L1.txt',
                              # '/data/train/stanford/bookstore_0.txt', '/data/train/stanford/bookstore_1.txt',
                              # '/data/train/stanford/bookstore_2.txt', '/data/train/stanford/bookstore_3.txt',
                              # '/data/train/stanford/coupa_3.txt', '/data/train/stanford/deathCircle_0.txt',
                              # '/data/train/stanford/deathCircle_1.txt', '/data/train/stanford/deathCircle_2.txt',
                              # '/data/train/stanford/deathCircle_3.txt',
                              # '/data/train/stanford/deathCircle_4.txt', '/data/train/stanford/gates_0.txt',
                              # '/data/train/stanford/gates_1.txt', '/data/train/stanford/gates_3.txt', '/data/train/stanford/gates_4.txt',
                              # '/data/train/stanford/gates_5.txt', '/data/train/stanford/gates_6.txt', '/data/train/stanford/gates_7.txt',
                              # '/data/train/stanford/gates_8.txt', '/data/train/stanford/hyang_4.txt',
                              # '/data/train/stanford/hyang_5.txt', '/data/train/stanford/hyang_6.txt', '/data/train/stanford/hyang_9.txt',
                              # '/data/train/stanford/nexus_0.txt', '/data/train/stanford/nexus_1.txt', '/data/train/stanford/nexus_2.txt',
                              # '/data/train/stanford/nexus_3.txt', '/data/train/stanford/nexus_4.txt', '/data/train/stanford/nexus_7.txt',
                              # '/data/train/stanford/nexus_8.txt', '/data/train/stanford/nexus_9.txt'
                              ]
        # dimensions of each file set
        self.dataset_dimensions = {'biwi': [720, 576], 'crowds': [720, 576], 'stanford': [595, 326], 'mot': [768, 576]}

        # List of data directories where raw data resides
        self.base_train_path = 'data/train/'
        self.base_test_path = 'data/test/'
        self.base_validation_path = 'data/validation/'

        # check infer flag, if true choose test directory as base directory
        if infer is False:
            self.base_data_dirs = base_train_dataset
        else:
            self.base_data_dirs = base_test_dataset

        # train_dataset\test_dataset\validation_dataset中存放的是每个文本文件的路径
        self.train_dataset = self.get_dataset_path(self.base_train_path, f_prefix)
        self.test_dataset = self.get_dataset_path(self.base_test_path, f_prefix)
        self.validation_dataset = self.get_dataset_path(self.base_validation_path, f_prefix)

        # if generate mode, use directly train base files,generate default value is False
        # train_dataset是一个列表，其中存放的是所有文本文件的路径，如：.\\data/test/biwi/biwi_eth.txt',...
        if generate:
            self.train_dataset = [os.path.join('./', dataset[1:]) for dataset in base_train_dataset]

        # request of use of validation dataset, num_of_validation default value is 0
        if num_of_validation > 0:
            self.additional_validation = True
        else:
            self.additional_validation = False

        # check validation dataset availibility and clip the reuqested number if it is bigger than available validation dataset
        if self.additional_validation:
            if len(self.validation_dataset) is 0:
                print("There is no validation dataset.Aborted.")
                self.additional_validation = False
            else:
                num_of_validation = np.clip(num_of_validation, 0, len(self.validation_dataset))
                self.validation_dataset = random.sample(self.validation_dataset, num_of_validation)

        # if not infer mode, use train dataset,data_dirs中存放的是{train_dataset\validation_dataset\test_dataset}中路径
        if infer is False:
            self.data_dirs = self.train_dataset
        else:
            # use validation dataset
            if self.additional_validation:
                self.data_dirs = self.validation_dataset
            # use test dataset
            else:
                self.data_dirs = self.test_dataset

        self.infer = infer
        self.generate = generate

        # Number of datasets
        self.numDatasets = len(self.data_dirs)

        # array for keeping target ped ids for each sequence
        self.target_ids = []

        # Data directory where the pre-processed pickle file resides
        # 在{train_data_dir、test_data_dir、val_data_dir}路径下存储.cpkl后缀的文件如，./data/train/trajectories_train.cpkl
        self.train_data_dir = os.path.join(f_prefix, self.base_train_path)
        self.test_data_dir = os.path.join(f_prefix, self.base_test_path)
        self.val_data_dir = os.path.join(f_prefix, self.base_validation_path)

        # Store the arguments
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.orig_seq_lenght = seq_length

        # Validation arguments
        self.val_fraction = 0

        # Define the path in which the process data would be stored
        # 在{data_file_tr、data_file_te、data_file_vl}中存储路径：如，./data/train/trajectories_train.cpkl
        self.data_file_tr = os.path.join(self.train_data_dir, "trajectories_train.cpkl")
        self.data_file_te = os.path.join(self.base_test_path, "trajectories_test.cpkl")
        self.data_file_vl = os.path.join(self.val_data_dir, "trajectories_val.cpkl")

        # for creating a dict key: folder names, values: files in this folder
        self.create_folder_file_dict()

        if self.additional_validation:
            # If the file doesn't exist or forcePreProcess is true
            if not (os.path.exists(self.data_file_vl)) or forcePreProcess:
                print("Creating pre-processed validation data from raw data")
                # Preprocess the data from the csv files of the datasets
                # Note that this data is processed in frames
                self.frame_preprocess(self.validation_dataset, self.data_file_vl, self.additional_validation)

        if self.infer:
            # if infer mode, and no additional files -> test preprocessing
            if not self.additional_validation:
                if not (os.path.exists(self.data_file_te)) or forcePreProcess:
                    print("Creating pre-processed test data from raw data")
                    # Preprocess the data from the csv files of the datasets
                    # Note that this data is processed in frames
                    print("Working on directory: ", self.data_file_te)
                    self.frame_preprocess(self.data_dirs, self.data_file_te)
            # if infer mode, and there are additional validation files -> validation dataset visualization
            else:
                print("Validation visualization file will be created")

        # if not infer mode
        else:
            # If the file doesn't exist or forcePreProcess is true -> training pre-process
            if not (os.path.exists(self.data_file_tr)) or forcePreProcess:
                print("Creating pre-processed training data from raw data")
                # Preprocess the data from the .txt files of the datasets
                # Note that this data is processed in frames，因为以帧为基准进行的处理
                self.frame_preprocess(self.data_dirs, self.data_file_tr)

        if self.infer:
            # Load the processed data from the pickle file
            if not self.additional_validation:  # test mode
                self.load_preprocessed(self.data_file_te)
            else:  # validation mode
                self.load_preprocessed(self.data_file_vl, True)

        else:  # training mode
            self.load_preprocessed(self.data_file_tr)

        # Reset all the data pointers of the dataloader object
        self.reset_batch_pointer(valid=False)
        self.reset_batch_pointer(valid=True)

    def frame_preprocess(self, data_dirs, data_file, validation_set=False):
        '''
        Function that will pre-process the .txt files of each dataset into data with occupancy grid that can be used
        params:
        data_dirs : List of directories where raw data resides
        data_file : The file into which all the pre-processed data needs to be stored
        validation_set: true when a dataset is in validation set
        '''
        # all_frame_data would be a list of numpy arrays corresponding to each dataset
        # Each numpy array will correspond to a frame and would be of size (Ped_IDx, 3) each row,x~[1,n]
        # containing pedID, x, y
        all_frame_data = []
        valid_frame_data = []

        # frameList_data would be a list of lists corresponding to each dataset
        # Each list would contain the frameIds of all the frames in the dataset
        frameList_data = []
        valid_numPeds_data = []

        # numPeds_data would be a list of lists corresponding to each dataset
        # Each list would contain the number of pedestrians in each frame in the dataset
        numPeds_data = []

        # pedsList_data would be a list of lists corresponding to each dataset
        # each list includes ped ids of this frame
        pedsList_data = []
        valid_pedsList_data = []

        # target ped ids for each sequence，target_ids是一个二维列表，每一个维度是一维列表，其中存储着所有unique的id信息
        target_ids = []
        orig_data = []  # 数据[data]的备份列表

        # Index of the current dataset, for each .txt file dataset_index += 1
        dataset_index = 0

        # For each dataset，遍历data_dirs列表中的每一个元素（每一个元素都代表着一个路径）
        for directory in data_dirs:

            # Load the data from the txt file，打印当前正在处理的文件。
            print("Now processing: ", directory)
            column_names = ['frame_num', 'ped_id', 'y', 'x']

            # if training mode, read train file to pandas dataframe and process
            if self.infer is False:
                df = pd.read_csv(directory, dtype={'frame_num': 'int', 'ped_id': 'int'}, delimiter=' ', header=None,
                                 names=column_names)
                self.target_ids = np.array(df.drop_duplicates(subset=['ped_id'], keep='first', inplace=False)['ped_id'])
            else:
                # if validation mode, read validation file to pandas dataframe and process
                if self.additional_validation:
                    df = pd.read_csv(directory, dtype={'frame_num': 'int', 'ped_id': 'int'}, delimiter=' ', header=None,
                                     names=column_names)
                    self.target_ids = np.array(
                        df.drop_duplicates(subset={'ped_id'}, keep='first', inplace=False)['ped_id'])

                # if test mode, read test file to pandas dataframe and process
                else:
                    column_names = ['frame_num', 'ped_id', 'y', 'x']
                    df = pd.read_csv(directory, dtype={'frame_num': 'int', 'ped_id': 'int'}, delimiter=' ', header=None,
                                     names=column_names,
                                     converters={c: lambda x: float('nan') if x == '?' else float(x) for c in
                                                 ['y', 'x']})
                    self.target_ids = np.array(
                        df[df['y'].isnull()].drop_duplicates(subset={'ped_id'}, keep='first', inplace=False)['ped_id'])

            # convert pandas -> numpy array
            data = np.array(df)

            # keep original copy of file
            orig_data.append(data)

            # swap x and y points (in txt file it is like -> y,x)，交换轴的位置，前后两个的位置没有关系
            data = np.swapaxes(data, 0, 1)

            # get frame numbers, frameList is a list corresponding frame_IDs of each .txt file
            frameList = data[0, :].tolist()

            # Number of frames, is a int number
            numFrames = len(frameList)

            # Add the list of frameIDs to the frameList_data
            frameList_data.append(frameList)

            # Initialize the list of numPeds for the current dataset
            numPeds_data.append([])
            valid_numPeds_data.append([])

            # Initialize the list of numpy arrays for the current dataset
            all_frame_data.append([])
            # Initialize the list of numpy arrays for the current dataset
            valid_frame_data.append([])

            # list of peds for each frame
            pedsList_data.append([])
            valid_pedsList_data.append([])

            target_ids.append(self.target_ids)
            # ind表示访问的索引（index），其值从0开始取
            for ind, frame in enumerate(frameList):

                # Extract all pedestrians in current frame
                # 该行代码的写法值得学习，显示当前帧下所有行人的信息：以这种形式[Frame_ID,Ped_ID,y,x]呈现
                pedsInFrame = data[:, data[0, :] == frame]
                # print("peds in %d: %s"%(frame,str(pedsInFrame)))

                # Extract peds list,即只要pedsInFrame中第二个列表中存的值，存放的是Ped_ID
                pedsList = pedsInFrame[1, :].tolist()

                # Initialize the row of the numpy array
                # pedsWithPos would be a list of lists corresponding to each dataset
                # each list contains Ped_ID,x,y and its format is:[Ped_ID,x,y]
                pedsWithPos = []

                # For each ped in the current frame
                for ped in pedsList:
                    # Extract their x and y positions，即获取每个行人的y和x坐标
                    current_x = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[2, pedsInFrame[1, :] == ped][0]

                    # Add their pedID, x, y to the row of the numpy array，其格式是：[Ped_ID,x,y]
                    pedsWithPos.append([ped, current_x, current_y])

                # At inference time, data generation and if dataset is a validation dataset, no validation data
                if (ind >= numFrames * self.val_fraction) or (self.infer) or (self.generate) or (validation_set):
                    # Add the details of all the peds in the current frame to all_frame_data
                    all_frame_data[dataset_index].append(np.array(pedsWithPos))
                    pedsList_data[dataset_index].append(pedsList)
                    numPeds_data[dataset_index].append(len(pedsList))
                else:
                    valid_frame_data[dataset_index].append(np.array(pedsWithPos))
                    valid_pedsList_data[dataset_index].append(pedsList)
                    valid_numPeds_data[dataset_index].append(len(pedsList))
            # 每访问一个文本文件，该dataset_index就会加1
            dataset_index += 1
        # Save the arrays in the pickle file
        f = open(data_file, "wb")
        # 这里依次说明下dump函数中的每一个数据项表示的意义:
        # frameList_data:每个文本文件中的所有帧组成一个小列表，每个小列表拼接成一个大列表，该大列表存储的是所有文本文件的帧
        # valid_frame_data:该帧下，[[Ped_ID1,x1,y1],[Ped_ID2,x2,y2],...,[Ped_IDn,xn,yn]]](at non-inference time)
        # all_frame_data:该帧下，[[Ped_ID1,x1,y1],[Ped_ID2,x2,y2],...,[Ped_IDn,xn,yn]](at inference time)
        # valid_pedsList_data:该帧下，[Ped_ID1,Ped_ID2,...,Ped_IDn](at non-inference time)
        # pedsList_data:该帧下，[Ped_ID1,Ped_ID2,...,Ped_IDn](at inference time)
        # valid_numPeds_data:该帧下，包含的不同Ped_IDs的总个数(at non-inference time)
        # numPeds_data:该帧下，包含的不同Ped_IDs的总个数(at inference time)
        # target_ids:每个小列表内存储的是当前文本文件下所有的不重复的Ped_IDs，它们组成一个小列表；所有的文本文件共同组成一个大列表
        # orig_data:每个小的np.array类型的列表保存当前文本文件的信息[[Frame_ID1,Ped_ID1,y1,x1],[Frame_ID2,Ped_ID2,y2,x2],...,[]]
        # orig_data:所有的这些小列表，共同组成了一个大列表
        # pickle.dump函数中的protocol=2表示以二进制的形式进行序列化，(all_frame_data,...,orig_data)是obj即“序列化对象”
        # f表示保存到的类文件对象
        pickle.dump((all_frame_data, frameList_data, numPeds_data, valid_numPeds_data, valid_frame_data, pedsList_data,
                     valid_pedsList_data, target_ids, orig_data), f, protocol=2)
        f.close()

    def load_preprocessed(self, data_file, validation_set=False):
        '''
        Function to load the pre-processed data into the DataLoader object
        params:
        data_file : the path to the pickled data file,for example:./data/train/trajectories_train.cpkl
        validation_set : flag for validation dataset
        '''
        # Load data from the pickled file
        if (validation_set):
            print("Loading validaton datasets: ", data_file)
        else:
            print("Loading train or test dataset: ", data_file)

        f = open(data_file, 'rb')
        self.raw_data = pickle.load(f)
        f.close()

        # Get all the data from the pickle file
        # data:该帧下，[[Ped_ID1,x1,y1],[Ped_ID2,x2,y2],...,[Ped_IDn,xn,yn]](at inference time)
        # frameList:每个文本文件中的所有帧组成一个小列表，每个小列表拼接成一个大列表，该大列表存储的是所有文本文件的帧
        # numPedsList:该帧下，包含的不同Ped_IDs的总个数(at inference time)
        # valid_numPedsList:该帧下，包含的不同Ped_IDs的总个数(at non-inference time)
        # valid_data:该帧下，[[Ped_ID1,x1,y1],[Ped_ID2,x2,y2],...,[Ped_IDn,xn,yn]]](at non-inference time)
        # pedsList:该帧下，[Ped_ID1,Ped_ID2,...,Ped_IDn](at inference time)
        # valid_pedsList:该帧下，[Ped_ID1,Ped_ID2,...,Ped_IDn](at non-inference time)
        # target_ids:每个小列表内存储的是当前文本文件下所有的不重复的Ped_IDs，它们组成一个小列表；所有的文本文件共同组成一个大列表
        # orig_data:每个小的np.array类型的列表保存当前文本文件的信息[[Frame_ID1,Ped_ID1,y1,x1],[Frame_ID2,Ped_ID2,y2,x2],...,[]]
        # orig_data:所有的这些小列表，共同组成了一个大列表
        self.data = self.raw_data[0]
        self.frameList = self.raw_data[1]
        self.numPedsList = self.raw_data[2]
        self.valid_numPedsList = self.raw_data[3]
        self.valid_data = self.raw_data[4]
        self.pedsList = self.raw_data[5]
        self.valid_pedsList = self.raw_data[6]
        self.target_ids = self.raw_data[7]
        self.orig_data = self.raw_data[8]

        counter = 0
        valid_counter = 0
        print('Sequence size(frame) ------>', self.seq_length)
        print('One batch size (frame)--->-', self.batch_size * self.seq_length)

        # For each dataset,即针对每个.txt类型的文本文件，都进行遍历；dataset每次+1
        for dataset in range(len(self.data)):
            # get the frame data for the current dataset,[Ped_IDn,xn,yn]
            all_frame_data = self.data[dataset]
            valid_frame_data = self.valid_data[dataset]
            dataset_name = self.data_dirs[dataset].split('/')[-1]  # .txt的文本文件的名字
            # calculate number of sequence,for example:2900/20=145=num_seq_in_dataset or num_valid_seq_in_dataset
            num_seq_in_dataset = int(len(all_frame_data) / (self.seq_length))
            num_valid_seq_in_dataset = int(len(valid_frame_data) / (self.seq_length))
            if not validation_set:
                print('Training data from training dataset(name, #frame, #sequence)--> ', dataset_name, ':',
                      len(all_frame_data), ':', (num_seq_in_dataset))
                print('Validation data from training dataset(name, # frame, #sequence)--> ', dataset_name, ':',
                      len(valid_frame_data), ':', (num_valid_seq_in_dataset))
            else:
                print('Validation data from validation dataset(name, # frame, #sequence)--> ', dataset_name, ':',
                      len(all_frame_data), ':', (num_seq_in_dataset))

            # Increment the counter with the number of sequences in the current dataset
            # For example,the value of {num_seq_in_dataset\num_valid_seq_in_dataset} is 145
            # for example,这里的{counter\valid_counter}每次+145；例如：每20个为一组，记录所有的文本文件中所有的组数
            counter += num_seq_in_dataset
            valid_counter += num_valid_seq_in_dataset

        # Calculate the number of batches,num_batches=counter/batch_size;valid_num_batches=valid_counter/batch_size
        self.num_batches = int(counter / self.batch_size)
        self.valid_num_batches = int(valid_counter / self.batch_size)

        if not validation_set:
            print('Total number of training batches:', self.num_batches)
            print('Total number of validation batches:', self.valid_num_batches)
        else:
            print('Total number of validation batches:', self.num_batches)

        # self.valid_num_batches = self.valid_num_batches * 2

    def next_batch(self):
        # Function to get the next batch of points
        # Source data,information in current sequence->[Ped_ID1,x1,y1],...,[Ped_IDn,xn,yn](current frame)
        x_batch = []
        # Target data,information in current sequence->[Ped_ID1,x1,y1],...,[Ped_IDn,xn,yn](next frame)
        y_batch = []
        # Dataset data，storage numbers in current file, and add 1 every time when traverse a file.
        d = []
        # 不同Ped_IDs的总个数 per sequence,当前序列下的信息->该序列跨度下，所有不同Ped_IDs的总个数(at inference time)(当前帧)
        numPedsList_batch = []
        # [Ped_ID1,Ped_ID2,...,Ped_IDn] per sequence,当前序列下的信息->[Ped_ID1,Ped_ID2,...,Ped_IDn](at inference time)(当前帧)
        PedsList_batch = []
        # return target_id,存取的是行人的ID
        target_ids = []
        # Iteration index
        i = 0
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            # dataset_pointer指的是文本数据集的个数
            # data->frame_data:该帧下，[[Ped_ID1,x1,y1],[Ped_ID2,x2,y2],...,[Ped_IDn,xn,yn]](at inference time)
            # numPedsList->numPedsList:该帧下，包含的不同Ped_IDs的总个数(at inference time)
            # pedsList->pedsList:该帧下，[Ped_ID1,Ped_ID2,...,Ped_IDn](at inference time)
            frame_data = self.data[self.dataset_pointer]
            numPedsList = self.numPedsList[self.dataset_pointer]
            pedsList = self.pedsList[self.dataset_pointer]
            # Get the frame pointer for the current dataset,frame_pointer的初始化为0
            idx = self.frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length - 1 < len(frame_data):
                # 当前帧的信息
                # seq_source_frame_data:当前序列下的信息->[Ped_ID1,x1,y1],...,[Ped_IDn,xn,yn]
                seq_source_frame_data = frame_data[idx:idx + self.seq_length]
                # seq_numPedsList:当前序列下的信息->该序列跨度下，所有不同Ped_IDs的总个数(at inference time)
                seq_numPedsList = numPedsList[idx:idx + self.seq_length]
                # seq_PedsList：当前序列下的信息->[Ped_ID1,Ped_ID2,...,Ped_IDn](at inference time)
                seq_PedsList = pedsList[idx:idx + self.seq_length]
                # 下一帧的信息
                seq_target_frame_data = frame_data[idx + 1:idx + self.seq_length + 1]
                # Number of unique peds in this sequence of frames
                # 当前帧...
                x_batch.append(seq_source_frame_data)
                # 下一帧...
                y_batch.append(seq_target_frame_data)
                numPedsList_batch.append(seq_numPedsList)
                PedsList_batch.append(seq_PedsList)
                # get correct target ped id for the sequence, target_ids中存取的是所有行人的id
                target_ids.append(
                    self.target_ids[self.dataset_pointer][math.floor((self.frame_pointer) / self.seq_length)])
                # for example, frame_pointer的上限值为2900。(这里只是举个例子，当然frame_pointer还可以取其他的值)
                # for example, math.floor((self.frame_pointer) / self.seq_length)的上限值为145，该值还可以取其他的值
                self.frame_pointer += self.seq_length
                d.append(self.dataset_pointer)
                i += 1
            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=False)
        return x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids

    def next_valid_batch(self):
        # Function to get the next Validation batch of points
        x_batch = []  # source data
        y_batch = []  # target data
        d = []  # dataset data
        numPedsList_batch = []  # pedlist per sequence
        PedsList_batch = []  # pedlist per sequence
        target_ids = []
        i = 0  # iteration index
        while i < self.batch_size:
            # Extract the frame data of the current dataset
            frame_data = self.valid_data[self.valid_dataset_pointer]
            numPedsList = self.valid_numPedsList[self.valid_dataset_pointer]
            pedsList = self.valid_pedsList[self.valid_dataset_pointer]
            # Get the frame pointer for the current dataset
            idx = self.valid_frame_pointer
            # While there is still seq_length number of frames left in the current dataset
            if idx + self.seq_length < len(frame_data):
                # All the data in this sequence
                # seq_frame_data = frame_data[idx:idx+self.seq_length+1]
                seq_source_frame_data = frame_data[idx:idx + self.seq_length]
                seq_numPedsList = numPedsList[idx:idx + self.seq_length]
                seq_PedsList = pedsList[idx:idx + self.seq_length]
                seq_target_frame_data = frame_data[idx + 1:idx + self.seq_length + 1]
                # Number of unique peds in this sequence of frames
                x_batch.append(seq_source_frame_data)
                y_batch.append(seq_target_frame_data)
                numPedsList_batch.append(seq_numPedsList)
                PedsList_batch.append(seq_PedsList)
                # get correct target ped id for the sequence
                target_ids.append(
                    self.target_ids[self.dataset_pointer][math.floor((self.valid_frame_pointer) / self.seq_length)])
                self.valid_frame_pointer += self.seq_length
                d.append(self.valid_dataset_pointer)
                i += 1
            else:
                # Not enough frames left
                # Increment the dataset pointer and set the frame_pointer to zero
                self.tick_batch_pointer(valid=True)
        return x_batch, y_batch, d, numPedsList_batch, PedsList_batch, target_ids

    def tick_batch_pointer(self, valid=False):
        # Advance the dataset pointer
        # if False, goto the top path. if True, goto the bottom path.
        if not valid:
            # Go to the next dataset, traverse the next dataset.
            self.dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.frame_pointer = 0
            # If all datasets are done, then go to the first one again, (traverse all files in a folder name)
            if self.dataset_pointer >= len(self.data):
                self.dataset_pointer = 0
            print("*******************")
            print("now processing: %s" % self.get_file_name())
        else:
            # Go to the next dataset
            self.valid_dataset_pointer += 1
            # Set the frame pointer to zero for the current dataset
            self.valid_frame_pointer = 0
            # If all datasets are done, then go to the first one again
            if self.valid_dataset_pointer >= len(self.valid_data):
                self.valid_dataset_pointer = 0
            print("*******************")
            print("now processing: %s" % self.get_file_name(pointer_type='valid'))

    def reset_batch_pointer(self, valid=False):
        '''
        Reset all pointers; if False, then goto the top path. if True, then goto the bottom path.
        '''
        if not valid:
            # Go to the first frame of the first dataset
            self.dataset_pointer = 0
            self.frame_pointer = 0
        else:
            self.valid_dataset_pointer = 0
            self.valid_frame_pointer = 0

    def switch_to_dataset_type(self, train=False, load_data=True):
        # function for switching between train and validation datasets during training session
        # if False, then goto the first path. if True, then goto the second path.(default is goto the first path)
        print('--------------------------------------------------------------------------')
        if not train:  # if train mode, switch to validation mode
            if self.additional_validation:
                print("Dataset type switching: training ----> validation")
                self.orig_seq_lenght, self.seq_length = self.seq_length, self.orig_seq_lenght
                self.data_dirs = self.validation_dataset
                self.numDatasets = len(self.data_dirs)
                if load_data:
                    self.load_preprocessed(self.data_file_vl, True)  # set True for next "validation" operation.
                    self.reset_batch_pointer(
                        valid=False)  # reset the "train" index to ensure from zero index of next "train" operation.
            else:
                print("There is no validation dataset.Aborted.")
                return
        else:  # if validation mode, switch to train mode
            print("Dataset type switching: validation -----> training")
            self.orig_seq_lenght, self.seq_length = self.seq_length, self.orig_seq_lenght
            self.data_dirs = self.train_dataset
            self.numDatasets = len(self.data_dirs)
            if load_data:
                self.load_preprocessed(self.data_file_tr)
                self.reset_batch_pointer(
                    valid=False)  # reset index of last train to ensure start from zero index of next "train" operation.
                self.reset_batch_pointer(
                    valid=True)  # reset index of last validation to ensure start from zero index of next "validation" operation.

    def convert_proper_array(self, x_seq, num_pedlist, pedlist):
        # converter function to appropriate format. Instead of direcly use ped ids, we are mapping ped ids to
        # array indices using a lookup table for each sequence -> speed
        # output: seq_lenght (real sequence lenght+1)*max_ped_id+1 (biggest id number in the sequence)*2 (x,y)

        # get unique ids from sequence
        unique_ids = pd.unique(np.concatenate(pedlist).ravel().tolist()).astype(int)
        # create a lookup table which maps ped ids -> array indices, indices start from 0.
        lookup_table = dict(zip(unique_ids, range(0, len(unique_ids))))
        # the dimension of seq_data:(seq_length,len(lookup_table), 2). for example:(20,145,2)
        seq_data = np.zeros(shape=(self.seq_length, len(lookup_table), 2))

        # create new structure of array
        for ind, frame in enumerate(x_seq):
            # corr_index: find the "array indices" corresponding to each "ped ID".
            # for example: lookup_table={5:0,6:1,8:2,24:3,25:4,...,38:6,39:9,40:10,...}
            # for example: corr_index=[0,1,2],"0，1，2" are derived due to "5,6,8" of ped ID.
            corr_index = [lookup_table[x] for x in frame[:, 0]]
            # the shape of seq_data is:(seq_length,len(lookup_table), 2); for example:(20,145,2)
            # put the coordinate information (x,y) of frame into "seq_data". in other word, the location of "array indices".
            # for more information you can look the function of "open_cpkl_two" for more information.
            seq_data[ind, corr_index, :] = frame[:, 1:3]
        # return_arr: put "seq_data" into type of "torch", then use "Variable" for gradient backhaul.
        return_arr = Variable(torch.from_numpy(np.array(seq_data)).float())
        return return_arr, lookup_table

    def add_element_to_dict(self, dict, key, value):
        # helper function to add an element to dictionary
        dict.setdefault(key, [])
        dict[key].append(value)

    def get_dataset_path(self, base_path, f_prefix):
        dataset = []
        # the type of "dir_names" is list which contain deduplicated folder name of all .txt files.
        dir_names = unique_list(self.get_all_directory_namelist())
        for dir_ in dir_names:
            # the example of dir_path: ./data/train/biwi/
            dir_path = os.path.join('./', base_path, dir_)
            ic(dir_path)
            file_names = get_all_file_names(dir_path)
            # the example of dataset: ./data/train/biwi/biwi_hotel.txt,...
            [dataset.append(dir_path + '/' + file_name) for file_name in file_names]
        return dataset

    def get_file_name(self, offset=0, pointer_type='train'):
        # return file name of processing or pointing by dataset pointer
        # the type of "data_dirs" is list which contain paths of every .txt files.
        if pointer_type is 'train':
            return self.data_dirs[self.dataset_pointer + offset].split('/')[-1]
        elif pointer_type is 'valid':
            return self.data_dirs[self.valid_dataset_pointer + offset].split('/')[-1]

    def create_folder_file_dict(self):
        # create a helper dictionary folder name:file name
        self.folder_file_dict = {}
        for dir_ in self.base_data_dirs:
            folder_name = dir_.split('/')[-2]
            file_name = dir_.split('/')[-1]
            self.add_element_to_dict(self.folder_file_dict, folder_name, file_name)

    def get_directory_name(self, offset=0):
        # return folder name of file of processing or pointing by dataset pointer
        folder_name = self.data_dirs[self.dataset_pointer + offset].split('/')[-2]
        return folder_name

    def get_directory_name_with_pointer(self, pointer_index):
        # get directory name using pointer index; in other word, find the folder name.
        folder_name = self.data_dirs[pointer_index].split('/')[-2]
        return folder_name

    def get_all_directory_namelist(self):
        # find the folder name of .txt files of base_data_dirs{base_test_dataset\base_train_dataset},then put them into list.
        folder_list = [data_dir.split('/')[-2] for data_dir in (self.base_data_dirs)]
        return folder_list

    def get_file_path(self, base, prefix, model_name='', offset=0):
        # return file path of file of processing or pointing by dataset pointer
        folder_name = self.data_dirs[self.dataset_pointer + offset].split('/')[-2]
        base_folder_name = os.path.join(prefix, base, model_name, folder_name)
        return base_folder_name

    def get_base_file_name(self, key):
        # return file name using folder- file dictionary
        return self.folder_file_dict[key]

    def get_len_of_dataset(self):
        # return the number of dataset in the mode
        return len(self.data)

    def clean_test_data(self, x_seq, target_id, obs_lenght, predicted_lenght):
        # remove (pedid, x , y) array if x or y is nan for each frame in observed part (for test mode)
        for frame_num in range(obs_lenght):
            nan_elements_index = np.where(np.isnan(x_seq[frame_num][:, 2]))
            try:
                x_seq[frame_num] = np.delete(x_seq[frame_num], nan_elements_index[0], axis=0)
            except ValueError:
                print("an error has been occured")
                pass

        for frame_num in range(obs_lenght, obs_lenght + predicted_lenght):
            nan_elements_index = x_seq[frame_num][:, 0] != target_id
            try:
                x_seq[frame_num] = x_seq[frame_num][~nan_elements_index]
            except ValueError:
                pass

    def clean_ped_list(self, x_seq, pedlist_seq, target_id, obs_lenght, predicted_lenght):
        # remove peds from pedlist after test cleaning
        target_id_arr = [target_id]
        for frame_num in range(obs_lenght + predicted_lenght):
            pedlist_seq[frame_num] = list(x_seq[frame_num][:, 0])

    def write_to_file(self, data, base, f_prefix, model_name):
        # write all files as txt format
        self.reset_batch_pointer()
        for file in range(self.numDatasets):
            path = self.get_file_path('./', base, model_name, file)
            file_name = self.get_file_name(file)
            self.write_dataset(data[file], file_name, path)

    def write_dataset(self, dataset_seq, file_name, path):
        # write a file in txt format; for example the path:./LSTM\\biwi
        print("Writing to file  path: %s, file_name: %s" % (path, file_name))
        out = np.concatenate(dataset_seq, axis=0)
        np.savetxt(path + '/' + file_name, out, fmt="%1d %1.1f %.3f %.3f", newline='\n')

    def write_to_plot_file(self, data, path):
        # write plot file for further visualization in pkl format
        self.reset_batch_pointer()
        for file in range(self.numDatasets):
            file_name = self.get_file_name(file)
            file_name = file_name.split('.')[0] + '.pkl'
            print("Writing to plot file  path: %s, file_name: %s" % (path, file_name))
            with open(os.path.join(path, file_name), 'wb') as f:
                pickle.dump(data[file], f)

    def get_frame_sequence(self, frame_lenght):
        # get the frame_number. begin and end of predicted frame numbers in this seq.
        begin_fr = (self.frame_pointer - frame_lenght)
        end_fr = (self.frame_pointer)
        frame_number = self.orig_data[self.dataset_pointer][begin_fr:end_fr, 0].transpose()
        return frame_number

    def get_id_sequence(self, frame_lenght):
        # get the id_number. begin and end of predicted frame numbers in this seq.
        begin_fr = (self.frame_pointer - frame_lenght)
        end_fr = (self.frame_pointer)
        id_number = self.orig_data[self.dataset_pointer][begin_fr:end_fr, 1].transpose()
        id_number = [int(i) for i in id_number]
        return id_number

    def get_dataset_dimension(self, file_name):
        # find file's dimension by using "file name".
        return self.dataset_dimensions[file_name]
