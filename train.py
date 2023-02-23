import functools
import torch
import numpy as np
from torch.autograd import Variable
import argparse
import os
import time
import pickle
import subprocess
from model import SocialModel
from utils import DataLoader
from grid import getSequenceGridMask
from helper import *
from icecream import ic


def main():
    parser = argparse.ArgumentParser()
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    # RNN size parameter (dimension of the output/hidden state)
    parser.add_argument('--rnn_size', type=int, default=128, help='size of RNN hidden state')
    # Size of each batch parameter
    parser.add_argument('--batch_size', type=int, default=5, help='minibatch size')
    # Length of sequence to be considered parameter
    parser.add_argument('--sequence_length', type=int, default=20, help='RNN sequence length')
    parser.add_argument('--pred_length', type=int, default=12, help='prediction length')
    # Number of epochs parameter
    parser.add_argument('--num_epochs', type=int, default=30, help='number of epochs')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=400, help='save frequency')
    # TODO: (resolve) Clipping gradients for now. No idea whether we should
    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10., help='clip gradients at this value')
    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.003, help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.95, help='decay rate for rmsprop')
    # Dropout probability parameter
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout probability')
    parser.add_argument('--embedding_size', type=int, default=64,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--neighborhood_size', type=int, default=32,
                        help='Neighborhood size to be considered for social grid')
    parser.add_argument('--grid_size', type=int, default=4, help='Grid size of the social grid')
    # Maximum number of pedestrians to be considered
    parser.add_argument('--maxNumPeds', type=int, default=27, help='Maximum Number of Pedestrians')
    parser.add_argument('--lambda_param', type=float, default=0.0005, help='L2 regularization parameter')
    parser.add_argument('--use_cuda', action="store_true", default=False, help='Use GPU or not')
    parser.add_argument('--gru', action="store_true", default=False, help='True : GRU cell, False: LSTM cell')
    parser.add_argument('--drive', action="store_true", default=False, help='Use Google drive or not')
    parser.add_argument('--num_validation', type=int, default=2,
                        help='Total number of validation dataset for validate accuracy')
    parser.add_argument('--freq_validation', type=int, default=1,
                        help='Frequency number(epoch) of validation using validation data')
    # frequency of optimazer learning decay
    parser.add_argument('--freq_optimizer', type=int, default=8,
                        help='Frequency number(epoch) of learning decay for optimizer')
    # store grids in epoch 0 and use further.2 times faster -> Intensive memory use around 12 GB
    parser.add_argument('--grid', action="store_true", default=True,
                        help='Whether store grids and use further epoch')
    parser.add_argument('--particle_number', type=int, default=1, help='Number of sampling particles')
    args = parser.parse_args()
    train(args)


def train(args):
    time_saved_list = []
    validation_dataset_executed = False
    prefix = ''
    f_prefix = '.'
    if args.drive is True:
        prefix = 'drive/semester_project/social_lstm_final/'
        f_prefix = 'drive/semester_project/social_lstm_final'  # revise to my dictionary path
    # np.clip将args.freq_validation的值限制在[0,num_epochs]之间
    args.freq_validation = np.clip(args.freq_validation, 0, args.num_epochs)
    validation_epoch_list = list(range(args.freq_validation, args.num_epochs + 1, args.freq_validation))
    validation_epoch_list[-1] -= 1
    # Create the data loader object. This object would preprocess the data in terms of
    # batches each of size args.batch_size, of length args.seq_length
    dataloader = DataLoader(f_prefix, args.batch_size, args.sequence_length, args.num_validation, forcePreProcess=True)
    model_name = "LSTM"
    method_name = "SOCIALLSTM"
    save_tar_name = method_name + "_lstm_model_"
    if args.gru:
        model_name = "GRU"
        save_tar_name = method_name + "_gru_model_"
    # Logging directory
    log_directory = os.path.join(prefix, 'log/')
    plot_directory = os.path.join(prefix, 'plot/', method_name, model_name)
    plot_train_file_directory = 'validation'
    # Logging files
    log_file_curve = open(os.path.join(log_directory, method_name, model_name, 'log_curve.txt'), 'w+')
    log_file = open(os.path.join(log_directory, method_name, model_name, 'val.txt'), 'w+')
    # model directory
    save_directory = os.path.join(prefix, 'model/')
    # Save the arguments int the config file
    with open(os.path.join(save_directory, method_name, model_name, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file
    def checkpoint_path(x):
        return os.path.join(save_directory, method_name, model_name, save_tar_name + str(x) + '.tar')

    # model creation
    net = SocialModel(args)
    if args.use_cuda:
        net = net.cuda()
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adagrad(net.parameters(), weight_decay=args.lambda_param)
    # optimizer = torch.optim.Adam(net.parameters(), weight_decay=args.lambda_param)
    learning_rate = args.learning_rate
    best_val_loss = 100
    best_val_data_loss = 100
    smallest_err_val = 100000
    smallest_err_val_data = 100000
    best_epoch_val = 0
    best_epoch_val_data = 0
    best_err_epoch_val = 0
    best_err_epoch_val_data = 0
    num_batch = 0
    dataset_pointer_ins_grid = -1
    grids, all_epoch_results = [], []
    # grids是一个大列表，每个遍历的文件都是一个小列表，且该小列表初始化为[]
    [grids.append([]) for dataset in range(dataloader.get_len_of_dataset())]
    # Training
    for epoch in range(args.num_epochs):
        print('****************Training epoch beginning******************')
        if dataloader.additional_validation and (epoch - 1) in validation_epoch_list:
            dataloader.switch_to_dataset_type(True)  # 从验证模式进入train模式
        dataloader.reset_batch_pointer(valid=False)  # 把上一次train的索引清零，保证下次train的时候从零索引
        loss_epoch, denom_loss_epoch = 0, 0
        # For each batch,num_batches指的是批量的个数,其计算方法是：num_batches=int(counter / self.batch_size)
        for batch in range(dataloader.num_batches):
            start = time.time()
            # Get batch data
            x, y, d, numPedsList, PedsList, target_ids = dataloader.next_batch()
            loss_batch, denom_loss_batch = 0, 0
            # if we are in a new dataset, zero the counter of batch，dataset_pointer指的是遍历的文本文件的个数
            if dataset_pointer_ins_grid is not dataloader.dataset_pointer and epoch is not 0:
                num_batch = 0
                dataset_pointer_ins_grid = dataloader.dataset_pointer
            # For each sequence,这里batch_size的大小为5
            for sequence in range(dataloader.batch_size):
                # Get the data corresponding to the current sequence
                # x_seq:[Ped_ID1,x1,y1],...,[Ped_IDn,xn,yn](当前帧)
                # _:[Ped_ID1,x1,y1],...,[Ped_IDn,xn,yn](下一帧)
                # d_seq:存储当前文件的号码，每新遍历一个文件，就依次加1
                # numPedsList_seq:不同Ped_IDs的总个数(at inference time)(当前帧)
                # PedsList_seq:[Ped_ID1,Ped_ID2,...,Ped_IDn](at inference time)(当前帧)
                # target_ids:存储所有文本文件中所有行人的ID
                x_seq, _, d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[
                    sequence], PedsList[sequence]
                target_id = target_ids[sequence]
                # get processing file name and then get dimensions of file
                # 找到访问的文本文件的目录的名字
                folder_name = dataloader.get_directory_name_with_pointer(d_seq)
                # dataset_data列表中元素的解释：[width,height]
                dataset_data = dataloader.get_dataset_dimension(folder_name)
                # x_seq:之前有提到，lookup_seq:一个table表，如{5:0,6:1,8:2,24:3,25:4,...,38:6,39:9,40:10,...}
                x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
                # grid mask calculation and storage depending on grid parameter,grid默认是True
                # dataset_pointer指的是遍历的文件的个数,grid_seq是表示在该序列长度下，所有的mask组成一个列表，且该列表的长度=该序列的长度
                # grids是一个大列表，将grid_seq添加进该大列表中
                # 所有的mask组成一个小列表，有关mask的具体信息，请查看函数：open_cpkl_two
                if (args.grid):
                    if (epoch is 0):
                        grid_seq = getSequenceGridMask(0, lookup_seq, x_seq, dataset_data, PedsList_seq,
                                                       args.neighborhood_size,
                                                       args.grid_size, args.use_cuda)
                        grids[dataloader.dataset_pointer].append(grid_seq)
                    else:
                        grid_seq = grids[dataloader.dataset_pointer][(num_batch * dataloader.batch_size) + sequence]
                else:
                    grid_seq = getSequenceGridMask(0, lookup_seq, x_seq, dataset_data, PedsList_seq,
                                                   args.neighborhood_size,
                                                   args.grid_size, args.use_cuda)
                # vectorize trajectories in sequence,经过vectorize_seq函数将绝对地址转换为相对地址，返回转换后的相对地址
                x_seq, _, x_seq_direction, x_seq_velocity = vectorize_seq(x_seq, PedsList_seq, lookup_seq)
                if args.use_cuda:
                    x_seq = x_seq.cuda()
                    x_seq_direction = x_seq_direction.cuda()
                    x_seq_velocity = x_seq_velocity.cuda()
                # number of peds in this sequence per frame
                numNodes = len(lookup_seq)
                hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    hidden_states = hidden_states.cuda()
                cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                if args.use_cuda:
                    cell_states = cell_states.cuda()
                # Zero out gradients
                net.zero_grad()
                optimizer.zero_grad()
                # Forward prop
                outputs, _, _ = net(x_seq, grid_seq, hidden_states, cell_states, PedsList_seq, numPedsList_seq,
                                    dataloader, lookup_seq, x_seq_direction, x_seq_velocity)
                # Compute loss
                loss, denom_loss = Gaussian2DLikelihood(outputs, x_seq, PedsList_seq, lookup_seq)
                denom_loss_batch += denom_loss.item()
                loss_batch += loss.item()
                # Compute gradients
                loss.backward()
                # Clip gradients，put this code between "backward" and "step".
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                # Update parameters
                optimizer.step()
            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            denom_loss_batch = denom_loss_batch / dataloader.batch_size
            loss_epoch += loss_batch
            denom_loss_epoch += denom_loss_batch
            num_batch += 1
            print('{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(
                epoch * dataloader.num_batches + batch,
                args.num_epochs * dataloader.num_batches,
                epoch,
                loss_batch, end - start))
            time_saved_list.append(end - start)
        loss_epoch /= dataloader.num_batches
        denom_loss_epoch /= dataloader.num_batches
        # Log loss values
        log_file_curve.write(
            "Training epoch: " + str(epoch) + " loss: " + str(loss_epoch) + "denom loss: " + str(
                denom_loss_epoch) + '\n')
        if dataloader.valid_num_batches > 0:
            print('****************Validation epoch beginning******************')
            # Validation
            dataloader.reset_batch_pointer(valid=True)
            loss_epoch, err_epoch, denom_loss_epoch = 0, 0, 0
            # For each batch
            for batch in range(dataloader.valid_num_batches):
                # Get batch data
                x, y, d, numPedsList, PedsList, target_ids = dataloader.next_valid_batch()
                # Loss for this batch
                loss_batch, err_batch, denom_loss_batch = 0, 0, 0
                # For each sequence
                for sequence in range(dataloader.batch_size):
                    # Get data corresponding to the current sequence
                    x_seq, _, d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[
                        sequence], PedsList[sequence]
                    target_id = target_ids[sequence]
                    # get processing file name and then get dimensions of file
                    folder_name = dataloader.get_directory_name_with_pointer(d_seq)
                    dataset_data = dataloader.get_dataset_dimension(folder_name)
                    # dense vector creation
                    x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
                    # get grid mask
                    grid_seq = getSequenceGridMask(0, lookup_seq, x_seq, dataset_data, PedsList_seq,
                                                   args.neighborhood_size,
                                                   args.grid_size, args.use_cuda)
                    x_seq, first_values_dict, x_seq_direction, x_seq_velocity = vectorize_seq(x_seq, PedsList_seq,
                                                                                              lookup_seq)
                    if args.use_cuda:
                        x_seq = x_seq.cuda()
                        x_seq_direction = x_seq_direction.cuda()
                        x_seq_velocity = x_seq_velocity.cuda()
                    # number of peds in this sequence per frame
                    numNodes = len(lookup_seq)
                    hidden_states = Variable(torch.zeros(numNodes, args.rnn_size))
                    if args.use_cuda:
                        hidden_states = hidden_states.cuda()
                    cell_states = Variable(torch.zeros(numNodes, args.rnn_size))
                    if args.use_cuda:
                        cell_states = cell_states.cuda()

                    # Forward prop
                    # 只要最后一个行人id，x，y作为net的输入
                    outputs, _, _ = net(x_seq[:-1], grid_seq[:-1], hidden_states, cell_states, PedsList_seq[:-1],
                                        numPedsList_seq, dataloader, lookup_seq, x_seq_direction, x_seq_velocity)
                    # Compute loss
                    loss, denom_loss = Gaussian2DLikelihood(outputs, x_seq[1:], PedsList_seq[1:], lookup_seq)
                    # Extract the mean, std and corr of the bivariate Gaussian
                    mux, muy, sx, sy, corr = getCoef(outputs)
                    # Sample from the bivariate Gaussian
                    next_x, next_y, next_values_cluster_ = sample_gaussian_2d(args.particle_number, mux.data, muy.data,
                                                                              sx.data, sy.data, corr.data,
                                                                              PedsList_seq[-1], lookup_seq)
                    weight_cluster_ = torch.zeros(args.particle_number, numNodes, 2)
                    cum_ = torch.zeros(args.particle_number, numNodes, 2)
                    next_values_cluster_copy = next_values_cluster_.clone()
                    particle_main = torch.zeros(numNodes, 2)
                    for i in range(numNodes):
                        particle_main_i = torch.from_numpy(np.mean(next_values_cluster_.clone().numpy()[i, :], axis=0))
                        particle_main[i] = particle_main_i
                    for i in range(next_values_cluster_.shape[1]):
                        particle_i = next_values_cluster_[:, i]
                        weight_cluster_[i] = 1 / (np.sqrt(2 * np.pi * (1 / 100))) * np.exp(
                            -(particle_main - particle_i) ** 2 / (2 * (1 / 100)))
                    weight_cluster_ = weight_cluster_ / (sum(weight_cluster_) + 1e-20)
                    for j in range(next_values_cluster_.shape[1]):
                        cum_[j] = functools.reduce(lambda x, y: x + y, weight_cluster_[:j + 1])
                    for i in range(numNodes):
                        i_x = resampling_process(cum_[:, i, 0], args.particle_number)
                        i_y = resampling_process(cum_[:, i, 1], args.particle_number)
                        next_values_cluster_copy[i, [k for k in range(args.particle_number)], 0] = next_values_cluster_[
                            i, i_x, 0]
                        next_values_cluster_copy[i, [k for k in range(args.particle_number)], 1] = next_values_cluster_[
                            i, i_y, 1]
                    next_values_cluster_copy = next_values_cluster_copy.mean(axis=1, keepdim=False)
                    new_x = next_values_cluster_copy[:, 0]
                    new_y = next_values_cluster_copy[:, 1]
                    next_vals = torch.FloatTensor(1, numNodes, 2)
                    next_vals[:, :, 0] = new_x
                    next_vals[:, :, 1] = new_y
                    # next_vals = torch.FloatTensor(1, numNodes, 2)
                    # next_vals[:, :, 0] = next_x
                    # next_vals[:, :, 1] = next_y
                    err = get_mean_error(next_vals, x_seq[-1].data[None, :, :], [PedsList_seq[-1]], [PedsList_seq[-1]],
                                         args.use_cuda, lookup_seq)
                    loss_batch += loss.item()
                    denom_loss_batch += denom_loss.item()
                    err_batch += err
                loss_batch = loss_batch / dataloader.batch_size
                denom_loss_batch = denom_loss_batch / dataloader.batch_size
                err_batch = err_batch / dataloader.batch_size
                loss_epoch += loss_batch
                denom_loss_epoch += denom_loss_batch
                err_epoch += err_batch
            if dataloader.valid_num_batches != 0:
                loss_epoch = loss_epoch / dataloader.valid_num_batches
                err_epoch = err_epoch / dataloader.num_batches
                denom_loss_epoch = denom_loss_epoch / dataloader.valid_num_batches
                # Update best validation loss until now
                if loss_epoch < best_val_loss:
                    best_val_loss = loss_epoch
                    best_epoch_val = epoch
                if err_epoch < smallest_err_val:
                    smallest_err_val = err_epoch
                    best_err_epoch_val = epoch
                print('(epoch {}), valid_loss = {:.3f}, valid_err = {:.3f}'.format(epoch, loss_epoch, err_epoch))
                print('Best epoch', best_epoch_val, 'Best validation loss', best_val_loss, 'Best error epoch',
                      best_err_epoch_val, 'Best error', smallest_err_val)
                log_file_curve.write(
                    "Validation epoch: " + str(epoch) + " loss: " + str(loss_epoch) + " err: " + str(
                        err_epoch) + "denom loss: " + str(denom_loss_epoch) + '\n')
        # Validation dataset
        if dataloader.additional_validation and (epoch) in validation_epoch_list:
            dataloader.switch_to_dataset_type()
            print('****************Validation with dataset epoch beginning******************')
            dataloader.reset_batch_pointer(valid=False)
            dataset_pointer_ins = dataloader.dataset_pointer
            validation_dataset_executed = True
            loss_epoch, denom_loss_epoch, err_epoch, f_err_epoch, num_of_batch = 0, 0, 0, 0, 0
            smallest_err = 100000
            # results of one epoch for all validation datasets
            epoch_result = []
            # results of one validation dataset
            results = []
            # For each batch
            for batch in range(dataloader.num_batches):
                # Get batch data
                x, y, d, numPedsList, PedsList, target_ids = dataloader.next_batch()
                if dataset_pointer_ins is not dataloader.dataset_pointer:
                    if dataloader.dataset_pointer is not 0:
                        print('Finished prosessed file : ', dataloader.get_file_name(-1), ' Avarage error : ',
                              err_epoch / num_of_batch)
                        num_of_batch = 0
                        epoch_result.append(results)
                    dataset_pointer_ins = dataloader.dataset_pointer
                    results = []
                # Loss for this batch
                loss_batch, denom_loss_batch, err_batch, f_err_batch = 0, 0, 0, 0
                # For each sequence
                for sequence in range(dataloader.batch_size):
                    # Get data corresponding to the current sequence
                    x_seq, _, d_seq, numPedsList_seq, PedsList_seq = x[sequence], y[sequence], d[sequence], numPedsList[
                        sequence], PedsList[sequence]
                    target_id = target_ids[sequence]
                    # get processing file name and then get dimensions of file
                    folder_name = dataloader.get_directory_name_with_pointer(d_seq)
                    dataset_data = dataloader.get_dataset_dimension(folder_name)
                    # dense vector creation
                    x_seq, lookup_seq = dataloader.convert_proper_array(x_seq, numPedsList_seq, PedsList_seq)
                    # will be used for error calculation
                    orig_x_seq = x_seq.clone()
                    # grid mask calculation
                    grid_seq = getSequenceGridMask(0, lookup_seq, x_seq, dataset_data, PedsList_seq,
                                                   args.neighborhood_size,
                                                   args.grid_size, args.use_cuda)
                    if args.use_cuda:
                        x_seq = x_seq.cuda()
                        orig_x_seq = orig_x_seq.cuda()
                    # vectorize datapoints
                    x_seq, first_values_dict, x_seq_direction, x_seq_velocity = vectorize_seq(x_seq, PedsList_seq,
                                                                                              lookup_seq)
                    if args.use_cuda:
                        x_seq = x_seq.cuda()
                        x_seq_velocity = x_seq_velocity.cuda()
                        x_seq_direction = x_seq_direction.cuda()
                    # sample predicted points from model
                    ret_x_seq, loss, denom_loss = sample_validation_data(x_seq, PedsList_seq, grid_seq, args, net,
                                                                         lookup_seq,
                                                                         numPedsList_seq, dataloader, x_seq_direction,
                                                                         x_seq_velocity)
                    # revert the points back to original space
                    ret_x_seq = revert_seq(ret_x_seq, PedsList_seq, lookup_seq, first_values_dict)
                    # get mean and final error
                    err = get_mean_error(ret_x_seq.data, orig_x_seq.data, PedsList_seq, PedsList_seq, args.use_cuda,
                                         lookup_seq)
                    f_err = get_final_error(ret_x_seq.data, orig_x_seq.data, PedsList_seq, PedsList_seq, lookup_seq)
                    loss_batch += loss.item()
                    denom_loss_batch += denom_loss.item()
                    err_batch += err
                    f_err_batch += f_err
                    print('Current file : ', dataloader.get_file_name(0), ' Batch : ', batch + 1, ' Sequence: ',
                          sequence + 1, ' Sequence mean error: ', err, ' Sequence final error: ', f_err, ' time: ',
                          end - start)
                    results.append((orig_x_seq.data.cpu().numpy(), ret_x_seq.data.cpu().numpy(), PedsList_seq,
                                    lookup_seq, dataloader.get_frame_sequence(args.sequence_length), target_id))
                loss_batch = loss_batch / dataloader.batch_size
                denom_loss_batch = denom_loss_batch / dataloader.batch_size
                err_batch = err_batch / dataloader.batch_size
                f_err_batch = f_err_batch / dataloader.batch_size
                num_of_batch += 1
                loss_epoch += loss_batch
                denom_loss_epoch += denom_loss_batch
                err_epoch += err_batch
                f_err_epoch += f_err_batch
            epoch_result.append(results)
            all_epoch_results.append(epoch_result)
            if dataloader.num_batches != 0:
                loss_epoch = loss_epoch / dataloader.num_batches
                denom_loss_epoch = denom_loss_epoch / dataloader.num_batches
                err_epoch = err_epoch / dataloader.num_batches
                f_err_epoch = f_err_epoch / dataloader.num_batches
                avarage_err = (err_epoch + f_err_epoch) / 2
                # Update best validation loss until now
                if loss_epoch < best_val_data_loss:
                    best_val_data_loss = loss_epoch
                    best_epoch_val_data = epoch
                if avarage_err < smallest_err_val_data:
                    smallest_err_val_data = avarage_err
                    best_err_epoch_val_data = epoch
                print('(epoch {}), valid_loss = {:.3f}, valid_mean_err = {:.3f}, valid_final_err = {:.3f}'.format(epoch,
                                                                                                                  loss_epoch,
                                                                                                                  err_epoch,
                                                                                                                  f_err_epoch))
                print('Best epoch', best_epoch_val_data, 'Best validation loss', best_val_data_loss, 'Best error epoch',
                      best_err_epoch_val_data, 'Best error', smallest_err_val_data)
                log_file_curve.write(
                    "Validation dataset epoch: " + str(epoch) + " loss: " + str(
                        loss_epoch) + "denom loss: " + str(denom_loss_epoch) + " mean_err: " + str(
                        err_epoch) + 'final_err: ' + str(f_err_epoch) + '\n')
            optimizer = time_lr_scheduler(optimizer, epoch, lr_decay_epoch=args.freq_optimizer)
        # Save the model after each epoch
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))

    if dataloader.valid_num_batches != 0:
        print('Best epoch', best_epoch_val, 'Best validation Loss', best_val_loss, 'Best error epoch',
              best_err_epoch_val, 'Best error', smallest_err_val)
        # Log the best epoch and best validation loss
        log_file.write(
            'Validation Best epoch:' + str(best_epoch_val) + ',' + ' Best validation Loss: ' + str(best_val_loss))

    if dataloader.additional_validation:
        print('Best epoch acording to validation dataset', best_epoch_val_data, 'Best validation Loss',
              best_val_data_loss, 'Best error epoch', best_err_epoch_val_data, 'Best error', smallest_err_val_data)
        log_file.write(
            "Validation dataset Best epoch: " + str(best_epoch_val_data) + ',' + ' Best validation Loss: ' + str(
                best_val_data_loss) + '\n')

    if validation_dataset_executed:
        dataloader.switch_to_dataset_type(load_data=False)
        create_directories(plot_directory, [plot_train_file_directory])
        dataloader.write_to_plot_file(all_epoch_results[len(all_epoch_results) - 1],
                                      os.path.join(plot_directory, plot_train_file_directory))
    # Close logging files
    log_file.close()
    log_file_curve.close()
    # store time_saved_list in .txt file
    path = os.getcwd()
    save_path = path + '\\time_SL_SIRF.txt'
    with open(save_path, 'w') as f:
        for i in range(len(time_saved_list)):
            f.write(str(time_saved_list[i]) + '\n')
    f.close()


if __name__ == '__main__':
    main()
