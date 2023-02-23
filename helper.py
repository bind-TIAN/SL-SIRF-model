import functools
import torch
from torch.autograd import Variable
import os
import shutil
from os import walk
import math
from model import SocialModel
from icecream import ic
import operator
import numpy as np


# one time set dictionary for a exist key
class WriteOnceDict(dict):
    def __setitem__(self, key, value):
        if not key in self:
            super(WriteOnceDict, self).__setitem__(key, value)


def get_method_name(index):
    return {
        1: 'SOCIALLSTM',
    }.get(index, 'SOCIALLSTM')


def get_model(index, arguments, infer=False):
    if index == 1:
        return SocialModel(arguments, infer)


def getCoef(outputs):
    mux, muy, sx, sy, corr = outputs[:, :, 0], outputs[:, :, 1], outputs[:, :, 2], outputs[:, :, 3], outputs[:, :, 4]
    sx = torch.exp(sx)
    sy = torch.exp(sy)
    corr = torch.tanh(corr)
    return mux, muy, sx, sy, corr


def sample_gaussian_2d(particle_, mux, muy, sx, sy, corr, nodesPresent, look_up):
    '''
    Parameters:
    mux, muy, sx, sy, corr
    Contains x-means, y-means, x-stds, y-stds and correlation
    nodesPresent : a list of nodeIDs present in the frame
    look_up : lookup table for determining which ped is in which array index

    Returns:
    next_x, next_y : a tensor of shape numNodes
    Contains sampled values from the 2D gaussian
    '''
    o_mux, o_muy, o_sx, o_sy, o_corr = mux[0, :].cpu(), muy[0, :].cpu(), sx[0, :].cpu(), sy[0, :].cpu(), corr[0,
                                                                                                         :].cpu()
    numNodes = mux.size()[1]
    next_x = torch.zeros(numNodes)
    next_y = torch.zeros(numNodes)
    next_values_cluster = torch.zeros(numNodes, particle_, 2)  # particle_ represents number of particles.
    converted_node_present = [look_up[node] for node in nodesPresent]
    for node in range(numNodes):
        if node not in converted_node_present:
            continue
        mean = [o_mux[node], o_muy[node]]
        cov = [[o_sx[node] * o_sx[node], o_corr[node] * o_sx[node] * o_sy[node]],
               [o_corr[node] * o_sx[node] * o_sy[node], o_sy[node] * o_sy[node]]]
        mean = np.array(mean, dtype='float')
        cov = np.array(cov, dtype='float')
        next_values = np.random.multivariate_normal(mean, cov, particle_)  # revised for particle filter.
        next_x[node] = next_values[0][0]
        next_y[node] = next_values[0][1]
        next_values_cluster[node, :, 0] = torch.from_numpy(next_values[:, 0])
        next_values_cluster[node, :, 1] = torch.from_numpy(next_values[:, 1])
    return next_x, next_y, next_values_cluster


def get_mean_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, using_cuda, look_up):
    '''
    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index
    Error : Mean euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = torch.zeros(pred_length)
    if using_cuda:
        error = error.cuda()
    for tstep in range(pred_length):
        counter = 0
        for nodeID in assumedNodesPresent[tstep]:
            nodeID = int(nodeID)
            if nodeID not in trueNodesPresent[tstep]:
                continue
            nodeID = look_up[nodeID]
            pred_pos = ret_nodes[tstep, nodeID, :]
            true_pos = nodes[tstep, nodeID, :]
            # true_pos = nodes[tstep, nodeID, :].cuda()
            error[tstep] += torch.norm(pred_pos - true_pos, p=2)
            counter += 1
        if counter != 0:
            error[tstep] = error[tstep] / counter
    return torch.mean(error)


def get_final_error(ret_nodes, nodes, assumedNodesPresent, trueNodesPresent, look_up):
    '''
    ret_nodes : A tensor of shape pred_length x numNodes x 2
    Contains the predicted positions for the nodes

    nodes : A tensor of shape pred_length x numNodes x 2
    Contains the true positions for the nodes

    nodesPresent lists: A list of lists, of size pred_length
    Each list contains the nodeIDs of the nodes present at that time-step

    look_up : lookup table for determining which ped is in which array index
    Error : Mean final euclidean distance between predicted trajectory and the true trajectory
    '''
    pred_length = ret_nodes.size()[0]
    error = 0
    counter = 0
    # Last time-step
    tstep = pred_length - 1
    for nodeID in assumedNodesPresent[tstep]:
        nodeID = int(nodeID)
        if nodeID not in trueNodesPresent[tstep]:
            continue
        nodeID = look_up[nodeID]
        pred_pos = ret_nodes[tstep, nodeID, :]
        true_pos = nodes[tstep, nodeID, :]
        # true_pos = nodes[tstep, nodeID, :].cuda()
        error += torch.norm(pred_pos - true_pos, p=2)
        counter += 1
    if counter != 0:
        error = error / counter
    return error


def Gaussian2DLikelihoodInference(outputs, targets, nodesPresent, pred_length, look_up):
    '''
    Computes the likelihood of predicted locations under a bivariate Gaussian distribution at test time
    outputs: Torch variable containing tensor of shape seq_length x numNodes x 1 x output_size
    targets: Torch variable containing tensor of shape seq_length x numNodes x 1 x input_size
    nodesPresent : A list of lists, of size seq_length. Each list contains the nodeIDs that are present in the frame
    '''
    seq_length = outputs.size()[0]
    obs_length = seq_length - pred_length
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)
    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2
    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    # Final PDF calculation
    result = result / denom
    # Numerical stability
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    loss = 0
    counter = 0
    for framenum in range(obs_length, seq_length):
        nodeIDs = nodesPresent[framenum]
        nodeIDs = [int(nodeID) for nodeID in nodeIDs]
        for nodeID in nodeIDs:
            nodeID = look_up[nodeID]
            loss = loss + result[framenum, nodeID]
            counter = counter + 1
    if counter != 0:
        return loss / counter
    else:
        return loss


def Gaussian2DLikelihood(outputs, targets, nodesPresent, look_up):
    '''
    outputs : predicted locations
    targets : true locations
    assumedNodesPresent : Nodes assumed to be present in each frame in the sequence
    nodesPresent : True nodes present in each frame in the sequence
    look_up : lookup table for determining which ped is in which array index
    '''
    seq_length = outputs.size()[0]
    # Extract mean, std devs and correlation
    mux, muy, sx, sy, corr = getCoef(outputs)
    # Compute factors
    normx = targets[:, :, 0] - mux
    normy = targets[:, :, 1] - muy
    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr ** 2
    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    # Final PDF calculation
    result = result / denom
    # Numerical stability
    epsilon = 1e-20
    # result_neg_denom = -torch.log(torch.clamp((1 / denom), min=epsilon))
    result_neg_denom = torch.exp(-z / (2 * negRho))
    denom_loss = 0
    result = -torch.log(torch.clamp(result, min=epsilon))
    loss = 0
    counter = 0
    for framenum in range(seq_length):
        nodeIDs = nodesPresent[framenum]
        nodeIDs = [int(nodeID) for nodeID in nodeIDs]
        for nodeID in nodeIDs:
            nodeID = look_up[nodeID]
            loss = loss + result[framenum, nodeID]
            denom_loss = denom_loss + result_neg_denom[framenum, nodeID]
            counter = counter + 1
    if counter != 0:
        return loss / counter, denom_loss / counter
    else:
        return loss, denom_loss


def remove_file_extention(file_name):
    # remove file extension (.txt) given filename
    return file_name.split('.')[0]


def add_file_extention(file_name, extention):
    # add file extension (.txt) given filename
    return file_name + '.' + extention


def clear_folder(path):
    # remove all files in the folder
    if os.path.exists(path):
        shutil.rmtree(path)
        print("Folder succesfully removed: ", path)
    else:
        print("No such path: ", path)


def delete_file(path, file_name_list):
    # delete given file list
    for file in file_name_list:
        file_path = os.path.join(path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print("File succesfully deleted: ", file_path)
            else:
                print("Error: %s file not found" % file_path)
        except OSError as e:  # if failed, report it back to the user #
            print("Error: %s - %s." % (e.filename, e.strerror))


def get_all_file_names(path):
    # return all file names given directory
    files = []
    # dirpath, dirnames and filenames represent "path", "folder list" and "file names" respectively.
    for (dirpath, dirnames, filenames) in walk(path):
        files.extend(filenames)
        break
    return files


def create_directories(base_folder_path, folder_list):
    # create folders using a folder list and path
    for folder_name in folder_list:
        directory = os.path.join(base_folder_path, folder_name)
        if not os.path.exists(directory):
            os.makedirs(directory)


def unique_list(l):
    # add deduplicated element into x.
    x = []
    for a in l:
        if a not in x:
            x.append(a)
    return x


def angle_between(p1, p2):
    # return angle between two points
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return ((ang1 - ang2) % (2 * np.pi))


def vectorize_seq(x_seq, PedsList_seq, lookup_seq):
    # substract first frame value to all frames for a ped.Therefore, convert absolute pos. to relative pos.
    first_values_dict = WriteOnceDict()  # 存放行人相对位置
    x_seq_velocity = torch.zeros(x_seq.shape)
    x_seq_direction = torch.zeros(x_seq.shape)
    vectorized_x_seq = x_seq.clone()
    result_tbd = list(enumerate(x_seq))
    for ind, frame in enumerate(x_seq):
        # traverse all person's ID in every frames.
        for ped in PedsList_seq[ind]:
            # 按照lookup_seq[ped]对应的索引访问frame中的元素；找到每帧每个ID对应的x和y坐标
            first_values_dict[ped] = frame[lookup_seq[ped], 0:2]
            vectorized_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] - first_values_dict[ped][0:2]
    for i in range(1, len(result_tbd)):
        ped_past = PedsList_seq[i - 1]
        ped_current = PedsList_seq[i]
        if (len(ped_past) < len(ped_current)) or (len(ped_past) > len(ped_current)):
            for ped in ped_current:
                if ped in ped_past:
                    past_position = result_tbd[i - 1][1][lookup_seq[ped], 0:2]
                    current_position = result_tbd[i][1][lookup_seq[ped], 0:2]
                    past_current_diff = current_position - past_position
                    past_current_diff_norm = np.linalg.norm(past_current_diff.cpu())
                    direction = past_current_diff / (past_current_diff_norm + 1e-8)
                    velocity = past_current_diff / 0.4
                    x_seq_velocity[i, lookup_seq[ped], 0:2] = velocity
                    x_seq_direction[i, lookup_seq[ped], 0:2] = direction
        elif len(ped_past) == len(ped_current):
            res_equal = operator.eq(ped_past, ped_current)
            if res_equal:
                for ped in ped_current:
                    past_position = result_tbd[i - 1][1][lookup_seq[ped], 0:2]
                    current_position = result_tbd[i][1][lookup_seq[ped], 0:2]
                    past_current_diff = current_position - past_position
                    past_current_diff_norm = np.linalg.norm(past_current_diff.cpu())
                    direction = past_current_diff / (past_current_diff_norm + 1e-8)
                    velocity = past_current_diff / 0.4
                    x_seq_velocity[i, lookup_seq[ped], 0:2] = velocity
                    x_seq_direction[i, lookup_seq[ped], 0:2] = direction
            else:
                similar_list = list(set(ped_past) & set(ped_current))
                # 若ped_past和ped_current中存在的ID是完全不同的
                if len(similar_list) == 0:
                    continue
                # 有部分ID，在两个列表中都存在，只需考虑当前帧即可
                else:
                    for ped in ped_current:
                        past_position = result_tbd[i - 1][1][lookup_seq[ped], 0:2]
                        current_position = result_tbd[i][1][lookup_seq[ped], 0:2]
                        past_current_diff = current_position - past_position
                        past_current_diff_norm = np.linalg.norm(past_current_diff.cpu())
                        direction = past_current_diff / (past_current_diff_norm + 1e-8)
                        velocity = past_current_diff / 0.4
                        x_seq_velocity[i, lookup_seq[ped], 0:2] = velocity
                        x_seq_direction[i, lookup_seq[ped], 0:2] = direction
    return vectorized_x_seq, first_values_dict, x_seq_direction, x_seq_velocity


def translate(x_seq, PedsList_seq, lookup_seq, value):
    # translate all trajectories given x and y values
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            vectorized_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] - value[0:2]
    return vectorized_x_seq


def revert_seq(x_seq, PedsList_seq, lookup_seq, first_values_dict):
    # convert velocity array to absolute position array
    absolute_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + first_values_dict[ped][0:2]
            # absolute_x_seq[ind, lookup_seq[ped], 0:2] = frame[lookup_seq[ped], 0:2] + first_values_dict[ped][0:2].cuda()
    return absolute_x_seq


def rotate(origin, point, angle):
    # Rotate a point counterclockwise by a given angle around a given origin.
    # The angle should be given in radians.
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    # return torch.cat([qx, qy])
    return [qx, qy]


def time_lr_scheduler(optimizer, epoch, lr_decay=0.5, lr_decay_epoch=10):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    # When epoch is exactly 10 times lr_decay_epoch, this time do not return optimizer
    # In other words, when epoch is exactly 10 times lr_decay_epoch, it is time to decay learning rate
    if epoch % lr_decay_epoch:
        return optimizer
    print("Optimizer learning rate has been decreased.")
    for param_group in optimizer.param_groups:
        param_group['lr'] *= (1. / (1. + lr_decay * epoch))
    return optimizer


def sample_validation_data(x_seq, Pedlist, grid, args, net, look_up, num_pedlist, dataloader, x_seq_direction,
                           x_seq_velocity):
    '''
    The validation sample function
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    num_pedlist : number of peds in each frame
    look_up : lookup table for determining which ped is in which array index
    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)
    total_loss = 0
    total_denom_loss = 0
    # Construct variables for hidden and cell states
    with torch.no_grad():
        hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
        if args.use_cuda:
            hidden_states = hidden_states.cuda()
        if not args.gru:
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size))
            if args.use_cuda:
                cell_states = cell_states.cuda()
        else:
            cell_states = None
        ret_x_seq = Variable(torch.zeros(args.sequence_length, numx_seq, 2))
        # Initialize the return data structure
        if args.use_cuda:
            ret_x_seq = ret_x_seq.cuda()
        ret_x_seq[0] = x_seq[0]
        weight_cluster_ = torch.zeros(args.particle_number, numx_seq, 2)
        # For the observed part of the trajectory
        for tstep in range(args.sequence_length - 1):
            cum_ = torch.zeros(args.particle_number, numx_seq, 2)
            loss = 0
            denom_loss = 0
            # Do a forward prop
            out_, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), [grid[tstep]], hidden_states,
                                                   cell_states, [Pedlist[tstep]], [num_pedlist[tstep]], dataloader,
                                                   look_up, x_seq_direction[tstep].view(1, numx_seq, 2),
                                                   x_seq_velocity[tstep].view(1, numx_seq, 2))
            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(out_)
            # Sample from the bivariate Gaussian
            next_x, next_y, next_values_cluster_ = sample_gaussian_2d(args.particle_number, mux.data, muy.data, sx.data,
                                                                      sy.data, corr.data, Pedlist[tstep], look_up)
            next_values_cluster_copy = next_values_cluster_.clone()
            particle_main = torch.zeros(numx_seq, 2)
            for i in range(numx_seq):
                particle_main_i = torch.from_numpy(np.mean(next_values_cluster_.clone().numpy()[i, :], axis=0))
                particle_main[i] = particle_main_i
            for i in range(next_values_cluster_.shape[1]):
                particle_i = next_values_cluster_[:, i]
                weight_cluster_[i] = 1 / (np.sqrt(2 * np.pi * (1 / 100))) * np.exp(
                    -(particle_main - particle_i) ** 2 / (2 * (1 / 100)))
            weight_cluster_ = weight_cluster_ / (sum(weight_cluster_) + 1e-20)
            for j in range(next_values_cluster_.shape[1]):
                cum_[j] = functools.reduce(lambda x, y: x + y, weight_cluster_[:j + 1])
            for i in range(numx_seq):
                i_x = resampling_process(cum_[:, i, 0], args.particle_number)
                i_y = resampling_process(cum_[:, i, 1], args.particle_number)
                next_values_cluster_copy[i, [k for k in range(args.particle_number)], 0] = next_values_cluster_[
                    i, i_x, 0]
                next_values_cluster_copy[i, [k for k in range(args.particle_number)], 1] = next_values_cluster_[
                    i, i_y, 1]
            next_values_cluster_copy = next_values_cluster_copy.mean(axis=1, keepdim=False)
            new_x = next_values_cluster_copy[:, 0]
            new_y = next_values_cluster_copy[:, 1]
            ret_x_seq[tstep + 1, :, 0] = new_x
            ret_x_seq[tstep + 1, :, 1] = new_y
            # ret_x_seq[tstep + 1, :, 0] = next_x
            # ret_x_seq[tstep + 1, :, 1] = next_y
            # should add extra return element for Gaussian2DLikelihood,12.30,revised
            loss, denom_loss = Gaussian2DLikelihood(out_[0].view(1, out_.size()[1], out_.size()[2]),
                                                    x_seq[tstep].view(1, numx_seq, 2), [Pedlist[tstep]], look_up)
            total_loss += loss
            total_denom_loss += denom_loss
    return ret_x_seq, total_loss / args.sequence_length, total_denom_loss / args.sequence_length


def sample_validation_data_vanilla(x_seq, Pedlist, args, net, look_up, num_pedlist, dataloader, x_seq_direction,
                                   x_seq_velocity):
    '''
    The validation sample function for vanilla method
    x_seq: Input positions
    Pedlist: Peds present in each frame
    args: arguments
    net: The model
    num_pedlist : number of peds in each frame
    look_up : lookup table for determining which ped is in which array index
    '''
    # Number of peds in the sequence
    numx_seq = len(look_up)
    total_loss = 0
    hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size), volatile=True)
    if args.use_cuda:
        hidden_states = hidden_states.cuda()
    if not args.gru:
        cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size), volatile=True)
        if args.use_cuda:
            cell_states = cell_states.cuda()
    else:
        cell_states = None
    ret_x_seq = Variable(torch.zeros(args.sequence_length, numx_seq, 2), volatile=True)
    if args.use_cuda:
        ret_x_seq = ret_x_seq.cuda()
    ret_x_seq[0] = x_seq[0]

    # For the observed part of the trajectory
    for tstep in range(args.sequence_length - 1):
        loss = 0
        out_, hidden_states, cell_states = net(x_seq[tstep].view(1, numx_seq, 2), hidden_states, cell_states,
                                               [Pedlist[tstep]], [num_pedlist[tstep]], dataloader, look_up,
                                               x_seq_direction[tstep].view(1, numx_seq, 2),
                                               x_seq_velocity[tstep].view(1, numx_seq, 2))
        # Extract the mean, std and corr of the bivariate Gaussian
        mux, muy, sx, sy, corr = getCoef(out_)
        # Sample from the bivariate Gaussian
        next_x, next_y, next_values_cluster_ = sample_gaussian_2d(args.particle_number, mux.data, muy.data, sx.data,
                                                                  sy.data, corr.data, Pedlist[tstep], look_up)
        ret_x_seq[tstep + 1, :, 0] = next_x
        ret_x_seq[tstep + 1, :, 1] = next_y
        loss = Gaussian2DLikelihood(out_[0].view(1, out_.size()[1], out_.size()[2]), x_seq[tstep].view(1, numx_seq, 2),
                                    [Pedlist[tstep]], look_up)
        total_loss += loss
    return ret_x_seq, total_loss / args.sequence_length


def rotate_traj_with_target_ped(x_seq, angle, PedsList_seq, lookup_seq):
    # rotate sequence given angle
    origin = (0, 0)
    vectorized_x_seq = x_seq.clone()
    for ind, frame in enumerate(x_seq):
        for ped in PedsList_seq[ind]:
            point = frame[lookup_seq[ped], 0:2]
            rotated_point = rotate(origin, point, angle)
            vectorized_x_seq[ind, lookup_seq[ped], 0] = rotated_point[0]
            vectorized_x_seq[ind, lookup_seq[ped], 1] = rotated_point[1]
    return vectorized_x_seq


def resampling_process(listname, n):
    ran_w = np.random.rand(n)  # 产生N个随机数
    dd = [0 for i in range(n)]
    for i in range(len(ran_w)):
        j = 0
        while ran_w[i] > listname[j]:  # 若随机数在区间之内，则将下标(j+1)存入dd中；listname中存储的是粒子的权重
            if j < n - 1:
                if ran_w[i] <= listname[j + 1]:
                    break
                else:
                    j += 1
            else:
                j = j - 1
                break
        dd[i] = j + 1
    return dd
