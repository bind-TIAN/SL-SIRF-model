import numpy as np
import torch
import itertools
from torch.autograd import Variable
from icecream import ic
import operator


def getGridMask(flag_, x_sequ, frame, dimensions, num_person, neighborhood_size, grid_size, is_occupancy=False):
    '''
    This function computes the binary mask that represents the
    occupancy of each ped in the other's grid
    params:
    frame : This will be a MNP x 3 matrix with each row being [pedID, x, y]
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered
    grid_size : Scalar value representing the size of the grid discretization
    num_person : number of people exist in given frame
    is_occupancy: A flag using for calculation of accupancy map
    flag_: if is True, then going to original version. if is False, then going to the revised version.
    '''
    if flag_ == 1:
        mnp = num_person
        width, height = dimensions[0], dimensions[1]
        if is_occupancy:
            frame_mask = np.zeros((mnp, grid_size ** 2))
        else:
            frame_mask = np.zeros((mnp, mnp, grid_size ** 2))
        # x_seq_np = x_sequ.data.numpy()
        frame_np = frame.data.numpy()
        # width_bound, height_bound = (neighborhood_size/(width*1.0)), (neighborhood_size/(height*1.0))
        width_bound, height_bound = (neighborhood_size / (width * 1.0)) * 2, (neighborhood_size / (height * 1.0)) * 2
        list_indices = list(range(0, mnp))
        for real_frame_index, other_real_frame_index in itertools.permutations(list_indices, 2):
            # current_x and current_y represent x coordinate and y coordinate respectively.
            current_x, current_y = frame_np[real_frame_index, 0], frame_np[real_frame_index, 1]
            width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
            height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2
            other_x, other_y = frame_np[other_real_frame_index, 0], frame_np[other_real_frame_index, 1]
            if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (other_y < height_low):
                continue
            cell_x = int(np.floor(((other_x - width_low) / width_bound) * grid_size))
            cell_y = int(np.floor(((other_y - height_low) / height_bound) * grid_size))
            if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                continue
            if is_occupancy:
                frame_mask[real_frame_index, cell_x + cell_y * grid_size] = 1
            else:
                frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y * grid_size] = 1
        return frame_mask
    else:
        mnp = num_person
        width, height = dimensions[0], dimensions[1]
        if is_occupancy:
            frame_mask = np.zeros((mnp, grid_size ** 2))
        else:
            frame_mask = np.zeros((mnp, mnp, grid_size ** 2))
        x_seq_np = x_sequ.data.numpy()
        frame_np = frame.data.numpy()
        # width_bound, height_bound = (neighborhood_size/(width*1.0)), (neighborhood_size/(height*1.0))
        width_bound, height_bound = (neighborhood_size / (width * 1.0)) * 2, (neighborhood_size / (height * 1.0)) * 2
        list_indices = list(range(0, mnp))
        for real_frame_index, other_real_frame_index in itertools.permutations(list_indices, 2):
            # Observed person's coordinate. [float32]
            current_x, current_y = frame_np[real_frame_index, 0], frame_np[real_frame_index, 1]
            # Observed person's direction of movement between t and t+1 frames.
            current_direction = x_seq_np[real_frame_index]
            # get the height and width of this "social area". and the shape of this area is rectangular. [float64]
            width_low, width_high = current_x - width_bound / 2, current_x + width_bound / 2
            height_low, height_high = current_y - height_bound / 2, current_y + height_bound / 2
            # computing the direction vector of "observed" pointing to "participator".
            aim_m_n = frame_np[other_real_frame_index] - frame_np[real_frame_index]
            # calculating the value of these two vectors called current_direction and aim_m_n respectively, [float32]
            current_direction_norm = np.linalg.norm(current_direction)
            aim_m_n_norm = np.linalg.norm(aim_m_n)
            # multiply two vectors called current_direction and aim_m_n respectively. [float32]
            multi_vector = np.dot(current_direction, aim_m_n)
            # multiply two vectors called current_direction and aim_m_n respectively,[float32]
            multi_norm = current_direction_norm * aim_m_n_norm
            # calculating the cosine value of current_direction and aim_m_n [float64]
            two_vector_cosine_value = multi_vector / (multi_norm + 1e-8)
            # 判断current_direction和aim_m_n俩向量之间夹角的余弦值必须≥1/2且≤1，若不满足则continue
            if two_vector_cosine_value >= 0.5 and two_vector_cosine_value <= 1 and aim_m_n_norm <= 2:
                # 计算current_direction和aim_m_n俩向量之间的外积, [float32]
                multi_vector_xx = np.cross(current_direction, aim_m_n)
                # 计算该外积的模, [float32]
                multi_vector_xx_norm = np.linalg.norm(multi_vector_xx)
                # 计算current_direction和aim_m_n俩向量之间的夹角的sin值, [float64]
                two_vector_sin_value = multi_vector_xx_norm / (multi_norm + 1e-8)
                # 计算落入扇形区域中的点，转成正方形之后，转换之后的坐标点
                # 1.计算sinθ和cosθ的绝对值, 即two_vector_sin_value和two_vector_cosine_value的绝对值，并求最大值
                abs_two_vector_sin_value = np.abs(two_vector_sin_value)
                abs_two_vector_cosine_value = np.abs(two_vector_cosine_value)
                max_between_sin_cos = max(abs_two_vector_sin_value, abs_two_vector_cosine_value)
                # 2.计算扇形区域的半径r与角度之间的乘积,aim_m_n_norm[float32]；two_vector_sin_value..[float64]
                r_multi_sin = aim_m_n_norm * two_vector_sin_value
                r_multi_cos = aim_m_n_norm * two_vector_cosine_value
                # 3.对落入扇形区域中的坐标点映射到正方形区域中
                other_x = r_multi_cos / (max_between_sin_cos + 1e-8)
                other_y = r_multi_sin / (max_between_sin_cos + 1e-8)
                if (other_x >= width_high) or (other_x < width_low) or (other_y >= height_high) or (
                        other_y < height_low):
                    continue
                # If in surrounding, calculate the grid cell
                cell_x = int(np.floor(((other_x - width_low) / width_bound) * grid_size))
                cell_y = int(np.floor(((other_y - height_low) / height_bound) * grid_size))
                # 在框范围之外的行人坐标不被纳入计算
                if cell_x >= grid_size or cell_x < 0 or cell_y >= grid_size or cell_y < 0:
                    continue
                if is_occupancy:
                    frame_mask[real_frame_index, cell_x + cell_y * grid_size] = 1
                else:
                    # Other ped is in the corresponding grid cell of current ped
                    frame_mask[real_frame_index, other_real_frame_index, cell_x + cell_y * grid_size] = 1
            else:
                continue
        return frame_mask


def getSequenceGridMask(flag_, lookup_seq_, sequence, dimensions, pedlist_seq, neighborhood_size, grid_size, using_cuda,
                        is_occupancy=False):
    '''
    Get the grid masks for all the frames in the sequence
    params:
    sequence : A numpy matrix of shape SL x MNP x 2
    dimensions : This will be a list [width, height]
    neighborhood_size : Scalar value representing the size of neighborhood considered,默认是32
    grid_size : Scalar value representing the size of the grid discretization,默认是4
    using_cuda: Boolean value denoting if using GPU or not,默认是False
    is_occupancy: A flag using for calculation of accupancy map
    pedlist_seq:[Ped_ID1,Ped_ID2,...,Ped_IDn]
    flag_ is a tag，if is True, then going to the original version. if is False, then going to the revised version.
    '''
    if flag_ == 1:
        sl = len(sequence)
        sequence_mask = []
        x_seq_direction = torch.zeros(sequence.shape)
        for i in range(sl):
            mask = Variable(torch.from_numpy(
                getGridMask(flag_, x_seq_direction[i], sequence[i], dimensions, len(pedlist_seq[i]), neighborhood_size,
                            grid_size, is_occupancy)).float())
            if using_cuda:
                mask = mask.cuda()
            sequence_mask.append(mask)
    else:
        sl = len(sequence)
        sequence_mask = []
        x_seq_direction = torch.zeros(sequence.shape)
        result_tian = list(enumerate(sequence))
        # 循环体需要找到t和t+1俩时刻中的所有ped对应的x和y坐标,通过x_seq、lookup_seq、PedsList_seq计算出”被观察者“方向向量传入getGridMask中
        for i in range(1, sl):
            ped_past = pedlist_seq[i - 1]
            ped_current = pedlist_seq[i]
            if (len(ped_past) < len(ped_current)) or (len(ped_past) > len(ped_current)):
                for ped in ped_current:
                    if ped in ped_past:
                        past_position = result_tian[i - 1][1][lookup_seq_[ped], 0:2]
                        current_position = result_tian[i][1][lookup_seq_[ped], 0:2]
                        past_current_diff = current_position - past_position
                        past_current_diff_norm = np.linalg.norm(past_current_diff.cpu())
                        direction = past_current_diff / (past_current_diff_norm + 1e-8)
                        x_seq_direction[i, lookup_seq_[ped], 0:2] = direction
            elif len(ped_past) == len(ped_current):
                res_equal = operator.eq(ped_past, ped_current)
                if res_equal:
                    for ped in ped_current:
                        past_position = result_tian[i - 1][1][lookup_seq_[ped], 0:2]
                        current_position = result_tian[i][1][lookup_seq_[ped], 0:2]
                        past_current_diff = current_position - past_position
                        past_current_diff_norm = np.linalg.norm(past_current_diff.cpu())
                        direction = past_current_diff / (past_current_diff_norm + 1e-8)
                        x_seq_direction[i, lookup_seq_[ped], 0:2] = direction
                else:
                    similar_list = list(set(ped_past) & set(ped_current))
                    if len(similar_list) == 0:
                        continue
                    else:
                        for ped in ped_current:
                            past_position = result_tian[i - 1][1][lookup_seq_[ped], 0:2]
                            current_position = result_tian[i][1][lookup_seq_[ped], 0:2]
                            past_current_diff = current_position - past_position
                            past_current_diff_norm = np.linalg.norm(past_current_diff.cpu())
                            direction = past_current_diff / (past_current_diff_norm + 1e-8)
                            x_seq_direction[i, lookup_seq_[ped], 0:2] = direction
        for i in range(sl):
            mask = Variable(torch.from_numpy(
                getGridMask(flag_, x_seq_direction[i], sequence[i], dimensions, len(pedlist_seq[i]), neighborhood_size,
                            grid_size,
                            is_occupancy)).float())
            if using_cuda:
                mask = mask.cuda()
            sequence_mask.append(mask)
    return sequence_mask
