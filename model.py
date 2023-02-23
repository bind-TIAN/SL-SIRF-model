import torch
import torch.nn as nn
import numpy as np

from torch.autograd import Variable
from icecream import ic


class SocialModel(nn.Module):

    def __init__(self, args, infer=False):

        # infer: Training or test time (true if test time)

        super(SocialModel, self).__init__()
        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda
        if infer:  # Test time
            self.sequence_length = 1
        else:  # Training time
            self.sequence_length = args.sequence_length

        # Store required sizes
        self.rnn_size = args.rnn_size
        self.grid_size = args.grid_size
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.maxNumPeds = args.maxNumPeds
        self.sequence_length = args.sequence_length
        self.gru = args.gru

        # [4*embedding_size, rnn_size]
        self.cell = nn.LSTMCell(4 * self.embedding_size, self.rnn_size)
        if self.gru:
            self.cell = nn.GRUCell(4 * self.embedding_size, self.rnn_size)

        # position -> [input_size, embedding_size]
        self.input_embedding_layer = nn.Linear(self.input_size, self.embedding_size)
        # social tensor -> [grid_size*grid_size*rnn_size, embedding_size]
        self.tensor_embedding_layer = nn.Linear(self.grid_size * self.grid_size * self.rnn_size, self.embedding_size)

        # Linear layer to map the hidden state of LSTM to output
        # [rnn_size, output_size]
        self.output_layer = nn.Linear(self.rnn_size, self.output_size)

        # ReLU and dropout unit
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

    def getSocialTensor(self, grid, hidden_states):
        '''
        Computes the social tensor for a given grid mask and hidden states of all peds
        grid : Grid masks
        hidden_states : Hidden states of all peds
        '''
        # Number of peds
        numNodes = grid.size()[0]

        # social_tensor: [numNodes, grid_size*grid_size, rnn_size]
        social_tensor = Variable(torch.zeros(numNodes, self.grid_size * self.grid_size, self.rnn_size))
        if self.use_cuda:
            social_tensor = social_tensor.cuda()

        # For each ped
        for node in range(numNodes):
            social_tensor[node] = torch.mm(torch.t(grid[node]), hidden_states)

        # Reshape the social tensor
        social_tensor = social_tensor.view(numNodes, self.grid_size * self.grid_size * self.rnn_size)
        return social_tensor

    # def forward(self, input_data, grids, hidden_states, cell_states ,PedsList, num_pedlist,dataloader, look_up):
    def forward(self, *args):

        '''
        Forward pass for the model
        input_data: Input positions, include x and y coordinates
        grids: Grid masks
        hidden_states: Hidden states of the peds,torch->[len(lookup_seq),rnn_size]
        cell_states: Cell states of the peds,torch->[len(lookup_seq),rnn_size]
        PedsList: id of peds in each frame for this sequence
        num_pedlist: the sum-up value of peds' IDs
        look_up:a table, {5:0,6:1,8:2,24:3,25:4,...,38:6,39:9,40:10,...}

        outputs_return: Outputs corresponding to bivariate Gaussian distributions
        hidden_states
        cell_states
        '''
        input_data = args[0]
        grids = args[1]
        hidden_states = args[2]
        cell_states = args[3]
        PedsList = args[4]
        num_pedlist = args[5]
        dataloader = args[6]
        look_up = args[7]
        input_data_direction = args[8]
        input_data_velocity = args[9]
        if self.gru:
            cell_states = None
        numNodes = len(look_up)
        # outputs: [seq_length * numNodes, output_size]
        outputs = Variable(torch.zeros(self.sequence_length * numNodes, self.output_size))
        if self.use_cuda:
            outputs = outputs.cuda()
        # For each frame in the sequence
        result_position = list(enumerate(input_data))
        result_direction = list(enumerate(input_data_direction))
        result_velocity = list(enumerate(input_data_velocity))
        for i in range(len(result_position)):  # we can also use result_direction or result_velocity.
            nodeIDs = [int(nodeID) for nodeID in PedsList[i]]
            if len(nodeIDs) == 0:
                continue
            list_of_nodes = [look_up[x] for x in nodeIDs]
            corr_index = Variable((torch.LongTensor(list_of_nodes)))
            if self.use_cuda:
                corr_index = corr_index.cuda()
            # Select the corresponding input positions
            nodes_current_position = result_position[i][1][list_of_nodes, :]
            # Select the corresponding input direction
            nodes_current_direction = result_direction[i][1][list_of_nodes, :]
            # Select the corresponding input velocity
            nodes_current_velocity = result_velocity[i][1][list_of_nodes, :]
            # Get the corresponding grid masks,grid_current里面的值只有0和1
            grid_current = grids[i]
            # Get the corresponding hidden and cell states，hidden_states表示：索引对象；0表示：按行索引；corr_index表示：索引序号
            hidden_states_current = torch.index_select(hidden_states, 0, corr_index)
            if not self.gru:
                # cell_states表示：索引对象；0表示：按行索引；corr_index表示：索引序号
                cell_states_current = torch.index_select(cell_states, 0, corr_index)
            # Compute the social tensor
            # grid_current里面的值只有0和1，hidden_states_current的维度是[x,128],即rnn_size=128
            # social_tensor的维度：[numNodes, grid_size*grid_size*rnn_size]
            social_tensor = self.getSocialTensor(grid_current, hidden_states_current)
            # Embed inputs
            # input_size -> embedding_size|relu|dropout |-> input_embedded_position
            input_embedded_position = self.dropout(self.relu(self.input_embedding_layer(nodes_current_position)))
            # input_size -> embedding_size|relu|dropout |-> input_embedded_direction
            input_embedded_direction = self.dropout(self.relu(self.input_embedding_layer(nodes_current_direction)))
            # input_size -> embedding_size|relu|dropout |-> input_embedded_velocity
            input_embedded_velocity = self.dropout(self.relu(self.input_embedding_layer(nodes_current_velocity)))
            # Embed the social tensor
            # numNodes->grid_size*grid_size*rnn_size->grid_size*grid_size*rnn_size -> embedding_size|relu|dropout
            # tensor_embedded shape is [numNodes,embedding_size]
            tensor_embedded = self.dropout(self.relu(self.tensor_embedding_layer(social_tensor)))
            # Concat input
            # input_embedded+tensor_embedded |-> concat_embedded    (2*embedding_size),[position,direction,velocity]
            concat_embedded = torch.cat(
                (input_embedded_position, input_embedded_direction, input_embedded_velocity, tensor_embedded), 1)
            if not self.gru:
                h_nodes, c_nodes = self.cell(concat_embedded, (hidden_states_current, cell_states_current))
            else:
                # cell dimension is: [4*embedding_size, rnn_size]
                h_nodes = self.cell(concat_embedded, (hidden_states_current))
            # after the layer of "output_layer", the dimension becomes to [4*embedding_size, output_size]
            outputs[i * numNodes + corr_index.data] = self.output_layer(h_nodes)
            # Update hidden and cell states
            hidden_states[corr_index.data] = h_nodes
            if not self.gru:
                cell_states[corr_index.data] = c_nodes
        # Reshape outputs
        outputs_return = Variable(torch.zeros(self.sequence_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.sequence_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum * numNodes + node, :]
        return outputs_return, hidden_states, cell_states
