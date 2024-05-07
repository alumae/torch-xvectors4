import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn

def to_device(device_object, tensor):
    """
    Select device for non-parameters tensor w.r.t model or tensor which has been specified a device.
    """
    if isinstance(device_object, torch.nn.Module):
        device = next(device_object.parameters()).device
    elif isinstance(device_object, torch.Tensor):
        device = device_object.device

    return tensor.to(device)


def get_tensors(tensor_sets):
    """Get a single tensor list from a nested tensor_sets list/tuple object,
    such as transforming [(tensor1,tensor2),tensor3] to [tensor1,tensor2,tensor3]
    """
    tensors = []
    
    for this_object in tensor_sets:
        # Only tensor
        if isinstance(this_object, torch.Tensor):
            tensors.append(this_object)
        if isinstance(this_object, np.ndarray):
            tensors.append(torch.from_numpy(this_object))
        elif isinstance(this_object, list) or isinstance(this_object, tuple):
            tensors.extend(get_tensors(this_object))

    return tensors


def for_device_free(function):
    """
    A decorator to make class-function with input-tensor device-free
    Used in libs.nnet.framework.TopVirtualNnet
    """
    def wrapper(self, *tensor_sets):
        transformed = []

        for tensor in get_tensors(tensor_sets):
            transformed.append(to_device(self, tensor))

        return function(self, *transformed)

    return wrapper


class TdnnAffine(torch.nn.Module):
    """ An implemented tdnn affine component by conv1d
        y = splice(w * x, context) + b
    @input_dim: number of dims of frame <=> inputs channels of conv
    @output_dim: number of layer nodes <=> outputs channels of conv
    @context: a list of context
        e.g.  [-2,0,2]
    If context is [0], then the TdnnAffine is equal to linear layer.
    """
    def __init__(self, input_dim, output_dim, context=[0], bias=True, pad=True, stride=1, groups=1, norm_w=False, norm_f=False):
        super(TdnnAffine, self).__init__()
        assert input_dim % groups == 0
        # Check to make sure the context sorted and has no duplicated values
        for index in range(0, len(context) - 1):
            if(context[index] >= context[index + 1]):
                raise ValueError("Context tuple {} is invalid, such as the order.".format(context))

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context = context
        self.bool_bias = bias
        self.pad = pad
        self.groups = groups

        self.norm_w = norm_w
        self.norm_f = norm_f

        # It is used to subsample frames with this factor
        self.stride = stride

        self.left_context = context[0] if context[0] < 0 else 0 
        self.right_context = context[-1] if context[-1] > 0 else 0 

        self.tot_context = self.right_context - self.left_context + 1

        # Do not support sphereConv now.
        if self.tot_context > 1 and self.norm_f:
            self.norm_f = False
            print("Warning: do not support sphereConv now and set norm_f=False.")

        kernel_size = (self.tot_context,)

        self.weight = torch.nn.Parameter(torch.randn(output_dim, input_dim//groups, *kernel_size))

        if self.bool_bias:
            self.bias = torch.nn.Parameter(torch.randn(output_dim))
        else:
            self.register_parameter('bias', None)

        # init weight and bias. It is important
        self.init_weight()

        # Save GPU memory for no skiping case
        if len(context) != self.tot_context:
            # Used to skip some frames index according to context
            self.mask = torch.tensor([[[ 1 if index in context else 0 \
                                        for index in range(self.left_context, self.right_context + 1) ]]])
        else:
            self.mask = None

        ## Deprecated: the broadcast method could be used to save GPU memory, 
        # self.mask = torch.randn(output_dim, input_dim, 0)
        # for index in range(self.left_context, self.right_context + 1):
        #     if index in context:
        #         fixed_value = torch.ones(output_dim, input_dim, 1)
        #     else:
        #         fixed_value = torch.zeros(output_dim, input_dim, 1)

        #     self.mask=torch.cat((self.mask, fixed_value), dim = 2)

        # Save GPU memory of thi case.

        self.selected_device = False

    def init_weight(self):
        # Note, var should be small to avoid slow-shrinking
        torch.nn.init.normal_(self.weight, 0., 0.01)

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.)


    def forward(self, inputs):
        """
        @inputs: a 3-dimensional tensor (a batch), including [samples-index, frames-dim-index, frames-index]
        """
        assert len(inputs.shape) == 3
        assert inputs.shape[1] == self.input_dim

        # Do not use conv1d.padding for self.left_context + self.right_context != 0 case.
        if self.pad:
            inputs = F.pad(inputs, (-self.left_context, self.right_context), mode="constant", value=0)

        assert inputs.shape[2] >=  self.tot_context

        if not self.selected_device and self.mask is not None:
            # To save the CPU -> GPU moving time
            # Another simple case, for a temporary tensor, jus specify the device when creating it.
            # such as, this_tensor = torch.tensor([1.0], device=inputs.device)
            self.mask = to_device(self, self.mask)
            self.selected_device = True

        filters = self.weight  * self.mask if self.mask is not None else self.weight

        if self.norm_w:
            filters = F.normalize(filters, dim=1)

        if self.norm_f:
            inputs = F.normalize(inputs, dim=1)

        outputs = F.conv1d(inputs, filters, self.bias, stride=self.stride, padding=0, dilation=1, groups=self.groups)

        return outputs

    def extra_repr(self):
        return '{input_dim}, {output_dim}, context={context}, bias={bool_bias}, stride={stride}, ' \
               'pad={pad}, groups={groups}, norm_w={norm_w}, norm_f={norm_f}'.format(**self.__dict__)

    @classmethod
    def thop_count(self, m, x, y):
        x = x[0]

        kernel_ops = torch.zeros(m.weight.size()[2:]).numel()  # Kw x Kh
        bias_ops = 1 if m.bias is not None else 0

        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
        total_ops = y.nelement() * (m.input_dim * kernel_ops + bias_ops)

        m.total_ops += torch.DoubleTensor([int(total_ops)])

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def contrastive_cosine_loss(output1, output2, label, margin=0):
    # Cosine similarity
    #breakpoint()
    cosine_similarity = F.cosine_similarity(output1.unsqueeze(0), output2.unsqueeze(0))
    # For similar pairs, loss is (1-cosine_similarity)
    # For dissimilar pairs, loss is max(0, cosine_similarity - margin)
    if label == 1:
        loss = (1 - cosine_similarity)
    else:
        loss = torch.clamp(cosine_similarity - margin, min=0.0)
    
    return loss

    import torch



def concordance_correlation_coefficient_batched(x, y):
    # Ensure x and y are of the same shape
    assert x.shape == y.shape, "Input matrices must have the same shape"

    # Initialize CCC list
    ccc_list = []
    
    # Iterate over columns
    for i in range(x.shape[1]):
        col_x = x[:, i]
        col_y = y[:, i]

        # Calculate means
        mean_x = torch.mean(col_x)
        mean_y = torch.mean(col_y)

        # Calculate variances
        var_x = torch.var(col_x, unbiased=False)
        var_y = torch.var(col_y, unbiased=False)

        # Calculate covariance
        covariance = torch.mean((col_x - mean_x) * (col_y - mean_y))

        # Compute CCC for the current column
        ccc = (2 * covariance) / (var_x + var_y + (mean_x - mean_y) ** 2)
        ccc_list.append(ccc)

    # Calculate mean CCC over all columns
    mean_ccc = torch.mean(torch.tensor(ccc_list))
    return mean_ccc


class CCCLoss(nn.Module):
    def __init__(self, input_dim, output_dim, likert_scale=7):
        """
        Initializes the CCC loss module.
        
        :param input_dim: Dimensionality of the input features.
        :param hidden_dim: Dimensionality of the hidden layer.
        :param output_dim: Dimensionality of the output features (K).
        """
        super(CCCLoss, self).__init__()
        #self.hidden_layer = nn.Linear(input_dim, input_dim)
        self.output_layer = nn.Linear(input_dim, output_dim)
        
        self.output_layer.bias.data.fill_(0.0)
        self.tanh = nn.Tanh()

    def get_predictions(self, x):
        # Process input through the hidden layer and the output layer
        #return self.output_layer(self.tanh(self.hidden_layer(x)))
        return self.output_layer(self.tanh((x)))

    def forward(self, x, y):
        """
        Forward pass to compute the CCC loss.
        
        :param x: Input tensor from the neural network (batch_size, input_dim).
        :param y: Target tensor (batch_size, output_dim).
        :return: Mean CCC loss.
        """
        x = self.get_predictions(x)

        
        # Ensure x and y are of the same shape
        assert x.shape == y.shape, "Shape of predicted and target tensors must match"
        
        # Compute CCC for each column and then the mean CCC over all columns
        mean_x = torch.mean(x, dim=0)
        mean_y = torch.mean(y, dim=0)
        var_x = torch.var(x, dim=0, unbiased=False)
        var_y = torch.var(y, dim=0, unbiased=False)
        covariance = torch.mean((x - mean_x) * (y - mean_y), dim=0)
        ccc = (2 * covariance) / (var_x + var_y + (mean_x - mean_y) ** 2)
        mean_ccc = torch.mean(ccc)
        
        # Since CCC indicates agreement, to use it as a loss we subtract from 1
        return 1 - mean_ccc
    
# class CCCLoss(nn.Module):
#     def __init__(self, input_dim, output_dim, likert_scale=7):
#         """
#         Initializes the CCC loss module.
        
#         :param input_dim: Dimensionality of the input features.
#         :param hidden_dim: Dimensionality of the hidden layer.
#         :param output_dim: Dimensionality of the output features (K).
#         """
#         super(CCCLoss, self).__init__()
#         self.hidden_layer = nn.Linear(input_dim, input_dim)
#         self.output_layer = nn.Linear(input_dim, output_dim)
        
#         self.output_layer.bias.data.fill_(0.0)
#         self.likert_scale = likert_scale
#         #self.tanh = nn.Tanh()

#     def get_predictions(self, x):
#         # Process input through the hidden layer and the output layer
#         #return self.output_layer(self.tanh(self.hidden_layer(x)))
#         #return self.output_layer(self.tanh((x)))
#         x = torch.relu(self.hidden_layer(x))
#         x = torch.sigmoid(self.output_layer(x))
#         return x * (self.likert_scale - 1) + 1  # Scale output to 1-scale range

#     def forward(self, x, y):
#         """
#         Forward pass to compute the CCC loss.
        
#         :param x: Input tensor from the neural network (batch_size, input_dim).
#         :param y: Target tensor (batch_size, output_dim).
#         :return: Mean CCC loss.
#         """
#         x = self.get_predictions(x)

        
#         # Ensure x and y are of the same shape
#         assert x.shape == y.shape, "Shape of predicted and target tensors must match"
        
#         # Compute CCC for each column and then the mean CCC over all columns
#         mean_x = torch.mean(x, dim=0)
#         mean_y = torch.mean(y, dim=0)
#         var_x = torch.var(x, dim=0, unbiased=False)
#         var_y = torch.var(y, dim=0, unbiased=False)
#         covariance = torch.mean((x - mean_x) * (y - mean_y), dim=0)
#         ccc = (2 * covariance) / (var_x + var_y + (mean_x - mean_y) ** 2)
#         mean_ccc = torch.mean(ccc)
        
#         # Since CCC indicates agreement, to use it as a loss we subtract from 1
#         return 1 - mean_ccc


