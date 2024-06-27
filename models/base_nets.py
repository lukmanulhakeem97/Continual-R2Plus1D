import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
import math

import abc

#from models.r2plus1d_nets import SpatioTemporalConv, SpatioTemporalResBlock, SpatioTemporalResLayer


class ContinualLearner(nn.Module, metaclass=abc.ABCMeta):
    """Abstract module to add CL capabilities to multitask classifier"""

    def __init__(self):
        super().__init__()
        self.scenario = None
        self.tasks = 0
        self.task_nr = 0
        self.classes_per_task = 0
        self.multihead = None

    @abc.abstractmethod
    def forward(self, x):
        pass


class BaseModel(ContinualLearner, metaclass=abc.ABCMeta):
    """Abstract module to train and test classifier"""

    def __init__(self):
        super().__init__()
        self.optimizer = None
        self.scheduler = None ####add

    @abc.abstractmethod
    def forward(self, x):
        pass

    def train_batch(self, X, y, loss_fn, device, active_classes=None):
        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        self.optimizer.zero_grad()
        pred = self.forward(X)

        if self.scenario == "class":
            pred = pred[:, :active_classes]

        loss = loss_fn(pred, y)
        # Backpropagation
        loss.backward(retain_graph=False)
        self.optimizer.step()
        return loss

    def train_batch_v2(self, X, y, loss_fn, device, active_classes=None):
        """Without backprop and weight update"""

        # Compute prediction and loss
        X, y = X.to(device), y.to(device)
        if self.scenario == "task":
            y = torch.remainder(y, self.classes_per_task)  # changing labels for split dataset task

        pred = self.forward(X)
        if self.scenario == "class":
            pred = pred[:, :active_classes]

        loss = loss_fn(pred, y)
        return loss

    def train_epoch(self, dataloader, loss_fn, device, task_nr, verbose=True):
        size = len(dataloader.dataset)
        self.task_nr = task_nr
        self.set_bn_layer(task_nr)

        if self.scenario == "class":
            # List of classes seen so far
            active_classes = self.classes_per_task * (task_nr + 1)
        else:
            active_classes = None

        self.scheduler.step()  ####add
        self.train()
        total_loss = 0.0
        for batch, (X, y) in enumerate(dataloader):
            if self.scenario == "task":
                # changing labels for spilt dataset task
                y = torch.remainder(y, self.classes_per_task)

            loss = self.train_batch(X, y, loss_fn, device, active_classes)
            total_loss += loss.item()
            if verbose:
                if batch % 100 == 0:
                    loss_value, current = loss.item(), batch * 64
                    print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        return total_loss

    def train_epoch_joint(self, MultipleDLs, loss_fn, device, verbose=True):
        self.train()
        for batch_nr, batch in enumerate(MultipleDLs):
            loss_value = 0.0
            losses = []
            for i, dl in enumerate(batch):
                # Unpack values
                X, y = dl[0], dl[1]
                # -------------TRAIN TASK ------------#
                self.task_nr = i
                # Train on batch
                loss = self.train_batch_v2(X, y, loss_fn, device)
                loss_value += loss.item()
                losses.append(loss)

            # Update parameters using combined loss
            self.optimizer.zero_grad()
            total_loss = 0.0
            for loss in losses:
                total_loss += loss
            total_loss.backward(retain_graph=False)
            self.optimizer.step()

            size = len(MultipleDLs.dataloaders[-1].dataset)
            if verbose:
                if batch_nr % 100 == 0:
                    current = batch_nr * 256
                    print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        return loss_value

    def test_epoch(self, dataloader, loss_fn, device, task_nr, verbose=True):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0.0, 0.0

        self.task_nr = task_nr
        self.set_bn_layer(task_nr)
        # List of classes seen so far
        active_classes = self.classes_per_task * (task_nr + 1)

        self.eval()
        with torch.no_grad():
            for X, y in dataloader:
                if self.scenario == "task":
                    # changing labels for spilt dataset task
                    y = torch.remainder(y, self.classes_per_task)

                X, y = X.to(device), y.to(device)
                pred = self.forward(X)

                if self.scenario == "class":
                    pred = pred[:, :active_classes]

                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        acc = 100 * correct
        if verbose:
            print(f"TASK {task_nr+1:>1d} Test Accuracy: \n Accuracy: {acc:>0.2f}%, Avg loss: {test_loss:>8f} \n")

        return acc, test_loss


class MultiBatchNorm(BaseModel):
    def __init__(self, num_features=None, tasks=4):
        super().__init__()
        self.batchnorms = nn.ModuleList([nn.BatchNorm3d(num_features) for _ in range(tasks)])
        self.tasks = tasks
        self.current_bn_layer = 0

    def set_bn_layer(self, index):
        self.current_bn_layer = index

    def forward(self, x):
        x = self.batchnorms[self.current_bn_layer](x)
        return x


class SpatioTemporalConv(BaseModel):
    r"""Applies a factored 3D convolution over an input signal composed of several input 
    planes with distinct spatial and time axes, by performing a 2D convolution over the 
    spatial axes to an intermediate subspace, followed by a 1D convolution over the time 
    axis to produce the final output.

    Args:
        in_channels (int): Number of channels in the input tensor
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, tasks, stride=1, padding=0, bias=True):
        super(SpatioTemporalConv, self).__init__()

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula 
        # from the paper section 3.5
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/ \
                            (kernel_size[1]* kernel_size[2] * in_channels + kernel_size[0] * out_channels)))

        # the spatial conv is effectively a 2D conv due to the 
        # spatial_kernel_size, followed by batch_norm and ReLU
        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                    stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = MultiBatchNorm(intermed_channels, tasks)
        self.relu = nn.ReLU()

        # the temporal conv is effectively a 1D conv, but has batch norm 
        # and ReLU added inside the model constructor, not here. This is an 
        # intentional design choice, to allow this module to externally act 
        # identical to a standard Conv3D, so it can be reused easily in any 
        # other codebase
        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size, 
                                    stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x

class SpatioTemporalResBlock(BaseModel):
    r"""Single block for the ResNet network. Uses SpatioTemporalConv in 
        the standard ResNet block layout (conv->batchnorm->ReLU->conv->batchnorm->sum->ReLU)
        
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the block.
            kernel_size (int or tuple): Size of the convolving kernels.
            downsample (bool, optional): If ``True``, the output size is to be smaller than the input. Default: ``False``
        """
    def __init__(self, in_channels, out_channels, kernel_size, tasks, downsample=False):
        super(SpatioTemporalResBlock, self).__init__()
        
        # If downsample == True, the first conv of the layer has stride = 2 
        # to halve the residual output size, and the input x is passed 
        # through a seperate 1x1x1 conv with stride = 2 to also halve it.

        # no pooling layers are used inside ResNet
        self.downsample = downsample
        
        # to allow for SAME padding
        padding = kernel_size//2

        if self.downsample:
            # downsample with stride =2 the input x
            self.downsampleconv = SpatioTemporalConv(in_channels, out_channels, 1, tasks, stride=2)
            self.downsamplebn = MultiBatchNorm(out_channels, tasks)

            # downsample with stride = 2when producing the residual
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, tasks, padding=padding, stride=2)
        else:
            self.conv1 = SpatioTemporalConv(in_channels, out_channels, kernel_size, tasks, padding=padding)

        self.bn1 = MultiBatchNorm(out_channels, tasks)
        self.relu1 = nn.ReLU()

        # standard conv->batchnorm->ReLU
        self.conv2 = SpatioTemporalConv(out_channels, out_channels, kernel_size, tasks, padding=padding)
        self.bn2 = MultiBatchNorm(out_channels, tasks)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))    
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        return self.outrelu(x + res)


class SpatioTemporalResLayer(BaseModel):
    r"""Forms a single layer of the ResNet network, with a number of repeating 
    blocks of same output size stacked on top of each other
        
        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels in the output produced by the layer.
            kernel_size (int or tuple): Size of the convolving kernels.
            layer_size (int): Number of blocks to be stacked to form the layer
            block_type (Module, optional): Type of block that is to be used to form the layer. Default: SpatioTemporalResBlock. 
            downsample (bool, optional): If ``True``, the first block in layer will implement downsampling. Default: ``False``
        """

    def __init__(self, in_channels, out_channels, kernel_size, layer_size, tasks, block_type=SpatioTemporalResBlock, downsample=False):
        
        super(SpatioTemporalResLayer, self).__init__()

        # implement the first block
        self.block1 = block_type(in_channels, out_channels, kernel_size, tasks, downsample)

        # prepare module list to hold all (layer_size - 1) blocks
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            # all these blocks are identical, and have downsample = False by default
            self.blocks += [block_type(out_channels, out_channels, kernel_size, tasks)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R2Plus1DNet(BaseModel):
    r"""Forms the overall ResNet feature extractor by initializng 5 layers, with the number of blocks in 
    each layer set by layer_sizes, and by performing a global average pool at the end producing a 
    512-dimensional vector for each element in the batch.
        
        Args:
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default: SpatioTemporalResBlock. 
        """
    def __init__(self, layer_sizes, tasks, block_type=SpatioTemporalResBlock):
        super(R2Plus1DNet, self).__init__()

        # first conv, with stride 1x2x2 and kernel size 3x7x7
        self.conv1 = SpatioTemporalConv(3, 64, [3, 7, 7], tasks, stride=[1, 2, 2], padding=[1, 3, 3])
        # output of conv2 is same size as of conv1, no downsampling needed. kernel_size 3x3x3
        self.conv2 = SpatioTemporalResLayer(64, 64, 3, layer_sizes[0], tasks, block_type=block_type)
        # each of the final three layers doubles num_channels, while performing downsampling 
        # inside the first block
        self.conv3 = SpatioTemporalResLayer(64, 128, 3, layer_sizes[1], tasks, block_type=block_type, downsample=True)
        self.conv4 = SpatioTemporalResLayer(128, 256, 3, layer_sizes[2], tasks, block_type=block_type, downsample=True)
        self.conv5 = SpatioTemporalResLayer(256, 512, 3, layer_sizes[3], tasks, block_type=block_type, downsample=True)

        # global average pooling of the output
        self.pool = nn.AdaptiveAvgPool3d(1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)
        
        return x.view(-1, 512)

class CLR2Plus1DClassifier(BaseModel):
    r"""Forms a complete ResNet classifier producing vectors of size num_classes, by initializng 5 layers, 
    with the number of blocks in each layer set by layer_sizes, and by performing a global average pool
    at the end producing a 512-dimensional vector for each element in the batch, 
    and passing them through a Linear layer.
        
        Args:
            num_classes(int): Number of classes in the data
            layer_sizes (tuple): An iterable containing the number of blocks in each layer
            block_type (Module, optional): Type of block that is to be used to form the layers. Default:    SpatioTemporalResBlock. 
        """
    def __init__(self, layer_sizes, classes, tasks, block_type=SpatioTemporalResBlock, scenario="task",
                 multihead=True):
        super(CLR2Plus1DClassifier, self).__init__()

        self.res2plus1d = R2Plus1DNet(layer_sizes, tasks, block_type)
        #self.linear = nn.Linear(512, num_classes)
        
        # CL element
        if multihead:
            # different head for each tasks
            self.outputs = nn.ModuleList([nn.Linear(512, classes) for _ in range(tasks)])
        else:
            self.output = (
                nn.Linear(512, classes * tasks) if scenario == "class" else nn.Linear(512, classes)
            )

        self.tasks = tasks
        self.scenario = scenario
        self.classes_per_task = classes
        self.multihead = multihead

    def forward(self, x):
        x = self.res2plus1d(x)
        #x = self.linear(x) 
        
        if self.multihead:
            # Only head active for current task
            logits = self.outputs[self.task_nr](x)
        else:
            logits = self.output(x)

        return logits


    def set_bn_layer(self, index):
        for module in self.modules():
            if isinstance(module, MultiBatchNorm):
                module.set_bn_layer(index)



