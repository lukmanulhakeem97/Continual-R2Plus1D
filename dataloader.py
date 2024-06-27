import os
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list 
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            directory (str): The path to the directory containing the train/val/test datasets
            mode (str, optional): Determines which folder of the directory the dataset will read from. Defaults to 'train'. 
            clip_len (int, optional): Determines how many frames are there in each clip. Defaults to 8. 
        """

    def __init__(self, directory, mode='train', clip_len=32):
        folder = Path(directory)/mode  # get the directory of the specified split

        self.clip_len = clip_len

        # the following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 112  
        self.resize_width = 112

        # obtain all the filenames of files inside all the class folders 
        # going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)     

        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label:index for index, label in enumerate(sorted(set(labels)))} 
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)        

    def __getitem__(self, index):
        # loading and preprocessing. TODO move them to transform classes
        buffer = self.loadvideo(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len)
        buffer = self.normalize(buffer)
        return buffer, self.label_array[index] #, self.fnames[index]

    def loadvideo(self, fname):
        # initialize a VideoCapture object to read video data into a numpy array
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        count = 0
        retaining = True

        # read in each frame, one at a time into the numpy buffer array
        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if (frame is not None): ####add
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # will resize frames if not already final size
                # NOTE: strongly recommended to resize them during the download process. This script
                # will process videos of any size, but will take longer the larger the video file.
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                buffer[count] = frame
                count += 1

        # release the VideoCapture once it is no longer needed
        capture.release()

        # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
        # D = Depth (in this case, time), H = Height, W = Width, C = Channels
        buffer = buffer.transpose((3, 0, 1, 2))

        return buffer 
    
    def crop(self, buffer, clip_len):
        # randomly select time index for temporal jittering
        time_index_ = (buffer.shape[1]/2) - (clip_len/2) + 1
        time_index = int(time_index_)
        # crop and jitter the video using indexing. The spatial crop is performed on 
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        #buffer_og = buffer.copy()
        buffer = buffer[:, time_index:time_index + clip_len, :, :]

        #if buffer.shape[1]!=clip_len:
        #    time_index = np.random.randint(buffer_og.shape[1] - clip_len)
        #    buffer = buffer_og[:, time_index:time_index + clip_len, :, :]

        return buffer 

    def normalize(self, buffer):
        # Normalize the buffer
        # NOTE: Default values of RGB images normalization are used, as precomputed 
        # mean and std_dev values (akin to ImageNet) were unavailable for Kinetics. Feel 
        # free to push to and edit this section to replace them if found. 
        buffer = (buffer - 128)/128
        return buffer

    def __len__(self):
        return len(self.fnames)


def get_multitask_data(tasks=9, data_dir="./datasets"):
    total_tasks = tasks
    train_data_list = []
    val_data_list = []
    for task_id in range(1, total_tasks+1):
        directory = f"{data_dir}/task{task_id}"
        train_data = VideoDataset(directory, clip_len=32)
        val_data = VideoDataset(directory, mode='val', clip_len=32)
        train_data_list.append(train_data)
        val_data_list.append(val_data)

    return train_data_list, val_data_list

def get_singletask_data(task_id=1, data_dir="./datasets"):
    
    directory = f"{data_dir}/task{task_id}"
    train_data = VideoDataset(directory, clip_len=32)
    val_data = VideoDataset(directory, mode='val', clip_len=32)

    return train_data, val_data

