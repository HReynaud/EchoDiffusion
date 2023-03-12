import os
import typing

import cv2 
import numpy as np
import torch
import tqdm
import skimage.draw

# Code taken from https://github.com/echonet/dynamic/blob/master/echonet/utils/__init__.py

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path) # type: ignore
    fps = cap.get(cv2.CAP_PROP_FPS) # type: ignore
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # type: ignore
    duration = frame_count / fps
    return duration


def loadvideo(filename: str, get_fps=False):
    """Loads a video from a file.
    Args:
        filename (str): filename of video
    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.
    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)  # type: ignore

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) # type: ignore
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) # type: ignore
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # type: ignore
    fps = capture.get(cv2.CAP_PROP_FPS) # type: ignore

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # type: ignore
        v[count, :, :] = frame

    v = v.transpose((3, 0, 1, 2))
    if get_fps:
        return v, fps
    else:
        return v


def savevideo(filename: str, array: np.ndarray, fps: typing.Union[float, int] = 1):
    """Saves a video to a file.
    Args:
        filename (str): filename of video
        array (np.ndarray): video of uint8's with shape (channels=3, frames, height, width)
        fps (float or int): frames per second
    Returns:
        None
    """

    c, _, height, width = array.shape

    if c != 3:
        raise ValueError("savevideo expects array of shape (channels=3, frames, height, width), got shape ({})".format(", ".join(map(str, array.shape))))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') # type: ignore
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height)) # type: ignore

    for frame in array.transpose((1, 2, 3, 0)):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # type: ignore
        out.write(frame)


def get_mean_and_std(dataset: torch.utils.data.Dataset, # type: ignore
                     samples: int = 128,
                     batch_size: int = 8,
                     num_workers: int = 4):
    """Computes mean and std from samples from a Pytorch dataset.
    Args:
        dataset (torch.utils.data.Dataset): A Pytorch dataset.
            ``dataset[i][0]'' is expected to be the i-th video in the dataset, which
            should be a ``torch.Tensor'' of dimensions (channels=3, frames, height, width)
        samples (int or None, optional): Number of samples to take from dataset. If ``None'', mean and
            standard deviation are computed over all elements.
            Defaults to 128.
        batch_size (int, optional): how many samples per batch to load
            Defaults to 8.
        num_workers (int, optional): how many subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
    Returns:
       A tuple of the mean and standard deviation. Both are represented as np.array's of dimension (channels,).
    """

    if samples is not None and len(dataset) > samples:  # type: ignore
        indices = np.random.choice(len(dataset), samples, replace=False) # type: ignore
        dataset = torch.utils.data.Subset(dataset, indices) # type: ignore
    dataloader = torch.utils.data.DataLoader( # type: ignore
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    n = 0  # number of elements taken (should be equal to samples by end of for loop)
    s1 = 0.  # sum of elements along channels (ends up as np.array of dimension (channels,))
    s2 = 0.  # sum of squares of elements along channels (ends up as np.array of dimension (channels,))
    for (x, *_) in tqdm.tqdm(dataloader):
        x = x.transpose(0, 1).contiguous().view(3, -1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x ** 2, dim=1).numpy()
    mean = s1 / n  # type: ignore # type: np.ndarray # type: ignore
    std = np.sqrt(s2 / n - mean ** 2)  # type: np.ndarray

    mean = mean.astype(np.float32) # type: ignore
    std = std.astype(np.float32)

    return mean, std

