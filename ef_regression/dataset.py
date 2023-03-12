import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom

from utils import loadvideo

class CachedVideoLoader():
    def __init__(self, path, deactivate=False):
        self.path = path
        self.video = {}
        self.deactivate = deactivate

    def __call__(self, fname):
        if self.deactivate:
            return loadvideo(os.path.join(self.path, fname)).astype(np.uint8) # type: ignore
        if not fname in self.video:
            self.video[fname] = loadvideo(os.path.join(self.path, fname)).astype(np.uint8) # type: ignore
        return self.video[fname]


class EchoVideoEF(Dataset):
    def __init__(self, config, splits=["TRAIN", "VAL", "TEST"], fix_samples=False, limit=-1):

        self.config = config
        self.splits = splits if isinstance(splits, list) else [splits]
        self.data_path = config.dataset.data_path
        self.sample_duration = config.dataset.get("seconds", 1)
        self.target_fps = config.dataset.get("target_fps", 12)
        self.videos_length = self.target_fps * self.sample_duration
        self.fix_samples = fix_samples
        self.limit = limit

        self.filelist = pd.read_csv(os.path.join(config.dataset.data_path, "FileList.csv"))

        new_fl = self.filelist[self.filelist['Split'] == ""]
        for split in self.splits:
            new_fl = pd.concat((new_fl, self.filelist[self.filelist['Split'] == split.upper()])) # type: ignore
        self.filelist = new_fl

        self.video_folder_path = os.path.join(config.dataset.data_path, "Videos")

        # filter out videos that are not in the video folder
        self.fnames = [f+".avi" if not f.endswith(".avi") else f for f in self.filelist["FileName"].tolist()]
        self.fnames = [f for f in self.fnames if os.path.exists(os.path.join(self.video_folder_path, f))]

        if self.limit > 0 and self.limit < len(self.fnames):
            self.fnames = np.random.choice(self.fnames, self.limit, replace=False)

        fps_dict = {f: v for f, v in zip(self.filelist["FileName"].tolist(), self.filelist["FPS"].tolist())}
        self.fps = [fps_dict[name[:-4]] for name in self.fnames]
        lvef_dict = {f: float(v) for f, v in zip(self.filelist["FileName"].tolist(), self.filelist["EF"].tolist())}
        self.lvef = [lvef_dict[name[:-4]] for name in self.fnames]

        if "VAL" in splits:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.Tensor(x/255.0)),
                transforms.Resize((config.dataset.image_size,)*2),
            ])
        elif "TRAIN" in splits:
            self.transform = transforms.Compose([
                transforms.Lambda(lambda x: torch.Tensor(x/255.0)),
                transforms.Pad(12),
                transforms.RandomCrop(112),
                transforms.Resize((int(config.dataset.image_size))),
            ])
        
        self.lazy_vloader = CachedVideoLoader(self.video_folder_path, deactivate=config.dataset.deactivate_cache)

        print(f"Loaded {len(self.fnames)} videos for {self.splits} split")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i, frame_idx=None):

        fname = self.fnames[i]
        video = self.lazy_vloader(fname)
        total_frames = video.shape[1]

        if frame_idx is None and self.fix_samples:
            frame_idx = 0

        if self.config.dataset.get("resample_to_num_frames_fps", False) == True:
            fps = self.fps[i]
            video = zoom(video, (1, self.target_fps/fps, 1, 1), order=1)

            if video.shape[1] < self.videos_length: # if the video is too short, pad it with zeros
                video = np.concatenate([video, np.zeros((3, self.videos_length-video.shape[1], 112, 112))], axis=1)
        
        if video.shape[1] == self.videos_length:
            frame_idx = 0
        else:
            frame_idx = np.random.randint(0, video.shape[1]-self.videos_length) if frame_idx==None else frame_idx 

        end_idx = frame_idx + self.videos_length
        frames = video[:, frame_idx:end_idx, :, :] # (3, T, 112, 112) -> (3, videos_length, 112, 112)
        
        frames = self.transform(frames) # Rescale and resize 

        lvef = self.lvef[i]

        return frames, lvef


class EchoVideoEFextended(Dataset):
    def __init__(self, config, splits=["TRAIN", "VAL", "TEST"], balanced_bins=-1, minef=0, maxef=100):
        super().__init__()
        self.config = config

        if "TRAIN" in splits:
            video_ex_path = config.video_ex_path
            csv_ex_path = config.csv_ex_path
            self.label_key = config.label_key

            data_ex = pd.read_csv(csv_ex_path)

            self.fnames = data_ex['ID'].tolist()
            self.lvef = data_ex[self.label_key].tolist()
        
        self.base_dataset = EchoVideoEF(config, splits=splits, fix_samples=True)

        # Make bins
        # each entry is (og/ex dataset, lvef, index)
        all_entries = {}
        if "TRAIN" in splits:
            for i, e in enumerate(self.lvef):
                bbin = int(float(e))
                if e < minef or e > maxef:
                    continue
                if bbin not in all_entries:
                    all_entries[bbin] = []
                all_entries[bbin].append(('ex', e, i))
        
        for i, e in enumerate(self.base_dataset.lvef):
            bbin = int(float(e))
            if e < minef or e > maxef:
                continue
            if bbin not in all_entries:
                all_entries[bbin] = []
            all_entries[bbin].append(('og', e, i))
        
        # Make balanced bins
        self.bbins = []
        for bbin in all_entries:
            if bbin > maxef or bbin < minef:
                continue
            entries = all_entries[bbin]
            if len(entries) > balanced_bins and balanced_bins > 0:
                indices = np.random.choice(np.arange(len(entries)), balanced_bins)
                entries = [entries[i] for i in indices]
            self.bbins.append(entries)
        
        self.index_to_bin_elm = []
        for i, bbin in enumerate(self.bbins):
            for j, e in enumerate(bbin):
                self.index_to_bin_elm.append((i, j))

        if "TRAIN" in splits:
            self.lazy_vloader = CachedVideoLoader(video_ex_path, deactivate=config.dataset.deactivate_cache) # type: ignore

        print(f"Loaded {len(self.index_to_bin_elm)} elements in {len(self.bbins)} balanced bins")
    
    def __len__(self):
        return len(self.index_to_bin_elm)
    
    def __getitem__(self, i):
        bin_index, elm_index = self.index_to_bin_elm[i]
        bbin = self.bbins[bin_index][elm_index]

        (pool, gt_ef, index) = bbin

        if pool == 'ex':
            fname = self.fnames[index]
            video = self.lazy_vloader(fname)
            video = self.base_dataset.transform(video)
            lvef = self.lvef[index]
        else:
            video, lvef = self.base_dataset.__getitem__(index) # frame_idx=0 to make sure we get the same frame every time
        assert lvef == gt_ef
        return video, lvef

    def get_distribution(self):
        self.distribution = {}
        for bbins in self.bbins:
            for bbin in bbins:
                (pool, gt_ef, index) = bbin

                if int(gt_ef) not in self.distribution:
                    self.distribution[int(gt_ef)] = {}

                keyy = 'fake' if pool == 'ex' else 'real'

                if keyy not in self.distribution[int(gt_ef)]:
                    self.distribution[int(gt_ef)][keyy] = 0

                self.distribution[int(gt_ef)][keyy] += 1
        return self.distribution


