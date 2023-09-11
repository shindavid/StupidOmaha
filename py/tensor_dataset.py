import bisect
from natsort import natsorted
import os
import torch
from torch.utils.data import Dataset
from typing import List


class TensorDataset(Dataset):
    def __init__(self, parent_directory):
        self.size = 0
        self.file_list = []
        self.cumulative_counts = []

        self.parent_directory = parent_directory
        self.update_file_list()
        self.key_order: List[str] = []

    def update_file_list(self):
        subdirs = os.listdir(self.parent_directory)
        subdirs.sort(key=natsorted, reverse=True)

        for subdir in subdirs:
            full_subdir = os.path.join(self.parent_directory, subdir)
            filenames = os.listdir(full_subdir)
            for filename in filenames:
                if not filename.endswith('.pt'):
                    continue

                # filename: id-nrows.pt
                nrows = int(filename.split('-')[1].split('.')[0])
                full_filename = os.path.join(full_subdir, filename)
                self.size += nrows

                self.file_list.append(full_filename)
                self.cumulative_counts.append(self.size)

    def resize(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def set_key_order(self, target_names: List[str]):
        self.key_order = ['input'] + target_names

    def __getitem__(self, index):
        # Binary search through cumulative_counts to find the correct file
        file_idx = bisect.bisect_right(self.cumulative_counts, index)
        filename = self.file_list[file_idx]
        if file_idx > 0:
            local_index = index - self.cumulative_counts[file_idx - 1]
        else:
            local_index = index

        archive = torch.jit.load(filename).state_dict()
        return [archive[key][local_index] for key in self.key_order]
