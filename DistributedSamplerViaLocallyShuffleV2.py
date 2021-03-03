#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/21/2020 1:06 PM
# @Author  : Jianjin Zhang
# @File    : DistributedSamplerVialocallyShuffle.py

import math
from torch.utils.data import Sampler
import torch.distributed as dist
import random
from concurrent.futures import ThreadPoolExecutor
import psutil, os, gc


class DistributedSamplerViaLocallyShuffle(Sampler):
    def __init__(self, dataset, reader, num_replicas=None, rank=None, shuffle=True,
                 shuffle_buffer=None, total_size=None, batch_size=1, file_buffer=10, debug=False, files_len=None):
        super().__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.files_len = dict()
        self.ori_total_size = total_size
        if files_len is not None:
            self.files_len = files_len
            self.ori_total_size = int(sum([self.files_len[k] for k in self.files_len.keys()]))
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle_buffer = shuffle_buffer
        self.batch_size = batch_size
        self.reader = reader
        self.file_buffer = file_buffer
        self.debug = debug
        assert total_size is not None, 'total size must be provided'
        self.num_samples = int(math.ceil(self.ori_total_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.files_data = []
        self.files = self.dataset.files
        self.past_files_samples = []
        self.blocks = list(range(self.num_replicas))
        self.start_num = self.num_samples * self.blocks[self.rank]
        self.count_batches = 0
        self.threadpool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="file_reader_")
        self.warm_start = False

    def get_file_data(self, file_path):
        if self.files_data[self.files.index(file_path)] is None:
            if self.debug:
                print(str(self.rank) + ': ' + 'load data from ' + str((self.count_batches, file_path)) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
            if self.thread_file_name != file_path:
                file_data, _ = self.reader(file_path, get_data=True)
            else:
                file_data, _ = self.thread_read_process.result()
            if self.files.index(file_path) + 1 < len(self.files_data) and \
                    self.files_data[self.files.index(file_path) + 1] is None:
                self.thread_file_name = self.files[self.files.index(file_path) + 1]
                self.thread_read_process = self.threadpool.submit(self.reader, self.thread_file_name, True)
            self.files_data[self.files.index(file_path)] = file_data
            if self.debug:
                print(str(self.rank) + ': ' + 'load data successfully from ' + str((self.count_batches, file_path)) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
            self.file_load_num += 1
            if self.file_load_num > self.file_buffer:
                for i in range(len(self.files_data)):
                    if self.files_data[i] is not None:
                        if (self.ori_total_size - self.start_num) < self.num_samples and \
                                i < (len(self.files) // self.num_replicas):
                            continue
                        tmp = self.files_data[i]
                        self.files_data[i] = None
                        del tmp
                        gc.collect()
                        self.file_load_num -= 1
                        if self.file_load_num <= self.file_buffer:
                            break
        else:
            if self.debug:
                print(str(self.rank) + ': ' + 'use cache data from ' + str((self.count_batches, file_path)) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
            file_data = self.files_data[self.files.index(file_path)]
            if self.debug:
                print(str(self.rank) + ': ' + 'use cache data successfully from ' +
                      str((self.count_batches, file_path)) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        return file_data

    def get_index(self):
        indices = []
        for i in range(self.batch_size):
            if len(self.batch_ids) == 0 and len(self.batch_ids2) == 0:
                break
            index = random.choice(self.batch_ids)
            self.batch_ids.remove(index)
            if len(self.batch_ids2) != 0:
                index2 = random.choice(self.batch_ids2)
                self.batch_ids2.remove(index2)
                self.batch_ids.append(index2)
            if len(self.batch_ids2) == 0:
                random.seed(self.epoch + self.buffers * 10000)
                self.buffers += 1
                self.batch_ids2 = list(range(self.start_num + (self.buffers + 1) * self.shuffle_buffer,
                                             min(self.start_num + (self.buffers + 2) * self.shuffle_buffer,
                                                 self.start_num + self.num_samples)))
            if index >= self.ori_total_size:
                index = index - self.ori_total_size
            indices.append(index)
        return indices

    def find_ckpt_position(self, step):
        self.warm_start = True
        self.init_iter()
        for i in range(step):
            _ = self.get_index()

    def init_iter(self):
        self.dataset.reset()
        self.buffers = 0
        self.file_ids = list(range(len(self.dataset.files)))
        self.file_load_num = 0
        if len(self.files_data) != 0:
            for garabage_data in self.files_data:
                del garabage_data
        del self.files_data
        gc.collect()
        self.files_data = [None] * len(self.dataset.files)
        self.batch_ids = list(range(self.start_num,
                                    min(self.start_num + self.shuffle_buffer, self.start_num + self.num_samples)))
        self.batch_ids2 = list(range(self.start_num + self.shuffle_buffer,
                                     min(self.start_num + 2 * self.shuffle_buffer, self.start_num + self.num_samples)))
        if self.debug:
            print(str(self.rank) + ': ' + 'start number ' + str(self.start_num) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        self.blocks = list(range(self.num_replicas))
        random.seed(self.epoch)
        random.shuffle(self.file_ids)
        random.seed(self.epoch + 1)
        random.shuffle(self.blocks)
        random.seed(self.epoch + 2)
        self.start_num = self.num_samples * self.blocks[self.rank]
        tmp_files = []
        for file_id in self.file_ids:
            tmp_files.append(self.files[file_id])
        self.files = tmp_files
        self.past_files_samples = [0]
        self.count_batches = 0
        self.last_file_position = 0
        self.threadpool.shutdown()
        self.threadpool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="file_reader_")
        self.thread_file_name = None
        self.thread_read_process = None

    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.warm_start:
            self.init_iter()
        else:
            print(str(self.rank) + ': warm start!!')
            self.warm_start = False
        return self

    def __next__(self):
        if len(self.batch_ids) == 0 and len(self.batch_ids2) == 0:
            raise StopIteration
        if self.debug:
            print(str(self.rank) + ': ' + 'generate indices for batch ' + str(self.count_batches) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        indices = self.get_index()
        if self.debug:
            print(str(self.rank) + ': ' + 'generate indices successfully for batch ' + str(self.count_batches) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

        batch_ids = []
        read_files = []
        tmp_count = 0
        for batch_id in indices:
            if batch_id >= self.past_files_samples[-1]:
                while batch_id >= self.past_files_samples[-1]:
                    if len(self.past_files_samples) > len(self.files):
                        break
                    if self.files[len(self.past_files_samples) - 1] in self.files_len.keys():
                        l = self.files_len[self.files[len(self.past_files_samples) - 1]]
                    else:
                        l = self.reader(self.files[len(self.past_files_samples) - 1], get_data=False)
                    self.past_files_samples.append(l + self.past_files_samples[-1])
                if batch_id >= self.past_files_samples[-1]:
                    batch_id = self.past_files_samples[-1] * 2 - batch_id
                    if batch_id == self.past_files_samples[-1]:
                        batch_id = self.past_files_samples[-1] - 1
                    indices.append(batch_id)
                    continue
                file_path = self.files[len(self.past_files_samples) - 2]
                batch_id = batch_id - self.past_files_samples[-2]
                self.last_file_position = len(self.past_files_samples) - 2
            else:
                if batch_id >= self.past_files_samples[self.last_file_position]:
                    while batch_id >= self.past_files_samples[self.last_file_position]:
                        if batch_id < self.past_files_samples[self.last_file_position + 1]:
                            file_path = self.files[self.last_file_position]
                            batch_id = batch_id - self.past_files_samples[self.last_file_position]
                            break
                        self.last_file_position += 1
                else:
                    while batch_id < self.past_files_samples[self.last_file_position]:
                        if batch_id >= self.past_files_samples[self.last_file_position - 1]:
                            file_path = self.files[self.last_file_position - 1]
                            batch_id = batch_id - self.past_files_samples[self.last_file_position - 1]
                            break
                        self.last_file_position -= 1
            tmp_count += 1
            if file_path in read_files:
                batch_ids[read_files.index(file_path)].append(batch_id)
            else:
                read_files.append(file_path)
                batch_ids.append([])
                batch_ids[-1].append(batch_id)

        if len(indices) > self.batch_size:
            print(str(self.rank) + ': ' + 'the number of the whole dataset might be larger than the real number' +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        if tmp_count == 1:
            raise StopIteration

        if self.debug:
            print(str(self.rank) + ': ' + 'process indices successfully for batch ' + str(self.count_batches) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

        read_files_data = []
        for file_path in read_files:
            file_data = self.get_file_data(file_path)
            read_files_data.append(file_data)

        target_datas = []
        for d in range(len(read_files_data)):
            t_data = dict()
            for k in read_files_data[d].keys():
                t_data[k] = read_files_data[d][k][batch_ids[d]]
            target_datas.append(t_data)
        self.count_batches += 1
        del batch_ids
        for item in read_files_data:
            del item
        del read_files_data
        del indices
        gc.collect()
        return [target_datas, None, read_files]

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch