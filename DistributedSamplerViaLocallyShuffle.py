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
        self.batch_ids = list(range(shuffle_buffer))
        self.batch_position = 0
        self.file_ids = list(range(len(self.dataset.files)))
        self.buffers = 0
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

    def init_iter(self):
        self.dataset.reset()
        self.batch_ids = list(range(min(self.shuffle_buffer, self.num_samples)))
        self.file_ids = list(range(len(self.dataset.files)))
        self.file_load_num = 0
        if len(self.files_data) != 0:
            for garabage_data in self.files_data:
                del garabage_data
        del self.files_data
        gc.collect()
        self.files_data = [None] * len(self.dataset.files)
        self.batch_position = 0
        self.buffers = 0
        if self.shuffle:
            random.seed(self.epoch)
            random.shuffle(self.batch_ids)
            random.seed(self.epoch + 1)
            random.shuffle(self.file_ids)
            random.seed(self.epoch + 2)
            random.shuffle(self.blocks)
            random.seed(self.epoch + 3)
            self.start_num = self.num_samples * self.blocks[self.rank]
            tmp_files = []
            for file_id in self.file_ids:
                tmp_files.append(self.files[file_id])
            self.files = tmp_files
        self.past_files_samples = [0]
        self.count_batches = 0
        self.last_file_position = 0
        self.threadpool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="file_reader_")
        self.thread_file_name = None
        self.thread_read_process = None
        self.last_file = None

    def find_ckpt_position(self, step):
        self.warm_start = True
        self.init_iter()
        already_sampled_num = step * self.batch_size
        self.buffers = already_sampled_num // self.shuffle_buffer
        self.batch_ids = list(range(min(self.shuffle_buffer, self.num_samples - self.buffers * self.shuffle_buffer)))
        self.batch_position = already_sampled_num - self.buffers * self.shuffle_buffer

    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.warm_start:
            self.init_iter()
        else:
            print(str(self.rank) + ': warm start!! ' + str(self.epoch))
            self.warm_start = False
        return self

    def __next__(self):
        if len(self.batch_ids) == 0:
            raise StopIteration
        if self.debug:
            print(str(self.rank) + ': ' + 'generate indices for batch ' + str(self.count_batches) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
        indices = []
        for i in range(self.batch_size):
            if len(self.batch_ids) == 0:
                break
            index = self.batch_ids[self.batch_position] + self.shuffle_buffer * self.buffers + self.start_num
            if index >= self.ori_total_size:
                index = index - self.ori_total_size
            self.batch_position += 1
            if self.batch_position >= len(self.batch_ids):
                self.buffers += 1
                self.batch_ids = list(range(min(self.shuffle_buffer, self.num_samples - self.buffers * self.shuffle_buffer)))
                if self.shuffle:
                    random.seed(self.epoch + self.buffers * 10000)
                    random.shuffle(self.batch_ids)
                self.batch_position = 0
            indices.append(index)

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
            print(str(self.rank) + ': ' + 'the number of the whole dataset might be larger than the real number')
        if tmp_count == 1:
            raise StopIteration

        if self.debug:
            print(str(self.rank) + ': ' + 'process indices successfully for batch ' + str(self.count_batches) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

        read_files_data = []
        for file_path in read_files:
            file_data = self.get_file_data(file_path)
            read_files_data.append(file_data)

        # if self.last_file != read_files:
        #     print(str(self.rank) + ': ' + str(read_files) + ' ' + str(batch_ids))
        #     self.last_file = read_files
        # else:
        #     print(str(self.rank) + ': ' + str(batch_ids))

        target_datas = []
        for d in range(len(read_files_data)):
            t_data = dict()
            for k in read_files_data[d].keys():
                t_data[k] = read_files_data[d][k][batch_ids[d]]
            target_datas.append(t_data)
        self.count_batches += 1
        if self.debug:
            print(str(self.rank) + ': ' + 'get data successfully for batch ' + str(self.count_batches) +
                      ' memory used:' + str(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
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


if __name__ == '__main__':
    import numpy as np

    class datapipetmp():
        def __init__(self, fileinfo, batch_offsets):
            self.past_files_samples = []
            self.files = fileinfo[0]
            self.batch_offsets = batch_offsets
            self.number = 0
        def reset(self):
            pass

    train_file_name = []
    train_file_name_simply = []
    for name in os.listdir('../labelnpzdata'):
        print(name)
        train_file_name.append(os.path.join('../labelnpzdata', name))

    print('process train file (npzfilelist)')
    filelist = train_file_name
    train_data = []
    train_labels = []
    j = 0
    for i in range(len(filelist)):
        f = filelist[i]
        if not f.endswith('npz'):
            continue
        train_data.append(f)
        train_labels = None

    train_data = [train_data, 1024]
    train_labels = train_labels

    def parse_files_len(base_path, file_name):
        dict_path = os.path.join(base_path, file_name)
        cnt = 0
        ret = dict()
        with open(dict_path, 'r', encoding = 'utf-8', errors='ignore') as fp:
            print('parsing file {f}'.format(f = dict_path))
            while True:
                line = fp.readline().strip('\n\r')
                if line == '':
                    break
                line = line.split('\t')
                data_name = line[0]
                l = line[1]
                ret[os.path.join(base_path, data_name)] = int(l)
                cnt += 1
        print(str(cnt) + ' files in ' + dict_path)
        return ret

    files_len = parse_files_len('../labelnpzdata', 'files_len_dict')
    train_data[0] = list(files_len.keys())

    print(train_data)

    def file_reader_test(type):
        if 'npz' in type:
            def npz_reader(filename, get_data=False):
                ori_data = np.load(filename)
                data = dict()
                l = 0
                for key in ori_data.keys():
                    data[key] = ori_data[key]
                    if l == 0:
                        l = len(data[key])
                    if not get_data:
                        return l
                # l = files_len[filename]
                # if not get_data:
                #     return l
                return data, l
            return npz_reader
        else:
            raise RuntimeError("only support npz!!")

    for j in range(4):
        step = 0
        train_sampler = DistributedSamplerViaLocallyShuffle(datapipetmp(train_data, train_labels),
                                                            file_reader_test('npz'),
                                                            num_replicas=4,
                                                            rank=j,
                                                            shuffle_buffer=4096,
                                                            total_size=2000000000,
                                                            batch_size=1024,
                                                            file_buffer=2,
                                                            debug=True,
                                                            files_len=files_len)
        train_sampler.find_ckpt_position(400799)
        loader = iter(train_sampler)
        count = 0
        savedata = {'arr_0': [], 'arr_1': [], 'arr_2': [], 'arr_3': [], 'arr_4': [], 'arr_5': []}
        while True:
            try:
                tmp = next(loader)
                count = 0
                for i in tmp[0]:
                    count += (i['arr_0'].shape)[0]
                if count != 1024:
                    print(count)
                    print(step)
                    print(i.shape)
                for k in savedata.keys():
                    savedata[k].append(tmp[0][0][k])
                del tmp
                # gc.collect()
                step += 1
                if step % 250 == 0:
                    break
            except StopIteration:
                print(step)
                break
        out_data = []
        for k in savedata.keys():
            out_data.append(np.concatenate(savedata[k], axis=0))
        np.savez_compressed('../labelnpzdata/a' + str(j), *out_data)
        tmp = np.sum(out_data[2], axis=1)
        for i in range(len(tmp)):
            if tmp[i] == 0:
                print(i)
        tmp = np.sum(out_data[4], axis=1)
        for i in range(len(tmp)):
            if tmp[i] == 0:
                print(i)