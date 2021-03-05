# PartiallyShuffle

This repo includes 2 versions of distribute sampler with Partially Shuffle for Pytorch. They are designed to replace DistributedSampler.py in Pytorch.

These samplers will try to distribute files evenly to available GPUs, and use cache and threads to reduce the number of file reads, which is friendly to distributed storage system, such as HDFS.

DistributedSamplerViaLocallyShuffle.py is a simpler verison which only have one shuffle pool.

DistributedSamplerViaLocallyShuffleV2.py has two shuffle pools and similier process like the Partially Shuffle in TensorFlow.

## Usage

You can use it just like DistributedSampler in Pytorch with some extra hyper-parameters.

The hyper-parameter reader is used to read the file before the dataloader to prevent reading a file too many times. Previous design in Pytorch will try to read the file every time when it generate a batch.

The file reader should be designed like this:

```
    def file_reader():
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
            return data, l
        return npz_reader
```

The files_len is a dict with <filename, samples> pairs. It will accelerate the initailization of sampler.

This work is done in Microsoft Ads team. For more information, you can connect jianjzh@microsoft.com.
