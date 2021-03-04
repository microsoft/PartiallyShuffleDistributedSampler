# PartiallyShuffle

This repo includes 2 versions of distribute sampler with Partially Shuffle for Pytorch. They are designed to replace DistributedSampler.py in Pytorch.

DistributedSamplerViaLocallyShuffle.py is a simpler verison which only have one shuffle pool.

DistributedSamplerViaLocallyShuffleV2.py has two shuffle pools and similier process like the Partially Shuffle in TensorFlow.

# Usage

You can use it just like DistributedSampler in Pytorch with some extra hyper-parameters.
The file reader should be designed like this:

>    def file_reader(self, type):
>        def npz_reader(filename, get_data=False):
>            ori_data = np.load(filename)
>            data = dict()
>            l = 0
>            for key in ori_data.keys():
>                data[key] = ori_data[key]
>                if l == 0:
>                    l = len(data[key])
>                if not get_data:
>                    return l
>            return data, l
>        return npz_reader

For more information, you can connect jianjzh@microsoft.com.
