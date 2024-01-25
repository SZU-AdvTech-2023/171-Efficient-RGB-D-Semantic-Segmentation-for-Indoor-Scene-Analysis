# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import copy
import os
import pickle
from torch.utils.data import DataLoader

from src import preprocessing
from src.datasets import Cityscapes
from src.datasets import NYUv2
from src.datasets import SceneNetRGBD
from src.datasets import SUNRGBD


def prepare_data(args, ckpt_dir=None, with_input_orig=False, split=None):
    # 存储训练数据预处理的参数
    train_preprocessor_kwargs = {}

    # 根据命令行参数args.dataset的值，选择相应的数据集类型，并设置数据集参数和验证集类型。如果args.dataset的值不在预定义的选项中，则抛出ValueError异常
    if args.dataset == 'sunrgbd':
        Dataset = SUNRGBD
        dataset_kwargs = {}
        valid_set = 'test'
    elif args.dataset == 'nyuv2':
        Dataset = NYUv2
        dataset_kwargs = {'n_classes': 40}
        valid_set = 'test'
    elif args.dataset == 'cityscapes':
        Dataset = Cityscapes
        dataset_kwargs = {
            'n_classes': 19,
            'disparity_instead_of_depth': True
        }
        valid_set = 'valid'
    elif args.dataset == 'cityscapes-with-depth':
        Dataset = Cityscapes
        dataset_kwargs = {
            'n_classes': 19,
            'disparity_instead_of_depth': False
        }
        valid_set = 'valid'
    elif args.dataset == 'scenenetrgbd':
        Dataset = SceneNetRGBD
        dataset_kwargs = {'n_classes': 13}
        valid_set = 'valid'
        if args.width == 640 and args.height == 480:
            # for SceneNetRGBD, we additionally scale up the images by factor
            # of 2
            train_preprocessor_kwargs['train_random_rescale'] = (1.0*2, 1.4*2)
    else:
        raise ValueError(f"Unknown dataset: `{args.dataset}`")
    # 如果args.aug_scale_min不等于1或args.aug_scale_max不等于1.4，
    # 则将train_preprocessor_kwargs中的键train_random_rescale设置为一个元组，
    # 元组中包含args.aug_scale_min和args.aug_scale_max的值。
    if args.aug_scale_min != 1 or args.aug_scale_max != 1.4:
        train_preprocessor_kwargs['train_random_rescale'] = (
            args.aug_scale_min, args.aug_scale_max)

    if split in ['valid', 'test']:
        valid_set = split

    if args.raw_depth:
        # We can not expect the model to predict depth values that are just
        # interpolated and not really there. It is better to let the model only
        # predict the measured depth values and ignore the rest.
        depth_mode = 'raw'
    else:
        depth_mode = 'refined'

    # 创建Dataset对象train_data，用于训练数据集。根据传入的参数和数据集类型，设置数据目录、数据集划分、深度模式和其他参数。
    # train data
    train_data = Dataset(
        data_dir=args.dataset_dir,
        split='train',
        depth_mode=depth_mode,
        with_input_orig=with_input_orig,
        **dataset_kwargs
    )

    # 创建训练数据的预处理器train_preprocessor，使用preprocessing.get_preprocessor()函数根据传入的参数和训练数据集的统计信息进行设置。
    train_preprocessor = preprocessing.get_preprocessor(
        height=args.height,
        width=args.width,
        depth_mean=train_data.depth_mean,
        depth_std=train_data.depth_std,
        depth_mode=depth_mode,
        phase='train',
        **train_preprocessor_kwargs
    )

    # 将训练数据集的预处理器设置为train_preprocessor
    train_data.preprocessor = train_preprocessor

    # 如果ckpt_dir不为None，则根据ckpt_dir和文件名生成pickle文件的路径。
    # 如果文件存在，则加载pickle文件中的深度统计信息。否则，将深度统计信息存储在字典depth_stats中，并将其保存到pickle文件中。
    if ckpt_dir is not None:
        pickle_file_path = os.path.join(ckpt_dir, 'depth_mean_std.pickle')
        if os.path.exists(pickle_file_path):
            with open(pickle_file_path, 'rb') as f:
                depth_stats = pickle.load(f)
            print(f'Loaded depth mean and std from {pickle_file_path}')
            print(depth_stats)
        else:
            # dump depth stats
            depth_stats = {'mean': train_data.depth_mean,
                           'std': train_data.depth_std}
            with open(pickle_file_path, 'wb') as f:
                pickle.dump(depth_stats, f)
    else:
        depth_stats = {'mean': train_data.depth_mean,
                       'std': train_data.depth_std}

    # 创建验证数据的预处理器valid_preprocessor，使用preprocessing.get_preprocessor()函数根据深度统计信息和其他参数进行设置。
    # valid data
    valid_preprocessor = preprocessing.get_preprocessor(
        height=args.height,
        width=args.width,
        depth_mean=depth_stats['mean'],
        depth_std=depth_stats['std'],
        depth_mode=depth_mode,
        phase='test'
    )

    # 如果命令行参数args.valid_full_res为True，则创建全分辨率验证数据的预处理器valid_preprocessor_full_res，使用深度统计信息和其他参数进行设置。
    if args.valid_full_res:
        valid_preprocessor_full_res = preprocessing.get_preprocessor(
            depth_mean=depth_stats['mean'],
            depth_std=depth_stats['std'],
            depth_mode=depth_mode,
            phase='test'
        )

    # 创建验证数据集对象valid_data，用于验证数据集。根据传入的参数和数据集类型，设置数据目录、数据集划分、深度模式和其他参数。
    valid_data = Dataset(
        data_dir=args.dataset_dir,
        split=valid_set,
        depth_mode=depth_mode,
        with_input_orig=with_input_orig,
        **dataset_kwargs
    )

    valid_data.preprocessor = valid_preprocessor

    # 如果args.dataset_dir为None，表示没有传入实际数据的路径，此时只能进行推理，因此返回验证数据集和相应的预处理器对象。
    if args.dataset_dir is None:
        # no path to the actual data was passed -> we cannot create dataloader,
        # return the valid dataset and preprocessor object for inference only
        if args.valid_full_res:
            return valid_data, valid_preprocessor_full_res
        else:
            return valid_data, valid_preprocessor

    # 创建训练数据加载器train_loader，使用DataLoader类将训练数据集加载为小批量数据。设置批量大小、工作进程数、是否舍弃最后一个小批量和是否打乱数据。
    # create the data loaders
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              drop_last=True,
                              shuffle=True)

    # for validation we can use higher batch size as activations do not
    # need to be saved for the backwards pass
    # 创建验证数据加载器valid_loader，使用DataLoader类将验证数据集加载为小批量数据。设置批量大小、工作进程数和是否打乱数据。
    batch_size_valid = args.batch_size_valid or args.batch_size
    valid_loader = DataLoader(valid_data,
                              batch_size=batch_size_valid,
                              num_workers=args.workers,
                              shuffle=False)
    # 如果args.valid_full_res为True，则创建全分辨率验证数据加载器valid_loader_full_res，
    # 将其设置为valid_loader的深拷贝，并将其数据集的预处理器设置为valid_preprocessor_full_res。
    # 然后返回训练数据加载器、验证数据加载器和全分辨率验证数据加载器。
    if args.valid_full_res:
        valid_loader_full_res = copy.deepcopy(valid_loader)
        valid_loader_full_res.dataset.preprocessor = valid_preprocessor_full_res
        return train_loader, valid_loader, valid_loader_full_res

    # 返回训练数据加载器和验证数据加载器
    return train_loader, valid_loader
