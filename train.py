# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from datetime import datetime
import json
import pickle
import os
import sys
import time
import warnings

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim
from torch.optim.lr_scheduler import OneCycleLR

from src.args import ArgumentParserRGBDSegmentation
from src.build_model import build_model
from src import utils
from src.prepare_data import prepare_data
from src.utils import save_ckpt, save_ckpt_every_epoch
from src.utils import load_ckpt
from src.utils import print_log

from src.logger import CSVLogger
from src.confusion_matrix import ConfusionMatrixTensorflow
from torch.cuda.amp import autocast as autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

# 设置使用哪个GPU
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 0, 3

scaler = GradScaler()


def set_config_nyuv2(args):
    args.dataset = 'nyuv2'
    args.dataset_dir = './datasets/nyuv2/'
    args.pretrained_dir = './trained_models'
    args.results_dir = './results'
    # args.last_ckpt = './add_train/R34.pth'
    args.height = 480
    args.width = 640
    args.batch_size = 8
    args.batch_size_valid = 24
    args.lr = 0.01
    args.optimizer = 'SGD'
    args.class_weighting = 'median_frequency'
    args.encoder = 'resnet18'
    args.encoder_block = 'NonBottleneck1D'
    args.nr_decoder_blocks = 3
    args.modality = 'rgbd'
    args.encoder_decoder_fusion = 'add'
    args.context_module = 'ppm'
    args.decoder_channels_mode = 'decreasing'
    args.fuse_depth_in_rgb_encoder = 'SE-add'
    args.upsampling = 'learned-3x3-zeropad'
    args.epochs = 1000
    args.activation = 'relu'
    args.workers = 8
    args.gpu = 1
    return args


def set_config_city(args):
    args.dataset = 'cityscapes-with-depth'
    args.dataset_dir = './datasets/cityscapes'
    args.pretrained_dir = './trained_models'
    args.results_dir = './results'
    args.raw_depth = True
    args.he_init = True
    args.aug_scale_min = 0.5
    args.aug_scale_max = 2.0
    args.valid_full_res = True
    args.height = 512
    args.width = 1024
    args.batch_size = 8
    args.batch_size_valid = 16
    args.lr = 1e-4
    args.optimizer = 'Adam'
    args.class_weighting = 'None'
    args.encoder = 'resnet18'
    args.encoder_block = 'NonBottleneck1D'
    args.nr_decoder_blocks = 3
    args.modality = 'rgb'
    args.encoder_decoder_fusion = 'add'
    args.context_module = 'appm-1-2-4-8'
    args.decoder_channels_mode = 'decreasing'
    args.fuse_depth_in_rgb_encoder = 'SE-add'
    args.upsampling = 'learned-3x3-zeropad'
    return args


def set_config_sunrgbd(args):
    args.dataset = 'sunrgbd'
    args.dataset_dir = './datasets/sunrgbd/'
    args.pretrained_dir = './trained_models'
    # args.last_ckpt = './add_train/R50.pth'
    args.results_dir = './results'
    args.height = 480
    args.width = 640
    args.batch_size = 8
    args.batch_size_valid = 24
    args.lr = 0.01
    args.optimizer = 'SGD'
    args.class_weighting = 'median_frequency'
    args.encoder = 'resnet50'
    args.encoder_block = 'BasicBlock'
    args.nr_decoder_blocks = 3
    args.modality = 'rgbd'
    args.encoder_decoder_fusion = 'add'
    args.context_module = 'ppm'
    args.decoder_channels_mode = 'decreasing'
    args.fuse_depth_in_rgb_encoder = 'SE-add'
    args.upsampling = 'learned-3x3-zeropad'
    args.epochs = 1000
    return args


def parse_args():
    # 创建一个ArgumentParserRGBDSegmentation对象作为参数解析器，并设置描述和格式化类。
    parser = ArgumentParserRGBDSegmentation(
        description='Efficient RGBD Indoor Sematic Segmentation (Training)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 设置常见的命令行参数
    parser.set_common_args()
    # 自动解析命令行参数，并将其存储在args对象中。
    args = parser.parse_args()

    args = set_config_nyuv2(args)
    # args = set_config_city(args)
    # args = set_config_sunrgbd(args)

    # The provided learning rate refers to the default batch size of 8.
    # When using different batch sizes we need to adjust the learning rate
    # accordingly:
    if args.batch_size != 8:
        args.lr = args.lr * args.batch_size / 8
        warnings.warn(f'Adapting learning rate to {args.lr} because provided '
                      f'batch size differs from default batch size of 8.')

    return args


def train_main():
    args = parse_args()
    # print("梯度裁剪")

    # directory for storing weights and other training related files
    training_starttime = datetime.now().strftime("%d_%m_%Y-%H_%M_%S-%f")
    ckpt_dir = os.path.join(args.results_dir, args.dataset,
                            f'checkpoints_{training_starttime}')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(os.path.join(ckpt_dir, 'confusion_matrices'), exist_ok=True)

    with open(os.path.join(ckpt_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    with open(os.path.join(ckpt_dir, 'argsv.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')

    # when using multi scale supervision the label needs to be downsampled.
    label_downsampling_rates = [8, 16, 32]

    # data preparation ---------------------------------------------------------
    print(args)

    data_loaders = prepare_data(args, ckpt_dir)

    if args.valid_full_res:
        train_loader, valid_loader, valid_loader_full_res = data_loaders
    else:
        train_loader, valid_loader = data_loaders
        valid_loader_full_res = None

    # 从训练数据加载器中获取相机信息和没有void类别的类别数量。
    cameras = train_loader.dataset.cameras
    n_classes_without_void = train_loader.dataset.n_classes_without_void
    # 如果命令行参数args.class_weighting不等于'None'，则调用训练数据加载器的compute_class_weights()方法，计算类别权重。
    # 根据命令行参数args.class_weighting的值和args.c_for_logarithmic_weighting的值，确定计算类别权重的方式和参数c。
    # 否则，将类别权重设置为全为1的数组
    if args.class_weighting != 'None':
        class_weighting = train_loader.dataset.compute_class_weights(
            weight_mode=args.class_weighting,
            c=args.c_for_logarithmic_weighting)
    else:
        class_weighting = np.ones(n_classes_without_void)

    # model building -----------------------------------------------------------
    # 调用build_model_original()函数构建模型，并将返回的模型和设备（CPU或GPU）存储在model和device变量中。
    model, device = build_model(args, n_classes=n_classes_without_void)

    # 如果命令行参数args.freeze大于0，则冻结除输出层以外的所有参数
    # 遍历模型的每个参数，如果参数名称中不包含字符串'out'，则将其requires_grad属性设置为False，即不计算梯度。
    if args.freeze > 0:
        print('Freeze everything but the output layer(s).')
        for name, param in model.named_parameters():
            if 'out' not in name:
                param.requires_grad = False

    # loss, optimizer, learning rate scheduler, csvlogger  ----------

    # loss functions (only loss_function_train is really needed.
    # The other loss functions are just there to compare valid loss to
    # train loss)
    # 使用utils.CrossEntropyLoss2d类构建训练损失函数loss_function_train，并传入类别权重和设备信息。
    loss_function_train = \
        utils.CrossEntropyLoss2d(weight=class_weighting, device=device)

    # 通过调用验证数据加载器的compute_class_weights()方法，计算验证数据的类别权重pixel_sum_valid_data。
    # 然后，使用utils.CrossEntropyLoss2dForValidData类构建验证损失函数loss_function_valid，并传入类别权重和加权像素和的信息。
    pixel_sum_valid_data = valid_loader.dataset.compute_class_weights(
        weight_mode='linear'
    )
    pixel_sum_valid_data_weighted = \
        np.sum(pixel_sum_valid_data * class_weighting)
    loss_function_valid = utils.CrossEntropyLoss2dForValidData(
        weight=class_weighting,
        weighted_pixel_sum=pixel_sum_valid_data_weighted,
        device=device
    )
    # 使用utils.CrossEntropyLoss2dForValidDataUnweighted类构建不加权的验证损失函数loss_function_valid_unweighted，并传入设备信息。
    loss_function_valid_unweighted = \
        utils.CrossEntropyLoss2dForValidDataUnweighted(device=device)

    # 根据命令行参数和模型，调用get_optimizer()函数获取优化器optimizer
    optimizer = get_optimizer(args, model)

    # 使用OneCycleLR类构建学习率调度器lr_scheduler，并传入优化器、最大学习率、总步数、学习率下降因子、起始学习率比例、退火策略和最终学习率下降因子
    # in this script lr_scheduler.step() is only called once per epoch
    if args.lr_scheduler == 'ExponentialLR':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    elif args.lr_scheduler == 'CyclicLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
        base_lr=0.001, max_lr=0.1, step_size_up=50, mode='triangular2')
    else:
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=[i['lr'] for i in optimizer.param_groups],
            total_steps=args.epochs,
            div_factor=25,
            pct_start=0.1,
            anneal_strategy='cos',
            final_div_factor=1e4
        )

    # 如果命令行参数args.last_ckpt存在，则加载指定路径的检查点文件，并从检查点中获取最后一个保存的训练周期、最好的mIoU值和对应的训练周期
    # 然后，将起始训练周期设置为最后一个保存的训练周期加1。否则，将起始训练周期设置为0，并将最好的mIoU值和对应的训练周期初始化为0。
    # load checkpoint if parameter last_ckpt is provided
    if args.last_ckpt:
        # ckpt_path = os.path.join(ckpt_dir, args.last_ckpt)
        ckpt_path = args.last_ckpt
        epoch_last_ckpt, best_miou, best_miou_epoch = \
            load_ckpt(model, optimizer, ckpt_path, device)
        start_epoch = epoch_last_ckpt + 1
    else:
        start_epoch = 0
        best_miou = 0
        best_miou_epoch = 0

    early_stop_counter = 0

    # 根据验证数据加载器的split属性，确定验证数据集的名称。
    # 然后，构建用于记录日志的键log_keys，包括mIoU和损失函数的键。
    # 如果命令行参数args.valid_full_res为True，则还包括全分辨率验证的mIoU键。
    valid_split = valid_loader.dataset.split

    # build the log keys for the csv log file and for the web logger
    log_keys = [f'mIoU_{valid_split}']
    if args.valid_full_res:
        log_keys.append(f'mIoU_{valid_split}_full-res')
        best_miou_full_res = 0

    log_keys_for_csv = log_keys.copy()

    # 接下来，根据摄像机的数量构建用于CSV日志的键log_keys_for_csv，包括每个摄像机的mIoU键和全分辨率验证的mIoU键。
    # mIoU for each camera
    for camera in cameras:
        log_keys_for_csv.append(f'mIoU_{valid_split}_{camera}')
        if args.valid_full_res:
            log_keys_for_csv.append(f'mIoU_{valid_split}_full-res_{camera}')

    # 然后，添加其他与训练和验证时间、学习率和损失函数相关的键。接下来，根据验证数据集的名称，构建每个类别的IoU键和损失函数键，并将它们添加到log_keys_for_csv中
    log_keys_for_csv.append('epoch')
    for i in range(len(lr_scheduler.get_lr())):
        log_keys_for_csv.append('lr_{}'.format(i))
    log_keys_for_csv.extend(['loss_train_total', 'loss_train_full_size'])
    for rate in label_downsampling_rates:
        log_keys_for_csv.append('loss_train_down_{}'.format(rate))
    log_keys_for_csv.extend(['time_training', 'time_validation',
                             'time_confusion_matrix', 'time_forward',
                             'time_post_processing', 'time_copy_to_gpu'])

    valid_names = [valid_split]
    if args.valid_full_res:
        valid_names.append(valid_split + '_full-res')
    for valid_name in valid_names:
        # iou for every class
        for i in range(n_classes_without_void):
            log_keys_for_csv.append(f'IoU_{valid_name}_class_{i}')
        log_keys_for_csv.append(f'loss_{valid_name}')
        if loss_function_valid_unweighted is not None:
            log_keys_for_csv.append(f'loss_{valid_name}_unweighted')

    csvlogger = CSVLogger(log_keys_for_csv, os.path.join(ckpt_dir, 'logs.csv'),
                          append=True)

    # 最后，使用ConfusionMatrixTensorflow类构建每个摄像机和整个验证数据集的混淆矩阵，并将它们存储在confusion_matrices字典中。
    # one confusion matrix per camera and one for whole valid data
    confusion_matrices = dict()
    for camera in cameras:
        confusion_matrices[camera] = \
            ConfusionMatrixTensorflow(n_classes_without_void)
        confusion_matrices['all'] = \
            ConfusionMatrixTensorflow(n_classes_without_void)

    # start training -----------------------------------------------------------
    for epoch in range(int(start_epoch), args.epochs):
        # unfreeze
        # 代码检查是否需要解冻模型。如果args.freeze等于当前周期epoch，并且没有指定args.finetune，则将模型的所有参数的requires_grad属性设置为True，即解冻模型。
        if args.freeze == epoch and args.finetune is None:
            print('Unfreezing')
            for param in model.parameters():
                param.requires_grad = True

        # 该函数的作用是对模型进行一次前向传播和反向传播，并更新模型的参数以最小化训练损失。训练过程中会记录一些训练日志，这些日志将在后面用于监控训练过程。
        logs = train_one_epoch(
            model, train_loader, device, optimizer, loss_function_train, epoch,
            lr_scheduler, args.modality,
            label_downsampling_rates, debug_mode=args.debug)

        # 计算模型的梯度范数
        # if model.parameters():
        #     grad_norm = torch.norm(torch.cat([param.grad.flatten() for param in model.parameters()]))
        #     print(f'Gradient norm: {grad_norm.item()}')

        # 大于200轮在验证集评估正确率 -----------------------------------------
        if epoch >= 200:
            # 代码调用validate()函数进行模型在验证集上的评估。
            miou, logs = validate(
                model, valid_loader, device, cameras,
                confusion_matrices, args.modality, loss_function_valid, logs,
                ckpt_dir, epoch, loss_function_valid_unweighted,
                debug_mode=args.debug
            )

            if args.valid_full_res:
                miou_full_res, logs = validate(
                    model, valid_loader_full_res, device, cameras,
                    confusion_matrices, args.modality, loss_function_valid, logs,
                    ckpt_dir,
                    epoch, loss_function_valid_unweighted,
                    add_log_key='_full-res', debug_mode=args.debug
                )

            logs.pop('time', None)
            csvlogger.write_logs(logs)

            # save weights
            print(miou['all'])
            save_current_checkpoint = False
            if miou['all'] > best_miou:
                best_miou = miou['all']
                best_miou_epoch = epoch
                early_stop_counter = 0
                save_current_checkpoint = True
            else:
                early_stop_counter += 1

            if args.valid_full_res and miou_full_res['all'] > best_miou_full_res:
                best_miou_full_res = miou_full_res['all']
                best_miou_full_res_epoch = epoch
                save_current_checkpoint = True

            # don't save weights for the first 10 epochs as mIoU is likely getting
            # better anyway
            if epoch >= 10 and save_current_checkpoint is True:
                save_ckpt(ckpt_dir, model, optimizer, epoch)

            # Perform early stopping check
            if early_stop_counter >= args.early_stop:
                print(f'Early stopping at epoch {epoch} as mIoU '
                      f'did not improve for {args.early_stop} consecutive epochs.')
                break

        # 梯度裁剪
        clip_grad_norm_(model.parameters(), max_norm=1.0)

        # save / overwrite latest weights (useful for resuming training)
        save_ckpt_every_epoch(ckpt_dir, model, optimizer, epoch, best_miou,
                              best_miou_epoch)

    # write a finish file with best miou values in order overview
    # training result quickly
    with open(os.path.join(ckpt_dir, 'finished.txt'), 'w') as f:
        f.write('best miou: {}\n'.format(best_miou))
        f.write('best miou epoch: {}\n'.format(best_miou_epoch))
        if args.valid_full_res:
            f.write(f'best miou full res: {best_miou_full_res}\n')
            f.write(f'best miou full res epoch: {best_miou_full_res_epoch}\n')

    print("Training completed ")


def train_one_epoch(model, train_loader, device, optimizer, loss_function_train,
                    epoch, lr_scheduler, modality,
                    label_downsampling_rates, debug_mode=False):
    training_start_time = time.time()
    # 根据当前周期更新学习率
    lr_scheduler.step(epoch)
    samples_of_epoch = 0

    # set model to train mode
    # 将模型设置为训练模式，这将启用模型中的训练特定操作，如Dropout。
    model.train()

    # loss for every resolution
    losses_list = []

    # summed loss of all resolutions
    total_loss_list = []

    for i, sample in enumerate(train_loader):
        start_time_for_one_step = time.time()

        # 根据模态类型加载数据并将其发送到GPU。
        # load the data and send them to gpu
        if modality in ['rgbd', 'rgb']:
            image = sample['image'].to(device)
            batch_size = image.data.shape[0]
        if modality in ['rgbd', 'depth']:
            depth = sample['depth'].to(device)
            batch_size = depth.data.shape[0]
        target_scales = [sample['label'].to(device)]
        if len(label_downsampling_rates) > 0:
            for rate in sample['label_down']:
                target_scales.append(sample['label_down'][rate].to(device))

        # optimizer.zero_grad()
        # this is more efficient than optimizer.zero_grad()
        # 清零模型参数的梯度，以便进行后续的反向传播。
        for param in model.parameters():
            param.grad = None

        with autocast():
            # forward pass
            if modality == 'rgbd':
                pred_scales = model(image, depth)
            elif modality == 'rgb':
                pred_scales = model(image)
            else:
                pred_scales = model(depth)

            # 根据模态类型进行前向传播，得到预测的分辨率。
            # loss computation
            losses = loss_function_train(pred_scales, target_scales)
            loss_segmentation = sum(losses)

            total_loss = loss_segmentation

            # total_loss.backward()
            # optimizer.step()

            # 对总损失进行梯度缩放，并进行反向传播和优化器更新。
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # append loss values to the lists. Later we can calculate the
        # mean training loss of this epoch
        # 记录损失值到losses_list和total_loss_list中。
        losses_list.append([loss.cpu().detach().numpy() for loss in losses])
        total_loss = total_loss.cpu().detach().numpy()
        total_loss_list.append(total_loss)

        # 如果总损失为NaN，则抛出异常
        if np.isnan(total_loss):
            raise ValueError('Loss is None')

        # print log
        # 计算并打印日志，包括当前周期、已处理样本数、批次大小、训练集大小、总损失、处理一个批次所需时间和学习率。
        samples_of_epoch += batch_size
        time_inter = time.time() - start_time_for_one_step

        learning_rates = lr_scheduler.get_lr()

        print_log(epoch, samples_of_epoch, batch_size,
                  len(train_loader.dataset), total_loss, time_inter,
                  learning_rates)

        # 如果处于调试模式，则仅处理一个批次，并跳出循环
        if debug_mode:
            # only one batch while debugging
            break

    # 填充日志字典到csv，包括训练时间、平均总损失、每个分辨率的平均损失、当前周期和学习率
    logs = dict()
    logs['time_training'] = time.time() - training_start_time
    logs['loss_train_total'] = np.mean(total_loss_list)
    losses_train = np.mean(losses_list, axis=0)
    logs['loss_train_full_size'] = losses_train[0]
    for i, rate in enumerate(label_downsampling_rates):
        logs['loss_train_down_{}'.format(rate)] = losses_train[i + 1]
    logs['epoch'] = epoch
    for i, lr in enumerate(learning_rates):
        logs['lr_{}'.format(i)] = lr
    return logs


def validate(model, valid_loader, device, cameras, confusion_matrices,
             modality, loss_function_valid, logs, ckpt_dir, epoch,
             loss_function_valid_unweighted=None, add_log_key='',
             debug_mode=False):
    valid_split = valid_loader.dataset.split + add_log_key

    print(f'Validation on {valid_split}')

    # we want to track how long each part of the validation takes
    validation_start_time = time.time()
    cm_time = 0  # time for computing all confusion matrices
    forward_time = 0
    post_processing_time = 0
    copy_to_gpu_time = 0

    # set model to eval mode
    model.eval()

    # we want to store miou and ious for each camera
    miou = dict()
    ious = dict()

    # reset loss (of last validation) to zero
    loss_function_valid.reset_loss()

    if loss_function_valid_unweighted is not None:
        loss_function_valid_unweighted.reset_loss()

    # validate each camera after another as all images of one camera have
    # the same resolution and can be resized together to the ground truth
    # segmentation size.
    for camera in cameras:
        with valid_loader.dataset.filter_camera(camera):
            confusion_matrices[camera].reset_conf_matrix()
            print(f'{camera}: {len(valid_loader.dataset)} samples')

            for i, sample in enumerate(valid_loader):
                # copy the data to gpu
                copy_to_gpu_time_start = time.time()
                if modality in ['rgbd', 'rgb']:
                    image = sample['image'].to(device)
                if modality in ['rgbd', 'depth']:
                    depth = sample['depth'].to(device)
                if not device.type == 'cpu':
                    torch.cuda.synchronize()
                copy_to_gpu_time += time.time() - copy_to_gpu_time_start

                # forward pass
                with torch.no_grad():
                    forward_time_start = time.time()
                    if modality == 'rgbd':
                        prediction = model(image, depth)
                    elif modality == 'rgb':
                        prediction = model(image)
                    else:
                        prediction = model(depth)
                    if not device.type == 'cpu':
                        torch.cuda.synchronize()
                    forward_time += time.time() - forward_time_start

                    # compute valid loss
                    post_processing_time_start = time.time()

                    loss_function_valid.add_loss_of_batch(
                        prediction,
                        sample['label'].to(device)
                    )

                    if loss_function_valid_unweighted is not None:
                        loss_function_valid_unweighted.add_loss_of_batch(
                            prediction, sample['label'].to(device))

                    # this label is not preprocessed and therefore still has its
                    # original size
                    label = sample['label_orig']
                    _, image_h, image_w = label.shape

                    # resize the prediction to the size of the original ground
                    # truth segmentation before computing argmax along the
                    # channel axis
                    prediction = F.interpolate(
                        prediction,
                        (image_h, image_w),
                        mode='bilinear',
                        align_corners=False)
                    prediction = torch.argmax(prediction, dim=1)

                    # ignore void pixels
                    mask = label > 0
                    label = torch.masked_select(label, mask)
                    prediction = torch.masked_select(prediction,
                                                     mask.to(device))

                    # In the label 0 is void, but in the prediction 0 is wall.
                    # In order for the label and prediction indices to match we
                    # need to subtract 1 of the label.
                    label -= 1

                    # copy the prediction to cpu as tensorflow's confusion
                    # matrix is faster on cpu
                    prediction = prediction.cpu()

                    label = label.numpy()
                    prediction = prediction.numpy()
                    post_processing_time += \
                        time.time() - post_processing_time_start

                    # finally compute the confusion matrix
                    cm_start_time = time.time()
                    confusion_matrices[camera].update_conf_matrix(label,
                                                                  prediction)
                    cm_time += time.time() - cm_start_time

                    if debug_mode:
                        # only one batch while debugging
                        break

            # After all examples of camera are passed through the model,
            # we can compute miou and ious.
            cm_start_time = time.time()
            miou[camera], ious[camera] = \
                confusion_matrices[camera].compute_miou()
            cm_time += time.time() - cm_start_time
            print(f'mIoU {valid_split} {camera}: {miou[camera]}')

    # confusion matrix for the whole split
    # (sum up the confusion matrices of all cameras)
    cm_start_time = time.time()
    confusion_matrices['all'].reset_conf_matrix()
    for camera in cameras:
        confusion_matrices['all'].overall_confusion_matrix += \
            confusion_matrices[camera].overall_confusion_matrix

    # miou and iou for all cameras
    miou['all'], ious['all'] = confusion_matrices['all'].compute_miou()
    cm_time += time.time() - cm_start_time
    print(f"mIoU {valid_split}: {miou['all']}")

    validation_time = time.time() - validation_start_time

    # save the confusion matrices of this epoch.
    # This helps if we want to compute other metrics later.
    with open(os.path.join(ckpt_dir, 'confusion_matrices',
                           f'cm_epoch_{epoch}.pickle'), 'wb') as f:
        pickle.dump({k: cm.overall_confusion_matrix
                     for k, cm in confusion_matrices.items()}, f,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # logs for the csv logger and the web logger
    logs[f'loss_{valid_split}'] = \
        loss_function_valid.compute_whole_loss()

    if loss_function_valid_unweighted is not None:
        logs[f'loss_{valid_split}_unweighted'] = \
            loss_function_valid_unweighted.compute_whole_loss()

    logs[f'mIoU_{valid_split}'] = miou['all']
    for camera in cameras:
        logs[f'mIoU_{valid_split}_{camera}'] = miou[camera]

    logs['time_validation'] = validation_time
    logs['time_confusion_matrix'] = cm_time
    logs['time_forward'] = forward_time
    logs['time_post_processing'] = post_processing_time
    logs['time_copy_to_gpu'] = copy_to_gpu_time

    # write iou value of every class to logs
    for i, iou_value in enumerate(ious['all']):
        logs[f'IoU_{valid_split}_class_{i}'] = iou_value

    return miou, logs


def get_optimizer(args, model):
    # set different learning rates fo different parts of the model
    # when using default parameters the whole model is trained with the same
    # learning rate
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            nesterov=True
        )
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            alpha=0.9
        )
    else:
        raise NotImplementedError(
            'Currently only SGD, Adam, AdamW and RMSprop as optimizers are '
            'supported. Got {}'.format(args.optimizer))

    print('Using {} as optimizer'.format(args.optimizer))
    return optimizer


if __name__ == '__main__':
    T1 = time.perf_counter()

    train_main()

    T2 = time.perf_counter()
    print('程序运行时间:%s秒' % (T2 - T1))