import math
import torch
import socket
import argparse
import os
import numpy as np
import scipy.misc
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import functools
from skimage.measure import compare_psnr as psnr_metric
from skimage.measure import compare_ssim as ssim_metric
from scipy import signal
from scipy import ndimage
from scipy.ndimage.filters import laplace
from PIL import Image, ImageDraw

from torchvision import datasets, transforms
from torch.autograd import Variable
import imageio

hostname = socket.gethostname()

def load_dataset(opt):
    if opt.dataset == 'towel':
        from data.towel import TowelPick 
        train_data = TowelPick(
                data_root=opt.data_root,
                type='train',
                seq_len=opt.n_past+opt.n_future,
                image_size=opt.image_width)
        val_data = TowelPick(
                data_root=opt.data_root,
                type='val',
                seq_len=opt.n_eval,
                image_size=opt.image_width)
        test_data = TowelPick(
                data_root=opt.data_root,
                type='test',
                seq_len=opt.n_eval,
                image_size=opt.image_width)
        
        return train_data, val_data, test_data

    elif opt.dataset == 'kitti':
        from data.kitti import KITTI 
        train_data = KITTI(
                data_root=opt.data_root,
                type='train',
                seq_len=opt.n_past+opt.n_future,
                image_size=opt.image_width)
        val_data = KITTI(
                data_root=opt.data_root,
                type='val',
                seq_len=opt.n_eval,
                image_size=opt.image_width)
        test_data = KITTI(
                data_root=opt.data_root,
                type='test',
                seq_len=opt.n_eval,
                image_size=opt.image_width)
        
        return train_data, val_data, test_data
        
    elif opt.dataset == 'bair':
        from data.bair import RobotPush 
        train_data = RobotPush(
                data_root=opt.data_root,
                train=True,
                seq_len=opt.n_past+opt.n_future,
                image_size=opt.image_width)
        test_data = RobotPush(
                data_root=opt.data_root,
                train=False,
                seq_len=opt.n_eval,
                image_size=opt.image_width)

    elif opt.dataset == 'kth':
        from data.kth import KTH 
        train_data = KTH(
                train=True, 
                data_root=opt.data_root,
                seq_len=opt.n_past+opt.n_future, 
                image_size=opt.image_width)
        test_data = KTH(
                train=False, 
                data_root=opt.data_root,
                seq_len=opt.n_eval, 
                image_size=opt.image_width)
    
    return train_data, test_data

def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]

def normalize_data(opt, dtype, sequence):
    if opt.dataset == 'kth' or opt.dataset == 'bair' or opt.dataset == 'towel' or opt.dataset == 'human' or opt.dataset == 'kitti':
        sequence.transpose_(0, 1)
        sequence.transpose_(3, 4).transpose_(2, 3)
    else:
        sequence.transpose_(0, 1)

    return sequence_input(sequence, dtype)

def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))

def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result

def save_np_img(fname, x):
    if x.shape[0] == 1:
        x = np.tile(x, (3, 1, 1))
    img = scipy.misc.toimage(x,
                             high=255*x.max(),
                             channel_axis=0)
    img.save(fname)

def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    return scipy.misc.toimage(tensor.numpy(),
                              high=255*tensor.numpy().max(),
                              channel_axis=0)

def draw_text_tensor(tensor, text):
    np_x = tensor.transpose(0, 1).transpose(1, 2).data.cpu().numpy()
    pil = Image.fromarray(np.uint8(np_x*255))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 64), text, (0,0,0))
    img = np.asarray(pil)
    return Variable(torch.Tensor(img / 255.)).transpose(1, 2).transpose(0, 1)

def save_gif(filename, inputs, duration=0.25):
    images = []
    for tensor in inputs:
        img = image_tensor(tensor, padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        img = 255*img.numpy()
        images.append(img.astype(np.uint8))
    imageio.mimsave(filename, images, duration=duration)

def save_gif_with_text(filename, inputs, text, fps=10):
    images = []
    for tensor, text in zip(inputs, text):
        img = image_tensor([draw_text_tensor(ti, texti) for ti, texti in zip(tensor, text)], padding=0)
        img = img.cpu()
        img = img.transpose(0,1).transpose(1,2).clamp(0,1)
        img = 255*img.numpy()
        images.append(img.astype(np.uint8))
    imageio.mimsave(filename, images, fps=10)

def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)

def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)

def prod(l):
    return functools.reduce(lambda x, y: x * y, l)

def batch_flatten(x):
    return x.resize(x.size(0), prod(x.size()[1:]))

def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

def sharpness_seq(seq):
    T = len(seq)
    bs = seq[0].shape[0]
    sharpness = np.zeros((bs, T))
    if seq[0][0].shape[0]==1: #1 channel
        for t in range(T):            
            seq[t] = seq[t].squeeze(axis=1)
    if seq[0][0].shape[0]==3: #3 channel
        for t in range(T):
            seq[t] = 0.2989 * seq[t][:,0,:,:]+0.5870 * seq[t][:,1,:,:] + 0.1140 * seq[t][:,2,:,:] # change image to grayscale
    for i in range(bs):
        for t in range(T): 
            sharpness[i, t] = np.var(laplace(seq[t][i].astype(np.float64)))

    return sharpness

def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[t][i].shape[0]):
                ssim[i, t] += ssim_metric(gt[t][i][c], pred[t][i][c])
                psnr[i, t] += psnr_metric(gt[t][i][c], pred[t][i][c])
            ssim[i, t] /= gt[t][i].shape[0]
            psnr[i, t] /= gt[t][i].shape[0]
            mse[i, t] = mse_metric(gt[t][i], pred[t][i])

    return mse, ssim, psnr

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

