import argparse
import os

import numpy as np
from PIL import Image
from skimage.measure import compare_ssim, compare_psnr
import torchvision
import torch
from torchvision.transforms.functional import to_tensor
from torch.autograd import Variable

from src.util.util import get_folder_paths_at_depth, makedir

#sharpness
from scipy.ndimage.filters import laplace

#lpips
from PerceptualSimilarity import models
from PerceptualSimilarity.util import util

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('qual_results_root', type=str)
    parser.add_argument('quant_results_root', type=str)
    parser.add_argument('K', type=int, help='Number of preceding frames')
    parser.add_argument('T', type=int, help='Number of middle frames')
    parser.add_argument('--depth', type=int, default=1,
                        help='Depth of the folders for each video (e.g. 2 for <qual_results_root>/<action>/<video>)')
    args = parser.parse_args()

    # Get the paths to the qualitative frames of each test video
    qual_frame_root_paths = get_folder_paths_at_depth(args.qual_results_root, args.depth)

    if len(qual_frame_root_paths) == 0:
        print('Failed to find any qualitative results (make sure you ran predict.py before this script). Quitting...')
        return

    print('Now computing quantitative results...')

    model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True,version='0.1')

    psnr_table = np.zeros((len(qual_frame_root_paths), args.T))
    ssim_table = np.zeros((len(qual_frame_root_paths), args.T))
    sharpness_table = np.zeros((len(qual_frame_root_paths), args.T))
    lpips_table = np.zeros((len(qual_frame_root_paths), args.T))
    video_list = []


    for i, qual_frame_root_path in enumerate(qual_frame_root_paths):
        print(f'Evaluating video {i+1}.')
        video_list.append(qual_frame_root_path)
        for t in range(args.K, args.K + args.T):
            try:
                gt_middle_frame = Image.open(os.path.join(qual_frame_root_path, 'gt_middle_%04d.png' % t))
            except IOError:
                raise RuntimeError('Failed to find GT middle frame at %s (did you generate GT middle frames and use '
                                   'the right values for K and T?)'
                                   % os.path.join(qual_frame_root_path, 'gt_middle_%04d.png' % t))
            pred_middle_frame = Image.open(os.path.join(qual_frame_root_path, 'pred_middle_%04d.png' % t))
            psnr_table[i, t - args.K] = compare_psnr(np.array(pred_middle_frame), np.array(gt_middle_frame))
            ssim_table[i, t - args.K] = compare_ssim(np.array(gt_middle_frame), np.array(pred_middle_frame),
                                                     multichannel=(gt_middle_frame.mode == 'RGB'))
 
            #sharpness
            if len(np.array(pred_middle_frame).shape)==3: #if color, to grayscale
                img_pred = 0.2989*np.array(pred_middle_frame)[:,:,0]+0.5870*np.array(pred_middle_frame)[:,:,1]+0.1140*np.array(pred_middle_frame)[:,:,2]
            else:
                img_pred = np.array(pred_middle_frame)

            sharpness_table[i, t - args.K] = np.var(laplace(img_pred/255.))
            


            # lpips
            if len(np.array(pred_middle_frame).shape)==2: #if grayscale, to color
                img_pred = util.im2tensor(np.repeat(np.array(pred_middle_frame)[:,:,np.newaxis], 3, axis=2))
                img_gt = util.im2tensor(np.repeat(np.array(gt_middle_frame)[:,:,np.newaxis], 3, axis=2))
            else: #color
                img_pred = util.im2tensor(np.array(pred_middle_frame))
                img_gt = util.im2tensor(np.array(gt_middle_frame))
            lpips_table[i, t - args.K] = model.forward(img_pred, img_gt)

    # Save PSNR and SSIM tables and video list to a file
    makedir(args.quant_results_root)
    video_list = np.array(video_list)

    np.savez(os.path.join(args.quant_results_root, 'results.npz'),
             psnr=psnr_table,
             ssim=ssim_table,
             sharpness=sharpness_table,
             lpips=lpips_table,
             video=video_list)

    print('Done computing quantitative results.')

if __name__ == '__main__':
    main()
