import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import progressbar
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use(['seaborn-poster'])

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data_root', default='data', help='root directory for data')
parser.add_argument('--model_path', default='', help='path to model')
parser.add_argument('--log_dir', default='', help='directory to save generations to')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--n_past', type=int, default=2, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=28, help='number of frames to predict')
parser.add_argument('--num_threads', type=int, default=0, help='number of data loading threads')
parser.add_argument('--nsample', type=int, default=100, help='number of samples')
parser.add_argument('--N', type=int, default=256, help='number of samples')
parser.add_argument('--data_type', default='test', help='train or test dataset')

opt = parser.parse_args()
os.makedirs('%s' % opt.log_dir, exist_ok=True)

opt.n_eval = opt.n_past+opt.n_future
opt.max_step = opt.n_eval

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- load the models  ----------------
tmp = torch.load(opt.model_path)
modelname = opt.model_path[:-4]
print(modelname)
model_log_dir = tmp['opt'].log_dir
frame_predictor = tmp['frame_predictor']
posterior = tmp['posterior']
prior = tmp['prior']
frame_predictor.eval()
prior.eval()
posterior.eval()
encoder = tmp['encoder']
decoder = tmp['decoder']
encoder.train()
decoder.train()
frame_predictor.batch_size = opt.batch_size
posterior.batch_size = opt.batch_size
prior.batch_size = opt.batch_size
opt.g_dim = tmp['opt'].g_dim
opt.z_dim = tmp['opt'].z_dim
opt.num_digits = tmp['opt'].num_digits

# --------- transfer to gpu ------------------------------------
frame_predictor.cuda()
posterior.cuda()
prior.cuda()
encoder.cuda()
decoder.cuda()

# ---------------- set the options ----------------
opt.dataset = tmp['opt'].dataset
opt.last_frame_skip = tmp['opt'].last_frame_skip
opt.channels = tmp['opt'].channels
opt.image_width = tmp['opt'].image_width

# --------- load a dataset ------------------------------------
train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.num_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.num_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch 
testing_batch_generator = get_testing_batch()

# --------- eval functions ------------------------------------

def evaluation(x, idx):
    nsample = opt.nsample
    ssim = np.zeros((opt.batch_size, nsample, opt.n_future))
    psnr = np.zeros((opt.batch_size, nsample, opt.n_future))

    sharpness = np.zeros((opt.batch_size, nsample, opt.n_future))

    progress = progressbar.ProgressBar(max_value=nsample).start()
    all_gen = []
    for s in range(nsample):
        progress.update(s+1)
        gen_seq = []
        gt_seq = []
        frame_predictor.hidden = frame_predictor.init_hidden()
        posterior.hidden = posterior.init_hidden()
        prior.hidden = prior.init_hidden()
        x_in = x[0]
        all_gen.append([])
        all_gen[s].append(x_in)
        for i in range(1, opt.n_eval):
            h = encoder(x_in)
            if opt.last_frame_skip or i <= opt.n_past:	
                h, skip = h
            else:
                h, _ = h
            h = h.detach()
            if i < opt.n_past:
                h_target = encoder(x[i])[0].detach()
                z_t, _, _ = posterior(h_target)
                prior(h)
                frame_predictor(torch.cat([h, z_t], 1))
                x_in = x[i]
                all_gen[s].append(x_in)
            else:
                z_t, _, _ = prior(h)
                h = frame_predictor(torch.cat([h, z_t], 1)).detach()
                x_in = decoder([h, skip]).detach()
                gen_seq.append(x_in.data.cpu().numpy())
                gt_seq.append(x[i].data.cpu().numpy())

                all_gen[s].append(x_in)
        _, ssim[:, s, :], psnr[:, s, :] = utils.eval_seq(gt_seq, gen_seq)
        sharpness[:, s, :] = utils.sharpness_seq(gen_seq)
        
    progress.finish()
    utils.clear_progressbar()

    return ssim, psnr, sharpness


ssim_stacked = np.empty((0, opt.nsample, opt.n_future))
psnr_stacked = np.empty((0, opt.nsample, opt.n_future))
sharpness_stacked = np.empty((0, opt.nsample, opt.n_future))

for i in range(0, opt.N, opt.batch_size):
    print(f'batch {i}~{i+opt.batch_size}...')
    
    if opt.data_type=='test':
        x = next(testing_batch_generator) 

    if opt.data_type=='train':        
        x = next(training_batch_generator) 
        
    ssim, psnr, sharpness = evaluation(x, i)

    ssim_stacked = np.append(ssim_stacked, ssim, axis=0)
    psnr_stacked = np.append(psnr_stacked, psnr, axis=0)
    sharpness_stacked = np.append(sharpness_stacked, sharpness, axis=0)

#### plot ####

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(30,7))
ax1.grid()
ax2.grid()
ax3.grid()

#ssim
ssim = np.mean(ssim_stacked, axis=1)
ssim = pd.DataFrame(ssim).melt()
ssim.columns=['Time Step', 'Average SSIM']
ssim['Time Step']+=tmp['opt'].n_past

ssim_average = np.mean(ssim['Average SSIM'])
ssim_plot = sns.lineplot(x="Time Step", y="Average SSIM", data=ssim, label='WVG', ci=90, ax=ax1)
ssim_plot.set_title('SSIM', fontsize=30, fontweight='bold')
ssim_plot.axes.set_xlim(0,)
ssim_plot.axvline(tmp['opt'].n_past+tmp['opt'].n_future, color='black')

#psnr
psnr = np.mean(psnr_stacked, axis=1)
psnr = pd.DataFrame(psnr).melt()
psnr.columns=['Time Step', 'Average PSNR']
psnr['Time Step']+=tmp['opt'].n_past

psnr_average = np.mean(psnr['Average PSNR'])
psnr_plot = sns.lineplot(x="Time Step", y="Average PSNR", data=psnr, label='WVG', ci=90, ax=ax2)
psnr_plot.set_title('PSNR', fontsize=30, fontweight='bold')
psnr_plot.axes.set_xlim(0,)
psnr_plot.axvline(tmp['opt'].n_past+tmp['opt'].n_future, color='black')

#sharpness
sharpness = np.mean(sharpness_stacked, axis=1)
sharpness = pd.DataFrame(sharpness).melt()
sharpness.columns=['Time Step', 'Average Sharpness']
sharpness['Time Step']+=tmp['opt'].n_past

sharpness_average = np.mean(sharpness['Average Sharpness'])
sharpness_plot = sns.lineplot(x="Time Step", y="Average Sharpness", data=sharpness, label='WVG', ci=90, ax=ax3)
sharpness_plot.set_title('Sharpness', fontsize=30, fontweight='bold')
sharpness_plot.axes.set_xlim(0,)
sharpness_plot.axvline(tmp['opt'].n_past+tmp['opt'].n_future, color='black')

fig.savefig(opt.log_dir + '/evaluation_plot.pdf', dpi=200, bbox_inches='tight')
