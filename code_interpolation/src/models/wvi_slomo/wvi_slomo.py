from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import cv2
import numpy as np
import logging
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
import sys
import os
sys.path.insert(
    0, os.path.realpath(os.path.dirname(__file__) + "/../../.."))
from src.data.base_dataset import ContiguousVideoClipDataset
from src.util.util import makedir, listopt, to_numpy, inverse_transform


##################################################
##############  SloMo PRIMITIVES  ###############
##################################################

class Encoder(nn.Module):

    def __init__(self, gf_dim, input_dim, alpha=0.1):
        """Constructor
        
        Parameters:
            gf_dim: The number of filters in the first layer
            input_dim: dimension of input
        """

        super(Encoder, self).__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(input_dim, gf_dim, 7, padding=3),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim, gf_dim, 7, padding=3),
            nn.LeakyReLU(alpha)
        )

        self.enc2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim, gf_dim * 2, 5, padding=2),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 2, gf_dim * 2, 5, padding=2),
            nn.LeakyReLU(alpha)
        )

        self.enc3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim * 2, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 4, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.enc4 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim * 4, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 8, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.enc5 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim * 8, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 16, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.enc6 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(gf_dim * 16, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 16, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

    def forward(self, input_imgs):

        enc1_out = self.enc1(input_imgs)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)
        output = self.enc6(enc5_out)

        res_in = [enc1_out, enc2_out, enc3_out, enc4_out, enc5_out]

        return output, res_in


class ComputeDecoder(nn.Module):

    def __init__(self, gf_dim, out_dim, alpha=0.1):
        """Constructor
        
        Parameters:
            gf_dim: The number of filters in the first layer
            out_dim: The dimension of output
        """

        super(ComputeDecoder, self).__init__()

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec1 = nn.Sequential(
            nn.Conv2d(gf_dim * 32, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 16, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec2 = nn.Sequential(
            nn.Conv2d(gf_dim * 16, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec3 = nn.Sequential(
            nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample4 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec4 = nn.Sequential(
            nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 2, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample5 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec5 = nn.Sequential(
            nn.Conv2d(gf_dim * 2, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.output = nn.Conv2d(gf_dim, out_dim, 1)
        self.tanh = nn.Tanh()


    def forward(self, encoded_input, res_in):

        upsample1_out = self.upsample1(encoded_input)
        dec1_out = self.dec1(
            torch.cat((upsample1_out, res_in[-1]), 1))
        upsample2_out = self.upsample2(dec1_out)
        dec2_out = self.dec2(torch.cat((upsample2_out, res_in[-2]), 1))
        upsample3_out = self.upsample3(dec2_out)
        dec3_out = self.dec3(torch.cat((upsample3_out, res_in[-3]), 1))
        upsample4_out = self.upsample4(dec3_out)
        dec4_out = self.dec4(torch.cat((upsample4_out, res_in[-4]), 1))
        upsample5_out = self.upsample5(dec4_out)
        dec5_out = self.dec5(torch.cat((upsample5_out, res_in[-5]), 1))
        output = self.output(dec5_out)
        output = self.tanh(output)

        return output


class RefineDecoder(nn.Module):

    def __init__(self, gf_dim, out_dim, alpha=0.1):
        """Constructor
        
        Parameters:
            gf_dim: The number of filters in the first layer
            out_dim: The dimension of output
        """

        super(RefineDecoder, self).__init__()

        self.upsample1 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec1 = nn.Sequential(
            nn.Conv2d(gf_dim * 32, gf_dim * 16, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 16, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample2 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec2 = nn.Sequential(
            nn.Conv2d(gf_dim * 16, gf_dim * 8, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample3 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec3 = nn.Sequential(
            nn.Conv2d(gf_dim * 8, gf_dim * 4, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample4 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec4 = nn.Sequential(
            nn.Conv2d(gf_dim * 4, gf_dim * 2, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim * 2, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.upsample5 = nn.Upsample(scale_factor=2, mode="bilinear")

        self.dec5 = nn.Sequential(
            nn.Conv2d(gf_dim * 2, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha),
            nn.Conv2d(gf_dim, gf_dim, 3, padding=1),
            nn.LeakyReLU(alpha)
        )

        self.output = nn.Conv2d(gf_dim, out_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, encoded_input, res_in):

        upsample1_out = self.upsample1(encoded_input)
        dec1_out = self.dec1(
            torch.cat((upsample1_out, res_in[-1]), 1))
        upsample2_out = self.upsample2(dec1_out)
        dec2_out = self.dec2(torch.cat((upsample2_out, res_in[-2]), 1))
        upsample3_out = self.upsample3(dec2_out)
        dec3_out = self.dec3(torch.cat((upsample3_out, res_in[-3]), 1))
        upsample4_out = self.upsample4(dec3_out)
        dec4_out = self.dec4(torch.cat((upsample4_out, res_in[-4]), 1))
        upsample5_out = self.upsample5(dec4_out)
        dec5_out = self.dec5(torch.cat((upsample5_out, res_in[-5]), 1))
        output = self.output(dec5_out)

        delta_F_t_0, delta_F_t_1, V_t_0 = torch.split(output, 2, dim=1)
        V_t_0 = self.sigmoid(V_t_0)
        delta_F_t_0 = self.tanh(delta_F_t_0)
        delta_F_t_1 = self.tanh(delta_F_t_1)

        return delta_F_t_0, delta_F_t_1, V_t_0


class FlowWarper(nn.Module):

    def forward(self, img, uv):

        super(FlowWarper, self).__init__()
        H = int(img.shape[-2])
        W = int(img.shape[-1])
        x = np.arange(0, W)
        y = np.arange(0, H)
        gx, gy = np.meshgrid(x, y)
        grid_x = Variable(torch.Tensor(gx), requires_grad=False).cuda()
        grid_y = Variable(torch.Tensor(gy), requires_grad=False).cuda()
        u = uv[:, 0, :, :]
        v = uv[:, 1, :, :]
        X = grid_x.unsqueeze(0) + u
        Y = grid_y.unsqueeze(0) + v
        X = 2 * (X / W - 0.5)
        Y = 2 * (Y / H - 0.5)
        grid_tf = torch.stack((X, Y), dim=3)
        img_tf = F.grid_sample(img, grid_tf)

        return img_tf

class Reparameterization(nn.Module):
    """A reparameterization layer."""

    def __init__(self, feature_size, num_features, bias=True):
        """Constructor

        :param feature_size: The kernel size of the convolutional layer
        :param num_features: Controls the number of input/output features of cell
        :param bias: Whether to use a bias for the convolutional layer
        """
        super(Reparameterization, self).__init__()

        self.feature_size = feature_size
        self.num_features = num_features
        
        self.mu_net = nn.Conv2d(num_features, num_features, feature_size, padding=int((feature_size-1)/2), bias=bias)
        self.logvar_net = nn.Conv2d(num_features, num_features, feature_size, padding=int((feature_size-1)/2), bias=bias)
        self.leakyrelu = nn.LeakyReLU(0.1)
        
    def forward(self, input):
        """Forward method

        :param input: The current input to the ConvLSTM
        :param state: The previous state of the ConvLSTM (the concatenated memory cell and hidden state)
        """
        mu, logvar = self.leakyrelu(self.mu_net(input)), self.leakyrelu(self.logvar_net(input))
        logvar = logvar.mul(0.5).exp_()
        eps = Variable(logvar.data.new(logvar.size()).normal_())
        z = eps.mul(logvar).add(mu)
        return z
    
    

class SloMo(nn.Module):
    """The SloMo video prediction network. """

    def __init__(self, gf_dim, c_input_dim, is_stochastic=False):

        super(SloMo, self).__init__()
        self.gf_dim = gf_dim
        self.c_input_dim = c_input_dim
        self.is_stochastic = is_stochastic
        self.compute_enc = Encoder(self.gf_dim, 2 * c_input_dim)
        self.compute_dec = ComputeDecoder(self.gf_dim, 4)
        self.flow_warper = FlowWarper()
        self.refine_enc_prior = Encoder(self.gf_dim, 4 * c_input_dim + 4)
        self.refine_enc_post = Encoder(self.gf_dim, 4 * c_input_dim + 4)
        
        if is_stochastic:
            self.reparameterization_prior = Reparameterization(1, self.gf_dim * 16, bias=True)
            self.reparameterization_post = Reparameterization(1, self.gf_dim * 16, bias=True)
        self.refine_dec = RefineDecoder(self.gf_dim, 5)
        
    def forward(self, T, I0, I1, gt_middle_frames=None, is_train=False):
        
        if is_train:
            # I0 and I1 have the format [batch_size, channel, W, H]
            img = torch.cat((I0, I1), 1)
            compute_enc_out, compute_res_in = self.compute_enc(img)
            h_cond = compute_enc_out.detach().cuda()
            compute_dec_out = self.compute_dec(compute_enc_out, compute_res_in)
            F_0_1 = compute_dec_out[:, :2, :, :]
            F_1_0 = compute_dec_out[:, 2:, :, :]
            first = True
            for t_ in range(T):
                t = (t_ + 1) / (T + 1)
                F_t_0 = -(1 - t) * t * F_0_1 + t ** 2 * F_1_0
                F_t_1 = (1 - t) * (1 - t) * F_0_1 - t * (1 - t) * F_1_0
                g_I0_F_t_0 = self.flow_warper(img[:, :self.c_input_dim, :, :], F_t_0)
                g_I1_F_t_1 = self.flow_warper(img[:, self.c_input_dim:, :, :], F_t_1)
                It = gt_middle_frames[:, t_, :, :, :]
                interp_input_prior = torch.cat((I0, g_I0_F_t_0, F_t_0, F_t_1, g_I1_F_t_1, I1), 1)
                interp_input_post = torch.cat((I0, It, F_t_0, F_t_1, It, I1), 1)
                interp_enc_out_prior, interp_res_in_prior = self.refine_enc_prior(interp_input_prior)
                interp_enc_out_post, _ = self.refine_enc_post(interp_input_post)
                if self.is_stochastic:
                    interp_enc_out_prior = self.reparameterization_prior(interp_enc_out_prior)
                    interp_enc_out_post = self.reparameterization_post(interp_enc_out_post)
                delta_F_t_0_post, delta_F_t_1_post, V_t_0_post = self.refine_dec(interp_enc_out_post, interp_res_in_prior)
                F_t_0_refine_post = delta_F_t_0_post + F_t_0
                F_t_0_refine_post = torch.clamp(F_t_0_refine_post, min=-1, max=1)
                F_t_1_refine_post = delta_F_t_1_post + F_t_1
                F_t_1_refine_post = torch.clamp(F_t_1_refine_post, min=-1, max=1)
                V_t_1_post = 1 - V_t_0_post
                g_I0_F_t_0_refine_post = self.flow_warper(I0, F_t_0_refine_post)
                g_I1_F_t_1_refine_post = self.flow_warper(I1, F_t_1_refine_post)
                normalization_post = (1 - t) * V_t_0_post + t * V_t_1_post
                interp_image_post = ((1 - t) * V_t_0_post * g_I0_F_t_0_refine_post + t * V_t_1_post * g_I1_F_t_1_refine_post) / normalization_post
                F_t_0 = torch.unsqueeze(F_t_0, 1)
                F_t_1 = torch.unsqueeze(F_t_1, 1)
                interp_image_post = torch.unsqueeze(interp_image_post, 1)
                if first:
                    predictions_post = interp_image_post
                    F_t_0_collector = F_t_0
                    F_t_1_collector = F_t_1
                    interp_enc_out_prior_seq = interp_enc_out_prior
                    interp_enc_out_post_seq = interp_enc_out_post
                    first = False
                else:
                    F_t_0_collector = torch.cat((F_t_0_collector, F_t_0), 1)
                    F_t_1_collector = torch.cat((F_t_1_collector, F_t_1), 1)
                    interp_enc_out_prior_seq = torch.cat([interp_enc_out_prior_seq, interp_enc_out_prior], dim=1)
                    interp_enc_out_post_seq = torch.cat([interp_enc_out_post_seq, interp_enc_out_post], dim=1)
                    predictions_post = torch.cat((predictions_post, interp_image_post), 1)
                    
            return predictions_post, F_0_1, F_1_0, F_t_0_collector, F_t_1_collector, interp_enc_out_prior_seq, interp_enc_out_post_seq, h_cond
        else:
            # I0 and I1 have the format [batch_size, channel, W, H]
            img = torch.cat((I0, I1), 1)
            compute_enc_out, compute_res_in = self.compute_enc(img)
            compute_dec_out = self.compute_dec(compute_enc_out, compute_res_in)
            F_0_1 = compute_dec_out[:, :2, :, :]
            F_1_0 = compute_dec_out[:, 2:, :, :]
            first = True
            for t_ in range(T):
                t = (t_ + 1) / (T + 1)
                F_t_0 = -(1 - t) * t * F_0_1 + t ** 2 * F_1_0
                F_t_1 = (1 - t) * (1 - t) * F_0_1 - t * (1 - t) * F_1_0
                g_I0_F_t_0 = self.flow_warper(img[:, :self.c_input_dim, :, :], F_t_0)
                g_I1_F_t_1 = self.flow_warper(img[:, self.c_input_dim:, :, :], F_t_1)
                interp_input = torch.cat((I0, g_I0_F_t_0, F_t_0, F_t_1, g_I1_F_t_1, I1), 1)
                interp_enc_out, interp_res_in = self.refine_enc_prior(interp_input)
                if self.is_stochastic:
                    interp_enc_out = self.reparameterization_prior(interp_enc_out)
                delta_F_t_0, delta_F_t_1, V_t_0 = self.refine_dec(interp_enc_out, interp_res_in)
                F_t_0_refine = delta_F_t_0 + F_t_0
                F_t_0_refine = torch.clamp(F_t_0_refine, min=-1, max=1)
                F_t_1_refine = delta_F_t_1 + F_t_1
                F_t_1_refine = torch.clamp(F_t_1_refine, min=-1, max=1)
                V_t_1 = 1 - V_t_0
                g_I0_F_t_0_refine = self.flow_warper(I0, F_t_0_refine)
                g_I1_F_t_1_refine = self.flow_warper(I1, F_t_1_refine)
                normalization = (1 - t) * V_t_0 + t * V_t_1
                interp_image = ((1 - t) * V_t_0 * g_I0_F_t_0_refine + t * V_t_1 * g_I1_F_t_1_refine) / normalization
                F_t_0 = torch.unsqueeze(F_t_0, 1)
                F_t_1 = torch.unsqueeze(F_t_1, 1)
                interp_image = torch.unsqueeze(interp_image, 1)
                if first:
                    predictions = interp_image
                    F_t_0_collector = F_t_0
                    F_t_1_collector = F_t_1
                    first = False
                else:
                    F_t_0_collector = torch.cat((F_t_0_collector, F_t_0), 1)
                    F_t_1_collector = torch.cat((F_t_1_collector, F_t_1), 1)
                    predictions = torch.cat((predictions, interp_image), 1)

            return predictions, F_0_1, F_1_0, F_t_0_collector, F_t_1_collector


class WVISloMoFillInModel(nn.Module):

    def __init__(self, gf_dim=32, c_input_dim=3):

        super(WVISloMoFillInModel, self).__init__()

        self.generator = SloMo(gf_dim, c_input_dim, is_stochastic=True)


    def forward(self, T, preceding_frames, following_frames):
        """Forward method

        :param T: The number of middle frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        """

        # Generate the forward and backward predictions
        pred, F_0_1, F_1_0, F_t_0_collector, F_t_1_collector = self.generator(T, preceding_frames[:, -1, :, :, :], following_frames[:, 0, :, :, :])

        return {
            'pred': pred,
            'F_0_1': F_0_1,
            'F_1_0': F_1_0,
            'F_t_0_collector': F_t_0_collector,
            'F_t_1_collector': F_t_1_collector
        }
    
    def forward_train(self, T, preceding_frames, gt_middle_frames, following_frames, gamma):
        """Forward method

        :param T: The number of middle frames to generate
        :param preceding_frames: The frames before the sequence to predict (B x K x C x H x W)
        :param gt_middle_frames: The frames of the sequence to be predicted (B x T x C x H x W)
        :param following_frames: The frames after the sequence to predict (B x F x C x H x W)
        """

        # Generate the forward and backward predictions
        predictions_post, F_0_1, F_1_0, F_t_0_collector, F_t_1_collector, interp_enc_out_prior_seq, interp_enc_out_post_seq, compute_enc_out = self.generator(T, preceding_frames[:, -1, :, :, :], following_frames[:, 0, :, :, :], gt_middle_frames, is_train=True)
        
        if gamma == 0.0:
            shuffled_idx, combination_post_shuffled, predictions_shuffled = None, None, None
        else:
            # Compute the shuffled time information to inject
            shuffled_idx = np.arange(preceding_frames.size(0))
            np.random.shuffle(shuffled_idx)
            
            compute_enc = getattr(self.generator, 'compute_enc')
            compute_dec = getattr(self.generator, 'compute_dec')
            flow_warper = getattr(self.generator, 'flow_warper')
            refine_enc_prior = getattr(self.generator, 'refine_enc_prior')
            refine_dec = getattr(self.generator, 'refine_dec')
            
            c_input_dim, gf_dim = getattr(self.generator, 'c_input_dim'), getattr(self.generator, 'gf_dim')
            
            # I0 and I1 have the format [batch_size, channel, W, H]
            I0_shuffled, I1_shuffled = preceding_frames[:, -1, :, :, :][shuffled_idx], following_frames[:, 0, :, :, :][shuffled_idx]
            img_shuffled = torch.cat((I0_shuffled, I1_shuffled), 1)
            compute_enc_out_shuffled, compute_res_in_shuffled = compute_enc(img_shuffled)
            compute_dec_out_shuffled = compute_dec(compute_enc_out_shuffled, compute_res_in_shuffled)
            F_0_1_shuffled = compute_dec_out_shuffled[:, :2, :, :]
            F_1_0_shuffled = compute_dec_out_shuffled[:, 2:, :, :]
            first = True
            for t_ in range(T):
                t = (t_ + 1) / (T + 1)
                F_t_0_shuffled = -(1 - t) * t * F_0_1_shuffled + t ** 2 * F_1_0_shuffled
                F_t_1_shuffled = (1 - t) * (1 - t) * F_0_1_shuffled - t * (1 - t) * F_1_0_shuffled
                g_I0_F_t_0_shuffled = flow_warper(img_shuffled[:, :c_input_dim, :, :], F_t_0_shuffled)
                g_I1_F_t_1_shuffled = flow_warper(img_shuffled[:, c_input_dim:, :, :], F_t_1_shuffled)
                interp_input_shuffled = torch.cat((I0_shuffled, g_I0_F_t_0_shuffled, F_t_0_shuffled, F_t_1_shuffled, g_I1_F_t_1_shuffled, I1_shuffled), 1)
                _, interp_res_in_shuffled = refine_enc_prior(interp_input_shuffled)
                interp_enc_out_unshuffled = interp_enc_out_post_seq[:, np.arange((t_*gf_dim*16),((t_+1)*gf_dim*16)), :, :]
                delta_F_t_0_shuffled, delta_F_t_1_shuffled, V_t_0_shuffled = refine_dec(interp_enc_out_unshuffled, interp_res_in_shuffled)
                F_t_0_refine_shuffled = delta_F_t_0_shuffled + F_t_0_shuffled
                F_t_0_refine_shuffled = torch.clamp(F_t_0_refine_shuffled, min=-1, max=1)
                F_t_1_refine_shuffled = delta_F_t_1_shuffled + F_t_1_shuffled
                F_t_1_refine_shuffled = torch.clamp(F_t_1_refine_shuffled, min=-1, max=1)
                V_t_1_shuffled = 1 - V_t_0_shuffled
                g_I0_F_t_0_refine_shuffled = flow_warper(I0_shuffled, F_t_0_refine_shuffled)
                g_I1_F_t_1_refine_shuffled = flow_warper(I1_shuffled, F_t_1_refine_shuffled)
                normalization_shuffled = (1 - t) * V_t_0_shuffled + t * V_t_1_shuffled
                interp_image_shuffled = ((1 - t) * V_t_0_shuffled * g_I0_F_t_0_refine_shuffled + t * V_t_1_shuffled * g_I1_F_t_1_refine_shuffled) / normalization_shuffled
                interp_image_shuffled = torch.unsqueeze(interp_image_shuffled, 1)
                if first:
                    predictions_shuffled = interp_image_shuffled
                    first = False
                else:
                    predictions_shuffled = torch.cat((predictions_shuffled, interp_image_shuffled), 1)
        
        return {
            'pred': predictions_post,
            'F_0_1': F_0_1,
            'F_1_0': F_1_0,
            'F_t_0_collector': F_t_0_collector,
            'F_t_1_collector': F_t_1_collector,
            'z_prior_seq': interp_enc_out_prior_seq,
            'z_post_seq': interp_enc_out_post_seq,
            'h_cond': compute_enc_out,
            'pred_shuffled': predictions_shuffled,
            'shuffled_idx': shuffled_idx
        }
