## Introduction
This supplementary file provides the implementation used for the results in "Conditional Wasserstein Generator (CWG)" submitted to IEEE TPAMI 2021. This code is based on the official Github code of [A Temporally-Aware Interpolation Network for Video Frame Inpainting](https://link.springer.com/chapter/10.1007/978-3-030-20893-6_16) by Ximeng Sun, Ryan Szeto and Jason J. Corso. [1].

The pretrained pytorch models can be downloaded from the link in the attached txt file.

## Setup the environment
In this section, we provide an instruction to help you set up the experiment environment. This code was tested for Python 2.7 on Ubuntu 16.04, and is based on PyTorch 0.3.1.

First, please make sure you have installed  `cuda(8.0.61)`, `cudnn(8.0-v7.0.5)` `opencv(4.0.0.21)`, and `ffmpeg(2.2.3)`.  Note that different versions might not work.

Then, to avoid conflicting with your current environment, we suggest to use `virtualenv`. For using `virtualenv`, please refer https://virtualenv.pypa.io/en/stable/. Below is an example of initializing and activating a virtual environment:

```bash
virtualenv .env
source .env/bin/activate
```

After activating a virtual environment, use the following command to install all dependencies you need to run this model.

```bash
pip install --upgrade 'setuptools<45.0.0'
pip install -r requirements.txt
```

### Compiling the separable convolution module

The separable convolution module uses CUDA code that must be compiled before it can be used. To compile, start by activating the virtual environment described above. Then, set the `CUDA_HOME` environment variable as the path to your CUDA installation. For example, if your CUDA installation is at `/usr/local/cuda-8.0`, run this command:

```bash
export CUDA_HOME=/usr/local/cuda-8.0
```

Then, identify a virtual architecture that corresponds to your GPU from [this site](http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list) (e.g. `compute_52` for Maxwell GPUs). Finally, to compile the CUDA code, run the install script with the virtual architecture as an argument, e.g.:

```bash
bash bashes/misc/install.bash compute_52
```


## Download datasets

In our paper, we use the KTH Actions, UCF-101, and HMDB-51 datasets to evaluate our method. Use the commands below to download these datasets. The sample commands below download the datasets into a new `datasets` directory, but you can replace this argument with another location if needed.


```bash
bash bashes/download/download_KTH.bash datasets
bash bashes/download/download_UCF.bash datasets
```

**Note: If you use a location other than `datasets`, change the files in `videolist` to point to the correct video paths. Alternatively, you can create a symlink from `datasets` to your custom folder.**



## Experiments on KTH action dataset

**Note: Make sure to run train and prediction script on Python 2.7 and evaluation script on Python 3.6 : outside the virtual environment installed above. You may have to install pacakges from `requirments_evaluate.txt` outside virtualenv as well.**

### KTH

[Train]
```
python train.py --K=5 --F=5 --T=5 --alt_T=10 --alt_K=7 --alt_F=7 --max_iter=100000 --train_video_list_path=videolist/KTH/train_data_list.txt --val_video_list_path=videolist/KTH/val_data_list_T=5.txt --val_video_list_alt_T_path=videolist/KTH/val_data_list_T=10.txt --val_video_list_alt_K_F_path=videolist/KTH/val_data_list_K=F=7.txt --vis_video_list_path=videolist/KTH/vis_data_list_T=5.txt --vis_video_list_alt_T_path=videolist/KTH/vis_data_list_T=10.txt --vis_video_list_alt_K_F_path=videolist/KTH/vis_data_list_K=F=7.txt --c_dim=1 --image_size=128 --sample_KTF --name=kth_CWG_TAI_beta=2e-1_gamma=0.000_L=1.0 --model_key=WVI_TAI_gray --not_validate_alt --gpu_id 0 1 --batch_size 4 --num_threads 16 --beta 2e-1 --gamma 0.000 --L 1.0 --seed 0 --save_latest_freq 10000
```

[Prediction]
```
python predict.py --K=5 --T=10 --F=5 --test_video_list_path=videolist/KTH/test_data_list_T\=10.txt --c_dim=1 --image_size=128 --batch_size=16 --model_key=WVI_TAI_gray --checkpoints_dir=checkpoints --name kth_CWG_TAI_beta=2e-1_gamma=0.000_L=1.0 --snapshot_file_name=model_best.ckpt --qual_result_root results/KTH-test_data_list_T=10/images/CWG_TAI_beta=2e-1_gamma=0.000_L=1.0
```

[Evaluation]
```
./bashes/experiments/compute_summarize_quant_results.sh results/KTH-test_data_list_T\=10/images/CWG_TAI_beta=2e-1_gamma=0.000_L=1.0/  results/KTH-test_data_list_T\=10/quantitative/CWG_TAI_beta=2e-1_gamma=0.000_L=1.0 5 10
```

## Experiments on UCF101 dataset

### UCF101, CWG-TAI

[Train]
```
python train.py --K=4 --F=4 --T=3 --alt_T=5 --alt_K=6 --alt_F=6 --max_iter=100000 --train_video_list_path=videolist/UCF-101/train_data_list.txt --val_video_list_path=videolist/UCF-101/val_data_list_T=3.txt --val_video_list_alt_T_path=videolist/UCF-101/val_data_list_T=5.txt --val_video_list_alt_K_F_path=videolist/UCF-101/val_data_list_K=F=6.txt --vis_video_list_path=videolist/UCF-101/vis_data_list_T=3.txt --vis_video_list_alt_T_path=videolist/UCF-101/vis_data_list_T=5.txt --vis_video_list_alt_K_F_path=videolist/UCF-101/vis_data_list_K=F=6.txt --c_dim=3 --image_size 160 208 --sample_KTF --name=UCF101_CWG_TAI_beta=2e-1_gamma=0.000_L=1.0 --model_key=WVI_TAI_color --not_validate_alt --gpu_id 0 1 --batch_size 4 --num_threads 16 --beta 2e-1 --gamma 0.000 --L 1.0 --seed 0 --save_latest_freq 10000
```

[Prediction]
```
python predict.py --K=4 --T=5 --F=4 --test_video_list_path=videolist/UCF-101/test_data_list_T\=5.txt --c_dim=3 --image_size 240 320 --batch_size=16 --model_key WVI_TAI_color --checkpoints_dir checkpoints --name UCF101_CWG_TAI_beta=2e-1_gamma=0.000_L=1.0 --snapshot_file_name model_best.ckpt --qual_result_root results/UCF-101-test_data_list_T=5/images/CWG_TAI_beta=2e-1_gamma=0.000_L=1.0
```

[Evaluation]
```
./bashes/experiments/compute_summarize_quant_results.sh results/UCF-101-test_data_list_T\=5/images/CWG_TAI_beta=2e-1_gamma=0.000_L=1.0 results/UCF-101-test_data_list_T\=5/quantitative/CWG_TAI_beta=2e-1_gamma=0.000_L=1.0  4 5
```

### UCF101, CWG-SloMo

[Train]
```
python train.py --K=4 --F=4 --T=3 --alt_T=5 --alt_K=6 --alt_F=6 --max_iter=100000 --train_video_list_path=videolist/UCF-101/train_data_list.txt --val_video_list_path=videolist/UCF-101/val_data_list_T=3.txt --val_video_list_alt_T_path=videolist/UCF-101/val_data_list_T=5.txt --val_video_list_alt_K_F_path=videolist/UCF-101/val_data_list_K=F=6.txt --vis_video_list_path=videolist/UCF-101/vis_data_list_T=3.txt --vis_video_list_alt_T_path=videolist/UCF-101/vis_data_list_T=5.txt --vis_video_list_alt_K_F_path=videolist/UCF-101/vis_data_list_K=F=6.txt --c_dim=3 --image_size 160 192 --sample_KTF --name=UCF101_CWG_SloMo_beta=2e-4_gamma=0.000_L=1.0_decay=1e-5 --model_key=WVI_SloMoFillInModel_color --not_validate_alt --gpu_id 0 1 --batch_size 4 --num_threads 16 --beta 2e-4 --gamma 0.000 --L 1.0 --seed 0 --weight_decay 1e-5  --save_latest_freq 10000
```

[Prediction]
```
python predict.py --K=4 --T=5 --F=4 --test_video_list_path=videolist/UCF-101/test_data_list_T\=5.txt --c_dim=3 --image_size 240 320 --batch_size=16 --model_key=WVI_SloMoFillInModel_color --padding_size 16 0 --checkpoints_dir checkpoints --name UCF101_CWG_SloMo_beta=2e-4_gamma=0.000_L=1.0_decay=1e-5 --snapshot_file_name model_best.ckpt --qual_result_root results/UCF-101-test_data_list_T=5/images/CWG_SloMo_beta=2e-4_gamma=0.000_L=1.0_decay=1e-5
```

[Evaluation]
```
./bashes/experiments/compute_summarize_quant_results.sh results/UCF-101-test_data_list_T\=5/images/CWG_SloMo_beta=2e-4_gamma=0.000_L=1.0_decay=1e-5  results/UCF-101-test_data_list_T\=5/quantitative/CWG_SloMo_beta=2e-4_gamma=0.000_L=1.0_decay=1e-5 4 5
```

*Note: For UCF-101, CWG-SuperSloMo uses different extra argument between training and testing time.*


## Visualize the training process

Losses and visualizations are saved in a TensorBoard file under `tb` for each experiment. In order to view these intermediate results, you can activate a TensorBoard instance with `tb` as the log directory:

```bash
tensorboard --logdir tb
```

## Reference
[1] Sun, Ximeng and Szeto, Ryan and Corso, Jason J. A Temporally-Aware Interpolation Network for Video Frame Inpainting. Asian Conference on Computer Vision(ACCV), 2018. URL: https://github.com/MichiganCOG/video-frame-inpainting/.

