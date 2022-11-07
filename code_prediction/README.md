# Introduction
This supplementary file provides the implementation used for the results in "Conditional Wasserstein Generator (CWG)" accepted for IEEE TPAMI 2022. This code is based on the official Github code of [stochastic video generation with a learned prior](https://arxiv.org/abs/1802.07687) by Emily Denton and Rob Fergus [1].

The pretrained pytorch models can be downloaded from the link in the attached txt file.

## Experiments on KTH action dataset
For downloading and pre-processing the KTH dataset, we modified codes of SVG-LP authors into python script. First download the KTH action recognition dataset by running:
```
sh data/download_kth.sh KTH
```
where KTH is the directory of downloaded data. Next, convert the downloaded .avi files into .png's for the data loader. The installation of [ffmpeg](https://ffmpeg.org/) is required.
```
python3 data/convert_kth.py --dataRoot KTH --imageSize 64
```
The ```--imageSize``` flag specifies the image resolution. 
To train the CWG on 64x64 KTH videos, run:
```
python3 train_cwg.py --dataset kth --image_width  64 --model vgg --g_dim 128 --z_dim 24 --beta 0.00002 --gamma 0.0001 --K 0.8 --n_past 10 --n_future 10 --channels 1 --lr 0.0008 --batch_size 10 --data_root KTH --log_dir logs/will/be/saved/here
```
To train the SVG-LP on 64x64 KTH videos, run:
```
python3 train_svg_lp.py --dataset kth --image_width  64 --model vgg --g_dim 128 --z_dim 24 --beta 0.000001 --n_past 10 --n_future 10 --channels 1 --lr 0.0008 --batch_size 10 --data_root KTH --log_dir logs/will/be/saved/here
```

## Experiments on BAIR robot pushing dataset
We follow downloading and pre-processing steps provided by SVG-LP authors. To download the BAIR robot pushing dataset, run:
```
sh data/download_bair.sh bair
```
This will download the dataset in tfrecord format into the specified directory. To train the pytorch models, we need to first convert the tfrecord data into .png images by running:
```
python3 data/convert_bair.py --data_dir bair
```
This may take some time. Images will be saved in ```bair/processed_data```.
Now we can train the CWG model by running:
```
python3 train_cwg.py --dataset bair --model vgg --g_dim 128 --z_dim 64 --beta 0.001 --gamma 0.001 --K 0.8 --n_past 2 --n_future 10 --channels 3 --batch_size 16 --data_root bair --log_dir logs/will/be/saved/here
```

We can train the SVG-LP model by running:
```
python3 train_svg_lp.py --dataset bair --model vgg --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --batch_size 16 --data_root bair --log_dir logs/will/be/saved/here
```

## Experiments on Towel Pick dataset
For Towel Pick dataset, 
each image file should be saved as following:

```
towel/{data_type}/{name_of_subfolder}/{number_of_video}/{number_of_image}.png
```
for example,
```
towel/train/traj_0_to_256/1/0.png
```

Train the CWG model by running:
```
python3 train_cwg.py --dataset towel --model vgg --g_dim 128 --z_dim 64 --beta 0.001 --gamma 0.001 --K 0.8 --n_past 2 --n_future 10 --channels 3 --batch_size 16 --data_root towel --log_dir logs/will/be/saved/here
```

We can train the SVG-LP model by running:
```
python3 train_svg_lp.py --dataset towel --model vgg --g_dim 128 --z_dim 64 --beta 0.0001 --n_past 2 --n_future 10 --channels 3 --batch_size 16 --data_root towel --log_dir logs/will/be/saved/here
```

## Experiments on KITTI dataset
For KITTI dataset, 
each image file should be saved as following:

```
KITTI/{data_type}/{name_of_subfolder}/{name_of_video}/{number_of_image}.png
```
for example,
```
KITTI/train/city/2011_09_26_drive_0001_sync/0000000008.png
```

Train the CWG model by running:
```
python3 train_cwg.py --dataset kitti --model vgg --g_dim 128 --z_dim 64 --beta 0.0001 --gamma 0.001 --K 0.8 --n_past 5 --n_future 10 --channels 3 --batch_size 16 --niter 200 --data_root KITTI --log_dir logs/will/be/saved/here
```

We can train the SVG-LP model by running:
```
python3 train_svg_lp.py --dataset bair --model kitti --g_dim 128 --z_dim 64 --beta 0.01 --n_past 5 --n_future 10 --channels 3 --batch_size 16 --niter 200 --data_root KITTI --log_dir logs/will/be/saved/here
```

## Generation of future videos with trained models

To generate images with a trained CWG model, run:
```
python3 generate.py --model_path cwg_bair_beta=0.00100-gamma=0.00100-K=0.800-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64.pth --data_root bair --log_dir generated/images/will/save/here --n_past 2 --n_future 28 --nsample 100 --N 256 --batch_size 16
```

To plot test scores of trained models as in the manuscript, run:
```
python3 evaluate_plot.py --data_type test --data_root bair --n_past 2 --n_future 28 --nsample 100 --N 256 --batch_size 16  --model_path cwg_bair_beta=0.00100-gamma=0.00100-K=0.800-n_past=2-n_future=10-lr=0.0020-g_dim=128-z_dim=64.pth --log_dir evaluation/plot/will/be/saved/here
```

## Dependencies
- python 3.7.6
- torch 1.2.0
- pillow 6.2.1
- numpy 1.15.4
- scipy 1.1.0
- torchfile 0.1.0

## Reference
[1] Emily Denton and Rob Fergus. Stochastic video generation with a learned prior. In Proceedings of the International Conference on Machine Learning (ICML), 2018. URL: https://github.com/edenton/svg.
