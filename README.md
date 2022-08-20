# PTGAN for test
We add GAN into this ReID model (AICITY2021_Track2_DMT),and our model is vehicle pose transform by generative adversarial network for ReID.
AICITY2021_Track2_DMT is the 1st place solution of track2 (Vehicle Re-Identification) in the NVIDIA AI City Challenge at CVPR 2021 Workshop. 

## Get Started

1. `cd` to folder where you want to download this repo


```bash
├── AIC21
│   ├── AIC21_Track2_ReID
│   │   ├── cropped_aic_test
│   │   ├── cropped_train
│   │   ├── image_query
│   │   ├── image_test
│   │   ├── image_train
│   │   ├── tools
│   │   └── training_part_seg
│   ├── AIC21_Track2_ReID_Simulation
│   │   ├── sys_image_train
│   │   └── sys_image_train_tr
│   └── veri_pose
│       ├── image_query
│       ├── image_test
│       ├── image_train
│       ├── query
│       ├── test
│       └── train
└── PTGAN_test
    ├── config
    │   ├── stage1
    │   └── stage2
    ├── gallery_features
    ├── gan
    │   └── weights
    ├── logs
    │   └── stage2
    ├── pretrained
    ├── reid_model
    │   ├── backbones
    │   ├── checkpoints
    │   └── layers
    ├── utils
    └── ...
```

2. Run `git clone `

3. Install dependencies: `pip install requirements.txt`

   We use cuda 11.0/python 3.7/torch 1.6.0/torchvision 0.7.0 for training and testing.

## 單張圖片ReID模型使用方式
GPU: 1080ti

We use cuda 11.0/python 3.7/torch 1.6.0/torchvision 0.7.0 for training and testing.

Install dependencies: `pip install requirements.txt`

目前使用的 query, gallery 圖片檔皆從 veri_pose 資料夾中取得

我們寫的dataset只會載入一張 query 圖片進行 ReID

1. 執行 UI_windows.py (會有8個模型需要測試，但僅先測 transreid 就可以)

```bash
python UI_windows.py --config_file
config/stage2/transreid_256_veri_gan.yml
MODEL.DEVICE_ID
"('0')"
TEST.WEIGHT
'./reid_model/checkpoints/stage2/transreid_256/v1/transformer_2.pth'
OUTPUT_DIR
'./logs/stage2/transreid_256/v1/veri_for_test'
```

2. Select query image, enter the rank-k number, then push the "set and run" button. 
Waiting for the load bar. 
The results will show in the left side of window.

Detailed code
process_for_test_CCK.py
```bash
# this place will return progressive distance matrix and gallery img name
distmats = compute_distmat(feat, query_data, gen_gallery, ori_gallery, evaluator, cfg, gen_P, gen_neg_vec, P, neg_vec)
```

3. after executing 執行之後會出現 sililiar_img_distmat.csv ，裡面存了圖片路徑和相對應圖片的距離 (值越大越不像，越小越像)

column: query image path

row: gallery image path




## GitHub Introduction (可以不用看)

**ReID Model Backbone** 
AICITY2021_Track2_DMT: Detailed information of NVIDIA AI City Challenge 2021 can be found [here](https://www.aicitychallenge.org/).

The code is modified from [AICITY2020_DMT_VehicleReID](https://github.com/heshuting555/AICITY2020_DMT_VehicleReID), [TransReID]( https://github.com/heshuting555/TransReID ), [reid_strong baseline]( https://github.com/michuanhaohao/reid-strong-baseline ),and [AICITY2021_Track2_DMT](https://github.com/michuanhaohao/AICITY2021_Track2_DMT).


## Get Started

1. `cd` to folder where you want to download this repo

2. Run `git clone `

3. Install dependencies: `pip install requirements.txt`

   We use cuda 11.0/python 3.7/torch 1.6.0/torchvision 0.7.0 for training and testing.

4. Prepare Datasets
		Download Original dataset, [Cropped_dataset](https://drive.google.com/file/d/1bxNjs_KZ_ocnhpsZmdMsIut93z8CqgBN/view?usp=sharing), [SPGAN_dataset](https://drive.google.com/file/d/1nPOTrK9WUEK38mwei9yAOCMlNiF1UJXV/view?usp=sharing), [Veri-776](https://github.com/JDAI-CV/VeRidataset),and [veri_pose](https://github.com/Zhongdao/VehicleReIDKeyPointData).
	
	The name of veri_pose folder is xxx_y_z (ex:000_0_3).
	-  xxx: identity
	-  y: color
	-  z: type
	
	and the numbers of 1~8 are the pose of vehicles.

```bash

├── AIC21/
│   ├── AIC21_Track2_ReID/
│   	├── image_train/
│   	├── image_test/
│   	├── image_query/
│   	├── train_label.xml
│   	├── ...
│   	├── training_part_seg/
│   	    ├── cropped_patch/
│   	├── cropped_aic_test
│   	    ├── image_test/
│   	    ├── image_query/		
│   ├── AIC21_Track2_ReID_Simulation/
│   	├── sys_image_train/
│   	├── sys_image_train_tr/
│   ├── veri_pose/
│   	├── train/
│   	    ├── 000_0_3/
│   	    	├── 0/
│   	    	├── 3/
│   	    	├── 4/
│   	    	├── ...
│   	    ├── ...
│   	├── query/
│   	    ├── 0002_c002_00030600_0.jpg
│   	    ├── ...
│   	├── test/
│   	    ├── 000_0_3/
│   	    	├── 1/
│   	    	├── 3/
│   	    	├── 4/
│   	    	├── ...
│   	    ├── ...
│   	├── train_label.xml
│   	├── query_label.xml
│   	├── test_label.xml
│   	├── ...
```

5. Put pre-trained models into ./pretrained/
	-  resnet101_ibn_a-59ea0ac6.pth, densenet169_ibn_a-9f32c161.pth, resnext101_ibn_a-6ace051d.pth and se_resnet101_ibn_a-fabed4e2.pth can be downloaded from [IBN-Net](https://github.com/XingangPan/IBN-Net)
	-  resnest101-22405ba7.pth can be downloaded from [ResNest](https://github.com/zhanghang1989/ResNeSt)
	-  jx_vit_base_p16_224-80ecf9dd.pth can be downloaded from [here](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth)

## GAN Model Training
We utilize 1 GPU 1080ti(11GB) for training.

1. `cd` to folder where you want to download this repo.

2. Run `git clone `, we use the code to train GAN models.

3. Prepare Datasets
		Download Original dataset, [Veri-776](https://github.com/JDAI-CV/VeRidataset),and [veri_pose](https://github.com/Zhongdao/VehicleReIDKeyPointData).
```bash
├── veri_pose/
│   	├── train/
│   	    ├── 000_0_3/
│   	    	├── 0/
│   	    	├── 3/
│   	    	├── 4/
│   	    	├── ...
│   	    ├── ...
│   	├── query/
│   	    ├── 0002_c002_00030600_0.jpg
│   	    ├── ...
│   	├── test/
│   	    ├── 000_0_3/
│   	    	├── 1/
│   	    	├── 3/
│   	    	├── 4/
│   	    	├── ...
│   	    ├── ...
│   	├── train_label.xml
│   	├── query_label.xml
│   	├── test_label.xml
│   	├── list_color.txt
│   	├── list_type.txt
│   	├── ...
```
4. Start training

Open Visdom to monitor training process

```bash
python -m visdom.server -port [xxxx]
```

Training Orthogonal Encoder

```bash
python orthogonal_encoder.py
```

Training GAN model

```bash
python train.py --stage 1 --dataset train --dataroot [path]
python train.py --stage 2 --dataset train --dataroot [path]
```

5. Put trained models into 
    1. ./code_GAN/gan/weights/
        - model_130.pth
    2. ./code_GAN/weights/GAN_stage_2
        - 17_net_Di.pth
        - 17_net_Dp.pth
        - 17_net_E.pth
        - 17_net_G.pth

```bash
├── PTGAN/
│   ├── code_GAN/
│   	├── gan/
│   	    ├── weights/
│   	        ├── model_130.pth
│   	├── weights/
│   	    ├── GAN_stage_2/
│   	        ├── 17_net_Di.pth
│   	        ├── 17_net_Dp.pth
│   	        ├── 17_net_E.pth
│   	        ├── 17_net_G.pth
│   	        ├── 17_net_Dp.pth
│   ├── ...
```


## ReID Model Training

We utilize 8 GPU 1080ti(11GB) for training. You can train one backbone as follow. 

```bash
# ResNext101-IBN-a
python train.py --config_file configs/stage1/resnext101a_384.yml MODEL.DEVICE_ID "('0')"
python train_stage2_v1.py --config_file configs/stage2/resnext101a_384.yml MODEL.DEVICE_ID "('0')" OUTPUT_DIR './logs/stage2/resnext101a_384/v1'
python train_stage2_v2.py --config_file configs/stage2/resnext101a_384.yml MODEL.DEVICE_ID "('0')" OUTPUT_DIR './logs/stage2/resnext101a_384/v2'
```

You should train camera and viewpoint models before the inference stage. You also can directly use our trained results (track_cam_rk.npy and track_view_rk.npy):

```bash
python train_cam.py --config_file configs/camera_view/camera_101a.yml
python train_view.py --config_file configs/camera_view/view_101a.yml
```

You can train all eight backbones by checking ***run.sh***. Then, you can ensemble all results:

```bash
python ensemble.py
```

All ReID trained models can be downloaded from [here](https://drive.google.com/drive/folders/1aCQmTbYQE-mq-07q86NIMLLZRc82mc5t?usp=sharing)


## Test

After Training all the above models, you can test one backbone as follow.

```bash
python PTGAN.py --config_file configs/stage2/resnext101a_384_veri_gan.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/resnext101a_384/v1/resnext101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/resnext101a_384/veri_gan_v1'
python PTGAN.py --config_file configs/stage2/resnext101a_384_veri_gan.yml MODEL.DEVICE_ID "('0')" TEST.WEIGHT './logs/stage2/resnext101a_384/v2/resnext101_ibn_a_2.pth' OUTPUT_DIR './logs/stage2/resnext101a_384/veri_gan_v2'
```

## VeRi-776 Result

We train the model on AICITY2021 dataset, and test on VeRi-776 by stage2/v1 ReID model.
(will be updated other results in the future)

**Baseline**
| Backbones          | mAP  | R-1  | R-5  | R-10 |
| ------------------ | ---- | ---- | ---- | ---- |
| ResNet101-IBN-a    | 48.8 | 86.1 | 86.4 | 91.5 |
| ResNet101-IBN-a+   | 48.7 | 87.0 | 87.3 | 91.8 |
| ResNet101-IBN-a ++ | 48.6 | 87.1 | 87.4 | 92.3 |
| ResNext101-IBN-a   | 48.1 | 86.1 | 86.4 | 91.1 |
| ResNest101         | 48.2 | 85.9 | 86.3 | 91.0 |
| SeResNet101-IBN    | 46.5 | 85.0 | 85.2 | 90.2 |
| DenseNet169-IBN    | 46.5 | 84.8 | 85.2 | 90.2 |
| TransReID          | 48.9 | 87.0 | 87.4 | 91.3 |

**Our Model**
Transform to gallery pose
| Backbones          | mAP  | R-1  | R-5  | R-10 |
| ------------------ | ---- | ---- | ---- | ---- |
| ResNet101-IBN-a    | 47.4 | 85.7 | 88.9 | 93.1 |
| ResNet101-IBN-a+   | 47.2 | 86.2 | 90.0 | 93.7 |
| ResNet101-IBN-a ++ | 47.5 | 86.7 | 90.6 | 94.0 |
| ResNext101-IBN-a   | 46.9 | 85.5 | 89.3 | 92.6 |
| ResNest101         | 46.3 | 84.9 | 90.2 | 94.3 |
| SeResNet101-IBN    | 47.0 | 84.9 | 90.2 | 91.7 |
| DenseNet169-IBN    | 45.1 | 84.2 | 87.9 | 92.1 |
| TransReID          | 47.0 | 85.6 | 91.1 | 94.0 |

Transform all photo
| Backbones          | mAP  | R-1  | R-5  | R-10 |
| ------------------ | ---- | ---- | ---- | ---- |
| ResNet101-IBN-a    | 47.1 | 86.1 | 88.9 | 92.3 |
| ResNet101-IBN-a+   | 46.9 | 86.2 | 89.7 | 92.8 |
| ResNet101-IBN-a ++ | 47.1 | 86.5 | 89.6 | 93.6 |
| ResNext101-IBN-a   | 46.5 | 85.8 | 88.9 | 92.3 |
| ResNest101         | 46.7 | 85.1 | 88.3 | 92.3 |
| SeResNet101-IBN    | 46.6 | 85.0 | 87.6 | 91.7 |
| DenseNet169-IBN    | 44.8 | 84.6 | 87.8 | 92.1 |
| TransReID          | 50.3 | 87.7 | 90.9 | 93.6 |

## Citation

If you find this work useful in your research, please consider citing:
```
@inproceedings{luo2021empirical,
 title={An Empirical Study of Vehicle Re-Identification on the AI City Challenge},
 author={Luo, Hao and Chen, Weihua and Xu Xianzhe and Gu Jianyang and Zhang, Yuqi and Chong Liu and Jiang Qiyi and He, Shuting and Wang, Fan and Li, Hao},
 booktitle={Proc. CVPR Workshops},
 year={2021}
}
```
