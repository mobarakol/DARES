# DARES

This is the official PyTorch implementation for training and testing depth estimation models using the method described in

> **DARES: Depth Anything in Robotic Endoscopic Surgery with Self-supervised Vector-LoRA of the Foundation Model**
>
> Mona Sheikh Zeinoddin, Chiara Lena, Jiongqi Qu, Luca Carlini, Mattia
Magro, Seunghoi Kim,  Elena De Momi,  Sophia Bano, Matthew
Grech-Sollars, Evangelos Mazomenos,  Daniel C. Alexander, Danail
Stoyanov, Matthew J. Clarkson and Mobarakol Islam 
>
> [accepted by the European Conference on Computer Vision 2024 Efficient Deep Learning for Foundation Models WorkShop (arXiv pdf)](https://arxiv.org/pdf/2408.17433)


#### Overview

<p align="center">
<img src='imgs/overview.png' width=800/> 
</p>

## ✏️ 📄 Citation

If you find our work useful in your research please consider citing our paper:

```
@article{zeinoddin2024dares,
  title={DARES: Depth Anything in Robotic Endoscopic Surgery with Self-supervised Vector-LoRA of the Foundation Model},
  author={Zeinoddin, Mona Sheikh and Lena, Chiara and Qu, Jiongqi and Carlini, Luca and Magro, Mattia and Kim, Seunghoi and De Momi, Elena and Bano, Sophia and Grech-Sollars, Matthew and Mazomenos, Evangelos and others},
  journal={arXiv preprint arXiv:2408.17433},
  year={2024}
}
```




## ⚙️ Setup

We ran our experiments with PyTorch 1.2.0, torchvision 0.4.0, CUDA 10.2, Python 3.7.3 and Ubuntu 18.04.



## 🖼️ Prediction for a single image or a folder of images

You can predict scaled disparity for a single image or a folder of images with:

```shell
CUDA_VISIBLE_DEVICES=0 python test_simple.py --model_path <your_model_path> --image_path <your_image_or_folder_path>
```

## Initializing with AF-Sfm Learner weights

You can download AF-Sfm Learners weights that we use in initialization with:

```shell
gdown 1kf7LjQ6a2ACKr6nX5Uyee3of3bXn1xWB
unzip -q Model_trained_end_to_end.zip
mv Model_trained_end_to_end af_sfmlearner_weights
```


## 💾 Datasets

You can download the [Endovis or SCARED dataset](https://endovissub2019-scared.grand-challenge.org) by signing the challenge rules and emailing them to max.allan@intusurg.com, the [EndoSLAM dataset](https://data.mendeley.com/datasets/cd2rtzm23r/1), the [SERV-CT dataset](https://www.ucl.ac.uk/interventional-surgical-sciences/serv-ct), and the [Hamlyn dataset](http://hamlyn.doc.ic.ac.uk/vision/).

**Endovis split**

The train/test/validation split for Endovis dataset used in our works is defined in the `splits/endovis` folder. 

**Endovis data preprocessing**

We use the ffmpeg to convert the RGB.mp4 into images.png:

```shell
find . -name "*.mp4" -print0 | xargs -0 -I {} sh -c 'output_dir=$(dirname "$1"); ffmpeg -i "$1" "$output_dir/%10d.png"' _ {}
```
We only use the left frames in our experiments and please refer to extract_left_frames.py. For dataset 8 and 9, we rephrase keyframes 0-4 as keyframes 1-5.

**Data structure**

The directory of dataset structure is shown as follows:

```
/path/to/endovis_data/
  dataset1/
    keyframe1/
      image_02/
        data/
          0000000001.png
```



## ⏳ Endovis training


```shell
CUDA_VISIBLE_DEVICES=0 python train_end_to_end.py --data_path <your_data_path> --log_dir <path_to_save_model (depth, pose, appearance flow, optical flow)>
```

## 📊 Endovis evaluation

To prepare the ground truth depth maps run:
```shell
CUDA_VISIBLE_DEVICES=0 python export_gt_depth.py --data_path endovis_data --split endovis
```
...assuming that you have placed the endovis dataset in the default location of `./endovis_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
CUDA_VISIBLE_DEVICES=0 python evaluate_depth.py --data_path <your_data_path> --load_weights_folder ~/mono_model/mdp/models/weights_19 --eval_mono
```

#### Depth Estimation

<p align="center">
<img src='imgs/depth.png' width=800/> 
</p>

#### Visual Odometry

<p align="center">
<img src='imgs/pose.png' width=800/> 
</p>

#### 3D Reconstruction

<p align="center">
<img src='imgs/reconstruction.png' width=500/> 
</p>


## Model zoo

| Model        | Abs Rel | Sq Rel | RMSE | RMSE log | Link |
| ------------ | ---------- | ------ | --------- | ---- | ---- |
| End-to-end best model weights | 0.052 | 0.356 | 4.483 | 0.073 |[google](https://drive.google.com/file/d/11C0sw396TcH2hMM7u6uMr-uBsCP4l2Kd/view?usp=sharing)|



## Contact

If you have any questions, please feel free to contact mona.zeinoddin.22@ucl.ac.uk or mobarakol.islam@ucl.ac.uk



## Acknowledgement

Our code is based on the implementation of [AF-Sfm Learner](https://github.com/ShuweiShao/AF-SfMLearner). We thank AF-Sfm Learner's authors for their excellent work and repository.
