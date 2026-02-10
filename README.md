## SD-IDD: Selective Distillation for Incremental Defect Detec-tion



###  Introduction
The surface defects of industrial production are complex and diverse. Hence, trained de-fect detection models based on deep learning must consistently adapt to newly emerging defect categories. The trained models generally suffer from catastrophic forgetting as it learns new defect categories. To address this issue, we proposes a selective distillation for incremental defect detection (SD-IDD) model based on GFLv1. Specifically, three selective distillation strategies are proposed, including high-confidence classification distillation, dual-stage cascade regression distillation, and Intersection over Union (IoU)-driven diffi-culty-aware feature distillation. The high-confidence classification distillation aims to preserve critical discriminative knowledge of old categories within semantic confusion re-gions of the classification head, reducing interference from low-value regions. Dual-stage cascade regression distillation focuses on high-quality anchors through geometric prior coarse filtering and statistical fine filtering, utilizing IoU-weighted KL divergence distilla-tion loss to accurately transfer localization knowledge. IoU-driven difficulty-aware feature distillation adaptively allocates distillation resources, prioritizing features of high-difficulty targets. These selective distillation strategies significantly mitigate cata-strophic forgetting while enhancing the detection accuracy of new classes, without relying on old samples. Experimental results demonstrate that SD-IDD respectively achieves mAP_old values of 58.2% and 99.3%, mAP_new values of 69% and 97.3%, and mAP_all values of 63.6% and 98.3% on the NEU-DET datasets and DeepPCB datasets, outperform-ing existing incremental detection methods. 

<p align='left'>
  <img src='figs/framework.jpg' width='721'/>
</p>

### ====== 2023.07.04 Updated  ======
### Migrate code to the following newer environment
- Python 3.8
- PyTorch 1.13.1
- CUDA 11.6
- [mmdetection](https://github.com/open-mmlab/mmdetection) 3.0.0
- [mmcv](https://github.com/open-mmlab/mmcv) 2.0.0

### Get Started

This repo is based on [MMDetection 3.0](https://github.com/open-mmlab/mmdetection). Please refer to [GETTING_STARTED.md](https://mmdetection.readthedocs.io/en/v3.0.0/get_started.html) for the basic configuration and usage of MMDetection.
Or follow the steps below to install

```python
conda create -n ERD python=3.8 -y

source activate ERD

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install tqdm

pip install -U openmim

mim install mmengine==0.7.3

mim install mmcv==2.0.0

# cd erd 
pip install -v -e .
```

You can run /script/select_categories.py to split the COCO dataset as you want,

```python
# to generate instances_train2017_sel_last_40_cats.json
python ./script/select_categories.py

```


### Train
```python
# assume that you are under the root directory of this project,
# and you have activated your virtual environment if needed.
# and with COCO dataset in '/dataset/coco/'

# train first 40 cats
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/gfl_increment/gfl_r50_fpn_1x_coco_first_40_cats.py 2 --work-dir=../ERD_results/gfl_increment/gfl_r50_fpn_1x_coco_first_40_cats
#train last 40 cats incrementally
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh configs/gfl_increment/gfl_r50_fpn_1x_coco_first_40_incre_last_40_cats.py 2 --work-dir=../ERD_results/gfl_increment/gfl_r50_fpn_1x_coco_first_40_incre_last_40_cats
```

### Test
```python
# test first 40 cats
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh configs/gfl_increment/gfl_r50_fpn_1x_coco_first_40_cats.py ../ERD_results/gfl_increment/gfl_r50_fpn_1x_coco_first_40_cats/epoch_12.pth 2 --cfg-options test_evaluator.classwise=True
#test all 80 cats on the incre model
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_test.sh configs/gfl_increment/gfl_r50_fpn_1x_coco_first_40_incre_last_40_cats.py ../ERD_results/gfl_increment/gfl_r50_fpn_1x_coco_first_40_incre_last_40_cats/epoch_12.pth 2 --cfg-options test_evaluator.classwise=True

```

### Citation
Please cite the following paper if this repo helps your research:
```bibtex
@InProceedings{ERD,
    author    = {Tao Feng and Mang Wang and Hangjie Yuan},
    title     = {Overcoming Catastrophic Forgetting in Incremental Object Detection via Elastic Response Distillation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2022}
}
```
