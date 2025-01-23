# RepLKan
# Usage：
## Recommended environment:
```python
python 3.12
torch 2.3.0
torchvision0.18.0
```
Please use pip install -r requirements.txt to install the dependencies.
<br>##Data preparation:
<br>&#8226; ACDC dataset: Download the preprocessed ACDC dataset from [Google Drive of MT-UNet](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view)
<br>&#8226; ISIC2018 dataset: Download the ISIC2018 dataset from [ISIC2018 dataset](https://challenge.isic-archive.com/data/#2018)
<br>&#8226; MOSmeddata dataset: Download the Mosmeddata dataset from [Mosmeddata dataset](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset)
<br>##Format Preparation:
<br>Then prepare the datasets in the following format for easy use of the code:
```python
├── data
    ├── train
    │   ├── images
    │   ├── labels
    └── valid
    │   ├── images
    │   ├── labels
    └── test
    │   ├── images
    │── ├── labels
```
<br>##Pretrained model:
<br>Download pretrained model:
<br>[ACDC](https://drive.google.com/file/d/1iuIpjGjefpQlUr7Gz2PQyZ0GVmSTjX9V/view?usp=drive_link)
<br>[ISIC2018](https://drive.google.com/file/d/1i8WHT_hcaDCEv6WfZ1HFaD9xaXaMZ_-q/view?usp=drive_link)
<br>[Mosmeddata](https://drive.google.com/file/d/103Y2f_QlvqczlsBQFjfVde3p5uUHz1KB/view?usp=drive_link)
<br>##Training:
```python
python train.py --dataset ACDC
```
<br>##Testing:
```python
python predict.py --dataset ACDC
```
<br>##Acknowledgement:
We are very grateful for these excellent works [RepLKNet](https://github.com/DingXiaoH/RepLKNet-pytorch), [U-Kan](https://github.com/duttapallabi2907/u-vixlstm), [MAN](https://github.com/icandle/MAN), [DY_smaple](https://github.com/tiny-smart/dysample), [SSRS](https://github.com/sstary/SSRS) and [UNet](https://github.com/milesial/Pytorch-UNet), which have provided the basis for our framework.
