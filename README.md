# WDANet

This repository contains official implementation for the paper titled "Enhanced Medical Image Segmentation via Wavelet-Deformable Attention Networks"

## 1.Prepare data

- The datasets we used are provided by TransUnet's authors. [Get processed data in this link]
（Synapse/BTCV：https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd和 
ACDC：https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4）。

## 2. Environment

- We recommend an evironment with python >= 3.8, and then install the following dependencies:

```bash
pip install -r requirements.txt
```

## 3.weights

Put -[pretrained weights]([https://github.com/Beckschen/TransUNet](https://huggingface.co/zxx00/Training_weight))  into folder "pretrained_ckpt/" under the main "WDANet" directory


## 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 24.
- train
 
```bash
python train.py 
```

- test
 
```bash
python test.py 
```

## References
- [TransUNet](https://github.com/Beckschen/TransUNet)  
- [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet)


  


