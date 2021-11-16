# The Triple Robust Deepfake Detection Model 
This is project code for [SKT AI Fellowship 3](https://www.sktaifellowship.com/) (2021)

---
#### Research Topic: Development of GAN-generated fake image detection technology
We aim to develop robust Deepfake detection model under the unseen environments as belows:
- Deepfake method shift
- Adversarial attack
- Physical condition shift
 
<p align="center">
  <img src="https://user-images.githubusercontent.com/43346577/142081393-7384e63c-048f-484f-a831-f2dc2cdc15ed.png">
</p>

---

## Environment settings
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch 
pip install matplotlib scikit-learn opencv-python albumentations tensorboard
```

## Train model
```
python scripts/train_model.py <GPU index> <model name> <train dataset> <flag> --<other options>
ex) python scripts/train_model.py 0,1,2,3 EfficientNet-B4 FaceForensics,UDAFV,CelebDF mymodel --img_size 299 
ex) python scripts/train_model.py 0 EfficientNet-B4 CelebDF mymodel --phase_concat --oudefend --self_supervised
```

## Test model
```
# single model
python scripts/train_model.py <GPU index> <model name> <seen dataset> <checkpoint path> --<other options>
python scripts/train_model.py 0 EfficientNet-B4 CelebDF ./checkpoints/mymodel/best_ckpt --phase_concat --oudefend --self_supervised

# ensenble model
ex) python scripts/test_model.py <GPU index> <None> <seen dataset> <None> --<other options> --ensemble 
--ensemble_ckpt_list  <checkpoint path 1> <checkpoint path 2> <checkpoint path 3> 
--ensemble_opt_list <option1_1:value/option1_2:value> <option2_1:value/option2_2:value/option2_3:value> <option3_1:value>
```

---

## Available Models
- Xception
- Xception (SPSL)
- Meso4, MesoInception4
- EfficientNet
- MobileNet-v2
- MobileNet-v3
- ResNet-50
- WideResNet-50

## Datasets
- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [UADFV](https://github.com/yuezunli/WIFS2018_In_Ictu_Oculi)
- [DeepfakeTIMIT](https://www.idiap.ch/dataset/deepfaketimit)
- [Celeb-DFv2](https://github.com/yuezunli/celeb-deepfakeforensics)

---

## Reference
Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain (CVPR 2021)  
Adversarial Perturbations Fool Deepfake Detectors (IJCNN 2020)  
Security of Facial Forensics Models Against Adversarial Attacks (ICIP 2020)  
A Simple Framework for Contrastive Learning of Visual Representations (ICML 2020)  
DeepfakeUCL: Deepfake Detection via Unsupervised Contrastive Learning (arXiv 2021.04)  
The Deepfake Detection Challenge Dataset (arXiv 2020.06)  
FaceForensics++: Learning to Detect Manipulated Facial Images (ICCV 2019)  
Exposing Deep Fakes Using Inconsistent Head Poses (IEEE ICASSP 2019)  
DeepFakes: a New Threat to Face Recognition? Assessment and Detection (arXiv 2018.12)  
Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics (CVPR 2020)
