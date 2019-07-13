# iNaturalist 2019 at FGVC6
Repository for Fine Grain Visual Classification competition on Kaggle.
This year competition features a smaller number of highly similar categories captured in a wide variety of situations, from all over the world. In total, the iNat Challenge 2019 dataset contains 1,010 species, with a combined training and validation set of 268,243 images that have been collected and verified by multiple users from iNaturalist.

See ** Data ** for more details regarding the details of Dataset

## Model

Network   - DenseNet-201.
Optimizer - Adam 


## Techniques Used 
The dataset was highly unbalanced (each class contain ~ 16-500 images). Weighted Random Sampling is used to deal with class imbalance.

Fine-grain Classification tries to differentiate between hard-to-distinguish object classes such as birds, flowers or animals. Thus, regularization plays a major role in attaining the desired accuracy. As regularization methods, I have used : 
- [Cutout](https://arxiv.org/abs/1708.04552)
- [Mixup](https://arxiv.org/pdf/1710.09412.pdf)
 
Apart from this to achieve faster convergence, a technique called [SuperConvergence](https://arxiv.org/abs/1708.07120) is used.

## Results
Top-1 accuracy of 65.6% (error rate - 0.344) was achieved.
