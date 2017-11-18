# Generative Adversarial Networks with Weight Normalization and ResNet

A GAN implementation based on my paper [*On the Effects of Batch and Weight Normalization in Generative Adversarial Networks*](https://arxiv.org/abs/1704.03971)

The [earlier repo](https://github.com/stormraiser/GAN-weight-norm) was for [the version submitted to NIPS 2017](https://arxiv.org/abs/1704.03971v3) but sadly the paper didn't get in, so we made some improvements and re-submitted to CVPR 2018 (will put that on arXiv soon). One major change was adding some tweaks so that the method could work with a residual network. For this we've changed the architecture substantially, completely switched to pytorch and removed any batchnorm code, thus we decided that we should open a new repository and keep the old one for reference (which also means we will not fix the numerous bugs there).

We are testing on direct training on CelebA-HQ, the result of which will determine how hard we are gonna sell this paper and whether we'll even bother documenting this code. Wait for it!

If you want to try it now:

```
python split_data.py --dataset folder --dataroot /path/to/img_align_celeba --test_num 200

python main.py --dataset folder --dataroot /path/to/img_align_celeba --image_size 160 --crop_size 160 --dis_feature 64 128 256 384 512 --dis_block 1 1 1 1 1 --gen_feature 64 128 256 384 512 --gen_block 1 1 1 1 1 --save_path /some/path
```

*The code is for Python 3*
