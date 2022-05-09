# PSPNet
This is the implementation of the network pyramid Scene Parsing Network https://arxiv.org/abs/1612.01105. This is a deep learning network used for general understanding of the scene. In this implementation we use it for Semantic Segmentation.
Training and Dataset:
Citys dataset should be downloaded. It is available at
https://www.cityscapes-dataset.com/
For a faster convergence, I have trained on the Coarse dataset and then trained on the Fine Labels. This can be done by simply changing the folder names in the cityscpaes.py program in the dataloader. The dataset need to be stored in the working directory of the train.py program.

Models:
The models are stored in the Models folder. The dilated residual network used is from the GitHub repo: https://github.com/fyu/drn. The model are stored in the folder models. And the weights are to be stored in the weights folder in the working directory.

Evaluation:
The evaluation results are stored in the test_results when we run the eval.py program.
