# Mixture of Weights

We realize controllable and dynamic model merging.

<img src='./png/fig_framework.png'>

## Get Started

### Dependencies

Please follow [task_vectors](https://github.com/mlfoundations/task_vectors) to install the dependencies.

Additionally, install transformers.

### Checkpoints 

You can download the fine-tuned checkpoints from the [task_vectors#checkpoints](https://github.com/mlfoundations/task_vectors#checkpoints).
The Google Drive folder is: [task_vectors_checkpoints](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw)

Please follow [doc](./checkpoints/README.md) to place these checkpoints.

### Datasets

Please follow [Adamerging](https://github.com/EnnengYang/AdaMerging?tab=readme-ov-file#datasets) to download the datasets.

Please follow [doc](./data/README.md) to place these datasets.


## Eval

Run AdaMerging++ (Layerwise) w/ MoW-Merging
> python src/main_mow_merge.py


## Results

Results will be saved in [logs](./logs). 


# Acknowledgements
Our implementation references the code below, thanks to them.

- FusionBench: https://github.com/tanganke/fusion_bench

- AdaMerging: https://github.com/EnnengYang/AdaMerging

- Task Arithmetic: https://github.com/mlfoundations/task_vectors

- TIES-MERGING: https://github.com/prateeky2806/ties-merging/tree/main

- Model Soups: https://github.com/mlfoundations/model-soups

