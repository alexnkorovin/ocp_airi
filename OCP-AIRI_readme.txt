Getting started

After cloning the ocp-airi repository:
1. place dataset in sibling folder ocp_dataset folder to ocp_airi dataset:
Folder structure should be the following:
root\ 
	ocp-airi/
	ocp-dataset/data/is2re/

where data is obtained by unpacking file:
https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz
file details read in DATASET.md

2.In create ocp-airi/checkpoints/schnet/
and place in it file:
https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/schnet_10k.pt
file details read in MODELS.md