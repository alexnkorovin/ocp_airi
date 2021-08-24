#Getting started

Some workflow description is also on [Habr](https://habr.com/ru/post/342116/) (rus).
### Install
**1. installation git**
```bash
   brew install git (on MacOS)
   cd <desired_root_dir>
   git clone https://github.com/alexnkorovin/ocp-airi
   git status #check status
```
### Workflow
**2. Creating temp branch for the current task.**

In general, it is supposed the routine testing and updating is done within **task or personal branches**.</br>
After testing is completed one should be merged to **dev** branch.</br>
**dev** should be merged to **main** on general releases.</br>
 ```bash
 git checkout dev
 git pull # update to dev branch
 git checkout -b <task or name> # pull to task or personal branch
```
**3. Finishing work**
```bash
   git add <file_name> # use * to select all files in dir
   git commit -m "comment (e.g., changed number of layers in the model X"
   git checkout dev
   git pull
   git merge <task or name>
   git push
   # merging to main !do this only on general release
   git checkout main
   git merge dev
   git push
```
Note: to push make a push to cloud repository (GitHub) via git in console or GitHubAPI (PyCharm, !not GitHubDesktop) </br>
you need to generate token for your github account [here](https://github.com/settings/tokens/) ([read manual](https://docs.github.com/en/github/authenticating-to-github/keeping-your-account-and-data-secure/creating-a-personal-access-token)) </br>

**4.Workaround**
In the case of erroneous commit it is possible to revert:
```bash
   # look on commits (press **_q_** to exit):
   git log --pretty=format:"%H [%cd]: %an - %s" --graph --date=format:%c
   git checkout <hash of commit> # switch to previous state in the case of
```
###GUI
Alterantively, one can use [GitHub Desktop App](https://desktop.github.com/). </br>

## ocp-airi repository file dependencies:
1. Place dataset in sibling folder ocp_dataset folder to **ocp-airi** dataset
Folder structure should be the following:
```bash
 root/ocp-airi
 root/ocp-dataset/data/is2re/
```
**data** folder is obtained by unpacking **tar.gz** file downloaded by the link
as it is noted in **DATASET.md** file:
https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz

Note in Windows it is useful to use the following command:
```bash
tar -xvzf /mnt/c/PATH/TO/TAR-FILE/Desktop/FILE-NAME.tar.gz -C /mnt/c/PATH/TO/DESTINATION/FOLDER
```
2. Create
```bash
 ocp-airi/checkpoints/schnet/ # or correspondent model
```
and place in it file downloaded by the link (details read in MODELS.md):</br>
https://dl.fbaipublicfiles.com/opencatalystproject/models/2021_02/is2re/schnet_10k.pt
