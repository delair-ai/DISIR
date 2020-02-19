


<img src="https://github.com/delair-ai/DISIR/blob/master/imgs/logo-delair.png" alt="drawing" width="200" align="left"/>

<img src="https://github.com/delair-ai/DISIR/blob/master/imgs/logo-onera.png" alt="drawing" width="200"  align="right"/>

<br />

# Overall presentation
This is an implementation of the training code of **DISIR**. 

# Dataset preprocessing
The training dataset should be stored in a folder *MyDataset* organized as follows:
 - a folder named `imgs` containing the RGB images.
 - a folder named `gts` containing the ground-truths.

This code does not handle additional data sources such as DSMs.

:warning: Ground-truth files must have the same names than their associated image.
    
#### Example for ISPRS Potsdam dataset:

```Shell
cd <PotsdamDataset>
sudo apt install rename
cd gts; rename 's/_label//' *; cd ../imgs; rename 's/_RGB//' *
```
The ground-truth maps have to be categorically encoded (i.e. not in a RGB format):
```Shell
cd DISIR/train
python format_ground_truth -n <dataset_name> -d <PathToMyDataset>/gts
```
It is only implemented for INRIA and ISPRS Potsdam datasets but can be easily converted to other datasets.
Note that `format_ground_truth` over-writes the current ground truth maps, it might be wise to make a copy of the initial ones before. 

# Config files

Configuration files are in `train/configs` and must be passed as `--config` argument in the CLIs. 

# Train a model

```Shell
python -m semantic_segmentation.train -d ~/data/Potsdam -c configs/conf_potsdam.yml
```

# Export the graph of a trained model using jit
In order to use a model in the QGIS plugin, it has to be saved as a torch sript before. To do so:
```Shell
python -m semantic_segmentation.export_graph -m data/models/<YourModel>.pth -c configs/conf_potsdam.yml -o <OutputDirectory>
```
