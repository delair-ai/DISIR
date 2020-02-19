<img src="https://github.com/delair-ai/DISIR/blob/master/imgs/logo-delair.png" alt="drawing" width="200" align="left"/>

<img src="https://github.com/delair-ai/DISIR/blob/master/imgs/logo-onera.png" alt="drawing" width="200"  align="right"/>

<br />

# Presentation
This repository contains the code of the QGIS plug-in part of **DISIR**. This is a [QGIS](https://www.qgis.org/en/site/) plugin designed for interactive semantic segmentation of geo-referenced images using deep-learning. 

It is separated into two parts:

1. [frontend](./frontend) deals with the QGIS interface and the interaction with the user
2. [backend](./backend) deals with a background program launched in a terminal outside QGIS to perform the heavy computations (e.g. the semantic segmentation).

# Set up

If the backend is used locally, make sure to be in a python environment with the requirements installed. 
## Install the plugin in QGIS
Compress this repository (`qgis_plugin`) into a zip file and install the plugin with the Qgis plugin manager.

## (Optionnal) Backend in a remote server
This can be useful if the plugin runs on a computer wih a small/no GPU and that a better GPU is accessible remotely. 
 - Connect to the remote server using `ssh`. Check that the local computer hostkey is in the remote server hostkeys: `ssh-keygen -H -F <hostname/IP adress>` returns some key.
    - If not, add it: `ssh-keyscan -H <hostname/IP adress> >> ~/.ssh/known_hosts`.
 - In `connexion_setup.yml`, set the ssh addresses according to your needs: 
    - `address_server` is the IP of the remote server.
    - `address_client` is the IP of the local computer.
    - `username` is your username in the local computer.
- Install the python dependencies in the remote server.


# How to start

 Enable `SSH connexion` if the backend is running in a remote server.
  - Launch the backend in a terminal with `disir` environment activated:
    - Locally: `python -m backend`
    - In a remote server: `python -m backend --ssh`
 - In QGIS:
    - Click on th `IL` button, select your parameters, close it to launch a first inference.
    - Once the first inference is done, click on the red and blue button to select the classes to correct.
    - Click on the misclassified areas
    - Run a new inference by clicking on the `NN` button

:warning: The neural network has to take as input a torch tensor of shape *C* x *H* x *W* where $C$ is sorted as follows: *[R, G, B, annot_channel_0, ..., annot_channel_n]* with *n* the number of classes.

# Tips
Modify the file `DISIR/qgis_plugin/frontend/ui/interact_learn_dialog_base.ui` using [Qt Designer](https://doc.qt.io/qt-5/qtdesigner-manual.html) to set the path of the neural network to your own neural network and to set the outuput path to your output directory.