
<img src="https://github.com/delair-ai/DISIR/blob/master/imgs/logo-delair.png" alt="drawing" width="200" align="left"/>

<img src="https://github.com/delair-ai/DISIR/blob/master/imgs/logo-onera.png" alt="drawing" width="200"  align="right"/>

<br />

# Presentation
This repository contains the code of **DISIR**: Deep Image Segmentation with Interactive Refinement. In a nutshell, it consists in neural networks trained to perform semantic segmentation with human guidance. You may refer to our [paper](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2020/877/2020/isprs-annals-V-2-2020-877-2020.pdf) for detailed explanations.

 This repository is divided into two parts:
 - `train` which contains the training code of the networks ([README](./train/README.md))
 - `qgs_plugin` which contains the code of the QGIS plugin used to perform the interactive segmentation ([README](./qgis_plugin/README.md))

# Install Python dependencies

```
conda create -n disir python=3.7 rtree gdal=2.4 opencv scipy shapely -c 'conda-forge' 
conda activate disir
pip install -r requirements.txt
```

 # To use
 Please note that this repository has been tested on Ubuntu 18.4, QGIS 3.8 and python 3.7 only.

1. Download a segmentation dataset such as [ISPRS Potsdam](http://www2.isprs.org/commissions/comm3/wg4/data-request-form2.html) or [INRIA dataset](https://project.inria.fr/aerialimagelabeling/download/).
2. Prepare this dataset according to `Dataset preprocessing` in `train/README.md`.
3. Train a model and convert it to a torch script still following `train/README.md`.
4. Install the [QGIS](https://www.qgis.org/en/site/) plugin following `qgs_plugin/README.md`.
5. Follow `How to start` in `qgs_plugin/README.md` and start segmenting your data !

 # References

If you use this work for your projects, please take the time to cite our [ISPRS Congress conference paper](https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2020/877/2020/isprs-annals-V-2-2020-877-2020.pdf):

```
@Article{isprs-annals-V-2-2020-877-2020,
AUTHOR = {Lenczner, G. and Le Saux, B. and Luminari, N. and Chan-Hon-Tong, A. and Le Besnerais, G.},
TITLE = {DISIR: DEEP IMAGE SEGMENTATION WITH INTERACTIVE REFINEMENT},
JOURNAL = {ISPRS Annals of Photogrammetry, Remote Sensing and Spatial Information Sciences},
VOLUME = {V-2-2020},
YEAR = {2020},
PAGES = {877--884},
URL = {https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/V-2-2020/877/2020/},
DOI = {10.5194/isprs-annals-V-2-2020-877-2020}
}
```

 
 # Licence

Code is released under the MIT license for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

See [LICENSE](./LICENSE) for more details.

# Authors

See [AUTHORS.md](./AUTHORS.md)

# Acknowledgements

This work has been jointly conducted at [Delair](https://delair.aero/)  and [ONERA-DTIS](https://www.onera.fr/en/dtis).