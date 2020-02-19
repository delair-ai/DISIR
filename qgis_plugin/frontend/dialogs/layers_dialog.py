import os

from PyQt5 import QtWidgets, uic

from ..utils import get_layers

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'ui/layers.ui'))


class LayersDialog(QtWidgets.QDialog, FORM_CLASS):
    """Window to select layers based on the current active layers in Qgis (get_layers)"""
    def __init__(self):
        super(LayersDialog, self).__init__()
        self.setupUi(self)
        _, layer_list = get_layers()
        self.listWidget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.listWidget.addItems(layer_list)
