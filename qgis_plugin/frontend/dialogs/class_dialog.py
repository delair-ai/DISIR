import os

from PyQt5 import QtWidgets, uic
from PyQt5.QtGui import QIntValidator

FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'ui/set_class.ui'))


class ClassDialog(QtWidgets.QDialog, FORM_CLASS):
    """Window to select layers based on the current active layers in Qgis (get_layers)"""
    def __init__(self):
        super(ClassDialog, self).__init__()
        self.setupUi(self)
        self.lineEdit_left.setValidator(QIntValidator())
        self.lineEdit_right.setValidator(QIntValidator())
