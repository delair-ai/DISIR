from PyQt5.QtGui import QCursor, QPixmap
from PyQt5.QtWidgets import QApplication

from qgis.core import Qgis, QgsFeature, QgsGeometry, QgsPointXY
from qgis.gui import QgsMapToolEmitPoint

from .dialogs import ClassDialog


class SetPoints(QgsMapToolEmitPoint):
    def __init__(self, iface, layer):
        QgsMapToolEmitPoint.__init__(self, iface.mapCanvas())
        self.iface = iface
        self.layer = layer
        self.id_class_left = None
        self.id_class_right = None
        self.dialog = ClassDialog()
        self.cursor = QCursor(QPixmap("frontend/icons/cursor.png"), 1, 1)

    def activate(self):
        self.iface.mapCanvas().setCursor(self.cursor)

    def canvasReleaseEvent(self, event):
        if self.id_class_left is None or self.id_class_right is None:
            self.iface.messageBar().pushMessage(
                "Set a class before annotating.", level=Qgis.Warning
            )
        else:
            # left click: 1; right click: 2
            feat = QgsFeature(self.layer.fields())
            if event.button() == 1:
                feat.setAttribute("class", self.id_class_left)
            elif event.button() == 2:
                feat.setAttribute("class", self.id_class_right)
            point_ = self.toMapCoordinates(event.pos())
            point = QgsPointXY(point_.x(), point_.y())
            feat.setGeometry(QgsGeometry.fromPointXY(point))
            self.layer.dataProvider().addFeatures([feat])
            # reload new annotation
            # self.layer.set(
            #     self.layer.source(), self.layer.name(), self.layer.providerType()
            # )
            self.iface.mapCanvas().refreshAllLayers()
            QApplication.instance().restoreOverrideCursor()

    def set_class(self):
        self.dialog.show()
        result = self.dialog.exec_()
        if result:
            self.id_class_left = int(self.dialog.lineEdit_left.text())
            self.id_class_right = int(self.dialog.lineEdit_right.text())
        return result
