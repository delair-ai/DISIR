# -*- coding: utf-8 -*-
"""
Copy coordinates of a bounding box in qgis
Inspired from: https://github.com/nextgis/copy_coords
also see: https://docs.qgis.org/testing/en/docs/pyqgis_developer_cookbook/canvas.html
"""

from PyQt5.QtCore import Qt

from qgis.core import QgsPointXY
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand


class CoordsTool(QgsMapToolEmitPoint):
    def __init__(self, canvas, dialog=None):
        QgsMapToolEmitPoint.__init__(self, canvas)

        self.canvas = canvas
        self.dialog = dialog
        if dialog is not None:
            self.dialog.hide()
        self.rubberBand = QgsRubberBand(canvas, True)
        self.rubberBand.setColor(Qt.red)
        self.rubberBand.setFillColor(Qt.red)
        self.rubberBand.setWidth(1)

        self.start_point = None
        self.end_point = None
        self.is_activated = False

    def canvasReleaseEvent(self, event):
        """Release left click => reopen dialog window and set end point with the bb coordinates"""
        self.end_point = self.toMapCoordinates(event.pos())

        self.is_activated = False
        self.rubberBand.reset()
        if self.dialog is not None:
            self.dialog.show()

    def canvasPressEvent(self, event):
        self.start_point = self.toMapCoordinates(event.pos())
        self.is_activated = True

    def canvasMoveEvent(self, event):
        if self.is_activated:
            self.end_point = self.toMapCoordinates(event.pos())
            self.show_rect()

    def show_rect(self):
        self.rubberBand.reset()
        point1 = QgsPointXY(self.start_point.x(), self.start_point.y())
        point2 = QgsPointXY(self.start_point.x(), self.end_point.y())
        point3 = QgsPointXY(self.end_point.x(), self.end_point.y())
        point4 = QgsPointXY(self.end_point.x(), self.start_point.y())

        self.rubberBand.addPoint(point1, False)
        self.rubberBand.addPoint(point2, False)
        self.rubberBand.addPoint(point3, False)
        self.rubberBand.addPoint(point4, True)  # true to update canvas
        self.rubberBand.show()

    def get_data(self):
        return [
            (self.start_point.x(), self.start_point.y()),
            (self.end_point.x(), self.end_point.y()),
        ]

    def set_data(self, data):
        self.start_point = QgsPointXY(*data[0])
        self.end_point = QgsPointXY(*data[1])
