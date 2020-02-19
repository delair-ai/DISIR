"""
Some util functions for Qgis front end
"""
import colorsys
from multiprocessing.connection import Client, Listener

import yaml
from bs4 import BeautifulSoup

from qgis.core import (
    QgsCategorizedSymbolRenderer,
    QgsColorRampShader,
    QgsProject,
    QgsRasterLayer,
    QgsRasterShader,
    QgsRendererCategory,
    QgsSingleBandPseudoColorRenderer,
    QgsSymbol,
    QgsVectorLayer,
)
from qgis.PyQt import QtGui


def qgs_func(function):
    """Decorator for qgis functions which need a `task` argument which is a taskwrapper"""

    def add_task_arg(task, *args, **kwargs):
        return function(*args, **kwargs)

    return add_task_arg


def get_layers():
    """
    Get the different layers currently opened in Qgis. Rename them based on their occurence to manage the case where
    different layers have the same name in Qgis.
    """
    layers = [
        tree_layer.layer()
        for tree_layer in QgsProject.instance().layerTreeRoot().findLayers()
    ]
    layer_list = [i.name() for i in layers]
    count = {i: layer_list.count(i) for i in layer_list if layer_list.count(i) > 1}
    for i, layer in enumerate(layer_list):
        if layer in count.keys() and isinstance(layers[i], QgsRasterLayer):
            bands, height, width = (
                layers[i].bandCount(),
                layers[i].height(),
                layers[i].width(),
            )
            layer_list[
                i
            ] += f" ({count[layer] - 1}); {bands} bands; H*W: {height}*{width}"
            count[layer] -= 1
    return layers, layer_list


def raster_to_file(selected_layers):
    """
    Find the files corresponding to the selected layers in layers_dialog
    """
    selected_layers = [i.text() for i in selected_layers]
    all_layers, layer_list = get_layers()
    layers = [i for (i, j) in zip(all_layers, layer_list) if j in selected_layers]
    files = []
    for layer in layers:
        file = find_file_from_layer(layer)
        files.append(file)
    return files


def find_file_from_layer(layer):
    html_meta = layer.htmlMetadata()
    soup = BeautifulSoup(html_meta, "lxml")
    file = soup.find_all("a")[0].get("href")[len("file://") :]
    return file


def file_in_layers(f):
    layers, _ = get_layers()
    for layer in layers:
        if f == find_file_from_layer(layer):
            return layer


@qgs_func
def client_to_server(data):
    """
    Sends input data to the server and collect modified inputs from the server.
    """
    with open("connexion_setup.yml", "r") as f:
        d = yaml.safe_load(f)
        if data["ssh"]:
            address_server = tuple(d["ssh"]["address_server"])
            address_client = tuple(d["ssh"]["address_client"])
        else:
            address_server = tuple(d["local"]["address_server"])
            address_client = tuple(d["local"]["address_client"])
    conn_client = Client(address_server, authkey=b"Who you gonna call?")
    conn_client.send(data)
    conn_client.close()

    listener = Listener(address_client, authkey=b"ghostbusters")  # , args=[inst])
    msg = Exception("Error receiving data from the server.")
    conn = listener.accept()
    try:
        msg = conn.recv()
    finally:
        conn.close()
        listener.close()
        if isinstance(msg, Exception):
            raise msg
        return msg


def random_colors(n, bright=True, first_transparent=False):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    opacity = [255 for i in range(n)]
    if first_transparent:
        opacity[0] = 0
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    colors = [tuple([int(j * 255) for j in i] + [j]) for (i, j) in zip(colors, opacity)]
    return colors


def set_renderer_vector(layer, n_classes):
    if not isinstance(layer, QgsVectorLayer):
        raise TypeError(
            f"Wrong kind of layer. Input: {type(layer)}. Expected QgsVectorLayer"
        )
    categories = []
    colors = random_colors(n_classes, False)
    for i in range(n_classes):
        symbol = QgsSymbol.defaultSymbol(layer.geometryType())
        symbol.setColor(QtGui.QColor(*colors[i]))
        symbol.setOpacity(0.4)
        category = QgsRendererCategory(i, symbol, str(i))
        categories.append(category)

    renderer = QgsCategorizedSymbolRenderer("class", categories)
    layer.setRenderer(renderer)


def set_renderer_raster(layer, n_classes):
    colors = random_colors(n_classes, False, True)
    color_ramp = QgsColorRampShader()
    color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
    color_list = []
    for i in range(n_classes):
        color_list.append(
            QgsColorRampShader.ColorRampItem(int(i), QtGui.QColor(*colors[i]), str(i))
        )
    color_ramp.setColorRampItemList(color_list)

    raster_shader = QgsRasterShader()
    raster_shader.setRasterShaderFunction(color_ramp)

    new_renderer = QgsSingleBandPseudoColorRenderer(
        layer.dataProvider(), 1, raster_shader
    )

    layer.setRenderer(new_renderer)
    layer.renderer().setOpacity(0.5)
    layer.triggerRepaint()


class WarnQgs:
    """Warning class used to pass Exceptions to Qgis without interrupting the daemon"""

    def __init__(self, msg, iface=None):
        self.msg = msg
        self.iface = iface

    def __str__(self):
        return "\033[91m" + self.msg + "\033[0m"

    def __call__(self, level):
        """ Print self.msg on qgis interface.
        ----------
        Parameters:
            level: int Qgis level, eg Qgis.Critical, Qgis.Info, Qgis.Warning, ..."""
        if self.iface:
            self.iface.messageBar().pushMessage(str(self.msg), level=level)
        else:
            raise TypeError("QgsWarn is not callable until iface has not been set")
