import os
from multiprocessing.connection import Client, Listener
from time import time

import buzzard as buzz
import GPUtil
import numpy as np
import rasterio as rio
import torch
import yaml
from shapely.geometry import box
from torch import jit

from .ssh_connexion import SshConnexion
from .utils import (check_inputs_and_net, find_n_classes, from_coord_to_patch,
                    make_batches, polygonize, print_warning, vec_to_list)


class Daemon:
    """Connect a daemon to a client and execute computer vision tasks in it.
    """

    def __init__(
        self,
        config_file="backend/config.yml",
        ssh=False,
        cache=True,
        connexion_file="connexion_setup.yml",
    ):
        with open(config_file, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.connexion_file = connexion_file
        self.ssh_server = SshConnexion(connexion_file) if ssh else None
        device = self._set_device()
        print(f"\u001b[36mDevice: {device}\033[0m")
        self.device = torch.device(device)
        self.cache = {} if cache else None
        self.original_fp = None
        self.n_classes = None
        self.tiled_fp = None

    def _save_output(self, data, output, nodata_mask, fp):
        input_file = data["input_files"][0]
        output_file = data["output_file"]
        if self.ssh_server:
            input_file = self.ssh_server.tmp_file(input_file)
            output_file = self.ssh_server.tmp_file(output_file)
        with buzz.Dataset().close as ds:
            proj = ds.aopen_raster(input_file).proj4_virtual

            if os.path.isfile(output_file):
                ds.open_raster("output", output_file, mode="w")
                if not ds.output.fp.poly.contains(fp.poly):
                    ds.output.close()
                    os.remove(output_file)
                    ds.create_raster(
                        "output",
                        output_file,
                        self.original_fp,
                        dtype=np.uint8,
                        channel_count=1,
                        sr=proj,
                        channels_schema={"nodata": self.n_classes},
                    )
            else:
                ds.create_raster(
                    "output",
                    output_file,
                    self.original_fp,
                    dtype=np.uint8,
                    channel_count=1,
                    sr=proj,
                    channels_schema={"nodata": self.n_classes},
                )

            nodata_mask = nodata_mask[fp.slice_in(self.original_fp)]
            output[nodata_mask] = self.n_classes

            ds.output.set_data(output.astype(np.uint8), fp, channels=0)
            ds.output.close()
        if self.ssh_server:
            self.ssh_server.put(data["output_file"])
        if data["polygonize"]:
            poylgon_file = (
                data["polygonize"]
                if not self.ssh_server
                else self.ssh_server.tmp_file(data["polygonize"])
            )
            polygonize(output_file, poylgon_file, proj)
            if self.ssh_server:
                self.ssh_server.put(data["polygonize"])

    @staticmethod
    def _set_device(threshold=3000):
        """Set gpu device when cuda is activated based of free available memory. 
        ---------
        Parameters:
            threshold (int): Minimal amount of free memory (Mo) to select this device"""
        if not torch.cuda.is_available():
            return "cpu"
        for d, i in enumerate(GPUtil.getGPUs()):
            if i.memoryFree > threshold:
                device = d
                break
            elif d + 1 == len(GPUtil.getGPUs()):
                return "cpu"
        return f"cuda:{device}"

    def _prepare_inputs(self, data, task):
        """
        Open rasters with buzzard and normalize them if it's uint (assuming it's RGB which needs to be normalized).
        Put data in cache if type(self.cache)==dict.
        Returned footprint: 
            -   Original if default option and no annots
            -   BB englobing annots if there are annots
            -   Manual bb if user has specified one
        """

        assert self.n_classes
        input_files, crop_coords, dist_map = (
            data["input_files"],
            data["crop_coords"],
            data["dist_map"],
        )

        if isinstance(self.cache, dict) and input_files != self.cache.get(
            "input_files"
        ):
            if self.ssh_server:
                for i, file in enumerate(input_files):
                    input_files[i] = self.ssh_server.get(file, cache=True)
            file = buzz.open_raster(input_files[0])
            fp = file.fp
            inputs = [
                buzz.open_raster(i)
                .get_data(fp=fp, channels=[0, 1, 2])
                .transpose((2, 0, 1))
                for i in input_files
            ]
            #TODO Use buzzard instead of rasterio
            nodata_mask = rio.open(input_files[0]).read(1, masked=True).mask
            for i, raster in enumerate(inputs):
                if raster.dtype == np.uint8:
                    inputs[i] = np.asarray(raster / 255, dtype=np.float32)
            self.tiled_fp = None  # new input => reset the tiled footprint
            if isinstance(self.cache, dict):
                self.cache["input_files"] = input_files
                self.cache["inputs"] = inputs
                self.cache["fp"] = fp
                self.cache["nodata_mask"] = nodata_mask
        else:
            inputs, fp, nodata_mask = (
                self.cache["inputs"],
                self.cache["fp"],
                self.cache["nodata_mask"],
            )
        self.original_fp = fp
        if data["interactive"]:
            if data["annot_layer"] is not None and data["interact_use_annots"]:
                file = data["annot_layer"]
                if self.ssh_server and file.endswith(".shp"):
                    for ext in ["shx", "cpg", "dbf", "prj", "qpj"]:
                        _ = self.ssh_server.get(file.replace(".shp", f".{ext}"))
                annot_file = self.ssh_server.get(file) if self.ssh_server else file
                annots = vec_to_list(annot_file, self.n_classes, fp, dist_map=dist_map)
                if not crop_coords:
                    # find footprint englobing the different points
                    bounds = buzz.open_vector(annot_file).bounds
                    # check if bound not empty
                    if np.sum(bounds):
                        buffer_value = self.cfg["segmentation"]["window_size"][0] / 4
                        buffer_x = buffer_value * np.abs(fp.scale[0])
                        buffer_y = buffer_value * np.abs(fp.scale[1])
                        bounds[0] -= buffer_x
                        bounds[2] += buffer_x
                        bounds[1] -= buffer_y
                        bounds[3] += buffer_y
                        bbox = box(*bounds)
                        try:
                            fp = fp.intersection(bbox)
                        except ValueError:
                            print(
                                "\u001b[31mIntersection between raster input and annotation points is empty.\033[0m"
                            )
            else:
                shape = fp.shape
                annots = [np.zeros((1, *shape)) for i in range(self.n_classes)]
        else:
            annots = []
        inputs_annots = inputs + annots
        inputs_annots = [
            torch.from_numpy(i).to(self.device, torch.float) for i in inputs_annots
        ]

        if crop_coords:
            min_x, max_x = (
                min(crop_coords[0][0], crop_coords[1][0]),
                max(crop_coords[0][0], crop_coords[1][0]),
            )
            min_y, max_y = (
                min(crop_coords[0][1], crop_coords[1][1]),
                max(crop_coords[0][1], crop_coords[1][1]),
            )
            bbox = box(min_x, min_y, max_x, max_y)
            fp = fp.intersection(bbox)
        return fp, inputs_annots, nodata_mask

    def segmentation(self, data):
        """
        Perform semantic segmentation
        Parameters
        ----------
        data: dict contain:
            input_files: str path to input layers
            neural_network: str path to neural network
            output_file: str path where to save the output

        Returns
        -------
        True if segmentation has been correctly perfomed. Raise an exception otherwise.
        """
        cfg = self.cfg["segmentation"]
        neural_net = data["neural_network"]
        print("\tStep 1: Load inputs.")
        if self.ssh_server:
            neural_net = self.ssh_server.get(neural_net, cache=True)

        if isinstance(self.cache, dict) and self.cache.get("net_file") == neural_net:
            net = self.cache["net"]
        else:
            net = jit.load(neural_net, map_location=self.device)
            if isinstance(self.cache, dict):
                self.cache["net"] = net
                self.cache["net_file"] = neural_net

        self.n_classes = find_n_classes(net)

        fp, inputs, nodata_mask = self._prepare_inputs(data, "segmentation")
        result = self._tile_fp(fp, cfg)
        if isinstance(result, str):
            return result
        else:
            tiled_fp = result
        # check the inputs
        sum_channs, channels_net = check_inputs_and_net(inputs, net)
        if sum_channs != channels_net:
            warn = f"Warning: {sum_channs} input channels while the chosen net expects {channels_net} channels."
            print_warning(warn)
            return warn
        print("\tStep 2: Inference")
        with torch.no_grad():
            pred = torch.zeros((self.n_classes, *fp.shape)).to(
                self.device, torch.double
            )
            # Slides a window across the image
            for batch_fps in make_batches(cfg["batch_size"], tiled_fp):
                inputs_patches = [
                    from_coord_to_patch(x, batch_fps, self.original_fp) for x in inputs
                ]
                outs = net(torch.cat(inputs_patches, dim=1)).data.to(torch.double)

                for out, sub_fp in zip(outs, batch_fps):
                    small_sub_slice = sub_fp.slice_in(fp, clip=True)
                    big_sub_slice = fp.slice_in(sub_fp, clip=True)
                    pred[:, small_sub_slice[0], small_sub_slice[1]] += out[
                        :, big_sub_slice[0], big_sub_slice[1]
                    ]

        mask = torch.argmax(pred, dim=0).cpu().numpy()
        print("\tStep 3: Save outputs")
        self._save_output(data, mask, nodata_mask, fp)
        return mask

    def run(self):
        """
        Enable the server and perform some function on the inputs from the client.
        """
        print("\033[94mServer for Qgis backend running here...\033[0m")
        if isinstance(self.cache, dict):
            print(
                "\u001b[33mInputs and neural network will be cached in RAM.",
                "Faster but can be troublesome with large inputs.\033[0m",
            )
        with open(self.connexion_file, "r") as f:
            d = yaml.safe_load(f)
            if self.ssh_server:
                address_server = tuple(d["ssh"]["address_server"])
                address_client = tuple(d["ssh"]["address_client"])
            else:
                address_server = tuple(d["local"]["address_server"])
                address_client = tuple(d["local"]["address_client"])
        while True:
            listener = Listener(address_server, authkey=b"Who you gonna call?")
            try:
                conn = listener.accept()  # ends once it has found a client
            except KeyboardInterrupt:
                print_warning("Keyboard interrupt.")
                exit()
            data = conn.recv()  # receive data
            conn.close()
            listener.close()
            if data["task"] == "Semantic segmentation":
                func = self.segmentation
            else:
                return "Only semantic segmentation task currently implemented."
            try:
                print(f"\u001b[36mRunning a {data['task'].lower()} task...\033[0m")
                tic = time()
                result = func(data)  # perform all the real work.
                toc = time()
                time_ = np.round(toc - tic, 1)
                if not isinstance(self.cache, dict) and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if isinstance(result, str):
                    new_msg = result
                else:
                    new_msg = self.n_classes
                    print(f"\033[92mDone in {time_} seconds.\033[0m")
            except KeyboardInterrupt:
                new_msg = "Task interrupted by the user."
                print_warning("\n" + new_msg)
            except Exception as error:
                new_msg = error
                print_warning(error)
            finally:
                conn_client = Client(address_client, authkey=b"ghostbusters")
                conn_client.send(new_msg)
                conn_client.close()
                if isinstance(new_msg, Exception):
                    raise (new_msg)
                else:
                    print("\u001b[35;1m\nReady for another task \U0001F60E \n\033[0m")

    def _tile_fp(self, fp, cfg):
        mode = "overlap"
        if self.tiled_fp is None:
            try:
                self.tiled_fp = self.original_fp.tile(
                    cfg["window_size"], cfg["stride"], cfg["stride"], mode
                ).ravel()
            except ValueError:
                msg = f"Cannot segment an image with a shape smaller than {cfg['window_size']}. Input shape: {self.original_fp.rsize}"
                print_warning(msg)
                return msg
        if not fp.almost_equals(self.original_fp):
            tiled_fp = [
                i for i in self.tiled_fp if i.poly.intersection(fp.poly).area > 0
            ]
        else:
            tiled_fp = self.tiled_fp
        tiled_fp = iter(tiled_fp)
        return tiled_fp
