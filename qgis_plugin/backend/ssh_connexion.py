"""
This module handles the SSH part.
"""
import getpass
import os
import tempfile

import ntpath
import paramiko
import yaml


class SshConnexion:
    """Simple class for SFTP"""

    def __init__(self, config_file):
        with open(config_file, "r") as f:
            self.cfg = yaml.safe_load(f)["ssh"]
        self.sftp = self.init_sftp(self.cfg)
        self.cache = []

    def tmp_file(self, file):
        """Convert a source file from the client to a temporary file in the server"""
        f = tempfile.NamedTemporaryFile()
        tmp_folder = os.path.dirname(f.name)
        f.close()
        return os.path.join(tmp_folder, ntpath.basename(file))

    def get(self, file, cache=False):
        """Get data in the server from the client if the data isn't in cache already"""
        temp_file = self.tmp_file(file)
        if cache:
            if temp_file not in self.cache:
                self.sftp.get(file, temp_file)
                self.cache.append(temp_file)
        else:
            self.sftp.get(file, temp_file)
        return temp_file

    def put(self, file):
        """Paste data from the server to the client"""
        self.sftp.put(self.tmp_file(file), file)

    @staticmethod
    def init_sftp(ssh_info):
        """Initialize an SFTP using `sftp_info`"""
        port = 22  # ssh port
        hostname = ssh_info["address_client"][0]
        username = ssh_info["username"]
        try:
            host_keys = paramiko.util.load_host_keys(
                os.path.expanduser("~/.ssh/known_hosts")
            )
        except IOError:
            try:
                # Windows can't have a folder named ~/.ssh/
                host_keys = paramiko.util.load_host_keys(
                    os.path.expanduser("~/ssh/known_hosts")
                )
            except IOError:
                raise Exception("Unable to open host keys file")

        if hostname in host_keys:
            hostkeytype = host_keys[hostname].keys()[0]
            hostkey = host_keys[hostname][hostkeytype]
        else:
            raise Exception(f"Hostname {hostname} not in host keys.")

        t = paramiko.Transport(
            (hostname, port)
        )  # does not work if hostname not valid and port != 22
        password = getpass.getpass(f"Password for {username}@{hostname}:")
        t.connect(hostkey, username, password)
        del password
        sftp = paramiko.SFTPClient.from_transport(t)
        return sftp
