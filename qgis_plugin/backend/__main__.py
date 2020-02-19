import argparse

from .daemon import Daemon


def parse_args():
    """parser"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--ssh", action="store_true", help="enable ssh connexion")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    is_ssh = args.ssh
    is_cache = not args.no_cache
    daemon = Daemon(ssh=is_ssh, cache=is_cache)
    daemon.run()


if __name__ == "__main__":
    main()
