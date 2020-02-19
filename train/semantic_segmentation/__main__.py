import click

from semantic_segmentation.train import train
from semantic_segmentation.export_graph import export_graph


@click.group()
def cli():
    pass


cli.add_command(export_graph)
cli.add_command(train)

if __name__ == "__main__":
    cli()
