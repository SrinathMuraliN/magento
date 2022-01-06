import logging
import traceback

import gcsfs
import click
import fsspec
import yaml
from magento_lib.inference.model_inference import start_inference
from magento_lib.preprocessing.data_loading import load_data
from magento_lib.preprocessing.data_preprocessing import start_preprocessing
from magento_lib.training.model_training import start_training


@click.group()
@click.option("-tc", "--training_config", required=True)
@click.pass_context
def cli(ctx, training_config):
    print("calling cli")
    interim_fs = gcsfs.GCSFileSystem(project="tiger-mle")
    with interim_fs.open(training_config, "r") as fp:
        params = yaml.safe_load(fp)
    ctx.obj["context"] = params


@cli.command()
@click.pass_context
def data_loading():
    print("Starting data_loading ***")
    load_data()
    print("Ending data_loading ***")
    return click.echo("success")


@cli.command()
@click.pass_context
def data_preprocessing(ctx):
    context = ctx.obj["context"]
    print("Starting data_preprocessing ***")
    start_preprocessing(context)
    print("Ending data_preprocessing ***")
    return click.echo("success")


@cli.command()
@click.pass_context
def model_training(ctx):
    context = ctx.obj["context"]
    start_training(context)
    return click.echo("success")


@cli.command()
@click.pass_context
def model_inference():
    context = ctx.obj["context"]
    print("Starting the Inference Method")
    start_inference(context)
    print("Ending the Inference Method")
    return click.echo("success")


@cli.command()
@click.pass_context
def main(ctxt=None):
    ctxt = ctxt or {}
    try:
        cli(obj={})
    except click.exceptions.Exit as e:
        if e.code != 0:
            raise
    except SystemExit as e:
        if e.code != 0:
            raise


if __name__ == "__main__":
    print("Inside the main function ****")
    try:
        main({})
    except click.exceptions.Exit as e:
        print("called click exception")
        if e.code != 0:
            raise
    except SystemExit as e:
        print("called System Exit")
        if e.code != 0:
            raise

    except:
        traceback.print_exc()
        raise
