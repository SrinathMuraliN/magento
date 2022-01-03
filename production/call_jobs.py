import logging
import traceback

import click
from magento_lib.core.context import create_context
from magento_lib.inference.model_inference import start_inference
from magento_lib.preprocessing.data_loading import load_data
from magento_lib.preprocessing.data_preprocessing import start_preprocessing
from magento_lib.training.model_training import start_training


@click.group()
@click.option("-tc", "--training_config", required=True)
@click.pass_context
def cli():
    print("calling cli")
    training_config = (
        "gs://pg-explore/data/magento/interim/argentina_parameters.yml"
    )
    context = create_context(training_config)
    session = context.CustomSparkSession(context.config)
    session.CreateSparkSession()
    spark = session.spark
    context.spark = spark
    print("Setting context to object")
    ctx.obj["context"] = context


@cli.command()
@click.pass_context
def load_data(ctx):
    load_data()
    return click.echo("success")


@cli.command()
@click.pass_context
def start_preprocessing(ctx):
    context = ctx.obj["context"]
    # start_preprocessing(context, context.spark, ctx.obj["mlflow_run_suffix"])
    start_preprocessing()
    return click.echo("success")


@cli.command()
@click.pass_context
def start_training(ctx):
    context = ctx.obj["context"]
    # start_training(context, context.spark, ctx.obj["mlflow_run_suffix"])
    start_training()
    return click.echo("success")


@cli.command()
@click.pass_context
def start_inference():
    # context = ctx.obj["context"]
    # start_inference(context, context.spark, ctx.obj["mlflow_run_suffix"])
    print("Starting the Inference Method")
    start_inference()
    print("Ending the Inference Method")
    return click.echo("success")


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
