"""Module defining a project context.

The module defines a class ``Context`` that acts as the single source of all
configuration details, credential information, service clients (e.g. spark).
This allows us to create stateless functions that accept the context object
as an argument and can get all the required state information from the object
instead of reaching out to some shared global config or environment variables.

As a design principle, we should avoid adding end-user methods on ``Context``
and instead write modules with stateless functions that take context as an
input.

The primary purpose of the class should be to provide access to configuration
information, initialize/construct resource handles.
"""
import logging  # noqa
import logging.config as lc  # noqa
import os.path as op
from copy import deepcopy


from pyspark.sql import SparkSession  # noqa

from .utils import get_fs_and_abs_path, load_yml
from .tracking import create_client


def create_context(config_file):
    """Create a context object from a config file path.

    Parameters
    ----------
    config_file : str
        Path for the .yml config file.

    Returns
    -------
    ta_lib.core.context.Context
        Context object generated using the config file.
    """
    ctx = Context.from_config_file(config_file)
    logger = logging.getLogger(__name__)
    logger.info(f"Context created from {config_file}")
    return ctx


class Context:
    """Class to hold any stateful information for the project."""

    def __init__(self, cfg):
        """Initialize the Context.

        Parameters
        ----------
        cfg : dict
            Config dictionary added from a relevant config file
        """
        self._cfg = cfg
        self.spark = None  # The Spark Session
        self.sc = None  # The Spark Context
        self.scConf = None  # The Spark Context conf

        # FIXME: add a utils.configure_logger function and ensure
        # to create folders if missing for fileloggers
        lc.dictConfig(cfg["log_catalog"])

        self._model_tracker = None

    @property
    def CreateSparkSession(self, appName=None, master=None):
        """Create Spark Session."""
        cmd = "SparkSession.builder"
        if master is not None:
            cmd += f'.master("{master}")'
        if appName is not None:
            cmd += f'.appName("{appName}")'
        for k, v in self._cfg["spark"].items():
            cmd += f'.config("{k}","{v}")'
        cmd += ".getOrCreate()"
        print("Running spark command")
        self.spark = eval(cmd)  # noqa
        print(self.spark)
        self.sc = self.spark.sparkContext
        print(self.sc)
        self.scConf = self.sc.getConf()
        print(self.scConf)

    @property
    def data_catalog(self):
        """Get the Data Catalog from the current project configuration.

        Returns
        -------
        dict
        """
        return self.config["data_catalog"]

    @property
    def training_job(self):
        """Get the training job from the current project configuration.

        Returns
        -------
        dict
        """
        return self.config["training_job"]

    @training_job.setter
    def training_job(self, x):
        """Get the training job from the current project configuration.

        Returns
        -------
        dict
        """
        self._cfg.update({"training_job": x})

    @property
    def hyper_params(self):
        """Get the hyper params

        Returns
        -------
        dict
        """
        return self.config["hyper_params"]

    @property
    def spark_config(self):
        """Get the spark config

        Returns
        -------
        dict
        """
        return self.config["spark"]

    @property
    def config(self):
        """Get the current project configuration."""
        return deepcopy(self._cfg)

    @property
    def model_tracker(self):
        """Get the model tracking client.

        Returns
        -------
        mlflow_client
        """
        if self._model_tracker is None:
            cfg = self.config.get("model_tracker", None)
            if cfg is None:
                self._model_tracker = None
            else:
                self._model_tracker = create_client(cfg)
        return self._model_tracker

    # ----------------
    # Construction API
    # ----------------
    @classmethod
    def from_config_file(cls, cfg_file):
        """Create the Context from a config file location path.

        Parameters
        ----------
        cfg_file : str
            Location path of the .yml config file.

        Returns
        -------
        ta_lib.core.context.Context
        """

        fs, _ = get_fs_and_abs_path(cfg_file)
        app_cfg = load_yml(cfg_file, fs=fs)
        app_cfg["config_file"] = op.abspath(cfg_file)

        return cls(app_cfg)

    class CustomSparkSession:  # noqa
        """A Public Sparksession Class."""

        def __init__(self, cfg):
            self._cfg = cfg
            self.spark = None  # The Spark Session
            self.sc = None  # The Spark Context
            self.scConf = None  # The Spark Context conf

        def CreateSparkSession(self, appName=None, master=None):
            """Create Spark Session."""
            cmd = "SparkSession.builder"
            if master is not None:
                cmd += f'.master("{master}")'
            if appName is not None:
                cmd += f'.appName("{appName}")'
            for k, v in self._cfg["spark"].items():
                cmd += f'.config("{k}","{v}")'
            cmd += ".getOrCreate()"
            self.spark = eval(cmd)  # noqa
            self.sc = self.spark.sparkContext
            self.scConf = self.sc.getConf()
