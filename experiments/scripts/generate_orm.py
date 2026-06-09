import logging
import os
from dataclasses import is_dataclass
from pathlib import Path

import numpy as np

import experiments
import pycram.orm.ormatic_interface
from krrood.class_diagrams import ClassDiagram
from krrood.ormatic.data_access_objects.alternative_mappings import AlternativeMapping
from krrood.ormatic.helper import get_classes_of_ormatic_interface
from krrood.ormatic.ormatic import ORMatic
from krrood.ormatic.type_dict import TypeDict
from krrood.ormatic.utils import classes_of_package
from krrood.utils import recursive_subclasses
from pycram.orm.model import NumpyType

# ----------------------------------------------------------------------------------------------------------------------
# This script generates the ORM classes for the pycram package
# Classes that are self_mapped and explicitly_mapped are already mapped in the model.py file. Look there for more
# information on how to map them.
# ----------------------------------------------------------------------------------------------------------------------


# Create an ORMatic object with the classes to be mapped
ormatic = ORMatic.from_package([experiments], [pycram.orm.ormatic_interface], [])
logging.getLogger("krrood").setLevel(logging.DEBUG)

# Generate the ORM classes
ormatic.make_all_tables()

ormatic_interface_path = (
    Path(__file__).parent.parent
    / "src"
    / "experiments"
    / "orm"
    / "ormatic_interface.py"
)
with open(ormatic_interface_path, "w") as f:
    ormatic.to_sqlalchemy_file(f)
