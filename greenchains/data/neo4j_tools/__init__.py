# -*- coding: utf-8 -*-
from pathlib import PurePosixPath
from datetime import datetime as dt
import pandas as pd

import logging
from greenchains.config import LOGGER_NAME
from greenchains import logs
logger = logging.getLogger(LOGGER_NAME)

# Timestamp we have to use to represent NULL
INVALID_TIMESTAMP = pd.Timedelta(25, 'days') + pd.Timestamp(dt.today())

IMPORT_DIR = PurePosixPath("/var/lib/neo4j/import")  # assumes a linux remote instance
EXPORT_FORMAT_CHOICES = ("graphml", "csv", "json", "cypher")
NEO4J_PASSWORD_ENV_KEY = "NEO4J_PASSWORD"  # check os.environ for this. CBB but good enough for now


ALLOWED_NEO4J_DATA_TYPES = [
    'int', 'long', 'float', 'double', 'boolean', 'byte', 'short',
    'char', 'string', 'point',
    'date', 'time', 'localtime', 'datetime', 'localdatetime', 'duration'
]

# Add array variants
ALLOWED_NEO4J_DATA_TYPES.extend([t + '[]' for t in ALLOWED_NEO4J_DATA_TYPES])
BOLT_PORT = '7687'


def get_allowed_neo4j_types():
    """
    Simple helper function to provide the list of data types that Neo4j is
    expecting. Also provides some useful guidance on quirks of data formatting
    needed for successful Neo4j ingestion.


    Parameters
    ----------
    None.


    Returns
    -------
    List of str data types.
    """

    logger.info("Types ending with '[]' are arrays of that type.")
    logger.info("'boolean' types must have all values converted to \
strings 'true' and 'false' (note that they are all lowercase).")
    return ALLOWED_NEO4J_DATA_TYPES[:]


def format_property_columns_for_ingest(
        df,
        renaming_rules=None,
        return_only_renamed=True
):
    """
    Renames pandas DataFrame columns according to the needs of neo4j's ingest
    engine, to provide it with properly-formatted property, label, etc.
    keys.


    Parameters
    ----------
    df: pandas DataFrame containing the data you intend to prepare for Neo4j
        ingest.

    renaming_rules: pandas DataFrame or list of dicts with columns/keys
        'old', 'new', and 'type'. These are the existing column names,
        new column names, and Neo4j data types that they should be associated
        with, resp.

        Example: [
            {'old': 'orig_col1', 'new': 'renamed_col1', 'type': 'string'},
            {'old': 'orig_col2', 'new': 'renamed_col2', 'type': 'int'}
        ]

        If 'type' is np.nan, the column is simply renamed to the value of
        'new' and is not given explicit typing (e.g. this is useful for the
        ":ID" column usually)

    return_only_renamed: bool. If True, subsets ``df`` to only include columns
        named in ``renaming_rules``. Otherwise returns all columns, including
        those never renamed.


    Returns
    -------
    A copy of ``df`` with columns renamed accordingly.
    """

    if isinstance(renaming_rules, pd.DataFrame):
        column_conversions = renaming_rules.copy()

    elif isinstance(renaming_rules, dict):
        column_conversions = pd.DataFrame(renaming_rules)

    else:
        raise ValueError(f"renaming_rules must be of type dict or \
pandas.DataFrame. Got {type(renaming_rules)} instead.")

    # Check that column_conversions only contains allowed neo4j data types
    types = column_conversions['type'].fillna("string")
    if types.isin(ALLOWED_NEO4J_DATA_TYPES).sum() < len(column_conversions):
        raise ValueError("At least one 'type' specified is not in the \
list of allowed Neo4j data types. Please run ``get_allowed_neo4j_types()`` to \
see what types may be used or follow the link provided in the docstring of \
this function for the most up-to-date information.")

    column_conversions['new_with_types'] = \
        column_conversions['new'] + ':' + column_conversions['type']

    # Make sure any that had data type NaN will not differ from 'new' column
    column_conversions['new_with_types'] = \
        column_conversions['new_with_types'].fillna(column_conversions['new'])

    columns_mapping = column_conversions.set_index('old')['new_with_types']

    if return_only_renamed:
        return df[column_conversions['old']].rename(columns=columns_mapping)

    else:
        return df.rename(columns=columns_mapping)
        