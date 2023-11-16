import os
import pathlib

import pandas as pd

from greenchains.data.neo4j_tools.utils import export_df, transform_timestamps, convert_types

import logging
from greenchains.config import LOGGER_NAME
from greenchains import logs
logger = logging.getLogger(LOGGER_NAME)

class Nodes:
    """
    A representation of a whole class of Neo4j nodes (e.g. nodes that all
    share the same label(s)) based off of a DataFrame that provides information
    at the individual node level for properties and such.
    """

    def __init__(
            self,
            parent_label,
            data,
            id_column,
            reference,
            additional_labels=None,
            properties=None
    ):
        """
        Example: [
            {'old': 'orig_col1', 'new': 'renamed_col1', 'type': 'string'},
            {'old': 'orig_col2', 'new': 'renamed_col2', 'type': 'int'}
        ]

        If 'type' is np.nan, that column is assumed to be of the 'string'
        data type.

        Parameters
        ----------
        parent_label: str. Indicates the highest-level descriptive node label
            to apply. For example, when describing a node for a paper author,
            "Person" would be the parent label and ["Author"] would be the
            additional labels.

        data: pandas DataFrame. Must have at least one column named the same
            as the value passed to ``id_column``. Defines instances of
            individual nodes. Extra columns should include values of properties
            each node will have, one column per property.

        id_column: str. Name of the column containing the unique IDs for the
            nodes.

        reference: str. An alias to use for the type of node when referring
            to it in other Nodes objects. E.g. 'paper'.

        additional_labels: list of str. Indicates what label(s) you want associated
            with the nodes. For example, when describing a node for a paper author,
            "Person" would be the parent label and ["Author"] would be the
            additional labels.

        properties: pandas DataFrame or list of dicts with columns/keys
        'old', 'new', and 'type'. These are the existing column names,
        new column names, and Neo4j data types that each property should be
        associated with, resp.
        """

        assert isinstance(additional_labels,
                          list) or additional_labels is None, "``additional_labels`` not list or None"

        if additional_labels is None:
            additional_labels = []

        self.parent_label = parent_label
        self.additional_labels = additional_labels
        self.labels = [parent_label] + additional_labels
        self.reference = reference

        self.id = f'id:ID({reference}-ref)'

        if id_column not in data.columns:
            raise ValueError(f"id_column value of '{id_column}' not found \
in data.columns")

        if isinstance(properties, pd.DataFrame):
            self.column_conversions = properties.copy()

        elif isinstance(properties, dict):
            self.column_conversions = pd.DataFrame(properties)

        elif properties is not None:
            raise ValueError(f"``properties`` must be of type dict or \
    pandas.DataFrame. Got {type(properties)} instead.")

        else:
            self.properties = None

        if properties is not None:
            self.properties = self.column_conversions['new'].tolist()
            self.data = format_property_columns_for_ingest(
                data,
                renaming_rules=self.column_conversions,
                return_only_renamed=True
            )

        else:
            logger.warn(f"No properties identified, so {self.reference}-type nodes will only have an ID and a label!")
            self.data = pd.DataFrame(data[id_column])

        if id_column in self.data.columns:
            self.data = self.data.rename(columns={id_column: self.id})

        else:
            self.data[self.id] = data[id_column]

        # TODO: how to make this capable of accounting for multiple possible
        # labels varying across rows?
        self.data[':LABEL'] = ';'.join(self.labels)

        # Make sure we have IDs for every node, otherwise drop
        num_null_ids = self.data[self.id].isnull().sum()
        if num_null_ids > 0:
            logger.warning(f"Found {num_null_ids} null node IDs! Dropping...")
            self.data.dropna(subset=[self.id], inplace=True)

        # Make sure we don't have any duplicate nodes
        # Drop all but the first duplicate if we do
        num_duplicate_nodes = self.data.duplicated(subset=self.id, keep='first').sum()
        if num_duplicate_nodes > 0:
            logger.warn(f"Found {num_duplicate_nodes} duplicate {self.reference} node IDs! \
Removing all but the first...")
            self.data.drop_duplicates(subset=self.id, keep='first', inplace=True)

    # TODO: incorporate concept of addition as a Nodes method to allow for combinations of Nodes
    #  into a single object like we do for funders + institutions

    def __str__(self):
        output = f"Nodes object with {len(self.data):,} unique nodes and with"

        if len(self.labels) > 1:
            output += f" labels {self.labels}."

        else:
            output += f" label '{self.labels[0]}'."

        if self.properties:
            output += f" Most of these nodes have properties {self.properties}"

        return output

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.data)

    def export_to_csv(self, filepath):
        """
        Given everything we know about these objects, export the results
        into a CSV that ``neo4j-admin load`` can ingest.


        Parameters
        ----------
        filepath: str. Indicates where to save the exported CSV. Should be of
            the form 'path/to/file.csv'.


        Returns
        -------
        Nothing.
        """
        self.filepath = filepath
        filepath_directories_only = os.path.split(filepath)[0]

        if not os.path.exists(filepath_directories_only) and filepath_directories_only != '':
            logger.warn(f"Filepath {filepath_directories_only} not found, creating it now...")
            pathlib.Path(filepath_directories_only).mkdir(parents=True, exist_ok=True)

        export_df(self.data, filepath, neo4j_object_type='nodes')

    def export_to_neo4j(self, graph, batch_size=1_000):
        """
        Exports Nodes data to running Neo4j instance

        Parameters
        ----------
        graph : Neo4jConnectionHandler object
            Graph connection to the target Neo4j instance
        batch_size : int
            The maximum number of nodes and their data to send at once to Neo4j.
            Note that this number should be adjusted downward if you have a large
            number of properties per node.

        Raises
        ------
        ValueError
            Checks if ``graph`` is of the proper class
        """
        if not isinstance(graph, Neo4jConnectionHandler):
            raise ValueError("``graph`` must be of type Neo4jConnectionHandler")

        data = self.data.rename(
            columns={
                self.id: 'id',
            }
        )
        # Get as many Neo4j-compatible dtypes as possible
        data = convert_types(data)

        # Find out which are datetime columns, if any
        data, datetime_columns = transform_timestamps(data)

        # Need to treat datetime columns as special when sending over
        if datetime_columns is not None:
            node_properties = ", ".join([f"n.{p} = row.{p}" for p in self.properties if p not in datetime_columns])
            node_properties += ", "
            node_properties += ", ".join([f"n.{c} = datetime(row.{c})" for c in datetime_columns])
            properties_clause = f"""
                SET
                    {node_properties}
                """

        elif self.properties is not None:
            node_properties = ", ".join([f"n.{p} = row.{p}" for p in self.properties])
            # Check that there are properties to set
            if len(node_properties) > 0 and self.properties:
                properties_clause = f"""
                SET
                    {node_properties}
                """
            else:
                raise ValueError(f"node properties not successfully set from self.properties: {self.properties}")
        else:
            properties_clause = ""

        # Check if there are labels to set beyond parent
        if self.additional_labels is not None and len(self.additional_labels) > 0:
            if len(properties_clause) > 0:
                properties_clause += ", " + f"n:{':'.join(self.additional_labels)}"
            else:
                properties_clause = f"n:{':'.join(self.additional_labels)}"

        query = f"""
        UNWIND $rows AS row
        MERGE (n:{self.parent_label} {{id: row.id}})
        {properties_clause}
        """

        logger.debug(f"Insertion is using the query \n{query}")

        graph.insert_data(
            query,
            data,
            batch_size=batch_size
        )


class Relationships:
    """
    A representation of a whole class of Neo4j nodes (e.g. nodes that all
    share the same label(s)) based off of a DataFrame that provides information
    at the individual node level for properties and such.
    """

    def __init__(
            self,
            type,
            data,
            start_node,
            end_node,
            id_column_start=None,
            id_column_end=None,
            allow_unknown_nodes=False,
            properties=None
    ):
        """
         Example: [
            {'old': 'orig_col1', 'new': 'renamed_col1', 'type': 'string'},
            {'old': 'orig_col2', 'new': 'renamed_col2', 'type': 'int'}
        ]

        If 'type' is np.nan, that column is assumed to be of the 'string'
        data type.

        Parameters
        ----------
        type: str. Indicates what the relationship type is (e.g.
            'AFFILIATED_WITH'). Can only be one value.

        data: pandas DataFrame. Must have at least one column named the
            equivalent of ``start_node.id`` and one named the equivalent of
            ``end_node.id``. Defines the desired connections from individual
            start nodes to individual end nodes. Extra columns should include
            values of properties each relationships will have, one column
            per property.

        start_node: Nodes object. This will be used to derive the node
            reference to be used at the start of the relationship, check that
            all relationships provided by ``data`` are comprised only of
            existing nodes, etc.

        end_node: Nodes object. This will be used to derive the node
            reference to be used at the end of the relationship, check that
            all relationships provided by ``data`` are comprised only of
            existing nodes, etc.

        id_column_start: str. If, for some reason, the column needed from
            ``data`` for the starting nodes mapping is named something
            different than what is provided by ``start_node.id``, provide it
            here. For example, papers citing other papers would result in two
            ID columns that refer to the same set of nodes, but both columns
            in ``data`` should not be called the same thing (start_node.id).

        id_column_end: str. If, for some reason, the column needed from
            ``data`` for the ending nodes mapping is named something
            different than what is provided by ``end_node.id``, provide it
            here.

        allow_unknown_nodes: bool. If True, relationships may be defined in
            ``data`` that do not have corresponding node IDs present in either
            ``start_node`` or ``end_node``.

        properties: pandas DataFrame or list of dicts with columns/keys
            'old', 'new', and 'type'. These are the existing column names,
            new column names, and Neo4j data types that each property should
            be associated with, resp.
        """

        self.type = type

        self.start_id = f':START_ID({start_node.reference}-ref)'
        self.start_node_labels = start_node.labels

        self.end_id = f':END_ID({end_node.reference}-ref)'
        self.end_node_labels = end_node.labels

        self.start_reference = start_node.reference
        if id_column_start is None:
            self._start_id_input = start_node.id

        else:
            self._start_id_input = id_column_start

        self.end_reference = end_node.reference
        if id_column_end is None:
            self._end_id_input = end_node.id

        else:
            self._end_id_input = id_column_end

        if self._start_id_input not in data.columns:
            raise ValueError(f"Start node ID column '{self._start_id_input}' not found \
in data.columns")

        elif self._end_id_input not in data.columns:
            raise ValueError(f"End node ID column '{self._end_id_input}' not found \
in data.columns")

        if isinstance(properties, pd.DataFrame):
            self.column_conversions = properties.copy()

        elif isinstance(properties, dict):
            self.column_conversions = pd.DataFrame(properties)

        elif properties is not None:
            raise ValueError(f"``properties`` must be of type dict or \
    pandas.DataFrame. Got {type(properties)} instead.")

        else:
            self.properties = None

        if properties is not None:
            self.properties = self.column_conversions['new'].tolist()
            self.data = format_property_columns_for_ingest(
                data,
                renaming_rules=self.column_conversions,
                return_only_renamed=True
            )

            self.data[self.start_id] = \
                data[self._start_id_input]

            self.data[self.end_id] = \
                data[self._end_id_input]

        else:
            self.data = data[[self._start_id_input, self._end_id_input]].rename(
                columns={
                    self._start_id_input: self.start_id,
                    self._end_id_input: self.end_id
                })

        id_columns = [self.start_id, self.end_id]
        num_duplicates = self.data.duplicated(subset=id_columns).sum()
        num_null_ids = self.data[id_columns].isnull().sum().sum()

        if num_null_ids > 0:
            logger.warning(f"Dropping {num_null_ids} relationships with at "
                           "least one null node ID...")
            self.data.dropna(subset=id_columns, inplace=True)

        if num_duplicates > 0:
            logger.warn(f"Dropping {num_duplicates} relationships that are \
duplicative.")
            self.data.drop_duplicates(subset=id_columns, inplace=True)

        # Make sure we aren't connecting nodes that we weren't given
        # unless we should!
        if not allow_unknown_nodes:
            bad_nodes_start = (
                ~self.data[self.start_id].isin(start_node.data[start_node.id])
            )
            num_bad_nodes_start = bad_nodes_start.sum()

            if num_bad_nodes_start > 0:
                logger.warn(f"Dropping {num_bad_nodes_start} relationship \
mappings for {start_node.reference}-type start nodes as they don't exist in \
the Nodes data provided...")

                self.data = self.data[~bad_nodes_start]

            bad_nodes_end = (
                ~self.data[self.end_id].isin(end_node.data[end_node.id])
            )
            num_bad_nodes_end = bad_nodes_end.sum()

            if num_bad_nodes_end > 0:
                logger.warn(f"Dropping {num_bad_nodes_end} relationship \
mappings for {end_node.reference}-type end nodes as they don't exist in \
the Nodes data provided...")

                self.data = self.data[~bad_nodes_end]

        # Check that we don't have any papers citing themselves!
        if self.start_reference == self.end_reference:
            self_connecting = (self.data[self.start_id] == self.data[self.end_id])
            num_self_connecting = self_connecting.sum()

            if num_self_connecting > 0:
                logger.warn(f"Dropping {num_self_connecting} relationships that \
start and end at the same node. How did that happen?!")
                self.data = self.data[~self_connecting]

        self.data[':TYPE'] = self.type

    def __str__(self):
        output = f"Relationships object structured as \
({self.start_reference})-[:{self.type}]->({self.end_reference}) with \
{len(self.data):,} unique relationships."

        if self.properties is not None:
            output += f" Most of these relationships have properties {self.properties}"

        return output

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.data)

    def export_to_csv(self, filepath):
        """
        Given everything we know about these objects, export the results
        into a CSV that ``neo4j-admin load`` can ingest.

        Parameters
        ----------
        filepath: str. Indicates where to save the exported CSV. Should be of
            the form 'path/to/file.csv'.

        Returns
        -------
        Nothing.
        """
        self.filepath = filepath
        filepath_directories_only = os.path.split(filepath)[0]

        if not os.path.exists(filepath_directories_only) and filepath_directories_only != '':
            logger.warn(f"Filepath {filepath_directories_only} not found, creating it now...")
            pathlib.Path(filepath_directories_only).mkdir(parents=True, exist_ok=True)

        export_df(self.data, filepath, neo4j_object_type='relationships')

    def export_to_neo4j(self, graph, batch_size=1_000):
        """
        Exports Relationships data to running Neo4j instance

        Parameters
        ----------
        graph : Neo4jConnectionHandler object
            Graph connection to the target Neo4j instance
        batch_size : int
            The maximum number of edges and their data to send at once to Neo4j.
            Note that this number should be adjusted downward if you have a large
            number of properties per node.

        Raises
        ------
        ValueError
            Checks if ``graph`` is of the proper class
        """
        if not isinstance(graph, Neo4jConnectionHandler):
            raise ValueError("``graph`` must be of type Neo4jConnectionHandler")

        data = self.data.rename(
            columns={
                self.start_id: 'source_node_id',
                self.end_id: 'target_node_id'
            }
        )
        # Find out which are datetime columns, if any
        data, datetime_columns = transform_timestamps(data)

        # Need to treat datetime columns as special
        if datetime_columns is not None:
            relationship_properties = ", ".join(
                [f"r.{p} = row.{p}" for p in self.properties if p not in datetime_columns])
            if len(relationship_properties) > 0:
                relationship_properties += ", "
            relationship_properties += ", ".join([f"r.{c} = datetime(row.{c})" for c in datetime_columns])

            properties_clause = f"""
                SET
                    {relationship_properties}
                """

        elif self.properties is not None:
            relationship_properties = ", ".join([f"r.{p} = row.{p}" for p in self.properties])
            # Check that there are properties to set
            if len(relationship_properties) > 0 and self.properties:
                properties_clause = f"""
                SET
                    {relationship_properties}
                """
            else:
                raise ValueError(f"relationship_properties not successfully set from self.properties: "
                                 f"{self.properties}")
        else:
            properties_clause = ""

        query = f"""
        UNWIND $rows AS row
        MATCH
        (source:{':'.join(self.start_node_labels)} {{id: row.source_node_id}}),
        (target:{':'.join(self.end_node_labels)} {{id: row.target_node_id}})

        MERGE (source)-[r:{self.type}]->(target)
        {properties_clause}
        """

        graph.insert_data(
            query,
            data,
            batch_size=batch_size
        )