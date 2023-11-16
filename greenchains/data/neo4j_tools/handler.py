from typing import List, Dict, Any, Union
from pprint import pprint

from tqdm import tqdm

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, Neo4jError

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

import logging
from greenchains.config import LOGGER_NAME
from greenchains import logs
logger = logging.getLogger(LOGGER_NAME)

class Neo4jConnectionHandler:
    '''
    Driver for connecting to a running Neo4j instance.
    '''
    def __init__(
            self,
            db_ip: str = 'host.docker.internal',
            bolt_port: str = '7687',
            database: str = 'neo4j',
            db_username: str = 'neo4j',
            db_password: str = None,
            secure_connection: bool = True
    ):
        """
        Creates a direct connection to a single Neo4j graph
        (not the full DBMS, but just one of its databases).


        Parameters
        ----------
        db_ip: str. IP address or domain of the DBMS being used.
            The default provided is the equivalent of using "localhost"
            or "127.0.0.1", but when inside a docker container.

        bolt_port: str. Indicates the port through which
            bolt-database-connections are pushed.

        database: str. Name of the database to access.
            Use the default if accessing Neo4j Community Edition
            (as this only allows a single database).

        db_username. str. Username to use for authentication purposes when
            accessing the database.

        db_password: str. Password to use for authentication purposes when
            accessing the database.

        secure_connection: bool. If True, will assume that protocols to be
            used are those associated with SSL-secured instances (e.g. bolt+s).
        """
        self.db_ip = db_ip
        self.bolt_port = bolt_port
        self.database_name = database
        self.user = db_username
        self.stats = None
        self.procedures = None
        self.plugins = None
        self.data_splitting_ids = None
        self.schema = None
        self.secure_connection = secure_connection

        if secure_connection:
            self.uri = f"neo4j+s://{db_ip}:{bolt_port}"
        else:
            self.uri = f"neo4j://{db_ip}:{bolt_port}"

        # Setup the drivers
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, db_password),
            max_connection_lifetime=10_000  # seconds, default is 3600
        )

    def close(self):
        '''Closes the db connection.'''
        if self._driver is not None:
            self._driver.close()

    def __str__(self):
        output = f"Database: {self.database_name}"
        output += f"\nLocation: {self.db_ip}"
        output += f"\nLogged in as user '{self.user}'"

        return output

    def __repr__(self):
        return self.__str__()

    def get_schema(self):
        """
        Returns a tabular representation of the graph's schema, including
        things like properties associated with certain node labels/relationship
        types, etc.


        Returns
        -------
        pandas DataFrame.
        """
        if not self._check_for_plugin('apoc'):
            raise RuntimeError("APOC plugin not found")
        query = "CALL apoc.meta.schema"
        return pd.DataFrame(self.cypher_query_to_dataframe(query).iloc[0, 0])

    def _check_for_plugin(self, name):
        """
        Checks if a plugin is installed in the DBMS and returns a bool.
        """

        if self.plugins is None:
            self.procedures = self.cypher_query_to_dataframe("CALL dbms.procedures()")

            # Procedure families provided by default with core Neo4j install
            defaults = ['db', 'dbms', 'jwt', 'tx']
            self.plugins = pd.Series(self.procedures['name'].str \
                                     .split('.', expand=True)[0].unique()) \
                .replace({d: np.nan for d in defaults}).dropna().tolist()

        return name in self.plugins

    def _identify_graph_projections(self):
        projections = self.cypher_query_to_dataframe("CALL gds.graph.list()")
        return projections['graphName'].tolist()

    @classmethod
    def _query_is_directed(cls, query):
        return '<-' in query or '->' in query

    def get_subgraph_queries(self, subgraph):
        """
        Given a GDS graph projection, determine the Cypher queries
        that were used to construct it, so you can mimic those queries
        as needed.


        Parameters
        ----------
        subgraph: str. Name of the graph projection in the GDS graph catalog.


        Returns
        -------
        2-tuple of strings of the form (node_query, relationship_query).
        """

        if not self._check_for_plugin('gds'):
            raise RuntimeError("GDS plugin missing")

        elif subgraph not in self._identify_graph_projections():
            raise ValueError(f"Subgraph '{subgraph}' not found in the graph catalog")

        query = f"""
            CALL gds.graph.list('{subgraph}')
        """

        results = self.cypher_query_to_dataframe(query)
        node_query, relationship_query = results.loc[0, ['nodeQuery', 'relationshipQuery']]

        return node_query, relationship_query

    def describe(self, subgraph=None, simple=True):
        """
        Runs summary statistics on graph, including things
        like node and relationship counts, as well as more
        advanced measures of connectedness.

        See https://transportgeography.org/contents/methods/graph-theory-measures-indices/
        for details on these measures.

        This can be a computationally expensive item to run.
        As such, this should only be run as needed and the results should be saved
        by the user. Some of the most basic stats (node counts, relationship counts)
        are stored in `self.stats`. Benchmark timings:
        - 130K nodes and 475K relationships: ~3 seconds
        - 57K nodes and 819K relationships: 35 seconds


        Parameters
        ----------
        subgraph: str. If not None, this is the name of the in-memory
            graph projection of interest (e.g. 'citations'). This
            graph will be analyzed instead of the full graph, assuming
            it exists.

        simple: bool. If True, spits out the bare minimum counts-based stats
            of the graph, skipping over more computationally-intense items like
            graph diameter calculations.


        Returns
        -------
        None, only prints the stats to stdout.
        """
        for plugin in ['apoc', 'gds']:
            if not self._check_for_plugin(plugin):
                raise RuntimeError(f'{plugin} plugin not found')

        if subgraph is None:
            subgraph = 'full_graph'

        else:
            if subgraph not in self._identify_graph_projections():
                raise ValueError(f"Graph projection '{subgraph}' does not exist")

        already_projected = self.cypher_query_to_dataframe(f"CALL gds.graph.exists('{subgraph}')",
                                                           verbose=False).loc[0, 'exists']
        if not already_projected and subgraph != 'full_graph':
            raise ValueError(f"Graph projection '{subgraph}' not found")

        elif not already_projected and subgraph == 'full_graph' and not simple:
            logger.warn("Full graph projection not found, generating now...")
            query = """
            CALL gds.graph.create.cypher(
                'full_graph',
                'MATCH (n) RETURN id(n) as id, labels(n) as labels',
                'MATCH (n)-[rel]->(m) RETURN id(n) AS source, id(m) AS target'
            )
            """
            _ = self.cypher_query_to_dataframe(query, verbose=False)
            logger.info("Full graph projection complete")

        logger.debug(f"Using graph projection {subgraph}")

        # Get basic counts and such
        if subgraph != 'full_graph':
            label_counts = None
            relationship_counts = None
            stats = self.cypher_query_to_dataframe(f"CALL gds.graph.list('{subgraph}')",
                                                   verbose=False)
            stats = stats[stats['database'] == self.database_name] \
                .rename(columns={'relationshipCount': 'relCount'})

        else:
            stats = self.cypher_query_to_dataframe("CALL apoc.meta.stats()")
            label_counts = stats.loc[0, 'labels']
            relationship_counts = stats.loc[0, 'relTypesCount']

        # Number of nodes/vertices
        v = stats.loc[0, 'nodeCount']

        # Number of relationships/edges
        e = stats.loc[0, 'relCount']

        # Get various measures of connectedness
        # Beta = e/v; e = relationships, v = nodes
        beta = e / v

        # Alpha = (e - v) / (0.5v**2 - 1.5v + 1)
        alpha = (e - v) / (0.5 * v ** 2 - 1.5 * v + 1)

        # Gamma = e / (0.5v**2 - 0.5v)
        gamma = e / (0.5 * v ** 2 - 0.5 * v)

        if simple:
            stats_temp = {
                'node_count': v,
                'relationship_count': e,
                'label_counts': label_counts,
                'relationship_counts': relationship_counts,
                'alpha': alpha,
                'beta': beta,
                'gamma': gamma
            }
            pprint(stats_temp)

            # Do some simple rules-based flagging of values
            if beta < 1.0:
                logger.info(f"beta value of {beta} indicates a sparse graph, "
                            "with a low number of links relative to nodes")

            if alpha < 0.5:
                logger.info(f"Alpha value of {alpha} is fairly low, suggesting a sparse "
                            "network due to low number of observed vs. possible cycles")

            if gamma < 0.5:
                logger.info(f"Gamma value of {gamma} is fairly low, meaning that the "
                            "amount of observed links is low relative to the max possible link count")

        else:
            # Get orphan node counts by label
            query = f"""
                CALL gds.degree.stream('{subgraph}')
                YIELD nodeId, score
                WHERE score = 0.0
                WITH gds.util.asNode(nodeId) AS n
                RETURN labels(n) AS NodeLabels, count(n) AS NumIsolates
            """
            self.isolates = self.cypher_query_to_dataframe(query, verbose=False)
            if self.isolates is None or self.isolates.empty:
                orphan_count = 0

            else:
                orphan_count = self.isolates['NumIsolates'].sum()

            # Transitivity via LCC
            query = f"""
                CALL gds.localClusteringCoefficient.stream('{subgraph}')
                YIELD nodeId, localClusteringCoefficient
                WHERE gds.util.isFinite(localClusteringCoefficient)
                RETURN avg(localClusteringCoefficient)
                """

            result = self.cypher_query_to_dataframe(query, verbose=False)
            transitivity = result.iloc[0, 0]

            # Diameter = length(longest shortest path)
            # Efficiency = average of all shortest path lengths
            query = f"""
                CALL gds.alpha.allShortestPaths.stream('{subgraph}')
                YIELD sourceNodeId, targetNodeId, distance
                WHERE gds.util.isFinite(distance) = true AND distance > 0.0
                RETURN max(distance) AS diameter, avg(distance) AS efficiency
                """

            diameter, efficiency = self.cypher_query_to_dataframe(query,
                                                                  verbose=False).loc[0]

            if subgraph == 'full_graph':
                self.properties = self.cypher_query_to_dataframe("CALL apoc.meta.data()",
                                                                 verbose=False)

                self.stats = {
                    'node_count': v,
                    'orphan_node_count': orphan_count,
                    'orphan_node_fraction': orphan_count / v,
                    'relationship_count': e,
                    'label_counts': label_counts,
                    'relationship_counts': relationship_counts,
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'transitivity': transitivity,
                    'diameter': diameter,
                    'efficiency': efficiency
                }

                logger.info(
                    "Attributes now available/updated: "
                    "[self.procedures, self.plugins, self.stats, self.properties, self.isolates]")
                pprint(self.stats)

            else:
                logger.warn("As this is a graph projection, stats will not "
                            "be cached to Neo4jConnectionHandler object, but rather just displayed here:")
                stats_temp = {
                    'node_count': v,
                    'orphan_node_count': orphan_count,
                    'orphan_node_fraction': orphan_count / v,
                    'relationship_count': e,
                    'label_counts': label_counts,
                    'relationship_counts': relationship_counts,
                    'alpha': alpha,
                    'beta': beta,
                    'gamma': gamma,
                    'transitivity': transitivity,
                    'diameter': diameter,
                    'efficiency': efficiency
                }
                pprint(stats_temp)

            # Do some simple rules-based flagging of values
            if beta < 1.0:
                logger.info(f"beta value of {beta} indicates a sparse graph, "
                            "with a low number of links relative to nodes")

            if diameter > 5:
                logger.info(f"Diameter of {int(diameter)} is greater than 5 hops, "
                            "suggesting a fairly sparse graph.")

            if efficiency > 2:
                logger.info(f"Efficiency of {efficiency} may be a bit high, indicating "
                            "a sparse network that takes a long time to get from A to B.")

            if alpha < 0.5:
                logger.info(f"Alpha value of {alpha} is fairly low, suggesting a sparse "
                            "network due to low number of observed vs. possible cycles")

            if gamma < 0.5:
                logger.info(f"Gamma value of {gamma} is fairly low, meaning that the "
                            "amount of observed links is low relative to the max possible link count")

            if transitivity < 0.5:
                logger.info(f"Transitivity is {transitivity}, which is fairly low and indicates "
                            "that many nearest neighbor nodes in the graph are not connected "
                            "(low average local clustering coefficient)")

            logger.info(f"Transitivity is {transitivity} and diameter is {diameter}. "
                        "Recall that complex, small-world graphs tend to have high transitivity "
                        "and low diameter...")

    def train_test_split(
            self,
            query,
            train_fraction=0.6,
            test_fraction=None,
            n_splits=1
    ):
        """
        Allows us to split nodes into train/test groups for modeling building. If there is a temporal
        element to the data (e.g. publish dates), this will abide by those restrictions such
        that no future data are used in training when they shouldn't be (e.g. no shuffling).

        WARNING: this method will assume the data are pre-sorted in order of increasing time
        via the query passed. See the ``query`` example for how to do this.


        Parameters
        ----------
        query: str. Cypher query to be used for generating node IDs that can be
            split into different datasets by this method. Make sure you sort by
            a temporal property if needed. Note that Neo4j doesn't like doing
            DISTINCT calls on IDs, so the WITH clause shown in the example below
            is needed. Make sure only IDs are returned (as a single column).

            TODO: a more performant approach may be to use SKIP and LIMIT to get the datasets, instead of passing IDs.

            Example:
                MATCH (n:Publication)-[:CITES]->(m:Publication)
                WITH DISTINCT n AS paper, n.publicationDate AS date
                ORDER BY date ASC
                RETURN ID(paper) as paperNodeID


        train_fraction: float between 0.0 and 0.99. Indicates the maximum amount of data
            allowed to be used for training in the final fold (if doing multi-fold).

        test_fraction: float between 0.0 and 0.99. Indicates the minimum viable dataset that
            will be heldout for final model testing purposes. If None, will be assumed to be
            n_samples * (1 - train_fraction).

        n_splits: int. Indicates how many train/test splits you want to return.


        Returns
        -------
        Either a single tuple of the form (train_indices_array, test_indices_array)
        if ``n_splits`` = 1, else a generator of such tuples that follows
        Walk-Forward Chaining for time series cross-validation (e.g. each new training dataset
        includes the old training data + more new points that were previously in the test dataset).
        """

        logger.info("Querying for node IDs...")
        all_ids = self.cypher_query_to_dataframe(query).iloc[:, 0]
        self.data_splitting_ids = all_ids
        logger.info("Query complete!")

        train_size = round(train_fraction * len(all_ids))
        if test_fraction is None:
            test_size = len(all_ids) - train_size

        else:
            test_size = round(test_fraction * len(all_ids))

        if n_splits == 1:
            splitter = TimeSeriesSplit(
                n_splits=2,  # trust me on this
                # gap=0,
                test_size=test_size
            )

            # Loop through until the last values are all you have,
            # the full split
            for train_idx, test_idx in splitter.split(all_ids):
                pass

            output = (train_idx, test_idx)

        elif n_splits > 1:
            logger.info("As n_splits > 1, train and test sizes are calculated not user-defined")
            splitter = TimeSeriesSplit(
                n_splits=n_splits,
                # gap=0
            )
            output = splitter.split(all_ids)

        else:
            raise ValueError("n_splits must be an integer >= 1")
        return output

    def get_train_test_data(self, node_properties, train_indices, test_indices, id_query=None):
        """
        Uses pre-determined train/test indices to index node IDs values from
        the graph and get the training/testing data for a single fold/split.


        Parameters
        ----------
        node_properties: list of str. Node properties that will be used to form
            the datasets.

        train_indices: list of int that can be used for indexing an array of node IDs
            to get the nodes that are part of the training dataset.

        test_indices: list of int that can be used for indexing an array of node IDs
            to get the nodes that are part of the test dataset.

        id_query: str. Cypher query used, if one is needed, to draw down the properly-sorted
            node IDs for indexing via ``train_indices`` and ``test_indices``. As it's easy
            to improperly sort these node IDs and thus improperly index them, it is recommended
            that this be left as None such that the graph attribute self.data_splitting_ids
            can be used instead. This attribute only exists, however, if self.train_test_split()
            was run to generate ``train_indices`` and ``test_indices``.


        Returns
        -------
        2-tuple of pandas DataFrames of the form (train_data, test_data).
        """

        if id_query is None and self.data_splitting_ids is None:
            raise ValueError("self.data_splitting_ids is not set, "
                             "please pass a Cypher query for the id_query parameter")

        elif id_query is None:
            all_ids = self.data_splitting_ids

        else:
            all_ids = self.cypher_query_to_dataframe(id_query).iloc[:, 0]

        training_ids = all_ids[train_indices].tolist()
        test_ids = all_ids[test_indices].tolist()

        properties_clause = ', '.join([f"n.{p} AS {p}" for p in node_properties])
        query = f"""
            MATCH (n) 
            WHERE ID(n) IN $ids
            RETURN {properties_clause}
        """
        df_train = self.cypher_query_to_dataframe(query,
                                                  parameters={'ids': training_ids})

        df_test = self.cypher_query_to_dataframe(query,
                                                 parameters={'ids': test_ids})

        return df_train, df_test

    @classmethod
    def _validate_export_format(cls, export_format):
        if export_format not in EXPORT_FORMAT_CHOICES:
            raise ValueError(f"export format {export_format} invalid, i know about {EXPORT_FORMAT_CHOICES}")

    def cypher_query_to_dataframe(
            self,
            query: str,
            parameters: Dict['str', Union[Any, List[Any]]] = None, 
            db: str = None, 
            verbose: bool = True,
            max_retries: int = 5):
        """
        Uses a Cypher query to manipulate data in a Neo4j instance
        via the native driver.

        Parameters
        ----------
        query : str
            Cypher query
        parameters : dict of form {'str': value(s)}, optional
            Value(s) to feed into the query by means of
            parameter insertion. Useful for sending over large
            amounts of data in a query efficiently, by default None
        db : str, optional
            Name of the database to query, if more than one
            are supported by the target Neo4j instance (usually
            only available for Enterprise Edition), by default None
        verbose : bool, optional
            If True, will provide logger messages
            when the query starts and finishes, by default True
        max_retries : int, optional
            Number of attempts to run the query that the user wants to allow 
            before giving up.

        Returns
        -------
        pandas DataFrame
            The results of the query. If no results are returned,
            will be an empty DataFrame.

        Raises
        ------
        neo_exc
            Generic Neo4j error that indicates a problem that can't
            be solved simply through retrying. Usually raised due to
            a bad Cypher query.
        value_exc
            Indicates a problem with the received/sent data, although
            usually a deque error indicating that the Neo4j data queue wasn't
            properly cleared before a new query was run.
        """
        assert self._driver is not None, "Driver not initialized!"
        session = None
        response = None
        q = query.replace("\n", " ").strip()

        i = 0
        while True and i < max_retries:
            try:
                if verbose:
                    logger.info("Neo4j query started...")
                    logger.debug("DB IP:`%s` query: `%s`", self.db_ip, q)
                if db is not None:
                    session = self._driver.session(database=db)
                else:
                    session = self._driver.session()
                results = session.run(q, parameters)
                response = pd.DataFrame(results, columns=results.keys())
                if verbose:
                    logger.info("Neo4j query complete!")
                break

            except ServiceUnavailable:
                logger.error("Not able to communicate with Neo4j due to "
                             "ServiceUnavailable, retrying...")

            except Neo4jError as neo_exc:
                raise neo_exc

            except ValueError as value_exc:
                if value_exc.args[0] == 'deque.remove(x): x not in deque':
                    logger.error("deque error detected, retrying query...")
                else:
                    raise value_exc
                
            i += 1

        # Return empty df if no successful runs
        if i >= max_retries:
            logger.warning("Max number of retries reached with no success")
            return pd.DataFrame()

        return response.replace({None: np.nan})

    def insert_data(self, query, rows, batch_size=1_000):
        """
        Inserts data in batches to a Neo4j instance.

        Parameters
        ----------
        query : str
            Cypher query for doing the data insertion.
            Should include 'UNWIND $rows AS row' near
            the top to ensure the data are pushed appropriately, but
            will be added at beginning of query if missing. Note that it's
            also usually best practice to have any MATCH clause be of the form
            "MATCH (n:Label {id: row.id})".
        rows : iterable of data, often pandas DataFrame
            Tabular data to be inserted in batches
        batch_size : int, optional
            Number of rows to insert per batch, by default 10000

        Returns
        -------
        Nothing.
        """
        if 'unwind $rows as row' not in query.lower():
            logger.warning("Query is missing UNWIND statement for ``rows``, "
                           "inserting it at the beginning of query now...")
            query = 'UNWIND $rows AS row\n' + query
            logger.info(f"New query: {query}")

        for start_index in tqdm(
                range(0, len(rows), batch_size),
                desc='Inserting data into Neo4j',
                unit_scale=batch_size
        ):
            end_index = start_index + batch_size
            if end_index > len(rows):
                end_index = len(rows)

            _ = self._query_with_native_driver(
                query,
                verbose=False,
                parameters={
                    'rows': rows.iloc[start_index: end_index].replace({np.nan: None}).to_dict('records')
                }
            )
            

def test_graph_connectivity(db_ip, db_password, db_username='neo4j'):
    """
    Tries to set up a Neo4j connection to make sure all credentials, etc. are
    good to go.

    Parameters
    ----------
    db_ip : str
        IP address/URL of the DBMS to test
    db_password : str
        Password for the given username
    db_username : str, optional
        Username to use for DBMS authentication, by default 'neo4j'
    """
    logger.debug("Testing Neo4j database connectivity...")
    Neo4jConnectionHandler(
        db_ip=db_ip,
        db_username=db_username,
        db_password=db_password
    )
