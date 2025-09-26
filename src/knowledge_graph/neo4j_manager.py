"""
Neo4j Database Connection and Management

This module provides the database connection and basic operations
for the ContractSense knowledge graph stored in Neo4j.

Key Features:
- Connection management with retry logic
- Schema setup and validation
- Batch operations for efficient data loading
- Query execution with error handling
- Transaction management
"""

import os
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from contextlib import contextmanager
import logging
from dataclasses import asdict

try:
    from neo4j import GraphDatabase, Driver, Session, Transaction
    from neo4j.exceptions import ServiceUnavailable, TransientError
except ImportError:
    # Provide fallback for development without Neo4j installed
    GraphDatabase = None
    Driver = None
    Session = None
    Transaction = None
    ServiceUnavailable = Exception
    TransientError = Exception

from .schema import (
    ContractGraphSchema, GraphNode, GraphRelationship,
    NodeType, RelationshipType, create_standard_clause_types
)

logger = logging.getLogger(__name__)

class Neo4jConnection:
    """Manages connection to Neo4j database"""
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", password: str = "password",
                 max_retry_attempts: int = 3, retry_delay: float = 1.0):
        """
        Initialize Neo4j connection
        
        Args:
            uri: Neo4j connection URI
            username: Database username
            password: Database password
            max_retry_attempts: Maximum number of connection retry attempts
            retry_delay: Delay between retry attempts in seconds
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay = retry_delay
        
        self._driver = None
        self.schema = ContractGraphSchema()
        self._is_connected = False
        
    def connect(self) -> bool:
        """
        Establish connection to Neo4j database with retry logic
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if GraphDatabase is None:
            logger.error("Neo4j driver not installed. Run: pip install neo4j")
            return False
        
        for attempt in range(self.max_retry_attempts):
            try:
                logger.info(f"Attempting to connect to Neo4j at {self.uri} (attempt {attempt + 1})")
                
                self._driver = GraphDatabase.driver(
                    self.uri, 
                    auth=(self.username, self.password)
                )
                
                # Test connection
                with self._driver.session() as session:
                    result = session.run("RETURN 1 as test")
                    test_value = result.single()["test"]
                    if test_value == 1:
                        self._is_connected = True
                        logger.info("Successfully connected to Neo4j")
                        return True
                        
            except (ServiceUnavailable, Exception) as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    logger.error(f"Failed to connect after {self.max_retry_attempts} attempts")
                    
        return False
    
    def disconnect(self):
        """Close the database connection"""
        if self._driver:
            self._driver.close()
            self._is_connected = False
            logger.info("Disconnected from Neo4j")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to database"""
        return self._is_connected and self._driver is not None
    
    @contextmanager
    def session(self):
        """Context manager for database sessions"""
        if not self.is_connected:
            raise ConnectionError("Not connected to Neo4j database")
        
        session = self._driver.session()
        try:
            yield session
        finally:
            session.close()
    
    def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records as dictionaries
        """
        if parameters is None:
            parameters = {}
            
        try:
            with self.session() as session:
                result = session.run(query, parameters)
                return [dict(record) for record in result]
                
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    def execute_write_transaction(self, query: str, parameters: Dict[str, Any] = None) -> Any:
        """
        Execute a write transaction
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            Transaction result
        """
        if parameters is None:
            parameters = {}
        
        def _write_tx(tx):
            return tx.run(query, parameters)
        
        try:
            with self.session() as session:
                result = session.execute_write(_write_tx)
                return result
                
        except Exception as e:
            logger.error(f"Write transaction failed: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    def setup_schema(self) -> bool:
        """
        Set up the database schema (constraints and indexes)
        
        Returns:
            bool: True if setup successful
        """
        try:
            logger.info("Setting up Neo4j schema...")
            
            # Execute schema setup statements
            schema_statements = self.schema.get_cypher_schema_setup()
            
            for statement in schema_statements:
                try:
                    self.execute_write_transaction(statement)
                    logger.debug(f"Executed: {statement}")
                except Exception as e:
                    # Some constraints/indexes might already exist - that's OK
                    logger.debug(f"Schema statement warning: {e}")
            
            # Create standard clause types
            clause_type_nodes = create_standard_clause_types()
            self.bulk_create_nodes(clause_type_nodes)
            
            logger.info("Schema setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Schema setup failed: {e}")
            return False
    
    def clear_database(self) -> bool:
        """
        Clear all data from the database (use with caution!)
        
        Returns:
            bool: True if successful
        """
        try:
            logger.warning("Clearing all data from Neo4j database...")
            
            # Delete all relationships first, then nodes
            self.execute_write_transaction("MATCH ()-[r]-() DELETE r")
            self.execute_write_transaction("MATCH (n) DELETE n")
            
            logger.info("Database cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear database: {e}")
            return False

class GraphDataManager:
    """High-level interface for managing graph data"""
    
    def __init__(self, connection: Neo4jConnection):
        """
        Initialize with a Neo4j connection
        
        Args:
            connection: Neo4jConnection instance
        """
        self.conn = connection
        self.schema = connection.schema
        
    def create_node(self, node: GraphNode) -> bool:
        """
        Create a single node in the graph
        
        Args:
            node: GraphNode instance to create
            
        Returns:
            bool: True if successful
        """
        # Validate node against schema
        validation_errors = self.schema.validate_node(node)
        if validation_errors:
            logger.error(f"Node validation failed: {validation_errors}")
            return False
        
        try:
            query = node.to_cypher_merge()  # Use MERGE to avoid duplicates
            parameters = {"node_id": node.node_id, **node.properties}
            
            self.conn.execute_write_transaction(query, parameters)
            logger.debug(f"Created node: {node.node_type.value} ({node.node_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create node {node.node_id}: {e}")
            return False
    
    def bulk_create_nodes(self, nodes: List[GraphNode], batch_size: int = 100) -> int:
        """
        Create multiple nodes efficiently using batching
        
        Args:
            nodes: List of GraphNode instances
            batch_size: Number of nodes to create per batch
            
        Returns:
            int: Number of nodes successfully created
        """
        created_count = 0
        
        # Process nodes in batches
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            
            try:
                # Build batch query
                query_parts = []
                parameters = {}
                
                for j, node in enumerate(batch):
                    # Validate node
                    validation_errors = self.schema.validate_node(node)
                    if validation_errors:
                        logger.warning(f"Skipping invalid node {node.node_id}: {validation_errors}")
                        continue
                    
                    # Add to batch query
                    node_var = f"n{j}"
                    props_str = ", ".join([f"{k}: $n{j}_{k}" for k in node.properties.keys()])
                    query_parts.append(f"MERGE ({node_var}:{node.node_type.value} {{id: $n{j}_node_id}}) SET {node_var} += {{{props_str}}}")
                    
                    # Add parameters
                    parameters[f"n{j}_node_id"] = node.node_id
                    for k, v in node.properties.items():
                        parameters[f"n{j}_{k}"] = v
                
                if query_parts:
                    query = "\n".join(query_parts)
                    self.conn.execute_write_transaction(query, parameters)
                    created_count += len([n for n in batch if not self.schema.validate_node(n)])
                    
                    logger.debug(f"Created batch of {len(batch)} nodes (total: {created_count})")
                    
            except Exception as e:
                logger.error(f"Failed to create batch starting at index {i}: {e}")
        
        logger.info(f"Successfully created {created_count} nodes")
        return created_count
    
    def create_relationship(self, relationship: GraphRelationship) -> bool:
        """
        Create a relationship between two nodes
        
        Args:
            relationship: GraphRelationship instance
            
        Returns:
            bool: True if successful
        """
        try:
            query = relationship.to_cypher()
            parameters = {
                "from_node_id": relationship.from_node_id,
                "to_node_id": relationship.to_node_id,
                **relationship.properties
            }
            
            self.conn.execute_write_transaction(query, parameters)
            logger.debug(f"Created relationship: {relationship.from_node_id} -[{relationship.relationship_type.value}]-> {relationship.to_node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    def bulk_create_relationships(self, relationships: List[GraphRelationship], batch_size: int = 100) -> int:
        """
        Create multiple relationships efficiently
        
        Args:
            relationships: List of GraphRelationship instances
            batch_size: Number of relationships to create per batch
            
        Returns:
            int: Number of relationships successfully created
        """
        created_count = 0
        
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            
            try:
                # Create each relationship in the batch
                for rel in batch:
                    if self.create_relationship(rel):
                        created_count += 1
                        
                logger.debug(f"Created batch of relationships (total: {created_count})")
                
            except Exception as e:
                logger.error(f"Failed to create relationship batch starting at index {i}: {e}")
        
        logger.info(f"Successfully created {created_count} relationships")
        return created_count
    
    def find_nodes_by_type(self, node_type: NodeType, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Find nodes by type
        
        Args:
            node_type: Type of nodes to find
            limit: Maximum number of results
            
        Returns:
            List of node data dictionaries
        """
        query = f"MATCH (n:{node_type.value}) RETURN n LIMIT $limit"
        results = self.conn.execute_query(query, {"limit": limit})
        return [result["n"] for result in results]
    
    def find_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Find a node by its ID
        
        Args:
            node_id: Unique node identifier
            
        Returns:
            Node data dictionary or None if not found
        """
        query = "MATCH (n {id: $node_id}) RETURN n"
        results = self.conn.execute_query(query, {"node_id": node_id})
        return results[0]["n"] if results else None
    
    def get_node_relationships(self, node_id: str, direction: str = "both") -> List[Dict[str, Any]]:
        """
        Get all relationships for a node
        
        Args:
            node_id: Node identifier
            direction: 'incoming', 'outgoing', or 'both'
            
        Returns:
            List of relationship data
        """
        if direction == "outgoing":
            query = "MATCH (n {id: $node_id})-[r]->(m) RETURN r, m"
        elif direction == "incoming":
            query = "MATCH (n {id: $node_id})<-[r]-(m) RETURN r, m"
        else:  # both
            query = "MATCH (n {id: $node_id})-[r]-(m) RETURN r, m"
        
        return self.conn.execute_query(query, {"node_id": node_id})
    
    def execute_sample_queries(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Execute the sample queries defined in the schema
        
        Returns:
            Dictionary mapping query names to results
        """
        results = {}
        sample_queries = self.schema.get_sample_queries()
        
        for query_name, query in sample_queries.items():
            try:
                # Use default parameters for sample queries
                default_params = {
                    "party_name": "Corp",
                    "clause_type": "limitation_of_liability",
                    "threshold": 10000,
                    "clause_id": "sample_clause_1"
                }
                
                result = self.conn.execute_query(query, default_params)
                results[query_name] = result
                logger.info(f"Sample query '{query_name}' returned {len(result)} results")
                
            except Exception as e:
                logger.error(f"Sample query '{query_name}' failed: {e}")
                results[query_name] = []
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current database state
        
        Returns:
            Dictionary with database statistics
        """
        stats = {}
        
        try:
            # Count nodes by type
            node_counts = {}
            for node_type in NodeType:
                query = f"MATCH (n:{node_type.value}) RETURN count(n) as count"
                result = self.conn.execute_query(query)
                node_counts[node_type.value] = result[0]["count"] if result else 0
            
            stats["node_counts"] = node_counts
            stats["total_nodes"] = sum(node_counts.values())
            
            # Count relationships by type
            rel_counts = {}
            for rel_type in RelationshipType:
                query = f"MATCH ()-[r:{rel_type.value}]-() RETURN count(r) as count"
                result = self.conn.execute_query(query)
                rel_counts[rel_type.value] = result[0]["count"] if result else 0
            
            stats["relationship_counts"] = rel_counts
            stats["total_relationships"] = sum(rel_counts.values())
            
            # Database info
            db_info_query = "CALL db.info()"
            db_info = self.conn.execute_query(db_info_query)
            if db_info:
                stats["database_info"] = db_info[0]
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            stats["error"] = str(e)
        
        return stats

# Factory function for easy instantiation
def create_graph_connection(uri: str = None, username: str = None, password: str = None) -> Tuple[Neo4jConnection, GraphDataManager]:
    """
    Factory function to create a Neo4j connection and data manager
    
    Args:
        uri: Neo4j URI (defaults to environment variable or localhost)
        username: Neo4j username (defaults to environment variable or 'neo4j')  
        password: Neo4j password (defaults to environment variable or 'password')
        
    Returns:
        Tuple of (Neo4jConnection, GraphDataManager)
    """
    # Use environment variables or defaults
    uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
    username = username or os.getenv("NEO4J_USERNAME", "neo4j")
    password = password or os.getenv("NEO4J_PASSWORD", "password")
    
    # Create connection
    connection = Neo4jConnection(uri=uri, username=username, password=password)
    
    # Create data manager
    data_manager = GraphDataManager(connection)
    
    return connection, data_manager