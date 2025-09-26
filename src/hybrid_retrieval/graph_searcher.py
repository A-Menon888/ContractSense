"""
Graph Searcher for Hybrid Retrieval Engine

This module implements sophisticated graph traversal and Cypher query generation
for legal document search, leveraging the knowledge graph built in Module 5.

Key Features:
- Automatic Cypher query generation from natural language
- Multi-hop reasoning through contract relationships
- Exact matching for entities, amounts, and dates
- Complex constraint handling and optimization

Author: ContractSense Team  
Date: 2025-09-25
Version: 1.0.0
"""

import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .hybrid_engine import SearchResult
import re
from enum import Enum

@dataclass
class GraphSearchConfig:
    """Configuration for graph search operations"""
    max_traversal_depth: int = 3
    enable_cypher_optimization: bool = True
    timeout_seconds: int = 10
    max_results: int = 50
    enable_caching: bool = True
    cache_ttl_seconds: int = 300

class QueryType(Enum):
    """Types of graph queries"""
    ENTITY_LOOKUP = "entity_lookup"
    RELATIONSHIP_TRAVERSAL = "relationship_traversal"  
    CLAUSE_SEARCH = "clause_search"
    MONETARY_FILTER = "monetary_filter"
    TEMPORAL_FILTER = "temporal_filter"
    COMPLEX_JOIN = "complex_join"

class GraphSearcher:
    """
    Graph searcher that converts natural language queries to Cypher
    and executes them against the knowledge graph from Module 5
    """
    
    def __init__(self, config: GraphSearchConfig, knowledge_graph_integrator):
        self.config = config
        self.kg_integrator = knowledge_graph_integrator
        self.logger = logging.getLogger(__name__)
        
        # Cypher query templates
        self._load_query_templates()
        
        # Query optimization patterns
        self._load_optimization_patterns()
        
        # Query cache
        self.query_cache: Dict[str, Tuple[List, float]] = {}
        
        # Statistics
        self.stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_execution_time": 0.0,
            "cypher_generation_time": 0.0,
            "graph_traversal_time": 0.0
        }
        
        self.logger.info("Graph searcher initialized")
    
    def _load_query_templates(self):
        """Load Cypher query templates for different search patterns"""
        
        self.templates = {
            # Find parties by name
            'party_lookup': """
                MATCH (p:Party)
                WHERE toLower(p.name) CONTAINS toLower($party_name)
                RETURN p
                """,
            
            # Find agreements involving specific parties
            'party_agreements': """
                MATCH (p:Party)-[:PARTY_TO]->(a:Agreement)
                WHERE toLower(p.name) CONTAINS toLower($party_name)
                RETURN a, p
                ORDER BY a.date_signed DESC
                """,
            
            # Find clauses of specific type
            'clause_by_type': """
                MATCH (a:Agreement)-[:CONTAINS]->(c:Clause)
                WHERE c.type = $clause_type
                RETURN a, c
                """,
            
            # Find clauses with monetary constraints  
            'monetary_clauses': """
                MATCH (a:Agreement)-[:CONTAINS]->(c:Clause)-[:HAS_MONETARY_TERM]->(m:MonetaryTerm)
                WHERE m.amount >= $min_amount AND m.amount <= $max_amount
                RETURN a, c, m
                ORDER BY m.amount DESC
                """,
            
            # Find clauses with temporal constraints
            'temporal_clauses': """
                MATCH (a:Agreement)-[:CONTAINS]->(c:Clause)-[:HAS_TEMPORAL_TERM]->(t:TemporalTerm)
                WHERE t.duration_days >= $min_days AND t.duration_days <= $max_days
                RETURN a, c, t
                ORDER BY t.duration_days
                """,
            
            # Multi-hop relationship traversal
            'relationship_path': """
                MATCH path=(start:Party)-[*1..{max_depth}]-(end:Party)
                WHERE toLower(start.name) CONTAINS toLower($start_party)
                  AND toLower(end.name) CONTAINS toLower($end_party)
                RETURN path, start, end
                """,
            
            # Complex search with multiple constraints
            'complex_search': """
                MATCH (p1:Party)-[:PARTY_TO]->(a:Agreement)<-[:PARTY_TO]-(p2:Party),
                      (a)-[:CONTAINS]->(c:Clause)
                WHERE c.type IN $clause_types
                  AND ($party_filter IS NULL OR 
                       toLower(p1.name) CONTAINS toLower($party_filter) OR
                       toLower(p2.name) CONTAINS toLower($party_filter))
                RETURN a, c, p1, p2
                """,
            
            # Find agreements with liability caps
            'liability_caps': """
                MATCH (a:Agreement)-[:CONTAINS]->(c:Clause {type: 'liability'})-[:HAS_MONETARY_TERM]->(m:MonetaryTerm)
                WHERE m.amount > 0
                OPTIONAL MATCH (p:Party)-[:PARTY_TO]->(a)
                RETURN a, c, m, collect(p.name) as parties
                ORDER BY m.amount DESC
                """,
            
            # Find termination clauses with notice periods
            'termination_notice': """
                MATCH (a:Agreement)-[:CONTAINS]->(c:Clause {type: 'termination'})-[:HAS_TEMPORAL_TERM]->(t:TemporalTerm)
                WHERE t.duration_days > 0
                OPTIONAL MATCH (p:Party)-[:PARTY_TO]->(a)
                RETURN a, c, t, collect(p.name) as parties
                ORDER BY t.duration_days
                """,
            
            # Full-text search on clause content
            'clause_content_search': """
                MATCH (a:Agreement)-[:CONTAINS]->(c:Clause)
                WHERE toLower(c.content) CONTAINS toLower($search_term)
                OPTIONAL MATCH (p:Party)-[:PARTY_TO]->(a)
                RETURN a, c, collect(p.name) as parties
                """,
            
            # Find similar clauses across agreements
            'similar_clauses': """
                MATCH (a1:Agreement)-[:CONTAINS]->(c1:Clause),
                      (a2:Agreement)-[:CONTAINS]->(c2:Clause)
                WHERE c1.type = c2.type 
                  AND id(c1) < id(c2)
                  AND a1.id <> a2.id
                RETURN c1, c2, a1, a2
                """,
            
            # Agreement timeline and relationships
            'agreement_timeline': """
                MATCH (a:Agreement)
                WHERE a.date_signed >= $start_date AND a.date_signed <= $end_date
                OPTIONAL MATCH (p:Party)-[:PARTY_TO]->(a)
                OPTIONAL MATCH (a)-[:CONTAINS]->(c:Clause)
                RETURN a, collect(DISTINCT p.name) as parties, count(c) as clause_count
                ORDER BY a.date_signed
                """
        }
    
    def _load_optimization_patterns(self):
        """Load query optimization patterns"""
        
        self.optimizations = {
            # Use indexes for party name lookups
            'party_name_index': r'WHERE\s+toLower\(p\.name\)\s+CONTAINS\s+toLower\(',
            
            # Optimize date range queries
            'date_range_index': r'WHERE\s+.*\.date_signed\s+>=.*AND.*\.date_signed\s+<=',
            
            # Use clause type indexes
            'clause_type_index': r'WHERE\s+c\.type\s+=',
            
            # Limit early in complex queries
            'early_limit': r'RETURN.*ORDER BY.*(?!LIMIT)',
        }
    
    async def search(self, query) -> List['SearchResult']:
        """
        Execute graph search based on processed query
        
        Args:
            query: SearchQuery object with extracted entities and metadata
            
        Returns:
            List of SearchResult objects from graph traversal
        """
        from .hybrid_engine import SearchResult  # Import from same package
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._build_cache_key(query)
            if self.config.enable_caching and cache_key in self.query_cache:
                cached_results, cache_time = self.query_cache[cache_key]
                if time.time() - cache_time < self.config.cache_ttl_seconds:
                    self.stats["cache_hits"] += 1
                    self.logger.debug(f"Graph search cache hit for query: {query.text}")
                    return cached_results
            
            self.stats["cache_misses"] += 1
            
            # Generate Cypher query
            cypher_start = time.time()
            cypher_queries = self._generate_cypher_queries(query)
            self.stats["cypher_generation_time"] += time.time() - cypher_start
            
            if not cypher_queries:
                self.logger.warning("No Cypher queries generated for query")
                return []
            
            # Execute queries against knowledge graph
            graph_start = time.time()
            all_results = []
            
            for cypher, params in cypher_queries:
                try:
                    # Execute query through knowledge graph integrator
                    graph_results = await self._execute_cypher_query(cypher, params)
                    
                    # Convert to SearchResult objects
                    search_results = self._convert_graph_results(graph_results, query)
                    all_results.extend(search_results)
                    
                except Exception as e:
                    self.logger.error(f"Cypher query execution failed: {e}")
                    self.logger.debug(f"Failed query: {cypher}")
                    continue
            
            self.stats["graph_traversal_time"] += time.time() - graph_start
            
            # Remove duplicates and sort by confidence
            unique_results = self._deduplicate_results(all_results)
            
            # Limit results
            final_results = unique_results[:self.config.max_results]
            
            # Cache results
            if self.config.enable_caching:
                self.query_cache[cache_key] = (final_results, time.time())
            
            # Update statistics
            total_time = time.time() - start_time
            self.stats["queries_executed"] += 1
            self.stats["total_execution_time"] += total_time
            
            self.logger.debug(
                f"Graph search completed: {len(final_results)} results "
                f"in {total_time:.3f}s"
            )
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Graph search failed: {e}")
            return []
    
    def _generate_cypher_queries(self, query) -> List[Tuple[str, Dict[str, Any]]]:
        """Generate Cypher queries based on query analysis"""
        
        queries = []
        
        # Get processing metadata if available
        metadata = getattr(query, 'processing_metadata', {})
        intent = metadata.get('intent', 'unknown')
        
        # Strategy 1: Party-based queries
        if query.entities:
            for entity in query.entities:
                # Find agreements involving this party
                queries.append((
                    self.templates['party_agreements'],
                    {'party_name': entity}
                ))
                
                # Find relationships between parties if multiple entities
                if len(query.entities) > 1:
                    for other_entity in query.entities:
                        if entity != other_entity:
                            template = self.templates['relationship_path'].format(
                                max_depth=self.config.max_traversal_depth
                            )
                            queries.append((
                                template,
                                {'start_party': entity, 'end_party': other_entity}
                            ))
        
        # Strategy 2: Clause type queries
        if query.clause_types:
            for clause_type in query.clause_types:
                queries.append((
                    self.templates['clause_by_type'],
                    {'clause_type': clause_type}
                ))
                
                # Add specific templates for known clause types
                if clause_type == 'liability' and query.amounts:
                    queries.append((
                        self.templates['liability_caps'],
                        {}
                    ))
                elif clause_type == 'termination':
                    queries.append((
                        self.templates['termination_notice'],
                        {}
                    ))
        
        # Strategy 3: Monetary constraints
        if query.amounts:
            for amount_data in query.amounts:
                amount_value = amount_data.get('value', 0)
                
                # Create range around the specified amount
                min_amount = amount_value * 0.8
                max_amount = amount_value * 1.2
                
                queries.append((
                    self.templates['monetary_clauses'],
                    {'min_amount': min_amount, 'max_amount': max_amount}
                ))
        
        # Strategy 4: Date range queries
        if query.date_ranges:
            for date_data in query.date_ranges:
                if date_data.get('type') == 'range':
                    start_date = date_data.get('start', '2000-01-01')
                    end_date = date_data.get('end', '2030-12-31')
                    
                    queries.append((
                        self.templates['agreement_timeline'],
                        {'start_date': start_date, 'end_date': end_date}
                    ))
        
        # Strategy 5: Complex multi-constraint queries
        if query.entities and query.clause_types:
            queries.append((
                self.templates['complex_search'],
                {
                    'clause_types': query.clause_types,
                    'party_filter': query.entities[0] if query.entities else None
                }
            ))
        
        # Strategy 6: Full-text search fallback
        if not queries or intent == 'clause_extraction':
            # Use original query text for content search
            search_terms = self._extract_search_terms(query.text)
            for term in search_terms[:3]:  # Limit to avoid too many queries
                queries.append((
                    self.templates['clause_content_search'],
                    {'search_term': term}
                ))
        
        # Optimize queries if enabled
        if self.config.enable_cypher_optimization:
            queries = self._optimize_queries(queries)
        
        self.logger.debug(f"Generated {len(queries)} Cypher queries")
        return queries
    
    def _extract_search_terms(self, text: str) -> List[str]:
        """Extract meaningful search terms from query text"""
        
        # Remove stop words and extract meaningful terms
        stop_words = {'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
                     'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'can', 'must'}
        
        # Extract words, removing punctuation
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out stop words
        meaningful_words = [word for word in words if word not in stop_words]
        
        # Return unique terms, prioritizing longer words
        return sorted(set(meaningful_words), key=len, reverse=True)
    
    def _optimize_queries(self, queries: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
        """Apply optimization patterns to Cypher queries"""
        
        optimized = []
        
        for query, params in queries:
            optimized_query = query
            
            # Apply optimization patterns
            for optimization, pattern in self.optimizations.items():
                if re.search(pattern, query, re.IGNORECASE):
                    if optimization == 'early_limit':
                        # Add LIMIT clause if missing in complex queries
                        if 'LIMIT' not in query and len(query.split('\n')) > 3:
                            optimized_query = query.rstrip() + f'\nLIMIT {self.config.max_results}'
            
            optimized.append((optimized_query, params))
        
        return optimized
    
    async def _execute_cypher_query(self, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute a Cypher query against the knowledge graph"""
        
        try:
            # Execute through knowledge graph integrator
            if hasattr(self.kg_integrator, 'execute_cypher_query'):
                return await self.kg_integrator.execute_cypher_query(cypher, params)
            elif hasattr(self.kg_integrator, 'query_graph'):
                return await self.kg_integrator.query_graph(cypher, params)
            else:
                # Fallback to direct graph access
                self.logger.warning("Knowledge graph integrator missing query method")
                return []
                
        except Exception as e:
            self.logger.error(f"Cypher execution error: {e}")
            raise
    
    def _convert_graph_results(self, graph_results: List[Dict[str, Any]], query) -> List['SearchResult']:
        """Convert raw graph results to SearchResult objects"""
        from .hybrid_engine import SearchResult  # Import from same package
        
        search_results = []
        
        for result in graph_results:
            try:
                # Extract relevant information from graph result
                content = ""
                document_id = ""
                clause_id = ""
                document_title = ""
                clause_type = ""
                source_path = []
                metadata = {}
                
                # Handle different result structures
                if 'c' in result and result['c']:  # Clause result
                    clause = result['c']
                    content = clause.get('content', '')
                    clause_id = clause.get('id', '')
                    clause_type = clause.get('type', '')
                    
                    if 'a' in result and result['a']:  # Associated agreement
                        agreement = result['a']
                        document_id = agreement.get('id', '')
                        document_title = agreement.get('title', '')
                        source_path.append(f"Agreement:{document_id}")
                        source_path.append(f"Clause:{clause_id}")
                
                elif 'a' in result and result['a']:  # Agreement-only result
                    agreement = result['a']
                    document_id = agreement.get('id', '')
                    document_title = agreement.get('title', '')
                    content = agreement.get('summary', agreement.get('title', ''))
                    source_path.append(f"Agreement:{document_id}")
                
                elif 'p' in result and result['p']:  # Party result
                    party = result['p']
                    content = f"Party: {party.get('name', '')}"
                    metadata['party_type'] = party.get('type', '')
                    source_path.append(f"Party:{party.get('id', '')}")
                
                # Calculate graph confidence score
                graph_score = self._calculate_graph_confidence(result, query)
                
                # Add metadata from result
                if 'parties' in result:
                    metadata['parties'] = result['parties']
                if 'm' in result and result['m']:  # Monetary term
                    metadata['amount'] = result['m'].get('amount')
                    metadata['currency'] = result['m'].get('currency', 'USD')
                if 't' in result and result['t']:  # Temporal term  
                    metadata['duration_days'] = result['t'].get('duration_days')
                
                if content and document_id:  # Only create result if we have meaningful content
                    search_result = SearchResult(
                        content=content,
                        document_id=document_id,
                        clause_id=clause_id,
                        final_score=graph_score,
                        graph_score=graph_score,
                        vector_score=0.0,  # Will be set during fusion
                        confidence=graph_score,
                        source_path=source_path,
                        document_title=document_title,
                        clause_type=clause_type,
                        metadata=metadata,
                        explanation=f"Found via graph traversal: {' -> '.join(source_path)}"
                    )
                    
                    search_results.append(search_result)
                
            except Exception as e:
                self.logger.error(f"Error converting graph result: {e}")
                continue
        
        return search_results
    
    def _calculate_graph_confidence(self, result: Dict[str, Any], query) -> float:
        """Calculate confidence score for graph search result"""
        
        base_confidence = 0.5
        
        # Boost for exact entity matches
        if query.entities:
            if 'parties' in result:
                for entity in query.entities:
                    if any(entity.lower() in party.lower() for party in result['parties']):
                        base_confidence += 0.2
        
        # Boost for clause type matches
        if query.clause_types and 'c' in result and result['c']:
            clause = result['c']
            if clause.get('type') in query.clause_types:
                base_confidence += 0.3
        
        # Boost for monetary matches
        if query.amounts and 'm' in result and result['m']:
            base_confidence += 0.2
        
        # Boost for temporal matches
        if query.date_ranges and 't' in result and result['t']:
            base_confidence += 0.2
        
        return min(base_confidence, 1.0)
    
    def _deduplicate_results(self, results: List['SearchResult']) -> List['SearchResult']:
        """Remove duplicate results based on content similarity"""
        
        unique_results = []
        seen_content = set()
        
        for result in results:
            # Use combination of document_id and clause_id as deduplication key
            dedup_key = f"{result.document_id}:{result.clause_id}:{result.clause_type}"
            
            if dedup_key not in seen_content:
                seen_content.add(dedup_key)
                unique_results.append(result)
        
        # Sort by graph confidence score
        return sorted(unique_results, key=lambda x: x.graph_score, reverse=True)
    
    def _build_cache_key(self, query) -> str:
        """Build cache key for query"""
        import hashlib
        
        key_components = [
            query.text,
            str(sorted(query.entities)),
            str(sorted(query.clause_types)),
            str(len(query.amounts)),
            str(len(query.date_ranges))
        ]
        
        key_string = '|'.join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get graph searcher statistics"""
        avg_execution_time = 0.0
        if self.stats["queries_executed"] > 0:
            avg_execution_time = self.stats["total_execution_time"] / self.stats["queries_executed"]
        
        cache_hit_rate = 0.0
        total_cache_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
        if total_cache_requests > 0:
            cache_hit_rate = self.stats["cache_hits"] / total_cache_requests
        
        return {
            **self.stats,
            "average_execution_time": avg_execution_time,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.query_cache)
        }
    
    def clear_cache(self):
        """Clear query cache"""
        self.query_cache.clear()
        self.logger.info("Graph searcher cache cleared")

# Factory function
def create_graph_searcher(config: GraphSearchConfig = None, kg_integrator=None) -> GraphSearcher:
    """Create and return a configured graph searcher"""
    if config is None:
        config = GraphSearchConfig()
    return GraphSearcher(config, kg_integrator)