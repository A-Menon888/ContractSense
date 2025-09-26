"""
Performance Monitor for Hybrid Retrieval Engine

This module implements comprehensive performance monitoring, metrics collection,
and optimization recommendations for the hybrid retrieval system.

Key Features:
- Real-time performance metrics
- Query latency tracking
- Cache hit rate monitoring  
- Resource utilization analysis
- Performance optimization recommendations

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import logging
import time
import asyncio
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict
from enum import Enum
import json

@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring"""
    enable_monitoring: bool = True
    metrics_window_size: int = 1000  # Keep last N metrics
    alert_threshold_latency: float = 1.0  # Alert if queries take > 1s
    alert_threshold_cache_miss: float = 0.5  # Alert if cache miss rate > 50%
    enable_resource_monitoring: bool = True
    resource_check_interval: int = 30  # Check resources every 30s
    enable_recommendations: bool = True
    log_slow_queries: bool = True
    slow_query_threshold: float = 2.0

@dataclass
class QueryMetrics:
    """Metrics for individual query execution"""
    query_id: str
    query_text: str
    start_time: float
    end_time: float
    total_latency: float
    
    # Component latencies
    preprocessing_time: float = 0.0
    graph_search_time: float = 0.0
    vector_search_time: float = 0.0
    fusion_time: float = 0.0
    
    # Result counts
    graph_results: int = 0
    vector_results: int = 0
    final_results: int = 0
    
    # Cache performance
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Quality metrics
    average_confidence: float = 0.0
    top_result_confidence: float = 0.0
    
    # Resource usage
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    
    # Success/failure
    success: bool = True
    error_message: Optional[str] = None

@dataclass
class SystemMetrics:
    """System-wide performance metrics"""
    timestamp: float
    
    # Query statistics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    
    # Latency statistics
    average_latency: float = 0.0
    median_latency: float = 0.0
    p95_latency: float = 0.0
    p99_latency: float = 0.0
    
    # Cache statistics
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    
    # Resource statistics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    
    # Component performance
    graph_search_avg_time: float = 0.0
    vector_search_avg_time: float = 0.0
    fusion_avg_time: float = 0.0
    
    # Quality metrics
    average_result_count: float = 0.0
    average_confidence: float = 0.0

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class PerformanceAlert:
    """Performance alert"""
    level: AlertLevel
    message: str
    timestamp: float
    metric_name: str
    threshold: float
    actual_value: float

class PerformanceMonitor:
    """
    Performance monitor for hybrid retrieval engine
    Tracks metrics, generates alerts, and provides optimization recommendations
    """
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.query_metrics: deque = deque(maxlen=config.metrics_window_size)
        self.system_metrics_history: deque = deque(maxlen=100)  # Keep 100 system snapshots
        
        # Current system metrics
        self.current_metrics = SystemMetrics(timestamp=time.time())
        
        # Alert management
        self.alerts: deque = deque(maxlen=200)  # Keep last 200 alerts
        self.alert_cooldowns: Dict[str, float] = {}  # Prevent alert spam
        
        # Performance tracking
        self.active_queries: Dict[str, QueryMetrics] = {}
        
        # Component stats
        self.component_stats = {
            'graph_searcher': {'total_time': 0.0, 'query_count': 0},
            'vector_searcher': {'total_time': 0.0, 'query_count': 0},
            'result_fusion': {'total_time': 0.0, 'query_count': 0},
            'query_processor': {'total_time': 0.0, 'query_count': 0}
        }
        
        # Resource monitoring task
        self.resource_monitor_task: Optional[asyncio.Task] = None
        
        if self.config.enable_monitoring:
            self._start_resource_monitoring()
        
        self.logger.info("Performance monitor initialized")
    
    def start_query(self, query_id: str, query_text: str) -> QueryMetrics:
        """Start tracking a new query"""
        
        if not self.config.enable_monitoring:
            return None
        
        metrics = QueryMetrics(
            query_id=query_id,
            query_text=query_text,
            start_time=time.time(),
            end_time=0.0,
            total_latency=0.0
        )
        
        self.active_queries[query_id] = metrics
        return metrics
    
    def end_query(self, query_id: str, success: bool = True, error_message: str = None) -> QueryMetrics:
        """End tracking for a query"""
        
        if not self.config.enable_monitoring or query_id not in self.active_queries:
            return None
        
        metrics = self.active_queries.pop(query_id)
        metrics.end_time = time.time()
        metrics.total_latency = metrics.end_time - metrics.start_time
        metrics.success = success
        metrics.error_message = error_message
        
        # Add resource usage if monitoring enabled
        if self.config.enable_resource_monitoring:
            metrics.cpu_usage = psutil.cpu_percent()
            metrics.memory_usage = psutil.virtual_memory().percent
        
        # Store metrics
        self.query_metrics.append(metrics)
        
        # Update system metrics
        self._update_system_metrics()
        
        # Check for alerts
        self._check_alerts(metrics)
        
        # Log slow queries
        if (self.config.log_slow_queries and 
            metrics.total_latency > self.config.slow_query_threshold):
            self.logger.warning(
                f"Slow query detected: {query_id} took {metrics.total_latency:.3f}s"
            )
        
        return metrics
    
    def record_component_time(self, component: str, duration: float):
        """Record execution time for a component"""
        
        if component in self.component_stats:
            stats = self.component_stats[component]
            stats['total_time'] += duration
            stats['query_count'] += 1
    
    def record_cache_hit(self, query_id: str):
        """Record a cache hit"""
        
        if query_id in self.active_queries:
            self.active_queries[query_id].cache_hits += 1
    
    def record_cache_miss(self, query_id: str):
        """Record a cache miss"""
        
        if query_id in self.active_queries:
            self.active_queries[query_id].cache_misses += 1
    
    def record_results(self, query_id: str, graph_count: int, vector_count: int, 
                      final_count: int, avg_confidence: float):
        """Record result counts and quality metrics"""
        
        if query_id in self.active_queries:
            metrics = self.active_queries[query_id]
            metrics.graph_results = graph_count
            metrics.vector_results = vector_count
            metrics.final_results = final_count
            metrics.average_confidence = avg_confidence
    
    def _update_system_metrics(self):
        """Update system-wide metrics"""
        
        if not self.query_metrics:
            return
        
        # Calculate query statistics
        recent_queries = list(self.query_metrics)
        successful = [q for q in recent_queries if q.success]
        failed = [q for q in recent_queries if not q.success]
        
        self.current_metrics.timestamp = time.time()
        self.current_metrics.total_queries = len(recent_queries)
        self.current_metrics.successful_queries = len(successful)
        self.current_metrics.failed_queries = len(failed)
        
        # Latency statistics
        if successful:
            latencies = [q.total_latency for q in successful]
            latencies.sort()
            
            self.current_metrics.average_latency = sum(latencies) / len(latencies)
            self.current_metrics.median_latency = latencies[len(latencies) // 2]
            
            if len(latencies) >= 20:  # Only calculate percentiles with sufficient data
                p95_idx = int(len(latencies) * 0.95)
                p99_idx = int(len(latencies) * 0.99)
                self.current_metrics.p95_latency = latencies[p95_idx]
                self.current_metrics.p99_latency = latencies[p99_idx]
        
        # Cache statistics
        total_hits = sum(q.cache_hits for q in recent_queries)
        total_misses = sum(q.cache_misses for q in recent_queries)
        total_cache_requests = total_hits + total_misses
        
        if total_cache_requests > 0:
            self.current_metrics.cache_hit_rate = total_hits / total_cache_requests
        
        # Component performance
        for component, stats in self.component_stats.items():
            if stats['query_count'] > 0:
                avg_time = stats['total_time'] / stats['query_count']
                
                if component == 'graph_searcher':
                    self.current_metrics.graph_search_avg_time = avg_time
                elif component == 'vector_searcher':
                    self.current_metrics.vector_search_avg_time = avg_time
                elif component == 'result_fusion':
                    self.current_metrics.fusion_avg_time = avg_time
        
        # Quality metrics
        if successful:
            confidences = [q.average_confidence for q in successful if q.average_confidence > 0]
            if confidences:
                self.current_metrics.average_confidence = sum(confidences) / len(confidences)
            
            result_counts = [q.final_results for q in successful]
            self.current_metrics.average_result_count = sum(result_counts) / len(result_counts)
        
        # Resource usage (if enabled)
        if self.config.enable_resource_monitoring:
            self.current_metrics.cpu_usage = psutil.cpu_percent()
            self.current_metrics.memory_usage = psutil.virtual_memory().percent
        
        # Store system metrics snapshot
        self.system_metrics_history.append(SystemMetrics(**self.current_metrics.__dict__))
    
    def _check_alerts(self, query_metrics: QueryMetrics):
        """Check for performance alerts"""
        
        current_time = time.time()
        
        # High latency alert
        if query_metrics.total_latency > self.config.alert_threshold_latency:
            self._create_alert(
                AlertLevel.WARNING,
                f"High query latency: {query_metrics.total_latency:.3f}s",
                "query_latency",
                self.config.alert_threshold_latency,
                query_metrics.total_latency
            )
        
        # Cache miss rate alert
        total_cache_requests = query_metrics.cache_hits + query_metrics.cache_misses
        if total_cache_requests > 0:
            cache_miss_rate = query_metrics.cache_misses / total_cache_requests
            if cache_miss_rate > self.config.alert_threshold_cache_miss:
                self._create_alert(
                    AlertLevel.WARNING,
                    f"High cache miss rate: {cache_miss_rate:.2f}",
                    "cache_miss_rate",
                    self.config.alert_threshold_cache_miss,
                    cache_miss_rate
                )
        
        # Resource alerts (if enabled)
        if self.config.enable_resource_monitoring:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            if cpu_usage > 90:
                self._create_alert(
                    AlertLevel.ERROR,
                    f"High CPU usage: {cpu_usage:.1f}%",
                    "cpu_usage",
                    90.0,
                    cpu_usage
                )
            
            if memory_usage > 90:
                self._create_alert(
                    AlertLevel.ERROR,
                    f"High memory usage: {memory_usage:.1f}%",
                    "memory_usage",
                    90.0,
                    memory_usage
                )
    
    def _create_alert(self, level: AlertLevel, message: str, metric_name: str, 
                     threshold: float, actual_value: float):
        """Create a performance alert with cooldown"""
        
        # Check cooldown to prevent spam
        cooldown_key = f"{metric_name}_{level.value}"
        cooldown_period = 300  # 5 minutes
        
        current_time = time.time()
        if (cooldown_key in self.alert_cooldowns and 
            current_time - self.alert_cooldowns[cooldown_key] < cooldown_period):
            return
        
        alert = PerformanceAlert(
            level=level,
            message=message,
            timestamp=current_time,
            metric_name=metric_name,
            threshold=threshold,
            actual_value=actual_value
        )
        
        self.alerts.append(alert)
        self.alert_cooldowns[cooldown_key] = current_time
        
        # Log alert
        if level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            self.logger.error(f"Performance Alert: {message}")
        elif level == AlertLevel.WARNING:
            self.logger.warning(f"Performance Alert: {message}")
        else:
            self.logger.info(f"Performance Alert: {message}")
    
    def _start_resource_monitoring(self):
        """Start background resource monitoring"""
        
        if not self.config.enable_resource_monitoring:
            return
        
        async def resource_monitor():
            while True:
                try:
                    # Update resource metrics
                    cpu_usage = psutil.cpu_percent()
                    memory_usage = psutil.virtual_memory().percent
                    
                    self.current_metrics.cpu_usage = cpu_usage
                    self.current_metrics.memory_usage = memory_usage
                    
                    # Check for resource alerts
                    if cpu_usage > 90:
                        self._create_alert(
                            AlertLevel.ERROR,
                            f"Sustained high CPU usage: {cpu_usage:.1f}%",
                            "sustained_cpu_usage",
                            90.0,
                            cpu_usage
                        )
                    
                    if memory_usage > 90:
                        self._create_alert(
                            AlertLevel.ERROR,
                            f"Sustained high memory usage: {memory_usage:.1f}%",
                            "sustained_memory_usage",
                            90.0,
                            memory_usage
                        )
                    
                    await asyncio.sleep(self.config.resource_check_interval)
                    
                except Exception as e:
                    self.logger.error(f"Resource monitoring error: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
        
        self.resource_monitor_task = asyncio.create_task(resource_monitor())
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        summary = {
            "current_metrics": self.current_metrics.__dict__,
            "recent_alerts": [
                {
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "metric_name": alert.metric_name
                }
                for alert in list(self.alerts)[-10:]  # Last 10 alerts
            ],
            "component_performance": {}
        }
        
        # Component performance summary
        for component, stats in self.component_stats.items():
            if stats['query_count'] > 0:
                summary["component_performance"][component] = {
                    "average_time": stats['total_time'] / stats['query_count'],
                    "total_queries": stats['query_count'],
                    "total_time": stats['total_time']
                }
        
        return summary
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        
        if not self.config.enable_recommendations:
            return []
        
        recommendations = []
        
        # Analyze recent metrics
        if len(self.query_metrics) < 10:
            return ["Insufficient data for recommendations"]
        
        recent_queries = list(self.query_metrics)[-100:]  # Last 100 queries
        successful_queries = [q for q in recent_queries if q.success]
        
        if not successful_queries:
            return ["No successful queries to analyze"]
        
        # Latency recommendations
        avg_latency = sum(q.total_latency for q in successful_queries) / len(successful_queries)
        if avg_latency > 1.0:
            recommendations.append(
                f"High average latency ({avg_latency:.3f}s). Consider optimizing query processing or adding more caching."
            )
        
        # Cache recommendations
        total_hits = sum(q.cache_hits for q in recent_queries)
        total_misses = sum(q.cache_misses for q in recent_queries)
        if total_hits + total_misses > 0:
            cache_hit_rate = total_hits / (total_hits + total_misses)
            if cache_hit_rate < 0.7:
                recommendations.append(
                    f"Low cache hit rate ({cache_hit_rate:.2f}). Consider increasing cache size or TTL."
                )
        
        # Component-specific recommendations
        graph_avg = self.current_metrics.graph_search_avg_time
        vector_avg = self.current_metrics.vector_search_avg_time
        
        if graph_avg > vector_avg * 2:
            recommendations.append(
                "Graph search is significantly slower than vector search. Consider optimizing Cypher queries or graph indexes."
            )
        elif vector_avg > graph_avg * 2:
            recommendations.append(
                "Vector search is significantly slower than graph search. Consider optimizing embeddings or vector index."
            )
        
        # Resource recommendations
        if self.current_metrics.cpu_usage > 80:
            recommendations.append(
                f"High CPU usage ({self.current_metrics.cpu_usage:.1f}%). Consider scaling horizontally or optimizing algorithms."
            )
        
        if self.current_metrics.memory_usage > 80:
            recommendations.append(
                f"High memory usage ({self.current_metrics.memory_usage:.1f}%). Consider optimizing caches or memory usage."
            )
        
        # Quality recommendations
        if self.current_metrics.average_confidence < 0.6:
            recommendations.append(
                f"Low average confidence ({self.current_metrics.average_confidence:.2f}). Consider improving query processing or result fusion algorithms."
            )
        
        return recommendations if recommendations else ["Performance is within acceptable ranges"]
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file"""
        
        export_data = {
            "current_metrics": self.current_metrics.__dict__,
            "query_metrics": [q.__dict__ for q in self.query_metrics],
            "system_metrics_history": [m.__dict__ for m in self.system_metrics_history],
            "component_stats": self.component_stats,
            "alerts": [
                {
                    "level": alert.level.value,
                    "message": alert.message,
                    "timestamp": alert.timestamp,
                    "metric_name": alert.metric_name,
                    "threshold": alert.threshold,
                    "actual_value": alert.actual_value
                }
                for alert in self.alerts
            ]
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            self.logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def reset_stats(self):
        """Reset all statistics"""
        
        self.query_metrics.clear()
        self.system_metrics_history.clear()
        self.alerts.clear()
        self.alert_cooldowns.clear()
        self.active_queries.clear()
        
        for component in self.component_stats:
            self.component_stats[component] = {'total_time': 0.0, 'query_count': 0}
        
        self.current_metrics = SystemMetrics(timestamp=time.time())
        
        self.logger.info("Performance statistics reset")
    
    def stop(self):
        """Stop the performance monitor"""
        
        if self.resource_monitor_task:
            self.resource_monitor_task.cancel()
        
        self.logger.info("Performance monitor stopped")

# Factory function
def create_performance_monitor(config: PerformanceConfig = None) -> PerformanceMonitor:
    """Create and return a configured performance monitor"""
    if config is None:
        config = PerformanceConfig()
    return PerformanceMonitor(config)