"""
Performance Monitor for Cross-Encoder Reranking

This module monitors system performance, tracks metrics, and provides
optimization recommendations for the reranking pipeline.

Key Features:
- Real-time performance monitoring
- Latency and throughput tracking
- Resource usage monitoring
- Performance bottleneck detection
- Optimization recommendations

Author: ContractSense Team
Date: 2025-09-25
Version: 1.0.0
"""

import asyncio
import logging
import time
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
import json
from datetime import datetime, timedelta
import statistics
import warnings
import os

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation"""
    operation_type: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    batch_size: int = 1
    memory_used: float = 0.0  # MB
    cpu_percent: float = 0.0
    gpu_memory: float = 0.0  # MB
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealth:
    """System health status"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory_usage: float
    disk_usage: float
    temperature: float
    healthy: bool
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class PerformanceAlert:
    """Performance alert definition"""
    alert_type: str
    threshold: float
    message: str
    severity: str  # "info", "warning", "error", "critical"
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False

@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    time_period: Tuple[float, float]
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_latency: float
    p95_latency: float
    p99_latency: float
    throughput: float  # operations per second
    error_rate: float
    system_health: SystemHealth
    bottlenecks: List[str]
    recommendations: List[str]
    detailed_metrics: Dict[str, Any]

class ResourceMonitor:
    """Monitors system resource usage"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._monitoring = False
        self._monitor_thread = None
        self._resource_history = deque(maxlen=3600)  # 1 hour of data
        self._lock = threading.Lock()
        
        # GPU monitoring (if available)
        self._gpu_available = False
        try:
            import GPUtil
            self._gpu_util = GPUtil
            self._gpu_available = True
        except ImportError:
            self.logger.warning("GPUtil not available, GPU monitoring disabled")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring"""
        
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_resources(self, interval: float):
        """Monitor system resources in background thread"""
        
        while self._monitoring:
            try:
                # CPU and memory
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # Disk usage
                disk = psutil.disk_usage('/')
                
                # GPU monitoring
                gpu_usage = 0.0
                gpu_memory = 0.0
                gpu_temperature = 0.0
                
                if self._gpu_available:
                    try:
                        gpus = self._gpu_util.getGPUs()
                        if gpus:
                            gpu = gpus[0]  # Use first GPU
                            gpu_usage = gpu.load * 100
                            gpu_memory = gpu.memoryUtil * 100
                            gpu_temperature = gpu.temperature
                    except Exception as e:
                        self.logger.debug(f"GPU monitoring error: {e}")
                
                # Temperature (CPU)
                cpu_temperature = 0.0
                try:
                    temps = psutil.sensors_temperatures()
                    if 'coretemp' in temps:
                        cpu_temperature = temps['coretemp'][0].current
                    elif 'cpu_thermal' in temps:
                        cpu_temperature = temps['cpu_thermal'][0].current
                except (AttributeError, OSError):
                    pass  # Temperature monitoring not supported
                
                # Store metrics
                resource_data = {
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_gb': memory.used / (1024**3),
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'gpu_usage': gpu_usage,
                    'gpu_memory': gpu_memory,
                    'cpu_temperature': cpu_temperature,
                    'gpu_temperature': gpu_temperature
                }
                
                with self._lock:
                    self._resource_history.append(resource_data)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval)
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        
        with self._lock:
            if self._resource_history:
                return self._resource_history[-1].copy()
        
        return {}
    
    def get_usage_history(self, duration_minutes: int = 60) -> List[Dict[str, float]]:
        """Get resource usage history"""
        
        cutoff_time = time.time() - (duration_minutes * 60)
        
        with self._lock:
            return [
                data for data in self._resource_history
                if data['timestamp'] > cutoff_time
            ]

class MetricsCollector:
    """Collects and aggregates performance metrics"""
    
    def __init__(self, max_history: int = 100000):
        self.logger = logging.getLogger(__name__)
        self.max_history = max_history
        
        # Metrics storage
        self._metrics_history = deque(maxlen=max_history)
        self._operation_counters = defaultdict(int)
        self._error_counters = defaultdict(int)
        
        # Real-time aggregation
        self._current_window = deque(maxlen=1000)  # Last 1000 operations
        
        # Thread safety
        self._lock = threading.Lock()
    
    def record_operation(self, metrics: PerformanceMetrics):
        """Record performance metrics for an operation"""
        
        with self._lock:
            self._metrics_history.append(metrics)
            self._current_window.append(metrics)
            
            # Update counters
            self._operation_counters[metrics.operation_type] += 1
            if not metrics.success:
                self._error_counters[metrics.operation_type] += 1
    
    def get_operation_stats(self, operation_type: Optional[str] = None,
                           time_window_minutes: int = 60) -> Dict[str, Any]:
        """Get aggregated statistics for operations"""
        
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        with self._lock:
            # Filter metrics by time and operation type
            relevant_metrics = [
                m for m in self._metrics_history
                if m.start_time > cutoff_time and
                (operation_type is None or m.operation_type == operation_type)
            ]
        
        if not relevant_metrics:
            return {
                "total_operations": 0,
                "successful_operations": 0,
                "failed_operations": 0,
                "error_rate": 0.0,
                "average_latency": 0.0,
                "median_latency": 0.0,
                "p95_latency": 0.0,
                "p99_latency": 0.0,
                "throughput": 0.0
            }
        
        # Calculate statistics
        total_ops = len(relevant_metrics)
        successful_ops = sum(1 for m in relevant_metrics if m.success)
        failed_ops = total_ops - successful_ops
        
        latencies = [m.duration for m in relevant_metrics]
        
        # Time-based throughput
        time_span = max(relevant_metrics, key=lambda m: m.end_time).end_time - \
                   min(relevant_metrics, key=lambda m: m.start_time).start_time
        throughput = total_ops / max(time_span, 1)
        
        return {
            "total_operations": total_ops,
            "successful_operations": successful_ops,
            "failed_operations": failed_ops,
            "error_rate": failed_ops / total_ops if total_ops > 0 else 0.0,
            "average_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "p95_latency": np.percentile(latencies, 95) if latencies else 0.0,
            "p99_latency": np.percentile(latencies, 99) if latencies else 0.0,
            "throughput": throughput,
            "time_window_minutes": time_window_minutes,
            "operation_type": operation_type or "all"
        }
    
    def get_operation_types(self) -> List[str]:
        """Get list of monitored operation types"""
        
        with self._lock:
            return list(self._operation_counters.keys())

class PerformanceMonitor:
    """
    Main performance monitoring system
    Orchestrates resource monitoring and metrics collection
    """
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Components
        self.resource_monitor = ResourceMonitor()
        self.metrics_collector = MetricsCollector()
        
        # Alert system
        self._alerts = deque(maxlen=1000)
        self._alert_thresholds = self._initialize_alert_thresholds()
        
        # Performance tracking
        self._active_operations = {}  # operation_id -> start_info
        
        # Optimization tracking
        self._bottleneck_history = deque(maxlen=100)
        self._recommendation_history = deque(maxlen=100)
        
        self.logger.info("Performance monitor initialized")
    
    def _initialize_alert_thresholds(self) -> Dict[str, PerformanceAlert]:
        """Initialize performance alert thresholds"""
        
        return {
            "high_cpu": PerformanceAlert(
                alert_type="cpu_usage",
                threshold=85.0,
                message="High CPU usage detected",
                severity="warning"
            ),
            "high_memory": PerformanceAlert(
                alert_type="memory_usage",
                threshold=90.0,
                message="High memory usage detected",
                severity="error"
            ),
            "high_latency": PerformanceAlert(
                alert_type="latency",
                threshold=5.0,  # 5 seconds
                message="High operation latency detected",
                severity="warning"
            ),
            "high_error_rate": PerformanceAlert(
                alert_type="error_rate",
                threshold=0.05,  # 5%
                message="High error rate detected",
                severity="error"
            ),
            "low_throughput": PerformanceAlert(
                alert_type="throughput",
                threshold=1.0,  # 1 operation per second
                message="Low throughput detected",
                severity="warning"
            )
        }
    
    async def initialize(self):
        """Initialize the performance monitor"""
        
        try:
            self.logger.info("Initializing performance monitor...")
            
            # Start resource monitoring
            self.resource_monitor.start_monitoring(interval=1.0)
            
            # Start periodic health checks
            asyncio.create_task(self._periodic_health_check())
            
            self.logger.info("Performance monitor ready!")
            
        except Exception as e:
            self.logger.error(f"Performance monitor initialization failed: {e}")
            raise
    
    async def start_operation(self, operation_type: str, 
                            operation_id: str, metadata: Optional[Dict] = None) -> str:
        """Start tracking a new operation"""
        
        start_time = time.time()
        
        # Get current resource usage
        resource_usage = self.resource_monitor.get_current_usage()
        
        # Store operation info
        self._active_operations[operation_id] = {
            "operation_type": operation_type,
            "start_time": start_time,
            "start_cpu": resource_usage.get("cpu_percent", 0.0),
            "start_memory": resource_usage.get("memory_used_gb", 0.0) * 1024,  # Convert to MB
            "start_gpu_memory": resource_usage.get("gpu_memory", 0.0),
            "metadata": metadata or {}
        }
        
        return operation_id
    
    async def end_operation(self, operation_id: str, success: bool = True,
                          error_message: str = "", batch_size: int = 1) -> PerformanceMetrics:
        """End tracking an operation and record metrics"""
        
        end_time = time.time()
        
        if operation_id not in self._active_operations:
            self.logger.warning(f"Unknown operation ID: {operation_id}")
            return None
        
        # Get operation info
        op_info = self._active_operations.pop(operation_id)
        
        # Get current resource usage
        resource_usage = self.resource_monitor.get_current_usage()
        
        # Calculate metrics
        duration = end_time - op_info["start_time"]
        memory_used = resource_usage.get("memory_used_gb", 0.0) * 1024 - op_info["start_memory"]
        cpu_percent = resource_usage.get("cpu_percent", 0.0)
        gpu_memory = resource_usage.get("gpu_memory", 0.0)
        
        # Create metrics object
        metrics = PerformanceMetrics(
            operation_type=op_info["operation_type"],
            start_time=op_info["start_time"],
            end_time=end_time,
            duration=duration,
            success=success,
            batch_size=batch_size,
            memory_used=memory_used,
            cpu_percent=cpu_percent,
            gpu_memory=gpu_memory,
            error_message=error_message,
            metadata=op_info["metadata"]
        )
        
        # Record metrics
        self.metrics_collector.record_operation(metrics)
        
        # Check for alerts
        await self._check_operation_alerts(metrics)
        
        return metrics
    
    async def _check_operation_alerts(self, metrics: PerformanceMetrics):
        """Check if operation metrics trigger any alerts"""
        
        # High latency alert
        if metrics.duration > self._alert_thresholds["high_latency"].threshold:
            alert = PerformanceAlert(
                alert_type="high_latency",
                threshold=self._alert_thresholds["high_latency"].threshold,
                message=f"Operation {metrics.operation_type} took {metrics.duration:.2f}s",
                severity="warning"
            )
            self._alerts.append(alert)
    
    async def _periodic_health_check(self):
        """Periodic system health check"""
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Get current system health
                health = await self.get_system_health()
                
                # Check for resource alerts
                if health.cpu_usage > self._alert_thresholds["high_cpu"].threshold:
                    alert = PerformanceAlert(
                        alert_type="high_cpu",
                        threshold=self._alert_thresholds["high_cpu"].threshold,
                        message=f"CPU usage: {health.cpu_usage:.1f}%",
                        severity="warning"
                    )
                    self._alerts.append(alert)
                
                if health.memory_usage > self._alert_thresholds["high_memory"].threshold:
                    alert = PerformanceAlert(
                        alert_type="high_memory",
                        threshold=self._alert_thresholds["high_memory"].threshold,
                        message=f"Memory usage: {health.memory_usage:.1f}%",
                        severity="error"
                    )
                    self._alerts.append(alert)
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
    
    async def get_system_health(self) -> SystemHealth:
        """Get current system health status"""
        
        try:
            resource_usage = self.resource_monitor.get_current_usage()
            
            cpu_usage = resource_usage.get("cpu_percent", 0.0)
            memory_usage = resource_usage.get("memory_percent", 0.0)
            gpu_usage = resource_usage.get("gpu_usage", 0.0)
            gpu_memory_usage = resource_usage.get("gpu_memory", 0.0)
            disk_usage = resource_usage.get("disk_percent", 0.0)
            temperature = resource_usage.get("cpu_temperature", 0.0)
            
            # Determine health status
            warnings_list = []
            recommendations = []
            
            if cpu_usage > 80:
                warnings_list.append(f"High CPU usage: {cpu_usage:.1f}%")
                recommendations.append("Consider reducing concurrent operations or optimizing CPU-intensive tasks")
            
            if memory_usage > 85:
                warnings_list.append(f"High memory usage: {memory_usage:.1f}%")
                recommendations.append("Consider implementing memory optimization or increasing available RAM")
            
            if gpu_memory_usage > 90:
                warnings_list.append(f"High GPU memory usage: {gpu_memory_usage:.1f}%")
                recommendations.append("Consider reducing batch sizes or implementing GPU memory optimization")
            
            if temperature > 80:
                warnings_list.append(f"High CPU temperature: {temperature:.1f}°C")
                recommendations.append("Check system cooling and reduce computational load")
            
            healthy = len(warnings_list) == 0 and cpu_usage < 90 and memory_usage < 90
            
            return SystemHealth(
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                gpu_usage=gpu_usage,
                gpu_memory_usage=gpu_memory_usage,
                disk_usage=disk_usage,
                temperature=temperature,
                healthy=healthy,
                warnings=warnings_list,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"System health check failed: {e}")
            return SystemHealth(
                cpu_usage=0.0, memory_usage=0.0, gpu_usage=0.0,
                gpu_memory_usage=0.0, disk_usage=0.0, temperature=0.0,
                healthy=False, warnings=[f"Health check failed: {e}"]
            )
    
    async def generate_performance_report(self, time_window_minutes: int = 60) -> PerformanceReport:
        """Generate comprehensive performance report"""
        
        try:
            # Get time period
            end_time = time.time()
            start_time = end_time - (time_window_minutes * 60)
            
            # Get operation statistics
            operation_stats = self.metrics_collector.get_operation_stats(
                time_window_minutes=time_window_minutes
            )
            
            # Get system health
            system_health = await self.get_system_health()
            
            # Identify bottlenecks
            bottlenecks = await self._identify_bottlenecks(operation_stats, system_health)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(
                operation_stats, system_health, bottlenecks
            )
            
            # Create detailed metrics
            detailed_metrics = {
                "operation_breakdown": {},
                "resource_usage_trend": self.resource_monitor.get_usage_history(time_window_minutes),
                "recent_alerts": [
                    {
                        "type": alert.alert_type,
                        "message": alert.message,
                        "severity": alert.severity,
                        "timestamp": alert.timestamp
                    }
                    for alert in list(self._alerts)[-20:]  # Last 20 alerts
                ]
            }
            
            # Add per-operation-type breakdown
            for op_type in self.metrics_collector.get_operation_types():
                detailed_metrics["operation_breakdown"][op_type] = \
                    self.metrics_collector.get_operation_stats(op_type, time_window_minutes)
            
            return PerformanceReport(
                time_period=(start_time, end_time),
                total_operations=operation_stats["total_operations"],
                successful_operations=operation_stats["successful_operations"],
                failed_operations=operation_stats["failed_operations"],
                average_latency=operation_stats["average_latency"],
                p95_latency=operation_stats["p95_latency"],
                p99_latency=operation_stats["p99_latency"],
                throughput=operation_stats["throughput"],
                error_rate=operation_stats["error_rate"],
                system_health=system_health,
                bottlenecks=bottlenecks,
                recommendations=recommendations,
                detailed_metrics=detailed_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")
            return PerformanceReport(
                time_period=(time.time() - 3600, time.time()),
                total_operations=0,
                successful_operations=0,
                failed_operations=0,
                average_latency=0.0,
                p95_latency=0.0,
                p99_latency=0.0,
                throughput=0.0,
                error_rate=0.0,
                system_health=await self.get_system_health(),
                bottlenecks=["Report generation failed"],
                recommendations=["Check system logs"],
                detailed_metrics={}
            )
    
    async def _identify_bottlenecks(self, operation_stats: Dict[str, Any], 
                                  system_health: SystemHealth) -> List[str]:
        """Identify performance bottlenecks"""
        
        bottlenecks = []
        
        # High latency bottleneck
        if operation_stats["average_latency"] > 2.0:
            bottlenecks.append(
                f"High average latency: {operation_stats['average_latency']:.2f}s"
            )
        
        # Low throughput bottleneck
        if operation_stats["throughput"] < 5.0:
            bottlenecks.append(
                f"Low throughput: {operation_stats['throughput']:.1f} ops/sec"
            )
        
        # Resource bottlenecks
        if system_health.cpu_usage > 85:
            bottlenecks.append("CPU bottleneck detected")
        
        if system_health.memory_usage > 85:
            bottlenecks.append("Memory bottleneck detected")
        
        if system_health.gpu_memory_usage > 85:
            bottlenecks.append("GPU memory bottleneck detected")
        
        # Error rate bottleneck
        if operation_stats["error_rate"] > 0.02:
            bottlenecks.append(f"High error rate: {operation_stats['error_rate']:.1%}")
        
        return bottlenecks
    
    async def _generate_recommendations(self, operation_stats: Dict[str, Any],
                                      system_health: SystemHealth,
                                      bottlenecks: List[str]) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # Performance optimizations
        if operation_stats["average_latency"] > 1.0:
            recommendations.append(
                "Consider optimizing model inference or reducing batch sizes for better latency"
            )
        
        if operation_stats["throughput"] < 10.0:
            recommendations.append(
                "Consider increasing batch sizes or parallelization for better throughput"
            )
        
        # Resource optimizations
        if system_health.memory_usage > 80:
            recommendations.append(
                "Implement memory optimization: model quantization, gradient checkpointing, or smaller batch sizes"
            )
        
        if system_health.gpu_memory_usage > 80:
            recommendations.append(
                "Optimize GPU memory: use mixed precision training or reduce model size"
            )
        
        # Error handling
        if operation_stats["error_rate"] > 0.01:
            recommendations.append(
                "Investigate error patterns and implement better error handling"
            )
        
        # General optimizations
        if not recommendations:
            recommendations.append("System performing well - consider load testing for scalability")
        
        return recommendations
    
    def get_alerts(self, severity: Optional[str] = None, limit: int = 50) -> List[PerformanceAlert]:
        """Get recent performance alerts"""
        
        alerts = list(self._alerts)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return alerts[-limit:]
    
    def clear_alerts(self):
        """Clear all alerts"""
        self._alerts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance monitor statistics"""
        
        return {
            "active_operations": len(self._active_operations),
            "total_alerts": len(self._alerts),
            "monitoring_uptime": time.time(),  # Simplified
            "operation_types_monitored": self.metrics_collector.get_operation_types(),
            "resource_monitoring_active": self.resource_monitor._monitoring
        }
    
    def cleanup(self):
        """Clean up resources"""
        
        self.resource_monitor.stop_monitoring()
        self._active_operations.clear()
        self._alerts.clear()
        
        self.logger.info("Performance monitor cleanup completed")

# Context manager for easy operation tracking
class OperationTracker:
    """Context manager for tracking operations"""
    
    def __init__(self, monitor: PerformanceMonitor, operation_type: str,
                 operation_id: Optional[str] = None, metadata: Optional[Dict] = None):
        self.monitor = monitor
        self.operation_type = operation_type
        self.operation_id = operation_id or f"{operation_type}_{time.time()}"
        self.metadata = metadata
        self.metrics = None
    
    async def __aenter__(self):
        await self.monitor.start_operation(
            self.operation_type, self.operation_id, self.metadata
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error_message = str(exc_val) if exc_val else ""
        
        self.metrics = await self.monitor.end_operation(
            self.operation_id, success, error_message
        )
        
        return False  # Don't suppress exceptions

# Factory function
def create_performance_monitor(config) -> PerformanceMonitor:
    """Create and return a configured performance monitor"""
    return PerformanceMonitor(config)