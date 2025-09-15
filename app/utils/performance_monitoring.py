"""
Performance Monitoring System for Face Embedding System
Real-time performance tracking, alerts, and optimization recommendations
"""

import time
import threading
import psutil
import os
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime, timedelta
import json
import logging
import statistics


@dataclass
class PerformanceAlert:
    """Performance alert data structure"""
    alert_type: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: float
    metric_value: float
    threshold: float
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int
    open_files: int


@dataclass
class ApplicationMetrics:
    """Application-specific performance metrics"""
    timestamp: float
    active_connections: int
    frames_processed: int
    faces_detected: int
    embeddings_generated: int
    liveness_checks: int
    database_queries: int
    cache_hits: int
    cache_misses: int
    average_response_time: float
    error_count: int


class MetricsCollector:
    """Collects and stores performance metrics"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.system_metrics = deque(maxlen=max_history)
        self.app_metrics = deque(maxlen=max_history)
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.MetricsCollector")
        
        # Application counters
        self.app_counters = {
            'active_connections': 0,
            'frames_processed': 0,
            'faces_detected': 0,
            'embeddings_generated': 0,
            'liveness_checks': 0,
            'database_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'error_count': 0,
        }
        
        # Response time tracking
        self.response_times = deque(maxlen=100)
        
    def collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect current system performance metrics"""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network
            network = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_percent=disk.percent,
                disk_free_gb=disk.free / (1024**3),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=len(psutil.pids()),
                thread_count=process.num_threads(),
                open_files=len(process.open_files())
            )
            
            with self.lock:
                self.system_metrics.append(metrics)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return None
            
    def collect_app_metrics(self) -> Optional[ApplicationMetrics]:
        """Collect current application performance metrics"""
        try:
            with self.lock:
                avg_response_time = (
                    statistics.mean(self.response_times) 
                    if self.response_times else 0.0
                )
                
                metrics = ApplicationMetrics(
                    timestamp=time.time(),
                    active_connections=self.app_counters['active_connections'],
                    frames_processed=self.app_counters['frames_processed'],
                    faces_detected=self.app_counters['faces_detected'],
                    embeddings_generated=self.app_counters['embeddings_generated'],
                    liveness_checks=self.app_counters['liveness_checks'],
                    database_queries=self.app_counters['database_queries'],
                    cache_hits=self.app_counters['cache_hits'],
                    cache_misses=self.app_counters['cache_misses'],
                    average_response_time=avg_response_time,
                    error_count=self.app_counters['error_count']
                )
                
                self.app_metrics.append(metrics)
                
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect app metrics: {e}")
            return None
            
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment an application counter"""
        with self.lock:
            if counter_name in self.app_counters:
                self.app_counters[counter_name] += value
                
    def set_counter(self, counter_name: str, value: int):
        """Set an application counter value"""
        with self.lock:
            if counter_name in self.app_counters:
                self.app_counters[counter_name] = value
                
    def add_response_time(self, response_time: float):
        """Add a response time measurement"""
        with self.lock:
            self.response_times.append(response_time)
            
    def get_metrics_summary(self, time_window_minutes: int = 5) -> Dict[str, Any]:
        """Get metrics summary for the specified time window"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        with self.lock:
            # Filter recent metrics
            recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
            recent_app = [m for m in self.app_metrics if m.timestamp >= cutoff_time]
            
            if not recent_system or not recent_app:
                return {"error": "Insufficient data for summary"}
                
            # Calculate system averages
            avg_cpu = statistics.mean([m.cpu_percent for m in recent_system])
            avg_memory = statistics.mean([m.memory_percent for m in recent_system])
            avg_disk = statistics.mean([m.disk_percent for m in recent_system])
            
            # Calculate application totals and averages
            total_frames = sum([m.frames_processed for m in recent_app])
            total_faces = sum([m.faces_detected for m in recent_app])
            total_embeddings = sum([m.embeddings_generated for m in recent_app])
            total_errors = sum([m.error_count for m in recent_app])
            avg_response_time = statistics.mean([m.average_response_time for m in recent_app])
            
            # Calculate rates (per minute)
            time_span_minutes = max(1, time_window_minutes)
            frame_rate = total_frames / time_span_minutes
            face_rate = total_faces / time_span_minutes
            embedding_rate = total_embeddings / time_span_minutes
            error_rate = total_errors / time_span_minutes
            
            return {
                "time_window_minutes": time_window_minutes,
                "system_metrics": {
                    "avg_cpu_percent": round(avg_cpu, 2),
                    "avg_memory_percent": round(avg_memory, 2),
                    "avg_disk_percent": round(avg_disk, 2),
                },
                "application_metrics": {
                    "frames_per_minute": round(frame_rate, 2),
                    "faces_per_minute": round(face_rate, 2),
                    "embeddings_per_minute": round(embedding_rate, 2),
                    "errors_per_minute": round(error_rate, 2),
                    "avg_response_time_ms": round(avg_response_time * 1000, 2),
                    "current_connections": self.app_counters['active_connections'],
                    "cache_hit_rate": self._calculate_cache_hit_rate()
                }
            }
            
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage"""
        hits = self.app_counters['cache_hits']
        misses = self.app_counters['cache_misses']
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


class PerformanceMonitor:
    """Main performance monitoring system with alerting"""
    
    def __init__(self, 
                 collection_interval: float = 10.0,
                 alert_thresholds: Dict[str, Dict[str, float]] = None):
        self.collection_interval = collection_interval
        self.metrics_collector = MetricsCollector()
        self.alerts: List[PerformanceAlert] = []
        self.alert_callbacks: List[Callable] = []
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'cpu_percent': {'medium': 70.0, 'high': 85.0, 'critical': 95.0},
            'memory_percent': {'medium': 70.0, 'high': 85.0, 'critical': 95.0},
            'disk_percent': {'medium': 80.0, 'high': 90.0, 'critical': 95.0},
            'response_time': {'medium': 1.0, 'high': 2.0, 'critical': 5.0},  # seconds
            'error_rate': {'medium': 5.0, 'high': 10.0, 'critical': 20.0},  # per minute
        }
        
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(f"{__name__}.PerformanceMonitor")
        
    def start(self):
        """Start performance monitoring"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.thread.start()
        self.logger.info("Performance monitoring started")
        
    def stop(self):
        """Stop performance monitoring"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.logger.info("Performance monitoring stopped")
        
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback function to be called when alerts are triggered"""
        self.alert_callbacks.append(callback)
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                system_metrics = self.metrics_collector.collect_system_metrics()
                app_metrics = self.metrics_collector.collect_app_metrics()
                
                # Check for alerts
                if system_metrics:
                    self._check_system_alerts(system_metrics)
                if app_metrics:
                    self._check_app_alerts(app_metrics)
                    
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                
            time.sleep(self.collection_interval)
            
    def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system performance alerts"""
        # CPU alert
        self._check_threshold_alert(
            "cpu_percent", 
            metrics.cpu_percent, 
            "High CPU usage detected",
            ["Consider optimizing CPU-intensive operations",
             "Check for background processes",
             "Consider scaling horizontally"]
        )
        
        # Memory alert
        self._check_threshold_alert(
            "memory_percent",
            metrics.memory_percent,
            "High memory usage detected",
            ["Check for memory leaks",
             "Optimize data structures",
             "Consider increasing available memory"]
        )
        
        # Disk alert
        self._check_threshold_alert(
            "disk_percent",
            metrics.disk_percent,
            "High disk usage detected",
            ["Clean up old log files",
             "Archive old data",
             "Consider adding more storage"]
        )
        
    def _check_app_alerts(self, metrics: ApplicationMetrics):
        """Check for application performance alerts"""
        # Response time alert
        self._check_threshold_alert(
            "response_time",
            metrics.average_response_time,
            "High response time detected",
            ["Optimize database queries",
             "Implement caching",
             "Profile application performance"]
        )
        
        # Error rate alert (calculated from recent errors)
        recent_summary = self.metrics_collector.get_metrics_summary(time_window_minutes=1)
        if 'application_metrics' in recent_summary:
            error_rate = recent_summary['application_metrics'].get('errors_per_minute', 0)
            self._check_threshold_alert(
                "error_rate",
                error_rate,
                "High error rate detected",
                ["Check application logs",
                 "Review recent deployments",
                 "Implement circuit breakers"]
            )
            
    def _check_threshold_alert(self, metric_name: str, current_value: float, 
                              message: str, recommendations: List[str]):
        """Check if metric exceeds threshold and create alert"""
        if metric_name not in self.alert_thresholds:
            return
            
        thresholds = self.alert_thresholds[metric_name]
        severity = None
        threshold = None
        
        if current_value >= thresholds.get('critical', float('inf')):
            severity = 'critical'
            threshold = thresholds['critical']
        elif current_value >= thresholds.get('high', float('inf')):
            severity = 'high'
            threshold = thresholds['high']
        elif current_value >= thresholds.get('medium', float('inf')):
            severity = 'medium'
            threshold = thresholds['medium']
            
        if severity:
            alert = PerformanceAlert(
                alert_type=metric_name,
                message=f"{message}: {current_value:.2f}",
                severity=severity,
                timestamp=time.time(),
                metric_value=current_value,
                threshold=threshold,
                recommendations=recommendations
            )
            
            self._trigger_alert(alert)
            
    def _trigger_alert(self, alert: PerformanceAlert):
        """Trigger an alert and notify callbacks"""
        self.alerts.append(alert)
        
        # Log the alert
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        self.logger.log(log_level, f"PERFORMANCE ALERT [{alert.severity.upper()}]: {alert.message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {e}")
                
    def get_current_status(self) -> Dict[str, Any]:
        """Get current performance status"""
        summary = self.metrics_collector.get_metrics_summary()
        recent_alerts = [
            {
                "type": alert.alert_type,
                "message": alert.message,
                "severity": alert.severity,
                "timestamp": alert.timestamp
            }
            for alert in self.alerts[-5:]  # Last 5 alerts
        ]
        
        summary["recent_alerts"] = recent_alerts
        return summary
        
    def reset_counters(self):
        """Reset application counters"""
        self.metrics_collector.app_counters = dict.fromkeys(self.metrics_collector.app_counters, 0)
        
    # Convenience methods for common operations
    def track_frame_processed(self):
        """Track a processed frame"""
        self.metrics_collector.increment_counter('frames_processed')
        
    def track_face_detected(self):
        """Track a detected face"""
        self.metrics_collector.increment_counter('faces_detected')
        
    def track_embedding_generated(self):
        """Track an generated embedding"""
        self.metrics_collector.increment_counter('embeddings_generated')
        
    def track_liveness_check(self):
        """Track a liveness check"""
        self.metrics_collector.increment_counter('liveness_checks')
        
    def track_database_query(self):
        """Track a database query"""
        self.metrics_collector.increment_counter('database_queries')
        
    def track_cache_hit(self):
        """Track a cache hit"""
        self.metrics_collector.increment_counter('cache_hits')
        
    def track_cache_miss(self):
        """Track a cache miss"""
        self.metrics_collector.increment_counter('cache_misses')
        
    def track_error(self):
        """Track an error"""
        self.metrics_collector.increment_counter('error_count')
        
    def track_response_time(self, response_time: float):
        """Track a response time"""
        self.metrics_collector.add_response_time(response_time)
        
    def set_active_connections(self, count: int):
        """Set the current number of active connections"""
        self.metrics_collector.set_counter('active_connections', count)


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


# Decorator for tracking function performance
def track_performance():
    """Decorator to track function performance"""
    def decorator(func):
        from functools import wraps
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time
                performance_monitor.track_response_time(response_time)
                return result
            except Exception:
                performance_monitor.track_error()
                raise
        return wrapper
    return decorator


# Alert callback for logging critical alerts
def log_critical_alert(alert: PerformanceAlert):
    """Log critical performance alerts"""
    if alert.severity == 'critical':
        logger = logging.getLogger("performance_alerts")
        logger.critical(f"CRITICAL ALERT: {alert.message}")
        logger.critical(f"Recommendations: {', '.join(alert.recommendations)}")


# Initialize with default alert callback
performance_monitor.add_alert_callback(log_critical_alert)
