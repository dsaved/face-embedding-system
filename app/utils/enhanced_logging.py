"""
Enhanced Logging System for Face Embedding System
Provides structured logging, performance monitoring, and error tracking
"""

import logging
import time
import json
import os
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from functools import wraps
import traceback
from dataclasses import dataclass, asdict


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class PerformanceMonitor:
    """Enhanced performance monitoring and metrics collection"""
    
    def __init__(self):
        self.metrics = []
        self.lock = threading.Lock()
        
    def log_metric(self, metric: PerformanceMetrics):
        """Log a performance metric"""
        with self.lock:
            self.metrics.append(metric)
            
    def get_metrics_summary(self, operation_filter: str = None) -> Dict[str, Any]:
        """Get summary of performance metrics"""
        with self.lock:
            filtered_metrics = self.metrics
            if operation_filter:
                filtered_metrics = [m for m in self.metrics if operation_filter in m.operation]
                
            if not filtered_metrics:
                return {"message": "No metrics found"}
                
            durations = [m.duration for m in filtered_metrics]
            successes = [m for m in filtered_metrics if m.success]
            failures = [m for m in filtered_metrics if not m.success]
            
            return {
                "total_operations": len(filtered_metrics),
                "successful_operations": len(successes),
                "failed_operations": len(failures),
                "success_rate": len(successes) / len(filtered_metrics) * 100,
                "average_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "recent_failures": [
                    {"operation": m.operation, "error": m.error_message, "time": m.end_time}
                    for m in failures[-5:]  # Last 5 failures
                ]
            }
    
    def clear_metrics(self):
        """Clear stored metrics"""
        with self.lock:
            self.metrics.clear()


class EnhancedLogger:
    """Enhanced logging system with structured logging and performance monitoring"""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.performance_monitor = PerformanceMonitor()
        self._setup_logger(log_level)
        
    def _setup_logger(self, log_level: str):
        """Setup enhanced logger with structured formatting"""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set level
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        json_formatter = JsonFormatter()
        
        # Console handler with detailed format
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with JSON format for machine processing
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        file_handler = logging.FileHandler('logs/enhanced_app.log')
        file_handler.setFormatter(json_formatter)
        self.logger.addHandler(file_handler)
        
        # Performance metrics file
        perf_handler = logging.FileHandler('logs/performance_metrics.log')
        perf_handler.setFormatter(json_formatter)
        perf_handler.setLevel(logging.INFO)
        
        # Create performance logger
        self.perf_logger = logging.getLogger(f"{self.logger.name}.performance")
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
        
    def info(self, message: str, extra_data: Dict[str, Any] = None):
        """Enhanced info logging"""
        self._log_with_context(logging.INFO, message, extra_data)
        
    def warning(self, message: str, extra_data: Dict[str, Any] = None):
        """Enhanced warning logging"""
        self._log_with_context(logging.WARNING, message, extra_data)
        
    def error(self, message: str, extra_data: Dict[str, Any] = None, exc_info: bool = True):
        """Enhanced error logging"""
        self._log_with_context(logging.ERROR, message, extra_data, exc_info)
        
    def debug(self, message: str, extra_data: Dict[str, Any] = None):
        """Enhanced debug logging"""
        self._log_with_context(logging.DEBUG, message, extra_data)
        
    def _log_with_context(self, level: int, message: str, extra_data: Dict[str, Any] = None, exc_info: bool = False):
        """Log with additional context"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "thread_id": threading.get_ident(),
        }
        
        if extra_data:
            log_data.update(extra_data)
            
        if exc_info and level >= logging.ERROR:
            log_data["traceback"] = traceback.format_exc()
            
        self.logger.log(level, json.dumps(log_data))
        
    def log_performance(self, operation: str, duration: float, success: bool = True, 
                       error_message: str = None, additional_data: Dict[str, Any] = None):
        """Log performance metrics"""
        metric = PerformanceMetrics(
            operation=operation,
            start_time=time.time() - duration,
            end_time=time.time(),
            duration=duration,
            success=success,
            error_message=error_message,
            additional_data=additional_data
        )
        
        self.performance_monitor.log_metric(metric)
        
        # Also log to performance file
        perf_data = asdict(metric)
        perf_data["timestamp"] = datetime.utcnow().isoformat()
        self.perf_logger.info(json.dumps(perf_data))
        
    def get_performance_summary(self, operation_filter: str = None) -> Dict[str, Any]:
        """Get performance metrics summary"""
        return self.performance_monitor.get_metrics_summary(operation_filter)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)


def performance_monitor(operation_name: str = None):
    """Decorator for monitoring function performance"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create logger
            logger_name = f"{func.__module__}.{func.__qualname__}"
            logger = EnhancedLogger(logger_name)
            
            operation = operation_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.log_performance(
                    operation=operation,
                    duration=duration,
                    success=True,
                    additional_data={"args_count": len(args), "kwargs_count": len(kwargs)}
                )
                
                logger.debug(f"Operation '{operation}' completed successfully", {
                    "duration": duration,
                    "function": func.__name__
                })
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                error_message = str(e)
                
                logger.log_performance(
                    operation=operation,
                    duration=duration,
                    success=False,
                    error_message=error_message
                )
                
                logger.error(f"Operation '{operation}' failed", {
                    "duration": duration,
                    "error": error_message,
                    "function": func.__name__
                })
                
                raise
                
        return wrapper
    return decorator


def log_system_health():
    """Log current system health metrics"""
    import psutil
    
    logger = EnhancedLogger("system_health")
    
    try:
        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        health_data = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_percent": disk.percent,
            "disk_free_gb": disk.free / (1024**3),
        }
        
        logger.info("System health check", health_data)
        
        # Alert on high resource usage
        if cpu_percent > 80:
            logger.warning("High CPU usage detected", {"cpu_percent": cpu_percent})
            
        if memory.percent > 85:
            logger.warning("High memory usage detected", {"memory_percent": memory.percent})
            
        if disk.percent > 90:
            logger.warning("High disk usage detected", {"disk_percent": disk.percent})
            
        return health_data
        
    except Exception as e:
        logger.error("Failed to collect system health metrics", {"error": str(e)})
        return None


# Global logger instance
app_logger = EnhancedLogger("face_embedding_system")


# Convenience functions
def log_info(message: str, extra_data: Dict[str, Any] = None):
    """Global info logging function"""
    app_logger.info(message, extra_data)


def log_warning(message: str, extra_data: Dict[str, Any] = None):
    """Global warning logging function"""
    app_logger.warning(message, extra_data)


def log_error(message: str, extra_data: Dict[str, Any] = None):
    """Global error logging function"""
    app_logger.error(message, extra_data)


def log_debug(message: str, extra_data: Dict[str, Any] = None):
    """Global debug logging function"""
    app_logger.debug(message, extra_data)


def log_performance(operation: str, duration: float, success: bool = True, 
                   error_message: str = None, additional_data: Dict[str, Any] = None):
    """Global performance logging function"""
    app_logger.log_performance(operation, duration, success, error_message, additional_data)
