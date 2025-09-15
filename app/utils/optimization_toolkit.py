"""
Performance Optimization Toolkit for Face Embedding System
Advanced optimization strategies and automated tuning capabilities
"""

import time
import threading
import numpy as np
import cv2
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import statistics
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools


@dataclass
class OptimizationResult:
    """Result of an optimization operation"""
    optimization_type: str
    parameter_name: str
    old_value: Any
    new_value: Any
    performance_improvement: float  # percentage
    timestamp: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class PerformanceProfile:
    """Performance profile for a specific operation"""
    operation_name: str
    execution_times: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    cpu_usage: List[float] = field(default_factory=list)
    success_count: int = 0
    error_count: int = 0
    last_updated: float = field(default_factory=time.time)


class AdaptiveConfigManager:
    """Manages adaptive configuration changes based on performance metrics"""
    
    def __init__(self, config_module, performance_monitor):
        self.config_module = config_module
        self.performance_monitor = performance_monitor
        self.optimization_history: List[OptimizationResult] = []
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.AdaptiveConfigManager")
        
        # Optimization rules
        self.optimization_rules = {
            'batch_size': self._optimize_batch_size,
            'thread_count': self._optimize_thread_count,
            'cache_size': self._optimize_cache_size,
            'detection_threshold': self._optimize_detection_threshold,
            'quality_threshold': self._optimize_quality_threshold,
        }
        
    def register_performance_data(self, operation_name: str, 
                                execution_time: float, 
                                memory_usage: float = 0.0,
                                cpu_usage: float = 0.0,
                                success: bool = True):
        """Register performance data for an operation"""
        with self.lock:
            if operation_name not in self.performance_profiles:
                self.performance_profiles[operation_name] = PerformanceProfile(operation_name)
                
            profile = self.performance_profiles[operation_name]
            profile.execution_times.append(execution_time)
            profile.memory_usage.append(memory_usage)
            profile.cpu_usage.append(cpu_usage)
            
            if success:
                profile.success_count += 1
            else:
                profile.error_count += 1
                
            profile.last_updated = time.time()
            
            # Keep only recent data (last 1000 measurements)
            if len(profile.execution_times) > 1000:
                profile.execution_times = profile.execution_times[-1000:]
                profile.memory_usage = profile.memory_usage[-1000:]
                profile.cpu_usage = profile.cpu_usage[-1000:]
                
    def analyze_performance_trends(self, operation_name: str) -> Dict[str, Any]:
        """Analyze performance trends for an operation"""
        if operation_name not in self.performance_profiles:
            return {"error": "No performance data available"}
            
        profile = self.performance_profiles[operation_name]
        
        if len(profile.execution_times) < 10:
            return {"error": "Insufficient data for analysis"}
            
        execution_times = profile.execution_times
        
        # Calculate trends
        recent_times = execution_times[-50:]  # Last 50 measurements
        historical_times = execution_times[:-50] if len(execution_times) > 50 else []
        
        recent_avg = statistics.mean(recent_times)
        historical_avg = statistics.mean(historical_times) if historical_times else recent_avg
        
        trend_direction = "improving" if recent_avg < historical_avg else "degrading"
        trend_percentage = abs((recent_avg - historical_avg) / historical_avg * 100) if historical_avg > 0 else 0
        
        return {
            "operation_name": operation_name,
            "total_measurements": len(execution_times),
            "recent_avg_time": recent_avg,
            "historical_avg_time": historical_avg,
            "trend_direction": trend_direction,
            "trend_percentage": round(trend_percentage, 2),
            "success_rate": profile.success_count / (profile.success_count + profile.error_count) * 100,
            "min_time": min(execution_times),
            "max_time": max(execution_times),
            "std_deviation": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        }
        
    def suggest_optimizations(self) -> List[Dict[str, Any]]:
        """Suggest optimizations based on performance analysis"""
        suggestions = []
        
        # Analyze system metrics
        system_status = self.performance_monitor.get_current_status()
        
        if 'system_metrics' in system_status:
            sys_metrics = system_status['system_metrics']
            
            # CPU optimization suggestions
            if sys_metrics.get('avg_cpu_percent', 0) > 80:
                suggestions.append({
                    "type": "cpu_optimization",
                    "priority": "high",
                    "suggestion": "High CPU usage detected. Consider reducing batch sizes or increasing thread pool size.",
                    "parameters": ["batch_size", "thread_count"],
                    "expected_impact": "20-30% CPU reduction"
                })
                
            # Memory optimization suggestions
            if sys_metrics.get('avg_memory_percent', 0) > 80:
                suggestions.append({
                    "type": "memory_optimization",
                    "priority": "high",
                    "suggestion": "High memory usage detected. Consider reducing cache sizes or implementing memory pooling.",
                    "parameters": ["cache_size", "memory_pool_size"],
                    "expected_impact": "15-25% memory reduction"
                })
                
        # Analyze application metrics
        if 'application_metrics' in system_status:
            app_metrics = system_status['application_metrics']
            
            # Response time optimization
            if app_metrics.get('avg_response_time_ms', 0) > 1000:
                suggestions.append({
                    "type": "response_time_optimization",
                    "priority": "medium",
                    "suggestion": "High response times detected. Consider optimizing detection thresholds or enabling parallel processing.",
                    "parameters": ["detection_threshold", "parallel_processing"],
                    "expected_impact": "30-50% response time reduction"
                })
                
            # Cache optimization
            cache_hit_rate = app_metrics.get('cache_hit_rate', 0)
            if cache_hit_rate < 70:
                suggestions.append({
                    "type": "cache_optimization",
                    "priority": "medium",
                    "suggestion": f"Low cache hit rate ({cache_hit_rate:.1f}%). Consider increasing cache size or adjusting cache strategy.",
                    "parameters": ["cache_size", "cache_ttl"],
                    "expected_impact": "10-20% performance improvement"
                })
                
        return suggestions
        
    def auto_optimize(self, parameters: List[str] = None) -> List[OptimizationResult]:
        """Automatically optimize specified parameters"""
        if parameters is None:
            parameters = list(self.optimization_rules.keys())
            
        results = []
        
        for param in parameters:
            if param in self.optimization_rules:
                try:
                    result = self.optimization_rules[param]()
                    results.append(result)
                    self.optimization_history.append(result)
                except Exception as e:
                    error_result = OptimizationResult(
                        optimization_type="auto_optimize",
                        parameter_name=param,
                        old_value=None,
                        new_value=None,
                        performance_improvement=0.0,
                        timestamp=time.time(),
                        success=False,
                        error_message=str(e)
                    )
                    results.append(error_result)
                    
        return results
        
    def _optimize_batch_size(self) -> OptimizationResult:
        """Optimize batch size based on performance metrics"""
        # Get current batch size
        current_batch_size = getattr(self.config_module, 'BATCH_SIZE', 32)
        
        # Analyze recent performance
        system_status = self.performance_monitor.get_current_status()
        cpu_usage = system_status.get('system_metrics', {}).get('avg_cpu_percent', 50)
        memory_usage = system_status.get('system_metrics', {}).get('avg_memory_percent', 50)
        
        # Calculate optimal batch size
        new_batch_size = current_batch_size
        
        if cpu_usage < 50 and memory_usage < 60:
            # Increase batch size for better throughput
            new_batch_size = min(current_batch_size * 2, 128)
        elif cpu_usage > 80 or memory_usage > 80:
            # Decrease batch size to reduce resource usage
            new_batch_size = max(current_batch_size // 2, 8)
            
        # Apply optimization
        if new_batch_size != current_batch_size:
            setattr(self.config_module, 'BATCH_SIZE', new_batch_size)
            improvement = abs(new_batch_size - current_batch_size) / current_batch_size * 100
        else:
            improvement = 0.0
            
        return OptimizationResult(
            optimization_type="batch_size",
            parameter_name="BATCH_SIZE",
            old_value=current_batch_size,
            new_value=new_batch_size,
            performance_improvement=improvement,
            timestamp=time.time(),
            success=True
        )
        
    def _optimize_thread_count(self) -> OptimizationResult:
        """Optimize thread count based on CPU usage"""
        import multiprocessing
        
        current_threads = getattr(self.config_module, 'MAX_WORKER_THREADS', 4)
        max_threads = multiprocessing.cpu_count()
        
        # Get CPU usage
        system_status = self.performance_monitor.get_current_status()
        cpu_usage = system_status.get('system_metrics', {}).get('avg_cpu_percent', 50)
        
        new_threads = current_threads
        
        if cpu_usage < 60:
            # Increase threads for better parallelism
            new_threads = min(current_threads + 2, max_threads)
        elif cpu_usage > 90:
            # Decrease threads to reduce contention
            new_threads = max(current_threads - 1, 2)
            
        if new_threads != current_threads:
            setattr(self.config_module, 'MAX_WORKER_THREADS', new_threads)
            improvement = abs(new_threads - current_threads) / current_threads * 100
        else:
            improvement = 0.0
            
        return OptimizationResult(
            optimization_type="thread_count",
            parameter_name="MAX_WORKER_THREADS",
            old_value=current_threads,
            new_value=new_threads,
            performance_improvement=improvement,
            timestamp=time.time(),
            success=True
        )
        
    def _optimize_cache_size(self) -> OptimizationResult:
        """Optimize cache size based on memory usage and hit rate"""
        current_cache_size = getattr(self.config_module, 'CACHE_SIZE', 1000)
        
        # Get metrics
        system_status = self.performance_monitor.get_current_status()
        memory_usage = system_status.get('system_metrics', {}).get('avg_memory_percent', 50)
        cache_hit_rate = system_status.get('application_metrics', {}).get('cache_hit_rate', 70)
        
        new_cache_size = current_cache_size
        
        if cache_hit_rate < 60 and memory_usage < 70:
            # Increase cache size for better hit rate
            new_cache_size = min(current_cache_size * 2, 10000)
        elif memory_usage > 85:
            # Decrease cache size to free memory
            new_cache_size = max(current_cache_size // 2, 100)
            
        if new_cache_size != current_cache_size:
            setattr(self.config_module, 'CACHE_SIZE', new_cache_size)
            improvement = abs(new_cache_size - current_cache_size) / current_cache_size * 100
        else:
            improvement = 0.0
            
        return OptimizationResult(
            optimization_type="cache_size",
            parameter_name="CACHE_SIZE",
            old_value=current_cache_size,
            new_value=new_cache_size,
            performance_improvement=improvement,
            timestamp=time.time(),
            success=True
        )
        
    def _optimize_detection_threshold(self) -> OptimizationResult:
        """Optimize detection threshold based on accuracy vs performance trade-off"""
        current_threshold = getattr(self.config_module, 'DETECTION_THRESHOLD', 0.5)
        
        # Analyze detection performance
        detection_profile = self.performance_profiles.get('face_detection')
        if not detection_profile or len(detection_profile.execution_times) < 50:
            return OptimizationResult(
                optimization_type="detection_threshold",
                parameter_name="DETECTION_THRESHOLD",
                old_value=current_threshold,
                new_value=current_threshold,
                performance_improvement=0.0,
                timestamp=time.time(),
                success=False,
                error_message="Insufficient detection performance data"
            )
            
        avg_detection_time = statistics.mean(detection_profile.execution_times[-50:])
        
        new_threshold = current_threshold
        
        # If detection is too slow, increase threshold (fewer detections)
        if avg_detection_time > 0.1:  # 100ms
            new_threshold = min(current_threshold + 0.05, 0.9)
        # If detection is very fast, decrease threshold (more accurate)
        elif avg_detection_time < 0.05:  # 50ms
            new_threshold = max(current_threshold - 0.05, 0.3)
            
        if abs(new_threshold - current_threshold) > 0.01:
            setattr(self.config_module, 'DETECTION_THRESHOLD', new_threshold)
            improvement = abs(new_threshold - current_threshold) / current_threshold * 100
        else:
            improvement = 0.0
            
        return OptimizationResult(
            optimization_type="detection_threshold",
            parameter_name="DETECTION_THRESHOLD",
            old_value=current_threshold,
            new_value=new_threshold,
            performance_improvement=improvement,
            timestamp=time.time(),
            success=True
        )
        
    def _optimize_quality_threshold(self) -> OptimizationResult:
        """Optimize quality threshold based on processing performance"""
        current_threshold = getattr(self.config_module, 'QUALITY_THRESHOLD', 0.7)
        
        # Get processing metrics
        system_status = self.performance_monitor.get_current_status()
        avg_response_time = system_status.get('application_metrics', {}).get('avg_response_time_ms', 500)
        
        new_threshold = current_threshold
        
        # If processing is slow, increase quality threshold (reject more low-quality faces)
        if avg_response_time > 1000:  # 1 second
            new_threshold = min(current_threshold + 0.05, 0.95)
        # If processing is fast, decrease threshold (accept more faces)
        elif avg_response_time < 300:  # 300ms
            new_threshold = max(current_threshold - 0.05, 0.5)
            
        if abs(new_threshold - current_threshold) > 0.01:
            setattr(self.config_module, 'QUALITY_THRESHOLD', new_threshold)
            improvement = abs(new_threshold - current_threshold) / current_threshold * 100
        else:
            improvement = 0.0
            
        return OptimizationResult(
            optimization_type="quality_threshold",
            parameter_name="QUALITY_THRESHOLD",
            old_value=current_threshold,
            new_value=new_threshold,
            performance_improvement=improvement,
            timestamp=time.time(),
            success=True
        )


class ImageProcessingOptimizer:
    """Optimizes image processing operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ImageProcessingOptimizer")
        
    @staticmethod
    def optimize_image_for_detection(image: np.ndarray, 
                                   target_size: Tuple[int, int] = (640, 480),
                                   quality_level: str = "balanced") -> np.ndarray:
        """Optimize image for face detection"""
        if image is None or image.size == 0:
            return image
            
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate optimal resize ratio
        scale_w = target_w / w
        scale_h = target_h / h
        scale = min(scale_w, scale_h)
        
        # Only resize if necessary
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Choose interpolation method based on quality level
            if quality_level == "fast":
                interpolation = cv2.INTER_NEAREST
            elif quality_level == "balanced":
                interpolation = cv2.INTER_LINEAR
            else:  # high quality
                interpolation = cv2.INTER_CUBIC
                
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            
        return image
        
    @staticmethod
    def preprocess_for_embedding(face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for embedding generation"""
        if face_image is None or face_image.size == 0:
            return face_image
            
        # Convert to RGB if needed
        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            # Assume BGR, convert to RGB
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
        # Normalize
        face_image = face_image.astype(np.float32) / 255.0
        
        # Apply histogram equalization for better contrast
        if len(face_image.shape) == 3:
            # Convert to LAB, equalize L channel, convert back
            lab = cv2.cvtColor(face_image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = cv2.equalizeHist((lab[:, :, 0] * 255).astype(np.uint8)) / 255.0
            face_image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            face_image = cv2.equalizeHist((face_image * 255).astype(np.uint8)) / 255.0
            
        return face_image


class CacheOptimizer:
    """Optimizes caching strategies"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.cache_performance: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.logger = logging.getLogger(f"{__name__}.CacheOptimizer")
        
    def record_access(self, key: str, hit: bool, access_time: float = None):
        """Record cache access for optimization analysis"""
        if access_time is None:
            access_time = time.time()
            
        self.access_patterns[key].append(access_time)
        
        # Keep only recent access patterns (last 1000 accesses)
        if len(self.access_patterns[key]) > 1000:
            self.access_patterns[key] = self.access_patterns[key][-1000:]
            
        # Update cache performance metrics
        if key not in self.cache_performance:
            self.cache_performance[key] = {'hits': 0, 'misses': 0, 'last_access': access_time}
            
        if hit:
            self.cache_performance[key]['hits'] += 1
        else:
            self.cache_performance[key]['misses'] += 1
            
        self.cache_performance[key]['last_access'] = access_time
        
    def get_eviction_candidates(self, count: int) -> List[str]:
        """Get cache keys that should be evicted based on access patterns"""
        current_time = time.time()
        candidates = []
        
        for key, perf in self.cache_performance.items():
            hit_rate = perf['hits'] / (perf['hits'] + perf['misses']) if (perf['hits'] + perf['misses']) > 0 else 0
            time_since_access = current_time - perf['last_access']
            
            # Score based on hit rate and recency (lower score = better eviction candidate)
            score = hit_rate - (time_since_access / 3600)  # Penalize old entries
            candidates.append((key, score))
            
        # Sort by score (ascending) and return worst performers
        candidates.sort(key=lambda x: x[1])
        return [key for key, _ in candidates[:count]]
        
    def suggest_cache_size(self) -> int:
        """Suggest optimal cache size based on access patterns"""
        if not self.access_patterns:
            return self.max_size
            
        # Analyze unique access patterns
        unique_keys = len(self.access_patterns)
        total_accesses = sum(len(accesses) for accesses in self.access_patterns.values())
        
        if total_accesses == 0:
            return self.max_size
            
        # Suggest cache size based on 80/20 rule
        # 80% of accesses typically come from 20% of keys
        suggested_size = int(unique_keys * 0.2 * 1.5)  # 50% buffer
        
        return min(max(suggested_size, 100), self.max_size * 2)


# Global instances
adaptive_config_manager = None
image_optimizer = ImageProcessingOptimizer()
cache_optimizer = CacheOptimizer()


def initialize_optimization_system(config_module, performance_monitor):
    """Initialize the optimization system"""
    global adaptive_config_manager
    adaptive_config_manager = AdaptiveConfigManager(config_module, performance_monitor)
    return adaptive_config_manager


# Performance tracking decorator for optimization
def track_for_optimization(operation_name: str):
    """Decorator to track function performance for optimization"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                success = False
                raise
            finally:
                execution_time = time.time() - start_time
                if adaptive_config_manager:
                    adaptive_config_manager.register_performance_data(
                        operation_name, execution_time, success=success
                    )
        return wrapper
    return decorator
