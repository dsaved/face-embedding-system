"""
Error Recovery System for Face Embedding System
Provides automatic error recovery, circuit breakers, and resilience patterns
"""

import time
import threading
from typing import Callable, Any, Dict, Optional, List
from functools import wraps
from enum import Enum
import logging
from dataclasses import dataclass
from collections import deque
import traceback


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class ErrorStats:
    """Error statistics tracking"""
    total_requests: int = 0
    failed_requests: int = 0
    success_requests: int = 0
    last_failure_time: Optional[float] = None
    consecutive_failures: int = 0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance"""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.state = CircuitState.CLOSED
        self.stats = ErrorStats()
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to a function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._call_with_circuit_breaker(func, *args, **kwargs)
        return wrapper
        
    def _call_with_circuit_breaker(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        with self.lock:
            # Check if circuit should transition states
            self._update_state()
            
            # If circuit is open, reject the request
            if self.state == CircuitState.OPEN:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Last failure: {self.stats.last_failure_time}"
                )
                
            # Track the request
            self.stats.total_requests += 1
            
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
            # Success - reset failure count if we're in HALF_OPEN
            with self.lock:
                self.stats.success_requests += 1
                if self.state == CircuitState.HALF_OPEN:
                    self._transition_to_closed()
                    
            return result
            
        except self.expected_exception as e:
            # Handle expected exceptions
            with self.lock:
                self.stats.failed_requests += 1
                self.stats.consecutive_failures += 1
                self.stats.last_failure_time = time.time()
                
                # Check if we should open the circuit
                if self.stats.consecutive_failures >= self.failure_threshold:
                    self._transition_to_open()
                    
            self.logger.error(f"Circuit breaker recorded failure: {e}")
            raise
            
    def _update_state(self):
        """Update circuit breaker state based on current conditions"""
        if self.state == CircuitState.OPEN:
            # Check if we should transition to HALF_OPEN
            if (time.time() - self.stats.last_failure_time) >= self.recovery_timeout:
                self._transition_to_half_open()
                
    def _transition_to_closed(self):
        """Transition to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.stats.consecutive_failures = 0
        self.logger.info("Circuit breaker transitioned to CLOSED")
        
    def _transition_to_open(self):
        """Transition to OPEN state"""
        self.state = CircuitState.OPEN
        self.logger.warning(
            f"Circuit breaker transitioned to OPEN after {self.stats.consecutive_failures} failures"
        )
        
    def _transition_to_half_open(self):
        """Transition to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.logger.info("Circuit breaker transitioned to HALF_OPEN")
        
    def get_stats(self) -> Dict[str, Any]:
        """Get current circuit breaker statistics"""
        with self.lock:
            return {
                "state": self.state.value,
                "total_requests": self.stats.total_requests,
                "failed_requests": self.stats.failed_requests,
                "success_requests": self.stats.success_requests,
                "failure_rate": self.stats.failure_rate,
                "consecutive_failures": self.stats.consecutive_failures,
                "last_failure_time": self.stats.last_failure_time
            }


class RetryPolicy:
    """Configurable retry policy with exponential backoff"""
    
    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.logger = logging.getLogger(f"{__name__}.RetryPolicy")
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply retry policy to a function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            return self._execute_with_retry(func, *args, **kwargs)
        return wrapper
        
    def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(self.max_attempts):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_attempts - 1:
                    # Last attempt failed
                    self.logger.error(
                        f"Function {func.__name__} failed after {self.max_attempts} attempts: {e}"
                    )
                    break
                    
                # Calculate delay for next attempt
                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    f"Function {func.__name__} failed on attempt {attempt + 1}, retrying in {delay:.2f}s: {e}"
                )
                time.sleep(delay)
                
        # All attempts failed
        raise RetryExhaustedError(
            f"Function {func.__name__} failed after {self.max_attempts} attempts"
        ) from last_exception
        
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the next retry attempt"""
        import random
        
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add random jitter to prevent thundering herd
            delay = delay * (0.5 + random.random() * 0.5)
            
        return delay


class HealthChecker:
    """System health monitoring and automatic recovery"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.health_checks: List[Callable] = []
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(f"{__name__}.HealthChecker")
        
    def add_health_check(self, check_func: Callable, name: str = None):
        """Add a health check function"""
        if name:
            check_func._health_check_name = name
        self.health_checks.append(check_func)
        
    def start(self):
        """Start the health checker"""
        if self.running:
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self.thread.start()
        self.logger.info("Health checker started")
        
    def stop(self):
        """Stop the health checker"""
        self.running = False
        if self.thread:
            self.thread.join()
        self.logger.info("Health checker stopped")
        
    def _health_check_loop(self):
        """Main health check loop"""
        while self.running:
            try:
                self._run_health_checks()
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                
            time.sleep(self.check_interval)
            
    def _run_health_checks(self):
        """Run all registered health checks"""
        for check_func in self.health_checks:
            try:
                check_name = getattr(check_func, '_health_check_name', check_func.__name__)
                result = check_func()
                
                if result:
                    self.logger.debug(f"Health check '{check_name}' passed")
                else:
                    self.logger.warning(f"Health check '{check_name}' failed")
                    
            except Exception as e:
                check_name = getattr(check_func, '_health_check_name', check_func.__name__)
                self.logger.error(f"Health check '{check_name}' raised exception: {e}")


class GracefulDegradation:
    """Graceful degradation when services fail"""
    
    def __init__(self):
        self.fallback_handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(f"{__name__}.GracefulDegradation")
        
    def register_fallback(self, service_name: str, fallback_func: Callable):
        """Register a fallback function for a service"""
        self.fallback_handlers[service_name] = fallback_func
        self.logger.info(f"Registered fallback for service: {service_name}")
        
    def with_fallback(self, service_name: str):
        """Decorator to add fallback behavior to a function"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.logger.warning(
                        f"Service '{service_name}' failed, attempting fallback: {e}"
                    )
                    
                    if service_name in self.fallback_handlers:
                        try:
                            fallback_result = self.fallback_handlers[service_name](*args, **kwargs)
                            self.logger.info(f"Fallback successful for service: {service_name}")
                            return fallback_result
                        except Exception as fallback_error:
                            self.logger.error(
                                f"Fallback also failed for service '{service_name}': {fallback_error}"
                            )
                            
                    # Re-raise original exception if no fallback or fallback failed
                    raise
                    
            return wrapper
        return decorator


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is in OPEN state"""
    pass


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted"""
    pass


# Global instances
database_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
api_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
websocket_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=45.0)

standard_retry_policy = RetryPolicy(max_attempts=3, base_delay=1.0)
aggressive_retry_policy = RetryPolicy(max_attempts=5, base_delay=0.5, max_delay=30.0)

graceful_degradation = GracefulDegradation()
health_checker = HealthChecker()


# Convenience decorators
def with_database_circuit_breaker(func: Callable) -> Callable:
    """Apply database circuit breaker to function"""
    return database_circuit_breaker(func)


def with_api_circuit_breaker(func: Callable) -> Callable:
    """Apply API circuit breaker to function"""
    return api_circuit_breaker(func)


def with_retry(func: Callable) -> Callable:
    """Apply standard retry policy to function"""
    return standard_retry_policy(func)


def with_aggressive_retry(func: Callable) -> Callable:
    """Apply aggressive retry policy to function"""
    return aggressive_retry_policy(func)
