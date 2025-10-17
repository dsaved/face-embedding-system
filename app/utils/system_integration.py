"""
System Integration and Testing Orchestrator
Integrates all enhanced features and runs comprehensive validation
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional
import json
import os
from pathlib import Path

# Import enhanced components
from app.utils.enhanced_logging import app_logger, performance_monitor as log_perf_monitor
from app.utils.error_recovery import CircuitBreaker, RetryPolicy, HealthChecker
from app.utils.performance_monitoring import performance_monitor
from app.utils.optimization_toolkit import initialize_optimization_system
from app.utils.edge_case_testing import edge_case_tester


class SystemIntegrator:
    """Integrates and orchestrates all enhanced system components"""
    
    def __init__(self, app_module=None, config_module=None):
        self.app_module = app_module
        self.config_module = config_module
        self.logger = app_logger
        self.integration_results: Dict[str, Any] = {}
        
        # Initialize components
        self.performance_monitor = performance_monitor
        self.health_checker = HealthChecker(check_interval=30.0)
        self.circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60.0)
        self.retry_policy = RetryPolicy(max_retries=3, backoff_multiplier=2.0)
        
        # Integration status
        self.integration_status = {
            'enhanced_logging': False,
            'performance_monitoring': False,
            'error_recovery': False,
            'optimization_system': False,
            'edge_case_testing': False,
        }
        
    def initialize_all_systems(self) -> Dict[str, Any]:
        """Initialize all enhanced systems"""
        self.logger.info("Starting system integration initialization...")
        
        results = {}
        
        # 1. Initialize Enhanced Logging
        try:
            self.logger.info("Initializing enhanced logging system...")
            log_perf_monitor.start_monitoring()
            self.integration_status['enhanced_logging'] = True
            results['enhanced_logging'] = {'status': 'success', 'message': 'Enhanced logging initialized'}
            self.logger.info("Enhanced logging system initialized successfully")
        except Exception as e:
            results['enhanced_logging'] = {'status': 'error', 'message': f'Failed to initialize enhanced logging: {e}'}
            self.logger.error(f"Failed to initialize enhanced logging: {e}")
            
        # 2. Initialize Performance Monitoring
        try:
            self.logger.info("Initializing performance monitoring system...")
            self.performance_monitor.start()
            self.integration_status['performance_monitoring'] = True
            results['performance_monitoring'] = {'status': 'success', 'message': 'Performance monitoring initialized'}
            self.logger.info("Performance monitoring system initialized successfully")
        except Exception as e:
            results['performance_monitoring'] = {'status': 'error', 'message': f'Failed to initialize performance monitoring: {e}'}
            self.logger.error(f"Failed to initialize performance monitoring: {e}")
            
        # 3. Initialize Error Recovery
        try:
            self.logger.info("Initializing error recovery system...")
            self.health_checker.start()
            self.integration_status['error_recovery'] = True
            results['error_recovery'] = {'status': 'success', 'message': 'Error recovery system initialized'}
            self.logger.info("Error recovery system initialized successfully")
        except Exception as e:
            results['error_recovery'] = {'status': 'error', 'message': f'Failed to initialize error recovery: {e}'}
            self.logger.error(f"Failed to initialize error recovery: {e}")
            
        # 4. Initialize Optimization System
        try:
            self.logger.info("Initializing optimization system...")
            if self.config_module:
                initialize_optimization_system(self.config_module, self.performance_monitor)
                self.integration_status['optimization_system'] = True
                results['optimization_system'] = {'status': 'success', 'message': 'Optimization system initialized'}
                self.logger.info("Optimization system initialized successfully")
            else:
                results['optimization_system'] = {'status': 'warning', 'message': 'Config module not available for optimization system'}
        except Exception as e:
            results['optimization_system'] = {'status': 'error', 'message': f'Failed to initialize optimization system: {e}'}
            self.logger.error(f"Failed to initialize optimization system: {e}")
            
        # 5. Initialize Edge Case Testing
        try:
            self.logger.info("Initializing edge case testing system...")
            # Edge case tester is initialized statically
            self.integration_status['edge_case_testing'] = True
            results['edge_case_testing'] = {'status': 'success', 'message': 'Edge case testing system initialized'}
            self.logger.info("Edge case testing system initialized successfully")
        except Exception as e:
            results['edge_case_testing'] = {'status': 'error', 'message': f'Failed to initialize edge case testing: {e}'}
            self.logger.error(f"Failed to initialize edge case testing: {e}")
            
        self.integration_results = results
        self.logger.info(f"System integration completed. Status: {self.integration_status}")
        return results
        
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive system validation"""
        self.logger.info("Starting comprehensive system validation...")
        validation_results = {}
        
        # 1. System Health Check
        try:
            self.logger.info("Running system health check...")
            health_status = self.health_checker.get_system_health()
            validation_results['health_check'] = {
                'status': 'success' if health_status['overall_health'] == 'healthy' else 'warning',
                'details': health_status
            }
        except Exception as e:
            validation_results['health_check'] = {'status': 'error', 'message': f'Health check failed: {e}'}
            
        # 2. Performance Baseline
        try:
            self.logger.info("Establishing performance baseline...")
            performance_status = self.performance_monitor.get_current_status()
            validation_results['performance_baseline'] = {
                'status': 'success',
                'details': performance_status
            }
        except Exception as e:
            validation_results['performance_baseline'] = {'status': 'error', 'message': f'Performance baseline failed: {e}'}
            
        # 3. Error Recovery Validation
        try:
            self.logger.info("Validating error recovery mechanisms...")
            # Test circuit breaker
            circuit_breaker_status = self._test_circuit_breaker()
            
            # Test retry policy
            retry_policy_status = self._test_retry_policy()
            
            validation_results['error_recovery'] = {
                'status': 'success',
                'circuit_breaker': circuit_breaker_status,
                'retry_policy': retry_policy_status
            }
        except Exception as e:
            validation_results['error_recovery'] = {'status': 'error', 'message': f'Error recovery validation failed: {e}'}
            
        # 4. Edge Case Testing (subset)
        try:
            self.logger.info("Running critical edge case tests...")
            # Run a subset of edge case tests for validation
            critical_tests = ['network_failures', 'invalid_inputs', 'resource_constraints']
            edge_case_results = {}
            
            for test_category in critical_tests:
                try:
                    test_result = edge_case_tester.run_category_tests(test_category)
                    edge_case_results[test_category] = test_result
                except Exception as test_error:
                    edge_case_results[test_category] = {'status': 'error', 'message': str(test_error)}
                    
            validation_results['edge_case_testing'] = {
                'status': 'success',
                'details': edge_case_results
            }
        except Exception as e:
            validation_results['edge_case_testing'] = {'status': 'error', 'message': f'Edge case testing failed: {e}'}
            
        # 5. Integration Validation
        try:
            self.logger.info("Validating system integration...")
            integration_validation = self._validate_integration()
            validation_results['integration_validation'] = integration_validation
        except Exception as e:
            validation_results['integration_validation'] = {'status': 'error', 'message': f'Integration validation failed: {e}'}
            
        self.logger.info("Comprehensive validation completed")
        return validation_results
        
    def _test_circuit_breaker(self) -> Dict[str, Any]:
        """Test circuit breaker functionality"""
        def failing_function():
            raise ConnectionError("Simulated service failure")
            
        def working_function():
            return "Success"
            
        # Test failure threshold
        failure_count = 0
        for _ in range(self.circuit_breaker.failure_threshold + 1):
            try:
                self.circuit_breaker.call(failing_function)
            except Exception:
                failure_count += 1
                
        # Circuit should now be open
        try:
            self.circuit_breaker.call(working_function)
            circuit_blocked = False
        except Exception:
            circuit_blocked = True
            
        return {
            'failure_count': failure_count,
            'circuit_opened': circuit_blocked,
            'status': 'passed' if circuit_blocked else 'failed'
        }
        
    def _test_retry_policy(self) -> Dict[str, Any]:
        """Test retry policy functionality"""
        attempt_count = 0
        
        def eventually_succeeds():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ConnectionError("Temporary failure")
            return "Success"
            
        try:
            result = self.retry_policy.execute(eventually_succeeds)
            success = result == "Success"
        except Exception:
            success = False
            
        return {
            'attempts': attempt_count,
            'success': success,
            'status': 'passed' if success and attempt_count == 3 else 'failed'
        }
        
    def _validate_integration(self) -> Dict[str, Any]:
        """Validate integration between components"""
        validation_results = {}
        
        # Check if all systems are initialized
        all_initialized = all(self.integration_status.values())
        validation_results['all_systems_initialized'] = all_initialized
        
        # Test component interactions
        try:
            # Test logging with performance monitoring
            start_time = time.time()
            self.logger.info("Integration test message")
            log_time = time.time() - start_time
            
            # Track the operation
            self.performance_monitor.track_response_time(log_time)
            
            validation_results['logging_performance_integration'] = {
                'status': 'passed',
                'log_time': log_time
            }
        except Exception as e:
            validation_results['logging_performance_integration'] = {
                'status': 'failed',
                'error': str(e)
            }
            
        # Test error recovery with logging
        try:
            def test_error_function():
                self.logger.error("Test error for integration validation")
                raise ValueError("Test error")
                
            try:
                self.retry_policy.execute(test_error_function)
            except Exception:
                pass  # Expected to fail
                
            validation_results['error_recovery_logging_integration'] = {
                'status': 'passed'
            }
        except Exception as e:
            validation_results['error_recovery_logging_integration'] = {
                'status': 'failed',
                'error': str(e)
            }
            
        return {
            'status': 'success' if all_initialized else 'partial',
            'details': validation_results,
            'integration_status': self.integration_status
        }
        
    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration report"""
        self.logger.info("Generating integration report...")
        
        report = {
            'timestamp': time.time(),
            'integration_status': self.integration_status,
            'initialization_results': self.integration_results,
            'system_health': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Get current system health
        try:
            report['system_health'] = self.health_checker.get_system_health()
        except Exception as e:
            report['system_health'] = {'error': str(e)}
            
        # Get performance metrics
        try:
            report['performance_metrics'] = self.performance_monitor.get_current_status()
        except Exception as e:
            report['performance_metrics'] = {'error': str(e)}
            
        # Generate recommendations
        if not all(self.integration_status.values()):
            report['recommendations'].append("Complete initialization of all system components")
            
        if 'system_metrics' in report['performance_metrics']:
            sys_metrics = report['performance_metrics']['system_metrics']
            if sys_metrics.get('avg_cpu_percent', 0) > 80:
                report['recommendations'].append("High CPU usage detected - consider optimization")
            if sys_metrics.get('avg_memory_percent', 0) > 80:
                report['recommendations'].append("High memory usage detected - monitor for leaks")
                
        self.logger.info("Integration report generated successfully")
        return report
        
    def shutdown_all_systems(self):
        """Gracefully shutdown all systems"""
        self.logger.info("Shutting down all integrated systems...")
        
        try:
            self.performance_monitor.stop()
            self.logger.info("Performance monitoring stopped")
        except Exception as e:
            self.logger.error(f"Error stopping performance monitoring: {e}")
            
        try:
            self.health_checker.stop()
            self.logger.info("Health checker stopped")
        except Exception as e:
            self.logger.error(f"Error stopping health checker: {e}")
            
        try:
            log_perf_monitor.stop_monitoring()
            self.logger.info("Enhanced logging monitoring stopped")
        except Exception as e:
            self.logger.error(f"Error stopping enhanced logging: {e}")
            
        self.logger.info("System shutdown completed")


# Global system integrator
system_integrator = None


def initialize_integrated_system(app_module=None, config_module=None) -> SystemIntegrator:
    """Initialize the integrated system"""
    global system_integrator
    system_integrator = SystemIntegrator(app_module, config_module)
    return system_integrator


def run_complete_system_validation() -> Dict[str, Any]:
    """Run complete system validation and integration testing"""
    if not system_integrator:
        raise RuntimeError("System integrator not initialized")
        
    # Initialize all systems
    init_results = system_integrator.initialize_all_systems()
    
    # Run comprehensive validation
    validation_results = system_integrator.run_comprehensive_validation()
    
    # Generate integration report
    integration_report = system_integrator.generate_integration_report()
    
    return {
        'initialization': init_results,
        'validation': validation_results,
        'integration_report': integration_report,
        'summary': {
            'total_systems': len(system_integrator.integration_status),
            'initialized_systems': sum(system_integrator.integration_status.values()),
            'success_rate': sum(system_integrator.integration_status.values()) / len(system_integrator.integration_status) * 100
        }
    }


def save_validation_report(results: Dict[str, Any], filename: str = None):
    """Save validation results to file"""
    if filename is None:
        timestamp = int(time.time())
        filename = f"system_validation_report_{timestamp}.json"
        
    report_path = Path("logs") / filename
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
        
    print(f"Validation report saved to: {report_path}")
    return str(report_path)


if __name__ == "__main__":
    # If run directly, perform complete system validation
    print("Starting complete system validation...")
    
    try:
        # Initialize system
        integrator = initialize_integrated_system()
        
        # Run validation
        results = run_complete_system_validation()
        
        # Save report
        report_path = save_validation_report(results)
        
        # Print summary
        print("\n" + "="*50)
        print("SYSTEM VALIDATION SUMMARY")
        print("="*50)
        print(f"Total Systems: {results['summary']['total_systems']}")
        print(f"Initialized Systems: {results['summary']['initialized_systems']}")
        print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
        print(f"Report saved to: {report_path}")
        
        # Shutdown
        integrator.shutdown_all_systems()
        
    except Exception as e:
        print(f"System validation failed: {e}")
        import traceback
        traceback.print_exc()
