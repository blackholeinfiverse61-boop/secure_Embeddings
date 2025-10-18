import time
import logging
from typing import Any, Callable, Optional, Type
from enum import Enum

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing if service recovered

class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass

class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time = None
        self._success_count = 0

    def _can_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self._last_failure_time is None:
            return False
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _reset(self):
        """Reset circuit breaker to closed state."""
        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info("Circuit breaker reset to CLOSED state")

    def _record_success(self):
        """Record a successful operation."""
        if self._state == CircuitBreakerState.HALF_OPEN:
            self._success_count += 1
            # Require a few successes before fully resetting
            if self._success_count >= 2:
                self._reset()

    def _record_failure(self):
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._state == CircuitBreakerState.HALF_OPEN:
            # Failed again, go back to open
            self._state = CircuitBreakerState.OPEN
            logger.warning("Circuit breaker returned to OPEN state")
        elif self._failure_count >= self.failure_threshold:
            # Threshold reached, open circuit
            self._state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self._failure_count} failures")

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self._state == CircuitBreakerState.OPEN:
            if self._can_attempt_reset():
                self._state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker testing recovery (HALF_OPEN)")
            else:
                raise CircuitBreakerOpenException("Circuit breaker is OPEN")

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except self.expected_exception as e:
            self._record_failure()
            raise e

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

# Global circuit breaker instances
pinecone_circuit_breaker = CircuitBreaker(
    failure_threshold=3,
    recovery_timeout=30,
    expected_exception=Exception
)

# Additional circuit breakers can be added here for other services