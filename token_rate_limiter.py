"""
Token rate limiter to prevent hitting Anthropic API rate limits.
Tracks token usage over time and throttles requests to stay under the limit.
"""
import time
import threading
from collections import deque
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TokenRateLimiter:
    """
    Rate limiter that tracks token usage and prevents exceeding API rate limits.
    Uses a sliding window to track token usage over the last minute.
    """

    def __init__(self, tokens_per_minute=400000, safety_margin=0.9):
        """
        Initialize the rate limiter.

        Args:
            tokens_per_minute: Maximum tokens allowed per minute (default 400,000)
            safety_margin: Use this fraction of the limit (default 0.9 = 90%)
        """
        self.max_tokens_per_minute = tokens_per_minute
        self.safety_margin = safety_margin
        self.effective_limit = int(tokens_per_minute * safety_margin)

        # Track token usage: deque of (timestamp, token_count) tuples
        self.token_history = deque()
        self.lock = threading.Lock()

        logger.info(f"TokenRateLimiter initialized: {tokens_per_minute:,} tokens/min limit, "
                   f"{self.effective_limit:,} effective limit (safety margin: {safety_margin:.0%})")

    def _cleanup_old_entries(self):
        """Remove entries older than 1 minute from the history."""
        cutoff_time = time.time() - 60  # 60 seconds ago

        while self.token_history and self.token_history[0][0] < cutoff_time:
            self.token_history.popleft()

    def _get_current_usage(self):
        """Get total tokens used in the last 60 seconds."""
        self._cleanup_old_entries()
        return sum(tokens for _, tokens in self.token_history)

    def wait_if_needed(self, estimated_tokens):
        """
        Wait if adding these tokens would exceed the rate limit.

        Args:
            estimated_tokens: Number of tokens about to be used
        """
        with self.lock:
            self._cleanup_old_entries()
            current_usage = self._get_current_usage()

            # If adding this request would exceed the limit, wait
            if current_usage + estimated_tokens > self.effective_limit:
                # Calculate how long to wait
                # Find the oldest entry that needs to expire to make room
                tokens_needed = (current_usage + estimated_tokens) - self.effective_limit

                # Find how many seconds until we have enough capacity
                running_total = 0
                wait_until = None

                for timestamp, tokens in self.token_history:
                    running_total += tokens
                    if running_total >= tokens_needed:
                        # We need to wait until this entry expires (60s after it was added)
                        wait_until = timestamp + 60
                        break

                if wait_until:
                    wait_time = wait_until - time.time()
                    if wait_time > 0:
                        logger.warning(
                            f"Rate limit: {current_usage:,}/{self.effective_limit:,} tokens used. "
                            f"Waiting {wait_time:.1f}s before next request "
                            f"(estimated: {estimated_tokens:,} tokens)..."
                        )
                        time.sleep(wait_time)
                        # Clean up again after waiting
                        self._cleanup_old_entries()

    def record_usage(self, actual_tokens):
        """
        Record actual token usage after a request completes.

        Args:
            actual_tokens: Number of tokens actually used
        """
        with self.lock:
            self.token_history.append((time.time(), actual_tokens))
            self._cleanup_old_entries()

            current_usage = self._get_current_usage()
            logger.debug(f"Recorded {actual_tokens:,} tokens. Current usage: {current_usage:,}/{self.effective_limit:,}")

    def get_stats(self):
        """Get current rate limiter statistics."""
        with self.lock:
            self._cleanup_old_entries()
            current_usage = self._get_current_usage()

            return {
                'current_usage': current_usage,
                'limit': self.effective_limit,
                'percentage': (current_usage / self.effective_limit * 100) if self.effective_limit > 0 else 0,
                'entries_in_window': len(self.token_history)
            }
