import logging
import asyncio
from prime_sandboxes import AsyncSandboxClient
import tenacity as tc

logger = logging.getLogger("SweGrepEnv")


class SandboxMetrics:
    def __init__(self):
        self.creation_success = 0
        self.creation_failed = 0
        self.clone_failed = 0
        self.setup_success = 0
        self.setup_failed = 0
        self.setup_retries = 0
        self.exec_retries = 0
        self.errors: list[dict] = []  # All errors for later grouping
        self._last_log_count = 0

    def track_error(self, error: Exception, operation: str):
        error_str = str(error)
        self.errors.append({
            "operation": operation,
            "type": type(error).__name__,
            "message": error_str[:200],
        })
        logger.warning(f"[{operation}] {type(error).__name__}: {error_str[:100]}")

    def maybe_log(self, every_n: int = 50):
        total = self.setup_success + self.setup_failed
        if total > 0 and total % every_n == 0 and total != self._last_log_count:
            self._last_log_count = total
            logger.info(
                f"[METRICS] setups={total} ok={self.setup_success} fail={self.setup_failed} "
                f"clone_fail={self.clone_failed} retries={self.exec_retries} errors={len(self.errors)}"
            )


RETRYABLE_ERRORS = ("502", "503", "409", "ConnectError")


def is_retryable(exception: Exception) -> bool:
    return any(tok in str(exception) for tok in RETRYABLE_ERRORS)


def retry_with_metrics(metrics: SandboxMetrics, operation: str, max_retries: int = 3):
    def before_sleep(retry_state):
        metrics.exec_retries += 1
        metrics.track_error(retry_state.outcome.exception(), operation)

    return tc.retry(
        retry=tc.retry_if_exception(is_retryable),
        stop=tc.stop_after_attempt(max_retries),
        wait=tc.wait_exponential(multiplier=1, min=1, max=8),
        before_sleep=before_sleep,
        reraise=True,
    )


async def execute_command(
    sandbox_client: AsyncSandboxClient,
    sandbox_id: str,
    command: str,
    metrics: SandboxMetrics,
    operation: str = "exec",
    max_retries: int = 3,
) -> tuple[bool, str]:
    @retry_with_metrics(metrics, operation, max_retries)
    async def _execute():
        result = await sandbox_client.execute_command(sandbox_id, command)
        return result.stdout or ""

    try:
        output = await _execute()
        return True, output
    except Exception as e:
        metrics.track_error(e, operation)
        return False, str(e)
