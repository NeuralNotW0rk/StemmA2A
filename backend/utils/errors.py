"""Errors that are reported to the user as a message alone, and helpers for recording failed jobs."""

import traceback


class EngineUnreachableError(ConnectionError):
    """
    The engine service can't be reached. Its message says all the user needs to know, so jobs that fail
    with it record no traceback.
    """


def job_traceback(error: BaseException) -> str | None:
    """The traceback to record on a failed job or error response, or None when the message says enough."""
    if isinstance(error, EngineUnreachableError):
        return None
    return "".join(traceback.format_exception(type(error), error, error.__traceback__))
