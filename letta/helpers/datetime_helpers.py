import time
from collections.abc import Callable
from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from time import strftime
from typing import Any

import pytz

_DATETIME_FORMAT = "%Y-%m-%d %I:%M:%S %p %Z%z"


def get_local_time_fast():
    formatted_time = strftime(_DATETIME_FORMAT)

    return formatted_time


def get_local_time_timezone(timezone: str = "America/Los_Angeles"):
    # Get the current time in UTC
    current_time_utc = datetime.now(pytz.utc)

    # Convert to San Francisco's time zone (PST/PDT)
    sf_time_zone = pytz.timezone(timezone)
    local_time = current_time_utc.astimezone(sf_time_zone)

    # You may format it as you desire, including AM/PM
    formatted_time = local_time.strftime(_DATETIME_FORMAT)

    return formatted_time


def get_local_time(timezone: str | None = None):
    if timezone is not None:
        time_str = get_local_time_timezone(timezone)
    else:
        # Get the current time, which will be in the local timezone of the computer
        local_time = datetime.now().astimezone()

        # You may format it as you desire, including AM/PM
        time_str = local_time.strftime(_DATETIME_FORMAT)

    return time_str.strip()


def get_utc_time() -> datetime:
    """Get the current UTC time"""
    return datetime.now(dt_timezone.utc)


def get_utc_time_int() -> int:
    return int(get_utc_time().timestamp())


def get_utc_timestamp_ns() -> int:
    """Get the current UTC time in nanoseconds"""
    return int(time.time_ns())


def ns_to_ms(ns: int) -> int:
    return ns // 1_000_000


def timestamp_to_datetime(timestamp_seconds: int) -> datetime:
    """Convert Unix timestamp in seconds to UTC datetime object"""
    return datetime.fromtimestamp(timestamp_seconds, tz=dt_timezone.utc)


def is_utc_datetime(dt: datetime) -> bool:
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) == timedelta(0)


class AsyncTimer:
    """An async context manager for timing async code execution.

    Takes in an optional callback_func to call on exit with arguments
    taking in the elapsed_ms and exc if present.

    Do not use the start and end times outside of this function as they are relative.
    """

    def __init__(self, callback_func: Callable[..., Any] | None = None):
        self._start_time_ns: int = 0
        self._end_time_ns: int = 0
        self.elapsed_ns: int = 0
        self.callback_func: Callable[..., Any] | None = callback_func

    async def __aenter__(self):
        self._start_time_ns = time.perf_counter_ns()
        return self

    async def __aexit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> bool:
        self._end_time_ns = time.perf_counter_ns()
        self.elapsed_ns = self._end_time_ns - self._start_time_ns
        if self.callback_func:
            from asyncio import iscoroutinefunction

            if iscoroutinefunction(self.callback_func):
                await self.callback_func(self.elapsed_ms, exc)
            else:
                self.callback_func(self.elapsed_ms, exc)
        return False

    @property
    def elapsed_ms(self):
        return ns_to_ms(self.elapsed_ns)
