from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, TypeVar

from tqdm.auto import tqdm

from dspy.utils.parallelizer import ParallelExecutor

from .types import ItemT, LogLevel, Verbosity, verbosity_rank

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass
class RuntimeTools:
    """Utility helpers shared across different APEX components."""

    verbosity: Verbosity
    num_threads: int
    logger: logging.Logger = logger

    def is_enabled(self, level: Verbosity) -> bool:
        return verbosity_rank(self.verbosity) >= verbosity_rank(level)

    def log(self, message: str, level: Verbosity = Verbosity.NORMAL, log_level: LogLevel = "info") -> None:
        if not self.is_enabled(level):
            return
        if log_level == "warning":
            self.logger.warning(message)
        elif log_level == "debug":
            self.logger.debug(message)
        elif log_level == "error":
            self.logger.error(message)
        else:
            self.logger.info(message)

    def iter_with_progress(
        self,
        iterable: Iterable[_T],
        *,
        description: str,
        level: Verbosity,
        total: int | None = None,
    ) -> Iterator[_T]:
        if not self.is_enabled(level):
            yield from iterable
            return
        progress_total = total
        if progress_total is None and hasattr(iterable, "__len__"):
            progress_total = len(iterable)  # type: ignore[arg-type]
        with tqdm(iterable, total=progress_total, desc=description, leave=False) as progress:
            yield from progress

    def parallel_execute(
        self,
        items: Iterable[ItemT],
        func: Callable[[ItemT], Any],
        *,
        description: str,
        level: Verbosity,
    ) -> list[Any]:
        items_list = list(items)
        if not items_list:
            return []
        if self.num_threads <= 1 or len(items_list) <= 1:
            results: list[Any] = []
            for item in self.iter_with_progress(items_list, description=description, level=level):
                results.append(func(item))
            return results
        executor = ParallelExecutor(
            num_threads=self.num_threads,
            disable_progress_bar=not self.is_enabled(level),
            max_errors=max(len(items_list), 1),
            provide_traceback=self.is_enabled(Verbosity.DETAILED),
        )
        return executor.execute(func, items_list)
