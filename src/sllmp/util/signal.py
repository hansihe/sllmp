import inspect
from typing import Callable, Union, List, Optional, TypeVar, Generic, Awaitable, cast, Any
from typing_extensions import ParamSpec
from enum import Enum
from dataclasses import dataclass, field
import time

# Type parameters
P = ParamSpec('P')  # Parameters for hooks/callbacks
T = TypeVar('T')    # Return type for hooks
R = TypeVar('R')    # Post-hook data type

class SignalState(Enum):
    OPEN = "open"
    CLOSED = "closed"


@dataclass
class CallbackResult(Generic[T]):
    """Result of a single callback execution."""
    callback: Callable[..., Union[T, Awaitable[T]]]
    success: bool
    return_value: Optional[T] = None
    exception: Optional[Exception] = None
    execution_time: Optional[float] = None

class Hooks(Generic[P, R]):
    """A reusable hooks container for pre/post execution hooks.

    Type Parameters:
        P: Parameters for pre-hooks (ParamSpec)
        R: Post-hook data type
    """

    def __init__(self) -> None:
        self._pre_hooks: List[Callable[P, Union[Any, Awaitable[Any]]]] = []
        self._post_hooks: List[Callable[[R], Union[Any, Awaitable[Any]]]] = []

    def add(self,
            pre_hook: Optional[Callable[P, Union[Any, Awaitable[Any]]]] = None,
            post_hook: Optional[Callable[[R], Union[Any, Awaitable[Any]]]] = None) -> None:
        """Add pre and/or post execution hooks.

        Args:
            pre_hook: Called before main execution, in add order.
            post_hook: Called after main execution, in reverse add order.
        """
        if pre_hook is not None:
            self._pre_hooks.append(pre_hook)
        if post_hook is not None:
            self._post_hooks.append(post_hook)

    def add_pre(self, hook: Callable[P, Union[Any, Awaitable[Any]]]) -> None:
        """Add a pre-execution hook. Runs in add order."""
        self._pre_hooks.append(hook)

    def add_post(self, hook: Callable[[R], Union[Any, Awaitable[Any]]]) -> None:
        """Add a post-execution hook. Runs in reverse add order."""
        self._post_hooks.append(hook)

    def remove_pre(self, hook: Callable[P, Union[Any, Awaitable[Any]]]) -> bool:
        """Remove a pre-execution hook. Returns True if found and removed."""
        try:
            self._pre_hooks.remove(hook)
            return True
        except ValueError:
            return False

    def remove_post(self, hook: Callable[[R], Union[Any, Awaitable[Any]]]) -> bool:
        """Remove a post-execution hook. Returns True if found and removed."""
        try:
            self._post_hooks.remove(hook)
            return True
        except ValueError:
            return False

    async def execute_pre(self, *args: P.args, **kwargs: P.kwargs) -> None:
        """Execute all pre-hooks in add order, silently catching exceptions."""
        for pre_hook in self._pre_hooks:
            try:
                if inspect.iscoroutinefunction(pre_hook):
                    await pre_hook(*args, **kwargs)
                else:
                    pre_hook(*args, **kwargs)
            except Exception:
                pass

    async def execute_post(self, data: R) -> None:
        """Execute all post-hooks in reverse add order, silently catching exceptions."""
        for post_hook in reversed(self._post_hooks):
            try:
                if inspect.iscoroutinefunction(post_hook):
                    await post_hook(data)
                else:
                    post_hook(data)
            except Exception:
                pass

    def clear(self) -> None:
        """Remove all hooks."""
        self._pre_hooks.clear()
        self._post_hooks.clear()

    def clear_pre(self) -> None:
        """Remove all pre-hooks."""
        self._pre_hooks.clear()

    def clear_post(self) -> None:
        """Remove all post-hooks."""
        self._post_hooks.clear()

    def __len__(self) -> int:
        """Return total number of hooks."""
        return len(self._pre_hooks) + len(self._post_hooks)

    def __bool__(self) -> bool:
        """Return True if any hooks are registered."""
        return len(self._pre_hooks) > 0 or len(self._post_hooks) > 0

S = TypeVar('S')    # Signal callback return type

@dataclass
class SignalExecutionResult(Generic[T]):
    """Result of signal emission containing detailed execution information."""
    completed: bool  # True if all callbacks executed, False if halted
    halted_by: Optional[Callable[..., Union[T, Awaitable[T]]]] = None
    callback_results: List[CallbackResult[T]] = field(default_factory=list)
    total_execution_time: Optional[float] = None
    callbacks_executed: int = 0
    callbacks_skipped: int = 0

    @property
    def success(self) -> bool:
        """True if execution completed without exceptions."""
        return self.completed and all(
            result.success for result in self.callback_results
        )

    @property
    def return_values(self) -> List[T]:
        """List of return values from all executed callbacks."""
        return [
            result.return_value for result in self.callback_results
            if result.success and result.return_value is not None
        ]

    @property
    def exceptions(self) -> List[Exception]:
        """List of all exceptions raised by callbacks."""
        return [
            result.exception for result in self.callback_results
            if result.exception is not None
        ]

class Signal(Generic[P, S]):
    """A typed signal that can emit events to connected callbacks.

    Type Parameters:
        P: The parameters that will be passed to callbacks when the signal is emitted
        T: The expected return type of callbacks
    """

    def __init__(self) -> None:
        self._state = SignalState.OPEN
        self._callbacks: List[Callable[P, Union[S, Awaitable[S]]]] = []
        self._executing = False
        self._pending_callbacks: List[Callable[P, Union[S, Awaitable[S]]]] = []
        self.hooks: Hooks[P, SignalExecutionResult[S]] = Hooks()

    @property
    def is_open(self) -> bool:
        return self._state == SignalState.OPEN

    @property
    def is_closed(self) -> bool:
        return self._state == SignalState.CLOSED

    def close(self) -> None:
        """Close the signal, preventing new callbacks from being added."""
        self._state = SignalState.CLOSED

    def open(self) -> None:
        """Reopen the signal, allowing new callbacks to be added."""
        self._state = SignalState.OPEN

    def connect(self, callback: Callable[P, Union[S, Awaitable[S]]]) -> None:
        """Add a callback to the signal.

        Args:
            callback: Function to call when signal is emitted. Can be sync or async.
                     Must match the signal's parameter and return type signatures.

        Raises:
            RuntimeError: If the signal is closed.
        """
        if self._state == SignalState.CLOSED:
            raise RuntimeError("Cannot add callback to closed signal")

        if self._executing:
            self._pending_callbacks.append(callback)
        else:
            self._callbacks.append(callback)

    def disconnect(self, callback: Callable[P, Union[S, Awaitable[S]]]) -> bool:
        """Remove a callback from the signal.

        Args:
            callback: The callback to remove.

        Returns:
            True if callback was found and removed, False otherwise.
        """
        try:
            self._callbacks.remove(callback)
            return True
        except ValueError:
            if self._executing and callback in self._pending_callbacks:
                self._pending_callbacks.remove(callback)
                return True
            return False


    async def emit(self, *args: P.args, **kwargs: P.kwargs) -> SignalExecutionResult[S]:
        """Emit the signal, calling all connected callbacks.

        Args:
            *args: Positional arguments to pass to callbacks.
            **kwargs: Keyword arguments to pass to callbacks.

        Returns:
            SignalExecutionResult containing detailed execution information.

        Raises:
            RuntimeError: If signal is already being executed.
        """
        if self._executing:
            raise RuntimeError("Signal is already being executed (recursive emission not allowed)")

        start_time = time.perf_counter()
        result: SignalExecutionResult[S] = SignalExecutionResult(completed=False)
        self._executing = True

        try:
            # Execute pre-hooks
            await self.hooks.execute_pre(*args, **kwargs)

            # Execute main callbacks
            all_callbacks = self._callbacks.copy()
            callback_index = 0

            while callback_index < len(all_callbacks):
                callback = all_callbacks[callback_index]
                callback_start = time.perf_counter()
                callback_result: CallbackResult[S] = CallbackResult(callback=callback, success=False)

                try:
                    if inspect.iscoroutinefunction(callback):
                        return_value = await callback(*args, **kwargs)
                    else:
                        return_value = cast(S, callback(*args, **kwargs))

                    callback_result.success = True
                    callback_result.return_value = return_value
                    result.callbacks_executed += 1

                except Exception as e:
                    callback_result.exception = e
                    callback_result.success = False
                    result.callbacks_executed += 1

                finally:
                    callback_result.execution_time = time.perf_counter() - callback_start
                    result.callback_results.append(callback_result)

                if self._pending_callbacks:
                    all_callbacks.extend(self._pending_callbacks)
                    self._pending_callbacks.clear()

                callback_index += 1

            if result.halted_by is None:
                result.completed = True

        finally:
            self._executing = False
            self._pending_callbacks.clear()
            result.total_execution_time = time.perf_counter() - start_time

            # Execute post-hooks
            await self.hooks.execute_post(result)

        return result

    def clear(self) -> None:
        """Remove all callbacks from the signal."""
        if self._executing:
            raise RuntimeError("Cannot clear callbacks while signal is executing")
        self._callbacks.clear()

    def clear_all(self) -> None:
        """Remove all callbacks and hooks from the signal."""
        if self._executing:
            raise RuntimeError("Cannot clear callbacks and hooks while signal is executing")
        self._callbacks.clear()
        self.hooks.clear()

    def __len__(self) -> int:
        """Return the number of connected callbacks."""
        return len(self._callbacks) + len(self._pending_callbacks)

    def __bool__(self) -> bool:
        """Return True if the signal has any connected callbacks."""
        return len(self._callbacks) > 0 or len(self._pending_callbacks) > 0
