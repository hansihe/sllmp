from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

# Error type definitions
@dataclass(frozen=True)
class PipelineError(ABC):
    """Base class for all pipeline errors."""
    message: str
    request_id: str
    error_type: str = field(init=False)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for API responses."""
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "request_id": self.request_id,
                **self._extra_fields()
            }
        }

    def _extra_fields(self) -> Dict[str, Any]:
        """Override in subclasses to add additional fields."""
        return {}


@dataclass(frozen=True)
class MiddlewareError(PipelineError):
    """Error that occurred in middleware execution."""
    middleware_name: str
    error_type: str = field(default="middleware_error", init=False)

    def _extra_fields(self) -> Dict[str, Any]:
        return {"middleware": self.middleware_name}


@dataclass(frozen=True)
class StreamError(PipelineError):
    """Error that occurred during streaming."""
    error_type: str = field(default="stream_error", init=False)


@dataclass(frozen=True)
class ValidationError(PipelineError):
    """Error due to invalid request parameters."""
    field_name: Optional[str] = None
    error_type: str = field(default="validation_error", init=False)

    def _extra_fields(self) -> Dict[str, Any]:
        return {"field": self.field_name} if self.field_name else {}


@dataclass(frozen=True)
class AuthenticationError(PipelineError):
    """Error due to authentication failure."""
    error_type: str = field(default="authentication_error", init=False)


@dataclass(frozen=True)
class InternalError(PipelineError):
    """Internal system error."""
    error_type: str = field(default="internal_error", init=False)
