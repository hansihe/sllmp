"""
Unit tests for limit middleware data models and backend.

Tests for the constraint/limit data classes (BudgetLimit, RateLimit, Constraint)
and the InMemoryLimitBackend. Integration tests for the limit_enforcement_middleware
are in test_limit_middleware_integration.py.
"""

import pytest
from sllmp.middleware.limit import (
    BudgetLimit,
    RateLimit,
    Constraint,
    InMemoryLimitBackend,
)


class TestBudgetLimit:
    """Test BudgetLimit dataclass validation."""

    def test_valid_budget_limit(self):
        limit = BudgetLimit(limit=10.0, window="1d")
        assert limit.limit == 10.0
        assert limit.window == "1d"

    def test_negative_limit_raises_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            BudgetLimit(limit=-1.0, window="1d")

    def test_zero_limit_raises_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            BudgetLimit(limit=0.0, window="1d")

    def test_invalid_window_raises_error(self):
        with pytest.raises(ValueError, match="Invalid window"):
            BudgetLimit(limit=10.0, window="2d")


class TestRateLimit:
    """Test RateLimit dataclass validation."""

    def test_valid_rate_limit(self):
        limit = RateLimit(per_minute=100)
        assert limit.per_minute == 100

    def test_negative_rate_raises_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            RateLimit(per_minute=-1)

    def test_zero_rate_raises_error(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="Input should be greater than 0"):
            RateLimit(per_minute=0)


class TestConstraint:
    """Test Constraint dataclass validation."""

    def test_valid_budget_constraint(self):
        constraint = Constraint(
            name="User Budget",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=50.0, window="1d"),
            rate_limit=None,
            description="Daily budget per user",
        )
        assert constraint.name == "User Budget"
        assert constraint.dimensions == ["user_id"]

    def test_valid_rate_constraint(self):
        constraint = Constraint(
            name="User Rate",
            dimensions=["user_id"],
            budget_limit=None,
            rate_limit=RateLimit(per_minute=10),
            description="Rate limit per user",
        )
        assert constraint.rate_limit.per_minute == 10

    def test_constraint_with_both_limits(self):
        constraint = Constraint(
            name="Combined",
            dimensions=["user_id"],
            budget_limit=BudgetLimit(limit=10.0, window="1d"),
            rate_limit=RateLimit(per_minute=5),
            description="Both limits",
        )
        assert constraint.budget_limit.limit == 10.0
        assert constraint.rate_limit.per_minute == 5

    def test_invalid_dimensions_raise_error(self):
        with pytest.raises(ValueError, match="Invalid dimensions"):
            Constraint(
                name="Bad",
                dimensions=["invalid_dim"],
                budget_limit=BudgetLimit(limit=10.0, window="1d"),
                rate_limit=None,
                description="Invalid dimension",
            )

    def test_duplicate_dimensions_raise_error(self):
        with pytest.raises(ValueError, match="Duplicate dimensions"):
            Constraint(
                name="Dupe",
                dimensions=["user_id", "user_id"],
                budget_limit=BudgetLimit(limit=10.0, window="1d"),
                rate_limit=None,
                description="Duplicate dimensions",
            )

    def test_no_limits_raises_error(self):
        with pytest.raises(
            ValueError, match="Either budget or rate limit must be specified"
        ):
            Constraint(
                name="No Limits",
                dimensions=["user_id"],
                budget_limit=None,
                rate_limit=None,
                description="No limits",
            )


class TestInMemoryLimitBackend:
    """Test in-memory limit backend."""

    @pytest.fixture
    def backend(self):
        return InMemoryLimitBackend()

    async def test_get_usage_empty(self, backend):
        usage = await backend.get_usage("user:test", "1d")
        assert usage == 0.0

    async def test_increment_and_get_usage(self, backend):
        await backend.increment_usage("user:test", 5.5, "1d")
        usage = await backend.get_usage("user:test", "1d")
        assert usage == 5.5

    async def test_multiple_increments(self, backend):
        await backend.increment_usage("user:test", 2.0, "1d")
        await backend.increment_usage("user:test", 3.0, "1d")
        usage = await backend.get_usage("user:test", "1d")
        assert usage == 5.0

    async def test_separate_keys(self, backend):
        await backend.increment_usage("user:test1", 10.0, "1d")
        await backend.increment_usage("user:test2", 20.0, "1d")

        usage1 = await backend.get_usage("user:test1", "1d")
        usage2 = await backend.get_usage("user:test2", "1d")

        assert usage1 == 10.0
        assert usage2 == 20.0

    async def test_get_rate_usage_empty(self, backend):
        usage = await backend.get_rate_usage("user:test")
        assert usage == 0

    async def test_increment_and_get_rate_usage(self, backend):
        await backend.increment_rate_usage("user:test")
        await backend.increment_rate_usage("user:test")
        usage = await backend.get_rate_usage("user:test")
        assert usage == 2

    async def test_rate_usage_different_keys(self, backend):
        await backend.increment_rate_usage("user:test1")
        await backend.increment_rate_usage("user:test2")
        await backend.increment_rate_usage("user:test2")

        usage1 = await backend.get_rate_usage("user:test1")
        usage2 = await backend.get_rate_usage("user:test2")

        assert usage1 == 1
        assert usage2 == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
