# Redis Limit Backend Testing

This document describes how to test the Redis limit backend implementation using both traditional Redis instances and testcontainers.

## Overview

The Redis limit backend provides production-ready Redis support for the limit enforcement middleware with:

- Budget tracking with time windows
- Rate limiting with sliding windows  
- Atomic operations for consistency
- Proper expiration handling
- Support for both standalone Redis and Redis Cluster

## Test Files

### 1. `tests/test_limit_redis.py`
Traditional Redis tests that require a manually running Redis instance.

**Markers**: `@pytest.mark.redis`

**Requirements**:
- Redis server running on `localhost:6379`
- Uses database 15 for testing to avoid conflicts

### 2. `tests/test_limit_redis_testcontainers.py`
Modern testcontainers-based tests that automatically manage Redis containers.

**Markers**: `@pytest.mark.asyncio`, `@pytest.mark.containers`

**Requirements**:
- Docker running and accessible
- Automatically starts/stops Redis containers

## Running Tests

### Option 1: Using Testcontainers (Recommended)

Testcontainers automatically start Redis containers for testing:

```bash
# Run all testcontainer tests
uv run pytest tests/test_limit_redis_testcontainers.py -v

# Run specific testcontainer test
uv run pytest tests/test_limit_redis_testcontainers.py::TestRedisLimitBackendWithContainers::test_container_health_check -v

# Skip container tests if Docker unavailable
uv run pytest -m "not containers"
```

### Option 2: Using Local Redis

Start a Redis instance manually and run traditional tests:

```bash
# Start Redis container
docker run -d -p 6379:6379 --name redis-test redis:7-alpine

# Run Redis tests
uv run pytest tests/test_limit_redis.py -v -m redis

# Stop Redis container
docker stop redis-test && docker rm redis-test
```

### Option 3: Skip Redis Tests

Skip all Redis-related tests:

```bash
# Skip both traditional and container Redis tests
uv run pytest -m "not redis and not containers"
```

## Test Markers

Configure test execution using pytest markers:

- `redis` - Tests requiring Redis (traditional approach)
- `containers` - Tests requiring Docker containers  
- `asyncio` - Async tests (automatically applied)

## Environment Setup

### Dependencies

The following dependencies are automatically installed:

```toml
[tool.uv]
dev-dependencies = [
    "testcontainers[redis]>=4.12.0",
    "redis>=5.0.0",
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
]
```

### Docker Requirements

For testcontainers to work:

1. Docker must be installed and running
2. Docker daemon must be accessible
3. User must have Docker permissions

Check Docker availability:
```bash
docker info
docker run hello-world
```

## Test Coverage

Both test suites cover:

### Core Functionality
- ✅ Basic usage increment/retrieval
- ✅ Multi-user isolation  
- ✅ Time window isolation
- ✅ Cumulative usage tracking
- ✅ Rate limiting functionality
- ✅ Detailed usage information
- ✅ Usage reset operations
- ✅ Key expiration behavior
- ✅ Health checks

### Advanced Features
- ✅ Concurrent operations
- ✅ Large usage values
- ✅ Negative values (credits/refunds)
- ✅ Custom key prefixes
- ✅ TTL calculations
- ✅ Error handling
- ✅ Middleware integration

### Error Scenarios
- ✅ Connection failures (fail-open behavior)
- ✅ Invalid Redis URLs
- ✅ Redis unavailable scenarios
- ✅ Docker unavailable scenarios

## Configuration

### Redis Backend Configuration

```python
backend = RedisLimitBackend(
    redis_url="redis://localhost:6379/0",
    key_prefix="llm_limit:",
    budget_key_ttl=None,  # Use window-based TTL
    rate_key_ttl=120      # 2 minutes cleanup
)
```

### Test-Specific Configuration

```python
# Test configuration uses separate DB and prefix
backend = RedisLimitBackend(
    redis_url="redis://localhost:6379/15",  # DB 15 for tests
    key_prefix="test_llm_limit:",
    rate_key_ttl=10  # Short TTL for tests
)
```

## Demo Script

Run the interactive demo to verify setup:

```bash
uv run python test_containers_demo.py
```

This script:
1. Checks Docker availability
2. Attempts testcontainers demo
3. Falls back to local Redis if needed
4. Provides setup instructions if both fail

## Troubleshooting

### Docker Issues

**Error**: `Error while fetching server API version`
**Solution**: 
- Ensure Docker Desktop is running
- Check Docker permissions: `docker ps`
- Try restarting Docker Desktop

**Error**: `Connection refused`
**Solution**:
- Verify Docker is accessible: `docker info`
- Check if another process is using Docker socket

### Redis Connection Issues

**Error**: `Redis not available`
**Solution**:
- Start Redis: `docker run -p 6379:6379 redis:7-alpine`
- Check Redis is accessible: `redis-cli ping`
- Verify port 6379 is not in use

### Test Skipping

Tests are automatically skipped when:
- Redis backend module not available
- Docker not available (for testcontainers)
- Redis connection fails (for traditional tests)

This ensures tests don't fail in CI/CD environments without proper setup.

## CI/CD Integration

For continuous integration:

```yaml
# GitHub Actions example
- name: Start Redis
  run: docker run -d -p 6379:6379 redis:7-alpine

- name: Run Redis tests
  run: uv run pytest tests/test_limit_redis.py -v -m redis

# Or with testcontainers (requires Docker-in-Docker)
- name: Run container tests  
  run: uv run pytest tests/test_limit_redis_testcontainers.py -v
```

## Performance Considerations

The testcontainers approach:
- **Pros**: Isolated, reproducible, no setup required
- **Cons**: Slower startup (~2-5 seconds per test session)

The local Redis approach:
- **Pros**: Fast execution, shared instance
- **Cons**: Requires manual setup, potential state conflicts

For development, use testcontainers. For CI/CD, consider local Redis for speed.