# Changelog

All notable changes to SLLMP will be documented in this file.

## [Unreleased]

### Added

- **Custom Provider Configuration**: Define provider aliases with custom options (api_base, organization, etc.) via `ProviderConfig` in YAML config. Allows mapping custom provider names (e.g., `my-openai`) to underlying providers with custom settings
- **ProviderBadRequestError**: New error type for 4xx responses from LLM providers
- **ContextLengthExceededError**: New error type for when input exceeds model's maximum context length
- **ClientRateLimitError**: Separate error type for middleware-enforced rate/budget limits (distinct from provider rate limits)
- **CLAUDE.md**: Project documentation and development guidelines for AI assistants
- **Limit Middleware Integration Tests**: Additional test coverage for rate/budget limiting

### Changed

- **any-llm-sdk upgraded to 1.6.2**: Major version upgrade from 0.16.0
- **Error Classification Refactored**: Now uses typed any-llm exceptions instead of string pattern matching for more reliable error handling
- **RateLimitError renamed to ProviderRateLimitError**: Clarifies that this error originates from the LLM provider, not our middleware
- **API Rename**: `create_from_config` renamed to `from_config` for cleaner API
- **CompletionParams**: Added `model_id` parameter to constructor

### Fixed

- **Limit Middleware Error Handling**: Catches exceptions when checking budget usage and allows requests through on backend failures (fail-open behavior)
- **Code Style**: Consistent double quote usage throughout limit middleware
