"""
Content safety and validation middleware.
"""

import re
from typing import List, Optional

from ..pipeline import Middleware, RequestContext


class ContentGuardrailMiddleware(Middleware):
    """
    Real-time content monitoring middleware.

    Monitors streaming responses for policy violations and can halt
    the stream immediately when violations are detected.
    """

    def __init__(self, policies: Optional[List[str]] = None, check_interval: int = 3, **config):
        super().__init__(**config)
        self.policies = policies or ["inappropriate", "pii"]
        self.check_interval = check_interval

    async def on_response_update(self, ctx: RequestContext, accumulated_content: str) -> RequestContext:
        """Check accumulated content for policy violations during streaming."""

        violations = self._check_policies(accumulated_content)
        if violations:
            # Log the violation
            ctx.metadata['guardrail_violation'] = {
                'violations': violations,
                'content_length': len(accumulated_content),
                'chunk_count': ctx.chunk_count
            }

            # Halt the stream
            self.halt_with_error(
                ctx,
                f"Content policy violation detected: {', '.join(violations)}",
                "content_policy_error"
            )

        return ctx

    async def on_response_complete(self, ctx: RequestContext, final_content: str) -> RequestContext:
        """Final check for non-streaming responses or end-of-stream validation."""

        violations = self._check_policies(final_content)
        if violations:
            ctx.metadata['guardrail_violation'] = {
                'violations': violations,
                'content_length': len(final_content),
                'final_check': True
            }

            self.halt_with_error(
                ctx,
                f"Final content check failed: {', '.join(violations)}",
                "content_policy_error"
            )

        return ctx

    def _check_policies(self, content: str) -> List[str]:
        """
        Check content against configured policies.

        TODO: Implement more sophisticated policy checking:
        - Integration with external content moderation APIs
        - ML-based detection models
        - Configurable policy rules
        - PII detection with regex/NER models
        """
        violations = []
        content_lower = content.lower()

        if "inappropriate" in self.policies:
            # Simple keyword-based detection (placeholder)
            inappropriate_patterns = [
                "inappropriate content",
                "explicit material",
                "harmful content"
            ]
            if any(pattern in content_lower for pattern in inappropriate_patterns):
                violations.append("inappropriate_content")

        if "pii" in self.policies:
            # Simple PII detection (placeholder)
            # TODO: Implement proper PII detection with regex patterns or NER
            pii_patterns = [
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN pattern
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
            ]
            if any(re.search(pattern, content) for pattern in pii_patterns):
                violations.append("pii_detected")

        return violations


class ResponseValidatorMiddleware(Middleware):
    """
    Response quality validation middleware.

    Validates complete responses for quality, completeness, and adherence
    to requirements. Runs after the full response is available.
    """

    def __init__(self, min_quality_score: float = 0.7, min_length: int = 10, **config):
        super().__init__(**config)
        self.min_quality_score = min_quality_score
        self.min_length = min_length

    async def on_response_complete(self, ctx: RequestContext, final_content: str) -> RequestContext:
        """Validate the complete response for quality and completeness."""

        validation_results = {
            'content_length': len(final_content),
            'quality_score': 0.0,
            'issues': []
        }

        # Basic length check
        if len(final_content) < self.min_length:
            validation_results['issues'].append('response_too_short')

        # TODO: Implement sophisticated quality scoring
        quality_score = self._assess_response_quality(final_content)
        validation_results['quality_score'] = quality_score

        if quality_score < self.min_quality_score:
            validation_results['issues'].append('quality_too_low')

        # Store validation results
        ctx.metadata['response_validation'] = validation_results

        # If validation fails, we could either halt or mark for retry
        if validation_results['issues']:
            # For now, just log the issues but don't halt
            # TODO: Make this configurable - halt vs retry vs log
            ctx.metadata['validation_failed'] = True

        return ctx

    def _assess_response_quality(self, content: str) -> float:
        """
        Assess response quality with a score from 0.0 to 1.0.

        TODO: Implement sophisticated quality assessment:
        - Coherence and relevance scoring
        - Grammar and language quality
        - Factual accuracy checking
        - Response completeness
        - Integration with external quality models
        """
        # Placeholder implementation - simple heuristics
        score = 0.5  # Base score

        # Length bonus
        if 50 <= len(content) <= 1000:
            score += 0.2

        # Sentence structure bonus
        sentence_count = len([s for s in content.split('.') if s.strip()])
        if 2 <= sentence_count <= 10:
            score += 0.2

        # Avoid repetition penalty
        words = content.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            uniqueness = len(unique_words) / len(words)
            if uniqueness > 0.7:
                score += 0.1

        return min(1.0, score)
