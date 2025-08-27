from typing import Dict, Optional
from any_llm.types.completion import CompletionUsage
from dataclasses import dataclass, field

@dataclass
class Pricing:
    input: float = field()
    cached_input: Optional[float] = field()
    output: float = field()

DEFAULT_PRICING = Pricing(
    input=2.0,
    cached_input=1.0,
    output=10.0,
)

# Per 1M tokens
PRICING_TABLE: Dict[str, Pricing] = {
    'openai:gpt-5': Pricing(input=1.25, cached_input=0.125, output=10.0),
    'openai:gpt-5-mini': Pricing(input=0.25, cached_input=0.025, output=2.0),
    'openai:gpt-5-nano': Pricing(input=0.05, cached_input=0.005, output=0.40),
    'openai:gpt-4o': Pricing(input=2.5, cached_input=1.25, output=10.0),
    'openai:gpt-4o-mini': Pricing(input=0.15, cached_input=0.075, output=0.6),
}

def calculate_usage_pricing(model_id: str, usage: CompletionUsage):
    pricing = PRICING_TABLE.get(model_id, DEFAULT_PRICING)

    # Prompt cost calculation
    prompt_cost = 0
    remaining_prompt_tokens = usage.prompt_tokens

    prompt_details = usage.prompt_tokens_details
    if prompt_details is not None:
        if prompt_details.cached_tokens is not None and pricing.cached_input is not None:
            cached_tokens = prompt_details.cached_tokens
            prompt_cost += cached_tokens * pricing.cached_input
            remaining_prompt_tokens -= cached_tokens

    prompt_cost += remaining_prompt_tokens * pricing.input

    # Completion cost calculation
    completion_cost = 0
    remaining_completion_tokens = usage.completion_tokens

    prompt_cost += remaining_completion_tokens * pricing.output

    # Full result
    return prompt_cost + completion_cost
