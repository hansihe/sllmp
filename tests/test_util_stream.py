"""
Tests for streaming utility classes in util.stream module.

Tests cover:
1. DeltaCollector functionality
2. MultiChoiceDeltaCollector functionality
3. Content accumulation
4. Function and tool call handling
5. Completion status tracking
"""

import pytest
from unittest.mock import Mock
from typing import Dict, Any

from sllmp.util.stream import DeltaCollector, MultiChoiceDeltaCollector


def create_mock_chunk_choice(
    index: int = 0,
    delta_content: str = None,
    delta_role: str = None,
    function_call: Dict[str, Any] = None,
    tool_calls: list = None,
    finish_reason: str = None,
):
    """Create a mock ChunkChoice for testing."""
    choice = Mock()
    choice.index = index
    choice.finish_reason = finish_reason

    delta = Mock()
    delta.content = delta_content
    delta.role = delta_role
    delta.function_call = None
    delta.tool_calls = None

    if function_call:
        func_call = Mock()
        func_call.name = function_call.get("name")
        func_call.arguments = function_call.get("arguments")
        delta.function_call = func_call

    if tool_calls:
        delta_tool_calls = []
        for tool_call in tool_calls:
            tc = Mock()
            tc.index = tool_call["index"]
            tc.id = tool_call.get("id")
            tc.type = tool_call.get("type")

            if "function" in tool_call:
                func = Mock()
                func.name = tool_call["function"].get("name")
                func.arguments = tool_call["function"].get("arguments")
                tc.function = func
            else:
                tc.function = None

            delta_tool_calls.append(tc)
        delta.tool_calls = delta_tool_calls

    choice.delta = delta
    return choice


class TestDeltaCollector:
    """Test DeltaCollector functionality."""

    def test_empty_collector(self):
        collector = DeltaCollector()

        assert collector.get_content() == ""
        assert collector.get_role() is None
        assert collector.get_function_call() is None
        assert collector.get_tool_calls() == []
        assert collector.get_finish_reason() is None
        assert not collector.is_complete()

    def test_content_accumulation(self):
        collector = DeltaCollector()

        # Add content chunks
        collector.accumulate(create_mock_chunk_choice(delta_content="Hello"))
        assert collector.get_content() == "Hello"

        collector.accumulate(create_mock_chunk_choice(delta_content=" world"))
        assert collector.get_content() == "Hello world"

        collector.accumulate(create_mock_chunk_choice(delta_content="!"))
        assert collector.get_content() == "Hello world!"

    def test_role_setting(self):
        collector = DeltaCollector()

        # Role should be set from first chunk that has it
        collector.accumulate(create_mock_chunk_choice(delta_role="assistant"))
        assert collector.get_role() == "assistant"

        # Subsequent role values should be ignored
        collector.accumulate(create_mock_chunk_choice(delta_role="user"))
        assert collector.get_role() == "assistant"

    def test_function_call_accumulation(self):
        collector = DeltaCollector()

        # Add function call name
        collector.accumulate(
            create_mock_chunk_choice(function_call={"name": "get_weather"})
        )

        result = collector.get_function_call()
        assert result == {"name": "get_weather"}

        # Add function arguments in chunks
        collector.accumulate(
            create_mock_chunk_choice(function_call={"arguments": '{"location": "'})
        )
        collector.accumulate(
            create_mock_chunk_choice(function_call={"arguments": 'San Francisco"}'})
        )

        result = collector.get_function_call()
        assert result == {
            "name": "get_weather",
            "arguments": '{"location": "San Francisco"}',
        }

    def test_tool_calls_accumulation(self):
        collector = DeltaCollector()

        # Add first tool call
        collector.accumulate(
            create_mock_chunk_choice(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather"},
                    }
                ]
            )
        )

        # Add arguments to first tool call
        collector.accumulate(
            create_mock_chunk_choice(
                tool_calls=[
                    {"index": 0, "function": {"arguments": '{"location": "NYC"}'}}
                ]
            )
        )

        # Add second tool call
        collector.accumulate(
            create_mock_chunk_choice(
                tool_calls=[
                    {
                        "index": 1,
                        "id": "call_456",
                        "type": "function",
                        "function": {"name": "get_time", "arguments": "{}"},
                    }
                ]
            )
        )

        result = collector.get_tool_calls()
        assert len(result) == 2

        # Check first tool call
        assert result[0].id == "call_123"
        assert result[0].type == "function"
        assert result[0].function.name == "get_weather"
        assert result[0].function.arguments == '{"location": "NYC"}'

        # Check second tool call
        assert result[1].id == "call_456"
        assert result[1].type == "function"
        assert result[1].function.name == "get_time"
        assert result[1].function.arguments == "{}"

    def test_finish_reason(self):
        collector = DeltaCollector()

        assert not collector.is_complete()

        collector.accumulate(create_mock_chunk_choice(finish_reason="stop"))
        assert collector.is_complete()
        assert collector.get_finish_reason() == "stop"

    def test_empty_delta_handling(self):
        collector = DeltaCollector()

        # Create choice with no delta
        choice = Mock()
        choice.delta = None
        choice.finish_reason = None

        collector.accumulate(choice)
        assert collector.get_content() == ""

    def test_fresh_collector_state(self):
        """Test that a fresh collector starts in the expected state."""
        collector = DeltaCollector()

        # Add some data
        collector.accumulate(
            create_mock_chunk_choice(
                delta_content="Hello", delta_role="assistant", finish_reason="stop"
            )
        )

        assert collector.get_content() == "Hello"
        assert collector.get_role() == "assistant"
        assert collector.is_complete()

        # Create a fresh collector to verify initial state
        fresh_collector = DeltaCollector()
        assert fresh_collector.get_content() == ""
        assert fresh_collector.get_role() is None
        assert not fresh_collector.is_complete()


class TestMultiChoiceDeltaCollector:
    """Test MultiChoiceDeltaCollector functionality."""

    def test_empty_multi_collector(self):
        collector = MultiChoiceDeltaCollector()

        assert collector.get_choice_indices() == []
        assert collector.get_all_content() == {}
        assert collector.get_content(0) == ""
        assert collector.get_collector(0) is None
        # Empty collector with no choices should be considered complete (vacuously true)
        assert collector.is_complete()
        assert collector.get_completed_indices() == []

    def test_single_choice_accumulation(self):
        collector = MultiChoiceDeltaCollector()

        collector.accumulate(
            create_mock_chunk_choice(
                index=0, delta_content="Hello", delta_role="assistant"
            )
        )

        assert collector.get_choice_indices() == [0]
        assert collector.get_content(0) == "Hello"
        assert collector.get_collector(0).get_role() == "assistant"

    def test_multiple_choices_interleaved(self):
        collector = MultiChoiceDeltaCollector()

        # Simulate interleaved chunks from multiple choices
        collector.accumulate(create_mock_chunk_choice(index=0, delta_content="First"))
        collector.accumulate(create_mock_chunk_choice(index=1, delta_content="Second"))
        collector.accumulate(
            create_mock_chunk_choice(index=0, delta_content=" response")
        )
        collector.accumulate(
            create_mock_chunk_choice(index=1, delta_content=" alternative")
        )

        assert set(collector.get_choice_indices()) == {0, 1}
        assert collector.get_content(0) == "First response"
        assert collector.get_content(1) == "Second alternative"

        all_content = collector.get_all_content()
        assert all_content == {0: "First response", 1: "Second alternative"}

    def test_completion_tracking(self):
        collector = MultiChoiceDeltaCollector()

        # Add content to both choices
        collector.accumulate(
            create_mock_chunk_choice(index=0, delta_content="Choice 0")
        )
        collector.accumulate(
            create_mock_chunk_choice(index=1, delta_content="Choice 1")
        )

        # Neither should be complete
        assert not collector.is_complete()
        assert not collector.is_complete(0)
        assert not collector.is_complete(1)
        assert collector.get_completed_indices() == []

        # Complete choice 0
        collector.accumulate(create_mock_chunk_choice(index=0, finish_reason="stop"))

        assert not collector.is_complete()  # Not all complete
        assert collector.is_complete(0)
        assert not collector.is_complete(1)
        assert collector.get_completed_indices() == [0]

        # Complete choice 1
        collector.accumulate(create_mock_chunk_choice(index=1, finish_reason="stop"))

        assert collector.is_complete()  # All complete
        assert collector.is_complete(0)
        assert collector.is_complete(1)
        assert set(collector.get_completed_indices()) == {0, 1}

    def test_get_nonexistent_choice(self):
        collector = MultiChoiceDeltaCollector()

        # Should handle requests for non-existent choices gracefully
        assert collector.get_content(5) == ""
        assert collector.get_collector(5) is None
        assert not collector.is_complete(5)

    def test_fresh_multi_collector_state(self):
        """Test that a fresh multi-collector starts in the expected state."""
        collector = MultiChoiceDeltaCollector()

        # Add data to multiple choices
        collector.accumulate(
            create_mock_chunk_choice(
                index=0, delta_content="Choice 0", finish_reason="stop"
            )
        )
        collector.accumulate(
            create_mock_chunk_choice(
                index=1, delta_content="Choice 1", finish_reason="stop"
            )
        )

        assert len(collector.get_choice_indices()) == 2
        assert collector.is_complete()

        # Create a fresh collector to verify initial state
        fresh_collector = MultiChoiceDeltaCollector()
        assert fresh_collector.get_choice_indices() == []
        assert fresh_collector.get_all_content() == {}
        assert fresh_collector.is_complete()  # Empty is complete


class TestIntegration:
    """Integration tests for both collectors."""

    def test_realistic_streaming_scenario(self):
        """Test a realistic multi-choice streaming scenario."""
        collector = MultiChoiceDeltaCollector()

        # Simulate realistic streaming chunks
        chunks = [
            # Initial role chunks
            create_mock_chunk_choice(index=0, delta_role="assistant"),
            create_mock_chunk_choice(index=1, delta_role="assistant"),
            # Interleaved content
            create_mock_chunk_choice(index=0, delta_content="The weather"),
            create_mock_chunk_choice(index=1, delta_content="Today is"),
            create_mock_chunk_choice(index=0, delta_content=" today is sunny"),
            create_mock_chunk_choice(index=1, delta_content=" a beautiful day"),
            create_mock_chunk_choice(index=0, delta_content=" and warm."),
            create_mock_chunk_choice(index=1, delta_content=" for a walk."),
            # Completion
            create_mock_chunk_choice(index=0, finish_reason="stop"),
            create_mock_chunk_choice(index=1, finish_reason="stop"),
        ]

        for chunk in chunks:
            collector.accumulate(chunk)

        # Verify results
        assert collector.is_complete()
        assert set(collector.get_choice_indices()) == {0, 1}

        content = collector.get_all_content()
        assert content[0] == "The weather today is sunny and warm."
        assert content[1] == "Today is a beautiful day for a walk."

        # Verify individual collectors have correct roles
        assert collector.get_collector(0).get_role() == "assistant"
        assert collector.get_collector(1).get_role() == "assistant"

    def test_function_calling_scenario(self):
        """Test function calling with DeltaCollector."""
        collector = DeltaCollector()

        chunks = [
            create_mock_chunk_choice(delta_role="assistant"),
            create_mock_chunk_choice(function_call={"name": "search_web"}),
            create_mock_chunk_choice(function_call={"arguments": '{"query": "'}),
            create_mock_chunk_choice(function_call={"arguments": "weather today"}),
            create_mock_chunk_choice(function_call={"arguments": '"}'}),
            create_mock_chunk_choice(finish_reason="function_call"),
        ]

        for chunk in chunks:
            collector.accumulate(chunk)

        assert collector.is_complete()
        assert collector.get_role() == "assistant"
        assert collector.get_finish_reason() == "function_call"

        func_call = collector.get_function_call()
        assert func_call["name"] == "search_web"
        assert func_call["arguments"] == '{"query": "weather today"}'


class TestChoiceReconstruction:
    """Test Choice object reconstruction from accumulated deltas."""

    def test_delta_collector_to_choice(self):
        """Test converting DeltaCollector to Choice object."""
        collector = DeltaCollector()

        # Accumulate content
        collector.accumulate(
            create_mock_chunk_choice(
                delta_role="assistant", delta_content="Hello world"
            )
        )
        collector.accumulate(
            create_mock_chunk_choice(delta_content="!", finish_reason="stop")
        )

        # Convert to Choice
        choice = collector.to_choice(index=0)

        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content == "Hello world!"
        assert choice.finish_reason == "stop"

    def test_delta_collector_to_choice_with_tool_calls(self):
        """Test converting DeltaCollector with tool calls to Choice."""
        collector = DeltaCollector()

        chunks = [
            create_mock_chunk_choice(delta_role="assistant"),
            create_mock_chunk_choice(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather"},
                    }
                ]
            ),
            create_mock_chunk_choice(
                tool_calls=[{"index": 0, "function": {"arguments": '{"city": "NYC"}'}}]
            ),
            create_mock_chunk_choice(finish_reason="tool_calls"),
        ]

        for chunk in chunks:
            collector.accumulate(chunk)

        choice = collector.to_choice(index=1)

        assert choice.index == 1
        assert choice.message.role == "assistant"
        assert choice.message.content is None
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) == 1
        assert choice.finish_reason == "tool_calls"

    def test_multi_collector_to_choice(self):
        """Test converting specific choice from MultiChoiceDeltaCollector."""
        collector = MultiChoiceDeltaCollector()

        # Add content to choice 0
        collector.accumulate(
            create_mock_chunk_choice(
                index=0, delta_role="assistant", delta_content="First response"
            )
        )
        collector.accumulate(create_mock_chunk_choice(index=0, finish_reason="stop"))

        # Add content to choice 1
        collector.accumulate(
            create_mock_chunk_choice(
                index=1, delta_role="assistant", delta_content="Second response"
            )
        )
        collector.accumulate(create_mock_chunk_choice(index=1, finish_reason="stop"))

        # Test individual choice conversion
        choice_0 = collector.to_choice(0)
        choice_1 = collector.to_choice(1)

        assert choice_0.index == 0
        assert choice_0.message.content == "First response"
        assert choice_1.index == 1
        assert choice_1.message.content == "Second response"

        # Test non-existent choice
        assert collector.to_choice(2) is None

    def test_multi_collector_to_choices(self):
        """Test converting all choices from MultiChoiceDeltaCollector."""
        collector = MultiChoiceDeltaCollector()

        # Add content to multiple choices (out of order indices)
        collector.accumulate(
            create_mock_chunk_choice(
                index=2,
                delta_role="assistant",
                delta_content="Third response",
                finish_reason="stop",
            )
        )
        collector.accumulate(
            create_mock_chunk_choice(
                index=0,
                delta_role="assistant",
                delta_content="First response",
                finish_reason="stop",
            )
        )
        collector.accumulate(
            create_mock_chunk_choice(
                index=1,
                delta_role="assistant",
                delta_content="Second response",
                finish_reason="stop",
            )
        )

        # Convert all to choices
        choices = collector.to_choices()

        # Should be sorted by index
        assert len(choices) == 3
        assert choices[0].index == 0
        assert choices[0].message.content == "First response"
        assert choices[1].index == 1
        assert choices[1].message.content == "Second response"
        assert choices[2].index == 2
        assert choices[2].message.content == "Third response"

    def test_empty_collectors_to_choice(self):
        """Test converting collectors to Choice objects."""
        # Collector with minimal data (role and finish_reason required)
        collector = DeltaCollector()
        collector.accumulate(
            create_mock_chunk_choice(delta_role="assistant", finish_reason="stop")
        )

        choice = collector.to_choice()

        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content is None
        assert choice.finish_reason == "stop"

        # Empty multi collector
        multi_collector = MultiChoiceDeltaCollector()
        choices = multi_collector.to_choices()
        assert choices == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
