from typing import Optional, List, Dict, Any, Sequence, Literal
from any_llm.types.completion import ChunkChoice, Choice, ChatCompletionMessage
from openai.types.chat import ChatCompletionMessageToolCall, ChatCompletionMessageToolCallUnion

class DeltaCollector:
    """
    Collects and accumulates delta content from streaming chunks.

    This utility helps reconstruct complete messages from streaming responses
    by accumulating delta content and other message properties.
    """

    def __init__(self):
        self.content_parts: List[str] = []
        self.role: Optional[str] = None
        self.function_call: Optional[Dict[str, Any]] = None
        self.tool_calls: List[Dict[str, Any]] = []
        self.finish_reason: Optional[Literal['stop', 'length', 'tool_calls', 'content_filter', 'function_call']] = None

    def accumulate(self, chunk_choice: ChunkChoice) -> None:
        """
        Accumulate delta content from a chunk choice.

        Args:
            chunk_choice: A choice from a streaming chunk containing delta information
        """
        if not chunk_choice.delta:
            return

        delta = chunk_choice.delta

        # Accumulate content
        if delta.content:
            self.content_parts.append(delta.content)

        # Set role (usually only in first chunk)
        if delta.role and not self.role:
            self.role = delta.role

        # Handle function calls if present
        if delta.function_call:
            if not self.function_call:
                self.function_call = {}

            if delta.function_call.name:
                self.function_call['name'] = delta.function_call.name

            if delta.function_call.arguments:
                if 'arguments' not in self.function_call:
                    self.function_call['arguments'] = ''
                self.function_call['arguments'] += delta.function_call.arguments

        # Handle tool calls if present
        if delta.tool_calls:
            for tool_call_delta in delta.tool_calls:
                # Ensure we have enough tool call slots
                while len(self.tool_calls) <= tool_call_delta.index:
                    self.tool_calls.append({})

                tool_call = self.tool_calls[tool_call_delta.index]

                if tool_call_delta.id:
                    tool_call['id'] = tool_call_delta.id

                if tool_call_delta.type:
                    tool_call['type'] = tool_call_delta.type

                if tool_call_delta.function:
                    if 'function' not in tool_call:
                        tool_call['function'] = {}

                    func = tool_call_delta.function
                    if func.name:
                        tool_call['function']['name'] = func.name

                    if func.arguments:
                        if 'arguments' not in tool_call['function']:
                            tool_call['function']['arguments'] = ''
                        tool_call['function']['arguments'] += func.arguments

        # Set finish reason (usually in last chunk)
        if chunk_choice.finish_reason:
            self.finish_reason = chunk_choice.finish_reason

    def get_content(self) -> str:
        """Get the accumulated content as a single string."""
        return ''.join(self.content_parts)

    def get_role(self) -> Optional[str]:
        """Get the message role."""
        return self.role

    def get_function_call(self) -> Optional[Dict[str, Any]]:
        """Get the accumulated function call if any."""
        return self.function_call

    def get_tool_calls(self) -> List[ChatCompletionMessageToolCallUnion]:
        """Get the accumulated tool calls."""
        result = []
        for call_dict in self.tool_calls:
            if not call_dict:  # Skip empty calls
                continue

            # Convert dict to proper tool call object
            tool_call = ChatCompletionMessageToolCall(
                id=call_dict.get('id', ''),
                type=call_dict.get('type', 'function'),
                function=call_dict.get('function', {})
            )
            result.append(tool_call)

        return result

    def get_finish_reason(self) -> Optional[str]:
        """Get the finish reason."""
        return self.finish_reason

    def is_complete(self) -> bool:
        """Check if the collection is complete (has finish reason)."""
        return self.finish_reason is not None

    def to_choice(self, index: int = 0) -> Choice:
        """
        Convert accumulated delta content to a Choice object.

        Args:
            index: The choice index for the reconstructed choice

        Returns:
            A Choice object equivalent to what would be returned in non-streaming response
        """

        assert self.role == "assistant"

        message = ChatCompletionMessage(
            role="assistant",
            content=self.get_content() or None
        )

        # Add tool calls if present
        if self.tool_calls:
            message.tool_calls = self.get_tool_calls()

        assert self.finish_reason is not None

        return Choice(
            index=index,
            message=message,
            finish_reason=self.finish_reason
        )

class MultiChoiceDeltaCollector:
    """
    Collects and accumulates delta content from streaming chunks with multiple choices.

    Handles OpenAI's n>1 parameter where multiple response alternatives are generated
    simultaneously and chunks arrive interleaved by choice index.
    """

    def __init__(self):
        self.collectors: Dict[int, DeltaCollector] = {}

    def accumulate_all(self, choices: Sequence[ChunkChoice]) -> None:
        for choice in choices:
            self.accumulate(choice)

    def accumulate(self, chunk_choice: ChunkChoice) -> None:
        """
        Accumulate delta content from a chunk choice, routing by index.

        Args:
            chunk_choice: A choice from a streaming chunk containing delta information
        """
        index = chunk_choice.index

        if index not in self.collectors:
            self.collectors[index] = DeltaCollector()

        self.collectors[index].accumulate(chunk_choice)

    def get_collector(self, index: int) -> Optional[DeltaCollector]:
        """Get the collector for a specific choice index."""
        return self.collectors.get(index)

    def get_content(self, index: int) -> str:
        """Get the accumulated content for a specific choice index."""
        collector = self.collectors.get(index)
        return collector.get_content() if collector else ""

    def get_all_content(self) -> Dict[int, str]:
        """Get accumulated content for all choice indices."""
        return {index: collector.get_content() for index, collector in self.collectors.items()}

    def get_choice_indices(self) -> List[int]:
        """Get all choice indices that have been encountered."""
        return list(self.collectors.keys())

    def is_complete(self, index: Optional[int] = None) -> bool:
        """
        Check if collection is complete.

        Args:
            index: If provided, check only this choice. If None, check all choices.
        """
        if index is not None:
            collector = self.collectors.get(index)
            return collector.is_complete() if collector else False

        # All choices must be complete
        return all(collector.is_complete() for collector in self.collectors.values())

    def get_completed_indices(self) -> List[int]:
        """Get indices of all completed choices."""
        return [index for index, collector in self.collectors.items() if collector.is_complete()]

    def to_choice(self, index: int) -> Optional[Choice]:
        """
        Convert accumulated delta content for a specific choice to a Choice object.

        Args:
            index: The choice index to convert

        Returns:
            A Choice object for the specified index, or None if index doesn't exist
        """
        collector = self.collectors.get(index)
        if not collector:
            return None

        return collector.to_choice(index)

    def to_choices(self) -> List[Choice]:
        """
        Convert all accumulated delta content to a list of Choice objects.

        Returns:
            A list of Choice objects equivalent to what would be returned in non-streaming response
        """
        choices = []
        for index in sorted(self.collectors.keys()):
            choice = self.to_choice(index)
            if choice:
                choices.append(choice)
        return choices
