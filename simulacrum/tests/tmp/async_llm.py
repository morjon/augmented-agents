#!/usr/bin/env python3
import asyncio
import pytest
from unittest.mock import patch, MagicMock

from models.llama_async import (
    generate,
    Llama,
    LLMResult,
    Generation,
)


# Test generate function
@pytest.mark.asyncio
async def test_generate():
    async for token in generate(prompt="test prompt"):
        assert isinstance(token, str)


# Test Llama class
def test_llama():
    llama = Llama()
    prompt = "test prompt"
    expected_result = LLMResult(generations=[[Generation(text="test response")]])

    # Test _agenerate method
    with patch("models.llama_async.generate") as mock_generate:
        mock_generate.return_value = generate(prompt=prompt)
        result = asyncio.run(llama._agenerate([prompt]))
        assert result == expected_result

    # Test _generate method
    with patch.object(asyncio, "new_event_loop") as mock_new_event_loop:
        mock_new_loop = MagicMock()
        mock_new_event_loop.return_value = mock_new_loop
        mock_new_loop.run_until_complete.return_value = expected_result
        result = llama._generate([prompt])
        assert result == expected_result

    # Test _call method
    with patch.object(Llama, "_generate") as mock_generate:
        mock_generate.return_value = expected_result
        result = llama._call(prompt)
        assert result == "test response"

    # Test _identifying_params property
    assert llama._identifying_params == {}

    # Test _llm_type property
    assert llama._llm_type == "llama"
