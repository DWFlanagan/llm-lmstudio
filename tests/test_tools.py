"""Tests for tool support in llm-lmstudio plugin."""

import pytest
import llm
from unittest.mock import Mock, MagicMock, patch
import json
import sys
import os

# Add the parent directory to the path so we can import llm_lmstudio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llm_lmstudio

# Mock data for testing
MOCK_RAW_MODEL_ID = "test-model-with-tools"
MODEL_ID = MOCK_RAW_MODEL_ID

MOCK_MODELS_LIST = [{
    'id': MOCK_RAW_MODEL_ID,
    'type': 'llm',
    'state': 'loaded',
    'publisher': 'test_publisher',
    'architecture': 'test_arch',
    'quantization': 'test_quant',
    'max_context_length': 4096
}]
MOCK_API_PATH = "/api/v0"
MOCK_FETCH_MODELS_RETURN_VALUE = (MOCK_MODELS_LIST, MOCK_API_PATH)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = llm_lmstudio.LMStudioModel(
        model_id=MODEL_ID,
        base_url="http://localhost:1234",
        raw_id=MOCK_RAW_MODEL_ID,
        api_path_prefix=MOCK_API_PATH,
        supports_images=False,
        metadata={}
    )
    return model


@pytest.fixture
def mock_async_model():
    """Create a mock async model for testing."""
    model = llm_lmstudio.LMStudioAsyncModel(
        model_id=MODEL_ID,
        base_url="http://localhost:1234",
        raw_id=MOCK_RAW_MODEL_ID,
        api_path_prefix=MOCK_API_PATH,
        supports_images=False,
        metadata={}
    )
    return model


@pytest.fixture
def sample_tool():
    """Sample tool for testing."""
    return llm.Tool(
        name="get_weather",
        description="Get current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    )


@pytest.fixture
def sample_tools(sample_tool):
    """Multiple sample tools for testing."""
    tool2 = llm.Tool(
        name="calculate",
        description="Perform mathematical calculations",
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    )
    return [sample_tool, tool2]


@pytest.fixture
def mock_prompt_with_tools(sample_tools):
    """Create a mock prompt with tools."""
    prompt = MagicMock()
    prompt.prompt = "What's the weather like and calculate 2+2?"
    prompt.system = None
    prompt.attachments = []
    prompt.options = None
    prompt.schema = None
    prompt.tools = sample_tools
    return prompt


@pytest.fixture
def mock_prompt_without_tools():
    """Create a mock prompt without tools."""
    prompt = MagicMock()
    prompt.prompt = "Hello world"
    prompt.system = None
    prompt.attachments = []
    prompt.options = None
    prompt.schema = None
    prompt.tools = []
    return prompt


def test_build_tools_with_single_tool(mock_model, sample_tool):
    """Test building tools payload with a single tool."""
    prompt = MagicMock()
    prompt.tools = [sample_tool]
    
    tools = mock_model._build_tools(prompt)
    
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"
    assert tools[0]["function"]["description"] == "Get current weather for a location"
    assert tools[0]["function"]["parameters"]["type"] == "object"
    assert "location" in tools[0]["function"]["parameters"]["properties"]
    assert tools[0]["function"]["parameters"]["required"] == ["location"]


def test_build_tools_with_multiple_tools(mock_model, sample_tools):
    """Test building tools payload with multiple tools."""
    prompt = MagicMock()
    prompt.tools = sample_tools
    
    tools = mock_model._build_tools(prompt)
    
    assert len(tools) == 2
    tool_names = [tool["function"]["name"] for tool in tools]
    assert "get_weather" in tool_names
    assert "calculate" in tool_names


def test_build_tools_with_no_tools(mock_model):
    """Test building tools payload when no tools are provided."""
    prompt = MagicMock()
    prompt.tools = []
    
    tools = mock_model._build_tools(prompt)
    
    assert tools == []


def test_build_tools_with_none_tools(mock_model):
    """Test building tools payload when tools is None."""
    prompt = MagicMock()
    prompt.tools = None
    
    tools = mock_model._build_tools(prompt)
    
    assert tools == []


def test_build_tools_tool_without_description(mock_model):
    """Test building tools payload with a tool that has no description."""
    tool_no_desc = llm.Tool(
        name="simple_tool",
        input_schema={"type": "object", "properties": {}}
    )
    
    prompt = MagicMock()
    prompt.tools = [tool_no_desc]
    
    tools = mock_model._build_tools(prompt)
    
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "simple_tool"
    # Should not have description field if None
    assert "description" not in tools[0]["function"]


def test_async_build_tools_with_single_tool(mock_async_model, sample_tool):
    """Test async model building tools payload with a single tool."""
    prompt = MagicMock()
    prompt.tools = [sample_tool]
    
    tools = mock_async_model._build_tools(prompt)
    
    assert len(tools) == 1
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["name"] == "get_weather"
    assert tools[0]["function"]["description"] == "Get current weather for a location"


def test_payload_building_with_tools(mock_model, mock_prompt_with_tools):
    """Test that the payload building logic includes tools correctly."""
    # This test focuses on the _build_tools method and payload structure
    # rather than the full HTTP mocking complexity
    
    tools = mock_model._build_tools(mock_prompt_with_tools)
    
    # Verify tools are built correctly
    assert len(tools) == 2
    assert tools[0]["type"] == "function"
    assert tools[1]["type"] == "function"
    
    tool_names = [tool["function"]["name"] for tool in tools]
    assert "get_weather" in tool_names
    assert "calculate" in tool_names
    
    # Verify the structure of one tool in detail
    weather_tool = next(tool for tool in tools if tool["function"]["name"] == "get_weather")
    assert weather_tool["function"]["description"] == "Get current weather for a location"
    assert "location" in weather_tool["function"]["parameters"]["properties"]
    assert weather_tool["function"]["parameters"]["required"] == ["location"]


def test_payload_building_without_tools(mock_model, mock_prompt_without_tools):
    """Test that the payload building logic works correctly without tools."""
    tools = mock_model._build_tools(mock_prompt_without_tools)
    assert tools == []


@patch('llm_lmstudio.LMStudioAsyncModel._is_model_loaded', return_value=True)
@pytest.mark.asyncio
async def test_async_execute_includes_tools_in_payload(mock_is_loaded, mock_async_model, mock_prompt_with_tools):
    """Test that async execute method includes tools in the API payload."""
    mock_response = MagicMock()
    
    # Mock the httpx.AsyncClient.post call properly
    with patch('llm_lmstudio.httpx.AsyncClient') as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        
        # Create a mock response
        mock_post_response = MagicMock()
        mock_post_response.status_code = 200
        mock_post_response.json.return_value = {
            "choices": [{
                "message": {"content": "Test response"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        mock_post_response.text = '{"choices": [{"message": {"content": "Test response"}, "finish_reason": "stop"}]}'
        
        # Make post return an awaitable
        async def mock_post(*args, **kwargs):
            # Store the call arguments for verification
            mock_post.call_args = (args, kwargs)
            return mock_post_response
        
        mock_post = MagicMock(side_effect=mock_post)
        mock_client.post = mock_post
        
        # Execute the async method
        result = mock_async_model.execute(mock_prompt_with_tools, stream=False, response=mock_response)
        
        # Consume the async generator to trigger the API call
        content = []
        async for chunk in result:
            content.append(chunk)
        
        # Verify that the API call was made with tools
        assert mock_post.call_count == 1
        call_args = mock_post.call_args
        payload = call_args[1]['json']
        
        assert "tools" in payload
        assert len(payload["tools"]) == 2
        assert payload["tool_choice"] == "auto"


def test_build_tools_preserves_input_schema_structure(mock_model):
    """Test that _build_tools preserves the exact input schema structure."""
    # Create a tool with a complex input schema
    complex_tool = llm.Tool(
        name="complex_tool",
        description="A tool with complex input schema",
        input_schema={
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"}
                    }
                },
                "array_field": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["nested"],
            "definitions": {
                "custom_type": {
                    "type": "string",
                    "enum": ["option1", "option2"]
                }
            }
        }
    )
    
    prompt = MagicMock()
    prompt.tools = [complex_tool]
    
    tools = mock_model._build_tools(prompt)
    
    assert len(tools) == 1
    function_params = tools[0]["function"]["parameters"]
    
    # Check that the complex schema is preserved exactly
    assert function_params["type"] == "object"
    assert "nested" in function_params["properties"]
    assert "array_field" in function_params["properties"]
    assert function_params["required"] == ["nested"]
    assert "definitions" in function_params
    assert function_params["definitions"]["custom_type"]["enum"] == ["option1", "option2"]