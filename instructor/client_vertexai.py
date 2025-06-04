from __future__ import annotations

from typing import Any, Union, Optional, get_origin

from vertexai.preview.generative_models import ToolConfig  # type: ignore
import vertexai.generative_models as gm  # type: ignore
from pydantic import BaseModel
import instructor
from instructor.dsl.parallel import get_types_array
import jsonref

GEMINI_ALLOWED_TYPES = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}
ALLOWED_GEMINI_TYPE_VALUES = set(GEMINI_ALLOWED_TYPES.values())

# Remove complex schema simplification in favor of using map_to_gemini_function_schema
# This makes schema handling consistent between from_gemini and from_vertexai clients
# particularly for handling optional fields

def _create_gemini_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    """Create Gemini JSON schema for VertexAI using the same approach as gemini_schema.
    
    This ensures consistent handling of optional fields between from_gemini and from_vertexai clients.
    """
    from instructor.utils import map_to_gemini_function_schema

    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
        raise TypeError(f"Expected concrete pydantic.BaseModel subclass, got {model_cls}")

    # Use OpenAI schema as an intermediate format
    try:
        # Handle both OpenAISchema and regular BaseModel
        if hasattr(model_cls, "openai_schema"):
            openai_schema = model_cls.openai_schema
        else:
            # Create schema in OpenAI format
            schema = model_cls.model_json_schema()
            docstring = getattr(model_cls, "__doc__", "") or ""

            parameters = {
                k: v for k, v in schema.items() if k not in ("title", "description")
            }
            # Set required fields
            parameters["required"] = sorted(
                k for k, v in parameters.get("properties", {}).items() if "default" not in v
            )
            # Set description
            description = schema.get("description") or (
                f"Correctly extracted `{model_cls.__name__}` with all required parameters with correct types"
            )

            openai_schema = {
                "name": schema.get("title", model_cls.__name__),
                "description": description,
                "parameters": parameters,
            }
    except Exception as e:
        raise RuntimeError(f"Failed to create OpenAI schema for {model_cls.__name__}: {e}") from e
 
    # Annotate any properties (or nested properties) with format=date
    def _annotate_date_fields(props: dict[str, Any]):
        for prop_name, prop_schema in props.items():
            # If this field is a date, add the ISO requirement
            if prop_schema.get("format") == "date":
                # Add or overwrite description
                prop_schema["description"] = "IMPORTANT: MUST BE IN ISO FORMAT"
            # Recurse into object properties
            if prop_schema.get("type") == "object" and "properties" in prop_schema:
                _annotate_date_fields(prop_schema["properties"])
            # Handle arrays of objects
            if prop_schema.get("type") == "array" and "items" in prop_schema:
                items = prop_schema["items"]
                if isinstance(items, dict) and items.get("type") == "object" and "properties" in items:
                    _annotate_date_fields(items["properties"])

    def _reassign_all_of(props: dict[str, Any]):
        for prop_name, prop_schema in props.items():
            print(prop_name, prop_schema)
            # If this field is a date, add the ISO requirement

            if isinstance(prop_schema, dict) and "allOf" in prop_schema and isinstance(prop_schema["allOf"], list) and  len(prop_schema["allOf"]) == 1 and "$ref" in prop_schema["allOf"][0]:
            
                prop_schema["$ref"] = prop_schema["allOf"][0]["$ref"]
                del prop_schema["allOf"]

            # Recurse into object properties

            if prop_schema.get("type") == "object" and "properties" in prop_schema:
                _reassign_all_of(prop_schema["properties"])
            # Handle arrays of objects
            if prop_schema.get("type") == "array" and "items" in prop_schema:
                items = prop_schema["items"]
                if isinstance(items, dict) and items.get("type") == "object" and "properties" in items:
                    _reassign_all_of(items["properties"])


    def _remove_unique_items(props: dict[str, Any]):
        for prop_name, prop_schema in list(props.items()):

            if isinstance(prop_schema, dict):
                if "uniqueItems" in prop_schema:
                    del prop_schema["uniqueItems"]
                # Recurse into object properties
                if prop_schema.get("type") == "object" and "properties" in prop_schema:
                    _remove_unique_items(prop_schema["properties"])
                # Handle arrays of objects
                if prop_schema.get("type") == "array" and "items" in prop_schema:
                    items = prop_schema["items"]
                    if isinstance(items, dict) and items.get("type") == "object" and "properties" in items:
                        _remove_unique_items(items["properties"])
                    elif isinstance(items, dict): # Handle cases where items is a dict but not an object with properties
                        _remove_unique_items(items)
                # Handle other nested dictionaries
                _remove_unique_items(prop_schema)
            elif isinstance(prop_schema, list):
                for item in prop_schema:
                    if isinstance(item, dict):
                        _remove_unique_items(item)

    # Invoke the date-field annotator on the schema's properties
    _annotate_date_fields(openai_schema["parameters"].get("properties", {}))

    _reassign_all_of(openai_schema["parameters"].get("properties", {}))

    _remove_unique_items(openai_schema["parameters"].get("properties", {}))

    # Transform to Gemini format using the same utility as gemini_schema
    try:
        gemini_parameters = map_to_gemini_function_schema(openai_schema["parameters"])
        gemini_schema = gemini_parameters
    except Exception as e:
        raise RuntimeError(
            f"Failed to transform schema to Gemini format: {e} with these params: {openai_schema['parameters']}"
        ) from e


    return gemini_schema


def _create_vertexai_tool(
    models: BaseModel | list[BaseModel] | type,
) -> gm.Tool:  # noqa: UP007
    """Creates a tool with function declarations for single model or list of models"""
    import google.generativeai.types as genai_types
    
    # Handle Iterable case first
    if get_origin(models) is not None:
        model_list = list(get_types_array(models))  # type: ignore
    else:
        # Handle both single model and list of models
        model_list = models if isinstance(models, list) else [models]

    declarations = []
    for model in model_list:  # type: ignore
        # Get schema using the same approach as gemini_schema
        try:
            # If model already has gemini_schema property (added by openai_schema decorator)
            if hasattr(model, "gemini_schema"):
                function = model.gemini_schema
            else:
                # Otherwise create it manually with our updated function
                parameters = _create_gemini_json_schema(model)  # type: ignore
                function = genai_types.FunctionDeclaration(
                    name=model.__name__,  # type: ignore
                    description=model.__doc__,  # type: ignore
                    parameters=parameters,
                )
            
            declarations.append(function)  # type: ignore
        except Exception as e:
            raise ValueError(f"Failed to create function declaration for {model.__name__}: {e}")  # type: ignore

    return gm.Tool(function_declarations=declarations)  # type: ignore


def vertexai_message_parser(
    message: dict[str, str | gm.Part | list[str | gm.Part]],
) -> gm.Content:
    if isinstance(message["content"], str):
        return gm.Content(
            role=message["role"],  # type:ignore
            parts=[gm.Part.from_text(message["content"])],
        )
    elif isinstance(message["content"], list):
        parts: list[gm.Part] = []
        for item in message["content"]:
            if isinstance(item, str):
                parts.append(gm.Part.from_text(item))
            elif isinstance(item, gm.Part):
                parts.append(item)
            else:
                raise ValueError(f"Unsupported content type in list: {type(item)}")
        return gm.Content(
            role=message["role"],  # type:ignore
            parts=parts,
        )
    else:
        raise ValueError("Unsupported message content type")


def _vertexai_message_list_parser(
    messages: list[dict[str, str | gm.Part | list[str | gm.Part]]],
) -> list[gm.Content]:
    contents = [
        vertexai_message_parser(message) if isinstance(message, dict) else message
        for message in messages
    ]
    return contents


def vertexai_function_response_parser(
    response: gm.GenerationResponse, exception: Exception
) -> gm.Content:
    return gm.Content(
        parts=[
            gm.Part.from_function_response(
                name=response.candidates[0].content.parts[0].function_call.name,
                response={
                    "content": f"Validation Error found:\n{exception}\nRecall the function correctly, fix the errors"
                },
            )
        ]
    )


def vertexai_process_response(
    _kwargs: dict[str, Any],
    model: Union[BaseModel, list[BaseModel], type],  # noqa: UP007
):
    messages: list[dict[str, str]] = _kwargs.pop("messages")
    contents = _vertexai_message_list_parser(messages)  # type: ignore

    tool = _create_vertexai_tool(models=model)

    tool_config = ToolConfig(
        function_calling_config=ToolConfig.FunctionCallingConfig(
            mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
        )
    )
    return contents, [tool], tool_config


def vertexai_process_json_response(_kwargs: dict[str, Any], model: BaseModel):
    messages: list[dict[str, str]] = _kwargs.pop("messages")
    contents = _vertexai_message_list_parser(messages)  # type: ignore

    config: dict[str, Any] | None = _kwargs.pop("generation_config", None)

    # Use the same approach for schema generation as gemini
    if hasattr(model, "model_json_schema"):
        raw_schema = model.model_json_schema()
        
        # Use our updated _create_gemini_json_schema function
        response_schema = _create_gemini_json_schema(model)
    else:
        # Fallback for older pydantic versions
        raw_schema = model.schema()
        response_schema = _create_gemini_json_schema(model)

    generation_config = gm.GenerationConfig(
        response_mime_type="application/json",
        response_schema=response_schema,
        **(config if config else {}),
    )

    return contents, generation_config


def from_vertexai(
    client: gm.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.VERTEXAI_TOOLS,
    _async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor:
    assert mode in {
        instructor.Mode.VERTEXAI_PARALLEL_TOOLS,
        instructor.Mode.VERTEXAI_TOOLS,
        instructor.Mode.VERTEXAI_JSON,
    }, "Mode must be instructor.Mode.VERTEXAI_TOOLS"

    assert isinstance(
        client, gm.GenerativeModel
    ), "Client must be an instance of vertexai.generative_models.GenerativeModel"

    create = client.generate_content_async if _async else client.generate_content

    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.VERTEXAI,
        mode=mode,
        **kwargs,
    )
