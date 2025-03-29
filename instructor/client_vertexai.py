from __future__ import annotations

from typing import Any, Union, get_origin

from vertexai.preview.generative_models import ToolConfig  # type: ignore
import vertexai.generative_models as gm  # type: ignore
from pydantic import BaseModel
import instructor
from instructor.dsl.parallel import get_types_array
import jsonref


import jsonref
import typing
from typing import Any, Union, Optional, get_origin, get_args
from pydantic import BaseModel
from pydantic_core import PydanticUndefined
from enum import Enum as PyEnum
import warnings
import copy

import copy
import warnings
from typing import Any, Optional

import jsonref
from pydantic import BaseModel

GEMINI_ALLOWED_TYPES = {
    "string": "STRING",
    "number": "NUMBER",
    "integer": "INTEGER",
    "boolean": "BOOLEAN",
    "array": "ARRAY",
    "object": "OBJECT",
}
ALLOWED_GEMINI_TYPE_VALUES = set(GEMINI_ALLOWED_TYPES.values())

class UnsupportedSchemaTypeError(TypeError):
    pass

def _simplify_pydantic_schema_for_gemini(schema: dict[str, Any]) -> Optional[dict[str, Any]]:
    if not isinstance(schema, dict):
        return schema

    processed_schema = copy.deepcopy(schema)

    processed_schema.pop("nullable", None)
    processed_schema.pop("default", None)
    processed_schema.pop("title", None)
    processed_schema.pop("$defs", None)

    if "anyOf" in processed_schema:
        any_of_list = processed_schema["anyOf"]
        non_null_schemas = [s for s in any_of_list if not (isinstance(s, dict) and s.get("type") == "null")]
        was_optional = len(non_null_schemas) != len(any_of_list)

        if not non_null_schemas:
             raise UnsupportedSchemaTypeError(f"Schema invalid: 'anyOf' contained only null type(s). Original: {any_of_list}")

        if len(non_null_schemas) == 1:
            simplified_t_schema = _simplify_pydantic_schema_for_gemini(non_null_schemas[0])
            if simplified_t_schema is None:
                 warnings.warn(f"Failed to simplify the single non-null part of an anyOf schema: {non_null_schemas[0]}. Field may be unusable.")
                 return None
            return simplified_t_schema

        else:
            simplified_anyof = []
            for sub_schema in non_null_schemas:
                 if isinstance(sub_schema, dict) and sub_schema.get("type") == "null":
                     raise UnsupportedSchemaTypeError(f"Unexpected 'null' type found nested within complex Union: {sub_schema}")

                 simplified_sub = _simplify_pydantic_schema_for_gemini(sub_schema)
                 if simplified_sub is not None:
                     simplified_anyof.append(simplified_sub)
                 else:
                     warnings.warn(f"Failed to simplify a sub-schema within complex 'anyOf': {sub_schema}. Skipping it.")

            if not simplified_anyof:
                 warnings.warn(f"All non-null sub-schemas failed simplification within 'anyOf', resulting schema may be invalid. Original non-null: {non_null_schemas}")
                 return None

            processed_schema["anyOf"] = simplified_anyof
            processed_schema.pop("type", None)

    elif "type" in processed_schema:
        schema_type = processed_schema["type"]

        if isinstance(schema_type, list):
            non_null_types = [t for t in schema_type if t != "null"]
            was_optional = len(non_null_types) != len(schema_type)

            if not non_null_types:
                raise UnsupportedSchemaTypeError(f"Schema invalid: 'type' list contained only null. Original: {schema_type}")

            if len(non_null_types) == 1:
                json_type = non_null_types[0]
                processed_schema['type'] = json_type
            else:
                processed_schema.pop("type")
                original_format = processed_schema.pop("format", None)

                converted_anyof = []
                for t in non_null_types:
                    if t == "null":
                         raise UnsupportedSchemaTypeError("Unexpected 'null' found during type list to anyOf conversion.")
                    elif t in GEMINI_ALLOWED_TYPES:
                         gemini_type = GEMINI_ALLOWED_TYPES[t]
                         sub_schema = {"type": gemini_type}
                         if gemini_type == "STRING" and original_format:
                             pass
                         simplified_sub = _simplify_pydantic_schema_for_gemini(sub_schema)
                         if simplified_sub:
                            converted_anyof.append(simplified_sub)
                         else:
                             warnings.warn(f"Failed to create simplified schema for type '{t}' during list conversion. Skipping.")
                    else:
                         warnings.warn(f"Unknown type '{t}' found in type list: {non_null_types}. Skipping.")
                         continue

                if converted_anyof:
                    processed_schema["anyOf"] = converted_anyof
                else:
                    warnings.warn(f"Could not convert type list to anyOf: {schema_type}. Schema may be invalid.")
                    return None

        if isinstance(processed_schema.get("type"), str):
             json_type = processed_schema["type"]

             if json_type == "null":
                  raise UnsupportedSchemaTypeError("Standalone 'type: null' is not permitted.")

             if "format" in processed_schema and json_type == "string":
                 processed_schema["type"] = GEMINI_ALLOWED_TYPES["string"]
                 processed_schema.pop("format", None)
             elif json_type in GEMINI_ALLOWED_TYPES:
                 processed_schema["type"] = GEMINI_ALLOWED_TYPES[json_type]
             else:
                 raise UnsupportedSchemaTypeError(f"Unknown JSON schema type '{json_type}' encountered.")

        elif not isinstance(schema_type, str):
             if "anyOf" not in processed_schema:
                  raise UnsupportedSchemaTypeError(f"Invalid 'type' value: {schema_type}. Expected string or list.")

    if "enum" in processed_schema:
         enum_base_type = processed_schema.get("type")
         if not enum_base_type or enum_base_type not in [GEMINI_ALLOWED_TYPES["string"], GEMINI_ALLOWED_TYPES["integer"], GEMINI_ALLOWED_TYPES["number"]]:
              warnings.warn(f"Enum found without a valid base type (STRING, INTEGER, NUMBER): {processed_schema}. Schema might be invalid.")

    current_type = processed_schema.get("type")

    if current_type == GEMINI_ALLOWED_TYPES["object"]:
        if "properties" in processed_schema:
            simplified_props = {}
            for prop_name, prop_schema in processed_schema["properties"].items():
                try:
                    if isinstance(prop_schema, dict) and prop_schema.get("type") == "null":
                         raise UnsupportedSchemaTypeError(f"Property '{prop_name}' has forbidden 'type: null'. Use Optional[T] instead.")

                    simplified = _simplify_pydantic_schema_for_gemini(prop_schema)
                    if simplified is not None:
                        simplified_props[prop_name] = simplified
                    else:
                        warnings.warn(f"Simplification failed for property '{prop_name}', excluding it. Original: {prop_schema}")
                except UnsupportedSchemaTypeError as e:
                     raise UnsupportedSchemaTypeError(f"Unsupported type in property '{prop_name}': {e}") from e
                except Exception as e:
                     warnings.warn(f"Excluding property '{prop_name}' due to unexpected error during simplification: {e}. Original: {prop_schema}")

            processed_schema["properties"] = simplified_props
            processed_schema.pop("required", None)
        else:
             processed_schema["properties"] = {}

    elif current_type == GEMINI_ALLOWED_TYPES["array"]:
        if "items" in processed_schema:
             items_schema = processed_schema["items"]
             try:
                 if isinstance(items_schema, dict) and items_schema.get("type") == "null":
                     raise UnsupportedSchemaTypeError("Array items cannot have forbidden 'type: null'. Use Optional[List[T]] or List[Optional[T]] if needed (check if List[Optional[T]] is supported).")

                 simplified_items = _simplify_pydantic_schema_for_gemini(items_schema)
                 if simplified_items is not None:
                      processed_schema["items"] = simplified_items
                 else:
                      raise RuntimeError(f"Failed to simplify items schema for array. Original items: {items_schema}")
             except UnsupportedSchemaTypeError as e:
                  raise UnsupportedSchemaTypeError(f"Unsupported type within array items: {e}") from e
             except Exception as e:
                  raise RuntimeError(f"Unexpected error simplifying array items: {e}") from e
        else:
             raise UnsupportedSchemaTypeError(f"Schema type is ARRAY but 'items' definition is missing: {processed_schema}")

    final_type = processed_schema.get("type")
    final_any_of = processed_schema.get("anyOf")

    if final_type:
         if final_type == "null":
              raise RuntimeError("Internal Error: 'type: null' survived simplification.")
         if final_type not in ALLOWED_GEMINI_TYPE_VALUES:
              raise RuntimeError(f"Internal Error: Schema simplification resulted in disallowed type '{final_type}'.")
    elif final_any_of:
         for sub in final_any_of:
             if isinstance(sub, dict) and sub.get("type") == "null":
                  raise RuntimeError("Internal Error: 'type: null' survived within 'anyOf'.")
    elif "enum" not in processed_schema:
         if not (processed_schema.get("properties") is not None and current_type == GEMINI_ALLOWED_TYPES["object"]):
              warnings.warn(f"Schema simplification resulted in a structure missing defining characteristics (type/anyOf/enum): {processed_schema}. Returning None.")
              return None

    processed_schema.pop("nullable", None)
    processed_schema.pop("default", None)
    processed_schema.pop("title", None)
    processed_schema.pop("$defs", None)
    processed_schema.pop("format", None)

    return processed_schema

def _create_gemini_json_schema(model_cls: type[BaseModel]) -> dict[str, Any]:
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
         raise TypeError(f"Expected concrete pydantic.BaseModel subclass, got {model_cls}")

    try:
        schema = model_cls.model_json_schema()
    except AttributeError:
        schema = model_cls.schema()

    schema_without_refs: dict[str, Any] = jsonref.replace_refs(schema, base_uri='local', jsonschema=True)

    if schema_without_refs.get("type") != "object" or "properties" not in schema_without_refs:
         raise ValueError(f"Schema for {model_cls} did not resolve to a standard object structure with properties.")

    simplified_properties = {}
    original_properties = schema_without_refs.get("properties", {})
    original_required_fields = set(schema_without_refs.get("required", []))
    processed_required = []
    model_description = schema_without_refs.get("description")

    for prop_name, prop_schema in original_properties.items():
        try:
            if isinstance(prop_schema, dict) and prop_schema.get("type") == "null":
                raise UnsupportedSchemaTypeError(f"Property '{prop_name}' uses forbidden 'type: null'. Use Optional[T] instead.")

            simplified_prop_schema = _simplify_pydantic_schema_for_gemini(prop_schema)

            if simplified_prop_schema is not None:
                if isinstance(simplified_prop_schema, dict) and simplified_prop_schema.get("type") == "null":
                     raise RuntimeError(f"Internal Error: Simplification of property '{prop_name}' incorrectly resulted in 'type: null'.")

                simplified_properties[prop_name] = simplified_prop_schema
                if prop_name in original_required_fields:
                    processed_required.append(prop_name)
            else:
                 warnings.warn(f"Property '{prop_name}' could not be simplified (returned None) and will be excluded.")

        except UnsupportedSchemaTypeError as e:
            raise UnsupportedSchemaTypeError(f"Failed to generate schema for model '{model_cls.__name__}'. Property '{prop_name}' has an unsupported structure or type: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error processing property '{prop_name}' in model '{model_cls.__name__}': {e}") from e

    gemini_schema: dict[str, Any] = {
        "type": "OBJECT",
        "properties": simplified_properties,
        **({"description": model_description} if model_description else {}),
        **({"required": sorted(processed_required)} if processed_required else {}),
    }

    return gemini_schema

def _create_vertexai_tool(
    models: BaseModel | list[BaseModel] | type,
) -> gm.Tool:  # noqa: UP007
    """Creates a tool with function declarations for single model or list of models"""
    # Handle Iterable case first
    if get_origin(models) is not None:
        model_list = list(get_types_array(models))  # type: ignore
    else:
        # Handle both single model and list of models
        model_list = models if isinstance(models, list) else [models]

    declarations = []
    for model in model_list:  # type: ignore
        parameters = _create_gemini_json_schema(model)  # type: ignore
        declaration = gm.FunctionDeclaration(
            name=model.__name__,  # type: ignore
            description=model.__doc__,  # type: ignore
            parameters=parameters,
        )
        declarations.append(declaration)  # type: ignore

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
