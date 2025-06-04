# type: ignore
from __future__ import annotations

from typing import Any, Literal, overload

import google.generativeai as genai

import instructor


def _create_gemini_response_model(model_cls: type[BaseModel]) -> type[BaseModel]:
    """Create a modified Pydantic model with uniqueItems keys removed from its schema.
    
    This function:
    1. Converts the Pydantic model to a JSON schema
    2. Removes all uniqueItems keys from the schema
    3. Creates a new Pydantic model from the modified schema
    4. Returns the new model class
    """
    from pydantic import create_model
    from pydantic.json_schema import model_json_schema_to_python
    import inspect

    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
        raise TypeError(f"Expected concrete pydantic.BaseModel subclass, got {model_cls}")

    # Convert Pydantic model to JSON schema
    try:
        # Get the JSON schema from the model
        schema = model_cls.model_json_schema()
    except Exception as e:
        raise RuntimeError(f"Failed to create schema for {model_cls.__name__}: {e}") from e

    # Define a recursive function to remove uniqueItems
    def _remove_unique_items(obj: Any) -> None:
        """Recursively remove uniqueItems keys from a schema object."""
        if isinstance(obj, dict):
            # Remove uniqueItems if present
            if "uniqueItems" in obj:
                del obj["uniqueItems"]
            
            # Recursively process all dictionary values
            for key, value in obj.items():
                _remove_unique_items(value)
        elif isinstance(obj, list):
            # Recursively process all list items
            for item in obj:
                _remove_unique_items(item)

    # Remove uniqueItems from the schema
    _remove_unique_items(schema)
    
    # Create a new model from the modified schema
    try:
        # Use the original model's name and docstring
        model_name = model_cls.__name__
        model_doc = inspect.getdoc(model_cls) or ""
        
        # Convert JSON schema back to Python types and fields
        fields = model_json_schema_to_python(schema)
        
        # Create a new model with the same name and fields
        new_model = create_model(
            model_name,
            __doc__=model_doc,
            **fields
        )
        
        return new_model
    except Exception as e:
        raise RuntimeError(f"Failed to create Pydantic model from modified schema: {e}") from e


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[True] = True,
    **kwargs: Any,
) -> instructor.AsyncInstructor:
    ...


@overload
def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: Literal[False] = False,
    **kwargs: Any,
) -> instructor.Instructor:
    ...


def from_gemini(
    client: genai.GenerativeModel,
    mode: instructor.Mode = instructor.Mode.GEMINI_JSON,
    use_async: bool = False,
    **kwargs: Any,
) -> instructor.Instructor | instructor.AsyncInstructor:
    assert mode in {
        instructor.Mode.GEMINI_JSON,
        instructor.Mode.GEMINI_TOOLS,
    }, "Mode must be one of {instructor.Mode.GEMINI_JSON, instructor.Mode.GEMINI_TOOLS}"

    assert isinstance(
        client,
        (genai.GenerativeModel),
    ), "Client must be an instance of genai.generativemodel"

    if use_async:
        create = client.generate_content_async
        return instructor.AsyncInstructor(
            client=client,
            create=instructor.patch(create=create, mode=mode),
            provider=instructor.Provider.GEMINI,
            mode=mode,
            **kwargs,
        )

    create = client.generate_content
    return instructor.Instructor(
        client=client,
        create=instructor.patch(create=create, mode=mode),
        provider=instructor.Provider.GEMINI,
        mode=mode,
        **kwargs,
    )
