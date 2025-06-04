# type: ignore
from __future__ import annotations

from typing import Any, Literal, overload

import google.generativeai as genai

import instructor


def _transform_type(tp):
    """
    Recursively walk an annotation and:
      • turn Set[...] into List[...]
      • clone inner BaseModels with their own sets → lists
    """
    from typing import Any, List, Set, Union, get_args, get_origin
    from pydantic import BaseModel
    
    origin = get_origin(tp)

    # -------------------- 1. primitive or concrete model --------------------
    if origin is None:                                # not a generic alias
        if isinstance(tp, type) and issubclass(tp, BaseModel):
            return replace_set_with_list(tp)          # clone nested model
        return tp

    # -------------------- 2. Set[...] → List[...] ---------------------------
    if origin in (set, Set):
        inner = _transform_type(get_args(tp)[0] if get_args(tp) else Any)
        return List[inner]

    # -------------------- 3. generic containers (List, Dict, Union, etc.) --
    new_args = tuple(_transform_type(a) for a in get_args(tp))

    if origin is Union:                               # special-case Union
        return Union[new_args]

    # most collections (list, dict, tuple, etc.) can be rebuilt by subscripting
    return origin[new_args]


def replace_set_with_list(model: type[BaseModel]) -> type[BaseModel]:
    """
    Return a *new* Pydantic model class equivalent to `model`
    but with every set-typed field replaced by a list-typed field,
    recursively through any depth of nested containers / models.
    Works on both Pydantic v1 and v2.
    """
    from pydantic import create_model
    
    raw_fields = getattr(model, "__fields__", None) or model.model_fields
    new_fields = {}

    for name, fld in raw_fields.items():
        anno = getattr(fld, "outer_type_", None) or getattr(fld, "annotation")
        anno = _transform_type(anno)

        default_val = ... if getattr(fld, "required", True) else fld.default
        new_fields[name] = (anno, default_val)

    return create_model(f"{model.__name__}", **new_fields)


def _create_gemini_response_model(model_cls: type[BaseModel]) -> type[BaseModel]:
    """
    Create a modified Pydantic model with Set types replaced by List types.
    
    This function uses replace_set_with_list to recursively transform a model,
    replacing all Set types with List types at any level of nesting.
    
    Args:
        model_cls: The Pydantic model class to transform
        
    Returns:
        A new Pydantic model class with all Set types replaced by List types
    """
    if not isinstance(model_cls, type) or not issubclass(model_cls, BaseModel):
        raise TypeError(f"Expected concrete pydantic.BaseModel subclass, got {model_cls}")

    try:
        return replace_set_with_list(model_cls)
    except Exception as e:
        raise RuntimeError(f"Failed to create modified model: {e}") from e


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
