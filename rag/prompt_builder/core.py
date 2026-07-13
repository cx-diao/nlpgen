from pathlib import Path
import yaml
from typing import Dict, Any, Callable
import re

from loguru import logger

from abc import ABC, abstractmethod
from rag.search.core import BaseSearch


def load_prompt_template(yaml_path: str | Path, template_name: str) -> str:
    """
    Load a prompt template by name from a YAML file.

    Args:
        template_name: The name of the template to fetch.
        yaml_path: Path to the YAML file containing templates.

    Returns:
        The prompt template string.
    """
    if not isinstance(yaml_path, Path):
        yaml_path = Path(yaml_path)
    assert yaml_path.exists(), f"YAML file not found: {yaml_path}"
    with open(yaml_path, "r") as f:
        templates = yaml.safe_load(f)
    assert template_name in templates, f"Template '{template_name}' not found in {yaml_path}"
    return templates[template_name]


def _build_prompt_from_template(template: str, placeholder2value: Dict[str, Any]) -> str:
    placeholder_keys = set(re.findall(r"\{(\w+)\}", template))
    missing_keys = placeholder_keys - placeholder2value.keys()
    assert (
        not missing_keys
    ), f"Missing required placeholders: {missing_keys}. Required: {placeholder_keys}, Provided: {set(placeholder2value.keys())}"
    # Only use the keys needed by the template
    filtered_values = {k: placeholder2value[k] for k in placeholder_keys}
    return template.format(**filtered_values)


def prompt_builder_from_yaml(
    yaml_path: str | Path,
    template_name: str,
) -> Callable[[Dict[str, Any]], str]:
    """
    Returns a prompt builder function that loads the template from YAML by name.

    Args:
        yaml_path: The path to the YAML file (e.g., "quantification.yaml")
        template_name: The name of the template to use (no default)

    Returns:
        A function that takes a placeholder dictionary and returns the formatted prompt
    """

    def build(placeholder2value: Dict[str, Any]) -> str:
        template = load_prompt_template(yaml_path, template_name)
        return _build_prompt_from_template(template, placeholder2value)

    return build
