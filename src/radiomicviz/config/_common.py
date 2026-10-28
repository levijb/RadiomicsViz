"""Shared machinery for strict configuration loading.

Configuration files are parsed with a YAML loader that rejects duplicate keys, then
checked against a fixed schema. Every problem is collected and reported together, each
prefixed by its dotted key path (for example ``modalities.T1.image_pattern``), so a user
can fix a file in one pass.
"""

from __future__ import annotations

import re
from pathlib import Path
from string import Formatter
from typing import Any, Iterable

import yaml

#: Identifiers (subject, session, region, modality, space) may contain only these
#: characters, so they are safe to use as file and folder names on every platform.
SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class ConfigError(ValueError):
    """A configuration file could not be read or does not satisfy its schema.

    Attributes:
        source: The file (or other source) that was being loaded.
        errors: One message per problem, each starting with the dotted key path.
    """

    def __init__(self, source: str, errors: Iterable[str]):
        self.source = source
        self.errors = list(errors)
        listing = "\n".join(f"  - {message}" for message in self.errors)
        noun = "problem" if len(self.errors) == 1 else "problems"
        super().__init__(f"{source}: {len(self.errors)} {noun} found:\n{listing}")


class _StrictLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys.

    Plain PyYAML keeps the last of two duplicate keys without warning, which would let a
    mistyped configuration silently override an earlier value.
    """

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        seen: set[Any] = set()
        for key_node, _value_node in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                continue
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in seen
            except TypeError:  # unhashable key; the base class reports it
                continue
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


def load_yaml_mapping(path: Path) -> dict:
    """Read ``path`` as UTF-8 YAML and return its top-level mapping.

    Raises:
        ConfigError: If the file is missing, unreadable, not valid YAML, contains a
            duplicate key, or does not hold a mapping at the top level.
    """
    source = str(path)
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        raise ConfigError(source, ["file not found"]) from None
    except IsADirectoryError:
        raise ConfigError(source, ["is a directory, not a file"]) from None
    except UnicodeDecodeError as exc:
        raise ConfigError(source, [f"file is not valid UTF-8 ({exc.reason} at byte {exc.start})"]) from None
    except OSError as exc:
        raise ConfigError(source, [f"cannot read file ({exc.strerror})"]) from None

    try:
        data = yaml.load(text, Loader=_StrictLoader)  # a SafeLoader subclass
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark
        where = f" (line {mark.line + 1}, column {mark.column + 1})" if mark else ""
        raise ConfigError(source, [f"invalid YAML: {exc.problem}{where}"]) from None
    except yaml.YAMLError as exc:
        raise ConfigError(source, [f"invalid YAML: {exc}"]) from None

    if data is None:
        raise ConfigError(source, ["file is empty"])
    if not isinstance(data, dict):
        raise ConfigError(source, [f"top level must be a mapping of keys to values, not {_type_name(data)}"])
    return data


class Checker:
    """Collects schema violations, each tagged with a dotted key path."""

    def __init__(self) -> None:
        self.errors: list[str] = []

    def add(self, path: str, message: str) -> None:
        self.errors.append(f"{path}: {message}" if path else message)

    def keys(
        self,
        mapping: dict,
        path: str,
        required: Iterable[str],
        optional: Iterable[str] = (),
    ) -> None:
        """Report unknown keys and missing required keys of ``mapping``."""
        required = tuple(required)
        allowed = required + tuple(optional)
        for key in mapping:
            if key not in allowed:
                where = _join(path, str(key))
                self.add(where, f"unknown key (allowed keys: {', '.join(allowed)})")
        for key in required:
            if key not in mapping:
                self.add(_join(path, key), "required key is missing")

    def mapping(self, value: Any, path: str) -> dict | None:
        if isinstance(value, dict):
            return value
        self.add(path, f"must be a mapping, not {_type_name(value)}")
        return None

    def string(self, value: Any, path: str, *, optional: bool = False) -> str | None:
        if value is None and optional:
            return None
        if not isinstance(value, str):
            hint = " (put quotes around it)" if isinstance(value, (int, float, bool)) else ""
            self.add(path, f"must be a string, not {_type_name(value)}{hint}")
            return None
        if not value.strip():
            self.add(path, "must not be empty")
            return None
        return value

    def identifier(self, value: Any, path: str, *, optional: bool = False) -> str | None:
        text = self.string(value, path, optional=optional)
        if text is None:
            return None
        if not SAFE_ID_PATTERN.match(text) or text in {".", ".."}:
            self.add(
                path,
                f"{text!r} is not a safe identifier (use only letters, digits, '_', '.', '-')",
            )
            return None
        return text

    def choice(self, value: Any, path: str, choices: Iterable[str]) -> str | None:
        choices = tuple(choices)
        if value not in choices or not isinstance(value, str):
            listed = ", ".join(repr(c) for c in choices)
            self.add(path, f"must be one of {listed} (got {value!r})")
            return None
        return value

    def integer(self, value: Any, path: str, *, minimum: int | None = None) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            self.add(path, f"must be an integer, not {_type_name(value)}")
            return None
        if minimum is not None and value < minimum:
            self.add(path, f"must be at least {minimum} (got {value})")
            return None
        return value

    def boolean(self, value: Any, path: str) -> bool | None:
        if not isinstance(value, bool):
            self.add(path, f"must be true or false, not {_type_name(value)}")
            return None
        return value

    def regex(self, value: Any, path: str, *, group: str, optional: bool = False) -> str | None:
        """Check that ``value`` compiles and defines the named group ``group``."""
        text = self.string(value, path, optional=optional)
        if text is None:
            return None
        try:
            compiled = re.compile(text)
        except re.error as exc:
            self.add(path, f"invalid regular expression ({exc})")
            return None
        if group not in compiled.groupindex:
            self.add(path, f"regular expression must define the named group (?P<{group}>...)")
            return None
        return text

    def path_template(
        self,
        value: Any,
        path: str,
        *,
        allowed: Iterable[str],
        required: Iterable[str] = (),
        optional: bool = False,
    ) -> str | None:
        """Check a path template whose ``{placeholders}`` are limited to ``allowed``."""
        text = self.string(value, path, optional=optional)
        if text is None:
            return None
        allowed = tuple(allowed)
        try:
            fields = [
                (name, spec, conversion)
                for _literal, name, spec, conversion in Formatter().parse(text)
                if name is not None
            ]
        except ValueError as exc:
            self.add(path, f"malformed placeholder ({exc}); write literal braces as '{{{{' and '}}}}'")
            return None
        ok = True
        for name, spec, conversion in fields:
            if name not in allowed or spec or conversion:
                listed = ", ".join("{" + a + "}" for a in allowed)
                self.add(path, f"unknown placeholder {{{name}}} (allowed: {listed})")
                ok = False
        names = {name for name, _spec, _conversion in fields}
        for name in required:
            if name not in names:
                self.add(path, f"must contain the placeholder {{{name}}}")
                ok = False
        return text if ok else None

    def raise_if_errors(self, source: str) -> None:
        if self.errors:
            raise ConfigError(source, self.errors)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, int):
        return "an integer"
    if isinstance(value, float):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, dict):
        return "a mapping"
    return type(value).__name__
