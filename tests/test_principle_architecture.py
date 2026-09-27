"""Architecture fitness test: enforces constitution principles P-013 (hexagonal
ports and adapters) and P-014 (Object Calisthenics) over every ``.py`` file
under ``src/lws``.

@principle:P-013 @principle:P-014

This file lives under ``tests/`` and is therefore itself exempt from the
rules it enforces (see constitution.md, "Not applicable" clauses).

Known, deliberate limitations (documented rather than silently skipped):

- Rule 4 (first-class collections) is checked structurally: a data-record
  class (a class whose body is only annotated field declarations, i.e. a
  dataclass or a ``typing.NamedTuple``) that has a field annotated as a
  container type (``list``/``dict``/``set``/``tuple``/``frozenset``,
  parametrized or not) must have exactly that one field. This does not (and
  cannot, via AST alone) verify that the collection is used purely as a
  collection at runtime.
- Rule 5 (one dot per line) is checked only for attribute chains rooted at
  ``self`` or at a local ``Name`` that is not a known module alias. A chain
  reached through an intermediate expression (e.g. ``(a + b).thing``) is not
  flagged, since it is not a dotted identifier chain.
- Rule 9 (no getters/setters) is checked mechanically: no ``@property``, no
  ``get_*``/``set_*`` method names. The "never assign another object's
  attributes" and "tell, don't ask" clauses are not mechanically checked
  (too easy to false-positive on ordinary object construction).
- Rule 3 (wrap primitives) is checked only on named, non-underscore,
  non-dunder function/method definitions. Dataclass/NamedTuple field
  annotations are governed by rules 4 and 8 instead, not by rule 3 — the
  auto-generated ``__init__`` of a leaf value object (e.g. ``Bits(value:
  float)``) is exactly the base case the rule's own wording describes
  ("single-field frozen value objects"), so it is not itself flagged.
- Composition root purity ("entrypoints ... parse arguments, wire adapters
  ... nothing else") is enforced only through rules 1/2/6/7 plus the import
  checks; there is no separate "this function only wires" check.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src" / "lws"
TESTS_ROOT = REPO_ROOT / "tests"

FORBIDDEN_LIBRARIES = (
    "torch",
    "cupy",
    "safetensors",
    "transformers",
    "huggingface_hub",
    "vllm",
    "fastapi",
    "uvicorn",
    "httpx",
)

BANNED_IDENTIFIERS = frozenset(
    {
        "cfg",
        "conf",
        "mgr",
        "tmp",
        "idx",
        "cnt",
        "num",
        "buf",
        "val",
        "res",
        "arr",
        "ptr",
        "calc",
        "util",
        "info",
        "obj",
        "ctx",
    }
)

# Module aliases that make an attribute chain "module-qualified" rather than
# a chain rooted at a local variable (rule 5 exempts these).
MODULE_ALIAS_NAMES = frozenset(
    {
        "numpy",
        "np",
        "torch",
        "cupy",
        "cp",
        "safetensors",
        "transformers",
        "huggingface_hub",
        "vllm",
        "fastapi",
        "uvicorn",
        "httpx",
        "os",
        "re",
        "sys",
        "math",
        "json",
        "typing",
        "dataclasses",
        "pathlib",
        "functools",
        "itertools",
        "argparse",
        "collections",
        "ast",
        "subprocess",
        "logging",
    }
)

RELAXABLE_RULES = frozenset({3, 4, 5, 8, 9})
ALLOW_COMMENT = re.compile(r"#\s*calisthenics:\s*allow\s+([\d,\s]+)")


@dataclasses.dataclass(frozen=True)
class SourceFile:
    path: pathlib.Path
    layer: str
    tree: ast.Module
    lines: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class Violation:
    file: SourceFile
    line: int
    rule: str
    message: str


def _layer_of(relative: pathlib.PurePosixPath) -> str:
    top = relative.parts[0]
    if top == "domain":
        return "domain"
    if top == "application":
        return "application"
    if top == "adapters":
        return "adapter"
    return "entrypoint"


def _read_source_files() -> list[SourceFile]:
    files = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        relative = pathlib.PurePosixPath(path.relative_to(SRC_ROOT).as_posix())
        text = path.read_text()
        tree = ast.parse(text, filename=str(path))
        lines = tuple(text.splitlines())
        files.append(SourceFile(path=path, layer=_layer_of(relative), tree=tree, lines=lines))
    return files


SOURCE_FILES = _read_source_files()


def _allowed_rules_on_line(file: SourceFile, line: int) -> frozenset[int]:
    if line < 1 or line > len(file.lines):
        return frozenset()
    match = ALLOW_COMMENT.search(file.lines[line - 1])
    if match is None:
        return frozenset()
    numbers = (piece.strip() for piece in match.group(1).split(","))
    return frozenset(int(piece) for piece in numbers if piece)


def _is_relaxed(file: SourceFile, rule_number: int) -> bool:
    return rule_number in RELAXABLE_RULES and file.layer in ("adapter", "entrypoint")


def _keep_unexcused(file: SourceFile, rule_number: int, violations: list[Violation]) -> list[Violation]:
    if not _is_relaxed(file, rule_number):
        return violations
    return [v for v in violations if rule_number not in _allowed_rules_on_line(file, v.line)]


# --------------------------------------------------------------------------
# P-013 — import direction and absolute imports
# --------------------------------------------------------------------------


def _module_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return None
    if isinstance(node, ast.ImportFrom):
        return node.module
    return None


def _check_imports(file: SourceFile) -> list[Violation]:
    violations: list[Violation] = []
    for node in ast.walk(file.tree):
        violations.extend(_check_one_import(file, node))
    return violations


def _check_one_import(file: SourceFile, node: ast.AST) -> list[Violation]:
    if isinstance(node, ast.ImportFrom):
        return _check_import_from(file, node)
    if isinstance(node, ast.Import):
        return _check_plain_import(file, node)
    return []


def _check_import_from(file: SourceFile, node: ast.ImportFrom) -> list[Violation]:
    violations = []
    if node.level > 0:
        violations.append(Violation(file, node.lineno, "P-013", "relative import (must be absolute)"))
    module = node.module or ""
    violations.extend(_check_module_path(file, node.lineno, module))
    return violations


def _check_plain_import(file: SourceFile, node: ast.Import) -> list[Violation]:
    violations = []
    for alias in node.names:
        violations.extend(_check_module_path(file, node.lineno, alias.name))
    return violations


def _check_module_path(file: SourceFile, line: int, module: str) -> list[Violation]:
    if file.layer not in ("domain", "application"):
        return []
    first_segment = module.split(".")[0]
    violations = []
    if first_segment in FORBIDDEN_LIBRARIES:
        violations.append(Violation(file, line, "P-013", f"{file.layer} imports forbidden library '{first_segment}'"))
    violations.extend(_check_lws_module_path(file, line, module))
    return violations


def _check_lws_module_path(file: SourceFile, line: int, module: str) -> list[Violation]:
    if not module.startswith("lws."):
        return []
    forbidden_for_domain = ("application", "adapters", "analyze", "pack", "device", "tune", "bench", "runtime", "server")
    forbidden_for_application = ("adapters", "analyze", "pack", "device", "tune", "bench", "runtime", "server")
    forbidden = forbidden_for_domain if file.layer == "domain" else forbidden_for_application
    second_segment = module.split(".")[1] if "." in module else ""
    if second_segment in forbidden:
        return [Violation(file, line, "P-013", f"{file.layer} imports 'lws.{second_segment}' (wrong direction)")]
    return []


def test_import_direction_and_absolute_imports() -> None:
    violations = [v for file in SOURCE_FILES for v in _check_imports(file)]
    assert not violations, _format(violations)


# --------------------------------------------------------------------------
# P-013 — every port has an adapter and a fake
# --------------------------------------------------------------------------


def _is_protocol_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        if isinstance(base, ast.Name) and base.id == "Protocol":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "Protocol":
            return True
    return False


def _public_method_names(node: ast.ClassDef) -> frozenset[str]:
    names = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and not item.name.startswith("_"):
            names.append(item.name)
    return frozenset(names)


def _find_protocols(files: list[SourceFile]) -> list[tuple[SourceFile, ast.ClassDef]]:
    ports_root = SRC_ROOT / "application" / "ports"
    found = []
    for file in files:
        if ports_root not in file.path.parents:
            continue
        for node in ast.walk(file.tree):
            if isinstance(node, ast.ClassDef) and _is_protocol_class(node):
                found.append((file, node))
    return found


def _classes_with_methods(files: list[SourceFile], under: pathlib.Path) -> list[ast.ClassDef]:
    classes = []
    for file in files:
        if under not in file.path.parents and file.path.parent != under:
            continue
        classes.extend(node for node in ast.walk(file.tree) if isinstance(node, ast.ClassDef))
    return classes


def _has_matching_implementation(required: frozenset[str], candidates: list[ast.ClassDef]) -> bool:
    for candidate in candidates:
        provided = _public_method_names(candidate)
        if required <= provided:
            return True
    return False


def test_every_port_has_an_adapter_and_a_fake() -> None:
    protocols = _find_protocols(SOURCE_FILES)
    adapters_root = SRC_ROOT / "adapters"
    fake_classes = _parse_fakes_module()
    missing = []
    for file, node in protocols:
        required = _public_method_names(node)
        if not required:
            continue
        adapter_classes = _classes_with_methods(SOURCE_FILES, adapters_root)
        has_adapter = _has_matching_implementation(required, adapter_classes)
        has_fake = _has_matching_implementation(required, fake_classes)
        if not has_adapter:
            missing.append(f"{file.path}:{node.lineno} port {node.name} has no matching adapter under src/lws/adapters")
        if not has_fake:
            missing.append(f"{file.path}:{node.lineno} port {node.name} has no matching fake in tests/fakes.py")
    assert not missing, "\n".join(missing)


def _parse_fakes_module() -> list[ast.ClassDef]:
    fakes_path = TESTS_ROOT / "fakes.py"
    if not fakes_path.exists():
        return []
    tree = ast.parse(fakes_path.read_text(), filename=str(fakes_path))
    return [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


# --------------------------------------------------------------------------
# P-014 rule 1 — one level of indentation per function
# --------------------------------------------------------------------------

COMPOUND_STATEMENTS = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)


def _max_compound_depth(node: ast.AST, depth: int = 0) -> int:
    best = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        child_depth = depth + 1 if isinstance(child, COMPOUND_STATEMENTS) else depth
        best = max(best, _max_compound_depth(child, child_depth))
    return best


def _check_rule_1(file: SourceFile) -> list[Violation]:
    violations = []
    for node in ast.walk(file.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            depth = _max_compound_depth(node)
            if depth > 1:
                violations.append(Violation(file, node.lineno, "1", f"function '{node.name}' nests {depth} levels deep (max 1)"))
    return _keep_unexcused(file, 1, violations)


def test_rule_1_one_level_of_indentation() -> None:
    violations = [v for file in SOURCE_FILES for v in _check_rule_1(file)]
    assert not violations, _format(violations)


# --------------------------------------------------------------------------
# P-014 rule 2 — no else / elif (redundant with the audit regex, checked
# here too so this test is self-contained without the external engine).
# --------------------------------------------------------------------------

ELSE_ELIF = re.compile(r"^\s*(else\s*:|elif\s)")


def _check_rule_2(file: SourceFile) -> list[Violation]:
    violations = [
        Violation(file, number, "2", "else/elif found (use a guard clause or early return)")
        for number, line in enumerate(file.lines, start=1)
        if ELSE_ELIF.match(line)
    ]
    return _keep_unexcused(file, 2, violations)


def test_rule_2_no_else_or_elif() -> None:
    violations = [v for file in SOURCE_FILES for v in _check_rule_2(file)]
    assert not violations, _format(violations)


# --------------------------------------------------------------------------
# P-014 rule 3 — wrap primitives at public function boundaries
# --------------------------------------------------------------------------

BARE_PRIMITIVE_NAMES = frozenset({"int", "float", "str", "bool"})


def _is_bare_primitive_annotation(annotation: ast.AST | None) -> str | None:
    if annotation is None:
        return None
    if isinstance(annotation, ast.Name) and annotation.id in BARE_PRIMITIVE_NAMES:
        return annotation.id
    if isinstance(annotation, ast.Attribute) and annotation.attr == "ndarray":
        return "ndarray"
    if isinstance(annotation, ast.Name) and annotation.id == "ndarray":
        return "ndarray"
    return None


def _is_checked_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if node.name.startswith("__") and node.name.endswith("__"):
        return False
    return not node.name.startswith("_")


def _check_rule_3(file: SourceFile) -> list[Violation]:
    if file.layer not in ("domain", "application"):
        return _keep_unexcused(file, 3, _check_rule_3_relaxed(file))
    violations = []
    for node in ast.walk(file.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_checked_function(node):
            violations.extend(_check_rule_3_signature(file, node))
    return violations


def _check_rule_3_relaxed(file: SourceFile) -> list[Violation]:
    violations = []
    for node in ast.walk(file.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_checked_function(node):
            violations.extend(_check_rule_3_signature(file, node))
    return violations


def _check_rule_3_signature(file: SourceFile, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Violation]:
    violations = []
    arguments = [a for a in node.args.args if a.arg != "self"]
    for argument in arguments:
        kind = _is_bare_primitive_annotation(argument.annotation)
        if kind is not None:
            violations.append(Violation(file, node.lineno, "3", f"'{node.name}' takes bare {kind} parameter '{argument.arg}'"))
    kind = _is_bare_primitive_annotation(node.returns)
    if kind is not None:
        violations.append(Violation(file, node.lineno, "3", f"'{node.name}' returns bare {kind}"))
    return violations


def test_rule_3_wrap_primitives() -> None:
    violations = [v for file in SOURCE_FILES for v in _check_rule_3(file)]
    assert not violations, _format(violations)


# --------------------------------------------------------------------------
# P-014 rule 4 — first-class collections
# --------------------------------------------------------------------------

COLLECTION_TYPE_NAMES = frozenset({"list", "dict", "set", "tuple", "frozenset", "List", "Dict", "Set", "Tuple", "FrozenSet", "Sequence", "Mapping", "Iterable"})


def _decorator_name(decorator: ast.AST) -> str | None:
    if isinstance(decorator, ast.Name):
        return decorator.id
    if isinstance(decorator, ast.Attribute):
        return decorator.attr
    if isinstance(decorator, ast.Call):
        return _decorator_name(decorator.func)
    return None


def _base_name(base: ast.AST) -> str | None:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return None


def _is_data_record_class(node: ast.ClassDef) -> bool:
    if _is_protocol_class(node):
        return False
    if any(_decorator_name(d) == "dataclass" for d in node.decorator_list):
        return True
    return any(_base_name(b) == "NamedTuple" for b in node.bases)


def _field_annotations(node: ast.ClassDef) -> list[ast.AST]:
    return [item.annotation for item in node.body if isinstance(item, ast.AnnAssign)]


def _annotation_root_name(annotation: ast.AST) -> str | None:
    if isinstance(annotation, ast.Name):
        return annotation.id
    if isinstance(annotation, ast.Subscript):
        return _annotation_root_name(annotation.value)
    if isinstance(annotation, ast.Attribute):
        return annotation.attr
    return None


def _check_rule_4(file: SourceFile) -> list[Violation]:
    violations = []
    for node in ast.walk(file.tree):
        if isinstance(node, ast.ClassDef) and _is_data_record_class(node):
            violations.extend(_check_rule_4_class(file, node))
    return _keep_unexcused(file, 4, violations)


def _check_rule_4_class(file: SourceFile, node: ast.ClassDef) -> list[Violation]:
    annotations = _field_annotations(node)
    has_collection = any(_annotation_root_name(a) in COLLECTION_TYPE_NAMES for a in annotations)
    if has_collection and len(annotations) > 1:
        return [Violation(file, node.lineno, "4", f"class '{node.name}' holds a collection and other fields")]
    return []


def test_rule_4_first_class_collections() -> None:
    violations = [v for file in SOURCE_FILES for v in _check_rule_4(file)]
    assert not violations, _format(violations)


# --------------------------------------------------------------------------
# P-014 rule 5 — one dot per line (chains rooted at self or a local name)
# --------------------------------------------------------------------------


def _chain_depth_and_root(node: ast.Attribute) -> tuple[int, str | None]:
    depth = 1
    current = node.value
    while isinstance(current, ast.Attribute):
        depth += 1
        current = current.value
    if isinstance(current, ast.Name):
        return depth, current.id
    return depth, None


def _attribute_parent_ids(tree: ast.Module) -> set[int]:
    # An Attribute node that is itself the `.value` of another Attribute is
    # an inner link of a longer chain — the OUTER node is where that chain's
    # full depth is visible, so only the outer one should be reported.
    parented = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            parented.add(id(node.value))
    return parented


def _check_rule_5(file: SourceFile) -> list[Violation]:
    parented_ids = _attribute_parent_ids(file.tree)
    violations = []
    for node in ast.walk(file.tree):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            violations.extend(_check_rule_5_attribute(file, node, parented_ids))
    return _keep_unexcused(file, 5, violations)


def _check_rule_5_attribute(file: SourceFile, node: ast.Attribute, parented_ids: set[int]) -> list[Violation]:
    if id(node) in parented_ids:
        return []
    depth, root = _chain_depth_and_root(node)
    if root is None or depth < 2:
        return []
    if root != "self" and root in MODULE_ALIAS_NAMES:
        return []
    return [Violation(file, node.lineno, "5", f"chain '{root}...' is {depth} dots deep (max 1)")]


def test_rule_5_one_dot_per_line() -> None:
    violations = [v for file in SOURCE_FILES for v in _check_rule_5(file)]
    assert not violations, _format(violations)


# --------------------------------------------------------------------------
# P-014 rule 6 — no abbreviations
# --------------------------------------------------------------------------


def _identifiers(file: SourceFile) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(file.tree):
        found.extend(_identifiers_of_node(node))
    return found


def _identifiers_of_node(node: ast.AST) -> list[tuple[int, str]]:
    if isinstance(node, ast.Name):
        return [(node.lineno, node.id)]
    if isinstance(node, ast.arg):
        return [(node.lineno, node.arg)]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [(node.lineno, node.name)]
    if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
        return [(node.lineno, node.attr)]
    return []


def _check_rule_6(file: SourceFile) -> list[Violation]:
    violations = [
        Violation(file, line, "6", f"identifier '{name}' is a banned abbreviation")
        for line, name in _identifiers(file)
        if name.lower() in BANNED_IDENTIFIERS
    ]
    return _keep_unexcused(file, 6, violations)


def test_rule_6_no_abbreviations() -> None:
    violations = [v for file in SOURCE_FILES for v in _check_rule_6(file)]
    assert not violations, _format(violations)


# --------------------------------------------------------------------------
# P-014 rule 7 — small entities
# --------------------------------------------------------------------------

MAX_CLASS_LINES = 50
MAX_MODULES_PER_PACKAGE = 10


def _check_rule_7_classes(file: SourceFile) -> list[Violation]:
    violations = []
    for node in ast.walk(file.tree):
        if isinstance(node, ast.ClassDef):
            span = (node.end_lineno or node.lineno) - node.lineno + 1
            if span > MAX_CLASS_LINES:
                violations.append(Violation(file, node.lineno, "7", f"class '{node.name}' is {span} lines (max {MAX_CLASS_LINES})"))
    return _keep_unexcused(file, 7, violations)


def test_rule_7_small_classes() -> None:
    violations = [v for file in SOURCE_FILES for v in _check_rule_7_classes(file)]
    assert not violations, _format(violations)


def test_rule_7_small_packages() -> None:
    problems = []
    for directory in sorted({f.path.parent for f in SOURCE_FILES}):
        if not (directory / "__init__.py").exists():
            continue
        modules = [p for p in directory.glob("*.py") if p.name != "__init__.py"]
        if len(modules) > MAX_MODULES_PER_PACKAGE:
            problems.append(f"{directory} has {len(modules)} modules (max {MAX_MODULES_PER_PACKAGE})")
    assert not problems, "\n".join(problems)


# --------------------------------------------------------------------------
# P-014 rule 8 — at most two instance variables per class
# --------------------------------------------------------------------------


def _self_assigned_attribute_names(node: ast.ClassDef) -> frozenset[str]:
    initializer = next((item for item in node.body if isinstance(item, ast.FunctionDef) and item.name == "__init__"), None)
    if initializer is None:
        return frozenset()
    return frozenset(_self_attribute_targets(statement) for statement in initializer.body if _is_self_attribute_assignment(statement)) - {None}


def _is_self_attribute_assignment(statement: ast.AST) -> bool:
    target = _assignment_target(statement)
    if not isinstance(target, ast.Attribute):
        return False
    return isinstance(target.value, ast.Name) and target.value.id == "self"


def _assignment_target(statement: ast.AST) -> ast.AST | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        return statement.targets[0]
    if isinstance(statement, ast.AnnAssign):
        return statement.target
    return None


def _self_attribute_targets(statement: ast.AST) -> str | None:
    target = _assignment_target(statement)
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


def _field_count(node: ast.ClassDef) -> int:
    if _is_data_record_class(node):
        return len(_field_annotations(node))
    return len(_self_assigned_attribute_names(node))


def _check_rule_8(file: SourceFile) -> list[Violation]:
    violations = []
    for node in ast.walk(file.tree):
        if isinstance(node, ast.ClassDef):
            count = _field_count(node)
            if count > 2:
                violations.append(Violation(file, node.lineno, "8", f"class '{node.name}' has {count} fields (max 2)"))
    return _keep_unexcused(file, 8, violations)


def test_rule_8_at_most_two_fields() -> None:
    violations = [v for file in SOURCE_FILES for v in _check_rule_8(file)]
    assert not violations, _format(violations)


# --------------------------------------------------------------------------
# P-014 rule 9 — no getters, setters or properties
# --------------------------------------------------------------------------


def _is_property_decorator(decorator: ast.AST) -> bool:
    if isinstance(decorator, ast.Name):
        return decorator.id == "property"
    if isinstance(decorator, ast.Attribute):
        return decorator.attr in ("property", "setter", "getter")
    return False


def _check_rule_9(file: SourceFile) -> list[Violation]:
    violations = []
    for node in ast.walk(file.tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.extend(_check_rule_9_function(file, node))
    return _keep_unexcused(file, 9, violations)


def _check_rule_9_function(file: SourceFile, node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Violation]:
    if any(_is_property_decorator(d) for d in node.decorator_list):
        return [Violation(file, node.lineno, "9", f"'{node.name}' uses @property")]
    if node.name.startswith("get_") or node.name.startswith("set_"):
        return [Violation(file, node.lineno, "9", f"'{node.name}' is a getter/setter by name")]
    return []


def test_rule_9_no_getters_or_setters() -> None:
    violations = [v for file in SOURCE_FILES for v in _check_rule_9(file)]
    assert not violations, _format(violations)


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------


def _format(violations: list[Violation]) -> str:
    lines = [f"{v.file.path}:{v.line} [rule {v.rule}, layer={v.file.layer}] {v.message}" for v in violations]
    return "\n" + "\n".join(lines)
