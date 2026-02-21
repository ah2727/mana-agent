from __future__ import annotations

from dataclasses import dataclass, field

from mana_analyzer.analysis.models import ClassDescriptor, ExportDescriptor


@dataclass(slots=True)
class ParsedModule:
    imports: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    classes: list[ClassDescriptor] = field(default_factory=list)
    constants: list[str] = field(default_factory=list)
    exports: list[ExportDescriptor] = field(default_factory=list)
    data_structures: list[ClassDescriptor] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    import_roots: set[str] = field(default_factory=set)
    parse_mode: str = "full"
