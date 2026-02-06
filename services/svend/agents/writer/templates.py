"""
Custom Template System

Users define their own document templates with:
- Sections (required/optional)
- Placeholders with constraints
- Length targets per section
- Quality requirements
"""

import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SectionSpec:
    """Specification for a document section."""
    name: str
    description: str = ""  # What should go here
    required: bool = True
    min_words: int = 0
    max_words: int = 0  # 0 = no limit
    placeholders: list[str] = field(default_factory=list)  # e.g., ["company_name", "budget"]
    example: str = ""  # Example content for this section


@dataclass
class TemplateSpec:
    """Full document template specification."""
    name: str
    description: str
    sections: list[SectionSpec]

    # Quality requirements
    target_reading_level: float = 0  # Flesch-Kincaid grade (0 = no target)
    max_reading_level: float = 0
    tone: str = ""  # e.g., "formal", "conversational"

    # Metadata
    domain: str = ""  # e.g., "grant_proposal", "sales", "technical"
    version: str = "1.0"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "sections": [
                {
                    "name": s.name,
                    "description": s.description,
                    "required": s.required,
                    "min_words": s.min_words,
                    "max_words": s.max_words,
                    "placeholders": s.placeholders,
                    "example": s.example,
                }
                for s in self.sections
            ],
            "target_reading_level": self.target_reading_level,
            "max_reading_level": self.max_reading_level,
            "tone": self.tone,
            "domain": self.domain,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateSpec":
        sections = [
            SectionSpec(
                name=s["name"],
                description=s.get("description", ""),
                required=s.get("required", True),
                min_words=s.get("min_words", 0),
                max_words=s.get("max_words", 0),
                placeholders=s.get("placeholders", []),
                example=s.get("example", ""),
            )
            for s in data.get("sections", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            sections=sections,
            target_reading_level=data.get("target_reading_level", 0),
            max_reading_level=data.get("max_reading_level", 0),
            tone=data.get("tone", ""),
            domain=data.get("domain", ""),
            version=data.get("version", "1.0"),
        )

    def to_prompt(self, variables: dict[str, str] = None) -> str:
        """Convert template to LLM prompt."""
        variables = variables or {}

        lines = [
            f"Write a {self.name}.",
            "",
            f"Description: {self.description}",
            "",
        ]

        if self.tone:
            lines.append(f"Tone: {self.tone}")
        if self.target_reading_level:
            lines.append(f"Target reading level: Grade {self.target_reading_level}")

        lines.extend(["", "## Required Sections", ""])

        for section in self.sections:
            if not section.required:
                continue

            section_desc = f"### {section.name}"
            if section.description:
                section_desc += f"\n{section.description}"

            constraints = []
            if section.min_words:
                constraints.append(f"minimum {section.min_words} words")
            if section.max_words:
                constraints.append(f"maximum {section.max_words} words")
            if constraints:
                section_desc += f"\n({', '.join(constraints)})"

            if section.example:
                section_desc += f"\n\nExample:\n{section.example}"

            lines.append(section_desc)
            lines.append("")

        optional = [s for s in self.sections if not s.required]
        if optional:
            lines.extend(["## Optional Sections", ""])
            for section in optional:
                lines.append(f"- {section.name}: {section.description}")
            lines.append("")

        # Add variables
        if variables:
            lines.extend(["## Variables to use", ""])
            for key, value in variables.items():
                lines.append(f"- {key}: {value}")

        return "\n".join(lines)


class TemplateManager:
    """Manage user's custom templates."""

    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path.home() / ".svend" / "templates"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, TemplateSpec] = {}

    def save(self, template: TemplateSpec) -> Path:
        """Save template to storage."""
        filename = re.sub(r'[^\w\-]', '_', template.name.lower()) + ".json"
        path = self.storage_dir / filename
        path.write_text(json.dumps(template.to_dict(), indent=2))
        self._cache[template.name] = template
        return path

    def load(self, name: str) -> TemplateSpec | None:
        """Load template by name."""
        if name in self._cache:
            return self._cache[name]

        # Try exact filename
        filename = re.sub(r'[^\w\-]', '_', name.lower()) + ".json"
        path = self.storage_dir / filename
        if path.exists():
            data = json.loads(path.read_text())
            template = TemplateSpec.from_dict(data)
            self._cache[name] = template
            return template

        # Search by name
        for path in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                if data.get("name", "").lower() == name.lower():
                    template = TemplateSpec.from_dict(data)
                    self._cache[name] = template
                    return template
            except (json.JSONDecodeError, KeyError):
                continue

        return None

    def list_templates(self) -> list[str]:
        """List all available templates."""
        templates = []
        for path in self.storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                templates.append(data.get("name", path.stem))
            except (json.JSONDecodeError, KeyError):
                continue
        return templates

    def delete(self, name: str) -> bool:
        """Delete a template."""
        filename = re.sub(r'[^\w\-]', '_', name.lower()) + ".json"
        path = self.storage_dir / filename
        if path.exists():
            path.unlink()
            self._cache.pop(name, None)
            return True
        return False


# Built-in templates
BUILTIN_TEMPLATES = {
    "grant_proposal": TemplateSpec(
        name="Grant Proposal",
        description="Funding request for research or project",
        domain="grant_proposal",
        tone="formal, persuasive",
        target_reading_level=12,
        sections=[
            SectionSpec(
                name="Executive Summary",
                description="Brief overview of the proposal, problem, and requested funding",
                min_words=150,
                max_words=300,
            ),
            SectionSpec(
                name="Problem Statement",
                description="Clear description of the problem being addressed",
                min_words=200,
                max_words=500,
            ),
            SectionSpec(
                name="Proposed Solution",
                description="Your approach to solving the problem",
                min_words=300,
                max_words=800,
            ),
            SectionSpec(
                name="Methodology",
                description="How you will implement the solution",
                min_words=200,
                max_words=600,
            ),
            SectionSpec(
                name="Timeline",
                description="Project milestones and schedule",
                min_words=100,
                max_words=300,
            ),
            SectionSpec(
                name="Budget",
                description="Itemized budget with justification",
                min_words=100,
                max_words=400,
            ),
            SectionSpec(
                name="Expected Outcomes",
                description="Measurable results and impact",
                min_words=150,
                max_words=400,
            ),
            SectionSpec(
                name="Team Qualifications",
                description="Why your team is qualified",
                min_words=100,
                max_words=300,
                required=False,
            ),
        ],
    ),

    "sales_proposal": TemplateSpec(
        name="Sales Proposal",
        description="Business proposal to potential client",
        domain="sales",
        tone="professional, confident, solution-focused",
        target_reading_level=10,
        sections=[
            SectionSpec(
                name="Executive Summary",
                description="Overview of how you'll solve their problem",
                min_words=100,
                max_words=200,
            ),
            SectionSpec(
                name="Understanding Your Needs",
                description="Demonstrate you understand their challenges",
                min_words=150,
                max_words=300,
            ),
            SectionSpec(
                name="Proposed Solution",
                description="Your product/service and how it helps",
                min_words=200,
                max_words=500,
            ),
            SectionSpec(
                name="Why Us",
                description="Your differentiators and credentials",
                min_words=100,
                max_words=300,
            ),
            SectionSpec(
                name="Investment",
                description="Pricing and payment terms",
                min_words=50,
                max_words=200,
            ),
            SectionSpec(
                name="Next Steps",
                description="Clear call to action",
                min_words=50,
                max_words=150,
            ),
        ],
    ),

    "technical_spec": TemplateSpec(
        name="Technical Specification",
        description="Technical document for software/system",
        domain="technical",
        tone="precise, clear, technical",
        target_reading_level=14,
        sections=[
            SectionSpec(
                name="Overview",
                description="High-level description of the system",
                min_words=100,
                max_words=300,
            ),
            SectionSpec(
                name="Requirements",
                description="Functional and non-functional requirements",
                min_words=200,
                max_words=600,
            ),
            SectionSpec(
                name="Architecture",
                description="System architecture and components",
                min_words=200,
                max_words=800,
            ),
            SectionSpec(
                name="Data Model",
                description="Data structures and storage",
                min_words=100,
                max_words=500,
                required=False,
            ),
            SectionSpec(
                name="API Specification",
                description="Endpoints and interfaces",
                min_words=100,
                max_words=600,
                required=False,
            ),
            SectionSpec(
                name="Security Considerations",
                description="Security requirements and measures",
                min_words=100,
                max_words=400,
            ),
            SectionSpec(
                name="Testing Strategy",
                description="How the system will be tested",
                min_words=100,
                max_words=300,
                required=False,
            ),
        ],
    ),
}


def get_builtin_template(name: str) -> TemplateSpec | None:
    """Get a built-in template by name."""
    return BUILTIN_TEMPLATES.get(name.lower().replace(" ", "_"))
