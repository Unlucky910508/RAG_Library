"""Render record fields as text, and assemble several of them into one
block according to a config recipe.

Shared because both ends of the pipeline assemble text out of the same
records from the same recipes: parse_chunks.py builds the text that gets
embedded, search.py builds the text that gets returned to the caller.
Keeping one vocabulary means a chunk_type can be reshaped in
config.CHUNK_FIELDS without either side growing its own private notion of
what a field looks like as text.

One function per JSON field. A new field needs an entry here; recombining
existing ones is a config edit only.
"""


def extract_name(record):
    return record.get("name")


def extract_kind(record):
    return record.get("kind")


def extract_explanation(record):
    return record.get("explanation")


def extract_doc(record):
    return record.get("doc")


def extract_code(record):
    return record.get("code")


def extract_heading_path(record):
    """The chain of headings a documentation section sits under, joined so
    it reads as the trail it is."""
    path = record.get("heading_path")
    return " > ".join(path) if path else None


def extract_section_text(record):
    """The body of a documentation section.

    Deliberately not called "doc": that field holds a docstring read off a
    library object, and this holds prose from a documentation file. Sharing
    one name would make every recipe naming it apply to both, which is not
    something a recipe can then take back."""
    return record.get("section_text")


def extract_source(record):
    source = record.get("source")
    return f"source: {source}" if source else None


def extract_license(record):
    license_name = record.get("license")
    return f"license: {license_name}" if license_name else None


def extract_signatures(record):
    signatures = record.get("signatures")
    return "\n".join(signatures) if signatures else None


def extract_parameter_names(record):
    names = sorted({p["name"] for overload in record.get("parameters", []) for p in overload})
    return "parameters: " + ", ".join(names) if names else None


def extract_apis_used(record):
    apis = record.get("apis_used")
    return "uses: " + ", ".join(apis) if apis else None


def extract_enum_members(record):
    members = record.get("members")
    return "members: " + ", ".join(m["name"] for m in members) if members else None


def extract_enum_of(record):
    enum_of = record.get("enum_of")
    return f"belongs to enum: {enum_of}" if enum_of else None


def extract_value(record):
    return f"value: {record['value']}" if "value" in record else None


FIELD_EXTRACTORS = {
    "name": extract_name,
    "kind": extract_kind,
    "explanation": extract_explanation,
    "doc": extract_doc,
    "code": extract_code,
    "heading_path": extract_heading_path,
    "section_text": extract_section_text,
    "source": extract_source,
    "license": extract_license,
    "signatures": extract_signatures,
    "parameter_names": extract_parameter_names,
    "apis_used": extract_apis_used,
    "enum_members": extract_enum_members,
    "enum_of": extract_enum_of,
    "value": extract_value,
}


def build_text(record, field_keys):
    """Concatenate the named fields, skipping ones this record lacks."""
    parts = [FIELD_EXTRACTORS[key](record) for key in field_keys]
    parts = [p for p in parts if p]
    return "\n".join(parts) if parts else None


def has_required_fields(record, required_keys):
    return all(FIELD_EXTRACTORS[key](record) for key in required_keys)
