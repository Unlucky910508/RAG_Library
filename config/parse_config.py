from pathlib import Path

parsed_module_name = "pycolmap"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def api_jsonl_path(version):
    return DATA_DIR / f"{parsed_module_name}_{version}_api.jsonl"