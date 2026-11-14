from pathlib import Path

SKILL_PATH = Path(__file__).parent / "SKILL.md"


def get_skill_path() -> Path:
    return SKILL_PATH


def get_skill_markdown() -> str:
    return SKILL_PATH.read_text(encoding="utf-8")


get_skill_text = get_skill_markdown


def install_skill(target_dir: str | Path) -> Path:
    dest_dir = Path(target_dir).expanduser().resolve() / "chandra"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / "SKILL.md"
    dest_file.write_text(get_skill_markdown(), encoding="utf-8")
    return dest_file
