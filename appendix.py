import ast
from pathlib import Path


def humanize_filename(filename):
    name = filename.rsplit(".", 1)[0]
    parts = name.replace("_", " ").split()
    return " ".join(p.capitalize() for p in parts)


def get_module_docstring(file_path):
    source = file_path.read_text(encoding="utf-8")
    module = ast.parse(source)
    return ast.get_docstring(module)


def collect_py_files(base):
    result = {}
    for entry in sorted(base.iterdir()):
        if entry.is_dir():
            py_files = sorted([f for f in entry.iterdir() if f.is_file() and f.suffix == ".py"], key=lambda p: p.name.lower())
            if py_files:
                result[entry.name] = py_files
    return result


def generate_appendix(structure):
    lines = []
    for rel_dir in sorted(structure.keys(), key=lambda d: d.lower()):
        if rel_dir:
            lines.append(f"## {rel_dir}")
        else:
            lines.append("## Top Level")
        for file_path in structure[rel_dir]:
            display_name = humanize_filename(file_path.name)
            lines.append(f"### {display_name}")
            lines.append("```python")
            lines.append(file_path.read_text(encoding="utf-8"))
            lines.append("```")
            lines.append("")
    return "\n".join(lines)


def main():
    base = Path().cwd()
    structure = collect_py_files(base)
    output_file = base / "APPENDIX.md"
    output_file.write_text(generate_appendix(structure), encoding="utf-8")


if __name__ == "__main__":
    main()
