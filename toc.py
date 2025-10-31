import ast
from pathlib import Path


def humanize_filename(filename):
    name = filename.rsplit(".", 1)[0]
    parts = name.replace("_", " ").split()
    return " ".join(p.capitalize() for p in parts)


def get_module_docstring(file_path):
    try:
        source = file_path.read_text(encoding="utf-8")
        module = ast.parse(source)
        return ast.get_docstring(module)
    except (OSError, SyntaxError):
        return None


def collect_py_files(base):
    result = {}
    for entry in sorted(base.iterdir()):
        if entry.is_dir():
            py_files = sorted(
                [f for f in entry.iterdir() if f.is_file() and f.suffix == ".py"],
                key=lambda p: p.name.lower(),
            )
            if py_files:
                result[entry.name] = py_files
    return result


def generate_markdown(structure, base):
    lines = []
    for folder, files in structure.items():
        header = " ".join(folder.strip().split("-")[1:]).capitalize()
        lines.append(f"## {header}\n")
        for f in files:
            nice_name = humanize_filename(f.name)
            rel_path = f.relative_to(base).as_posix()
            lines.append(f"* [{nice_name}]({rel_path})")
            doc = get_module_docstring(f)
            if doc:
                d = doc.strip().replace("\n", "\n    ")
                lines.append(f"  - {d}")
        lines.append("")
    return "\n".join(lines).strip()


def main():
    base = Path().cwd()
    structure = collect_py_files(base)
    print(
        """# Algorithm engineering

A repository of Python snippets for different algorithms.

See the [appendix for source code](APPENDIX.md) for all of the algorithms.

""",
    )
    print(generate_markdown(structure, base))


if __name__ == "__main__":
    main()
