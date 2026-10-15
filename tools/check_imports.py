"""Static import check (no GPU, no dependencies): resolve every dlmrt import, relative or absolute, to its module file
and check that the imported names exist there, and that every attribute read off an imported dlmrt module
(`gk.post_attn`) is defined in it. Covers dlmrt/, tests/, examples/, benchmarks/ and tools/.

usage: python tools/check_imports.py [repo root]
"""
import ast
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent)
PKG = ROOT / "dlmrt"


def module_file(dotted):
    p = ROOT.joinpath(*dotted.split("."))
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    return None


def top_names(path):
    tree = ast.parse(path.read_text())
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {n.id for t in node.targets for n in ast.walk(t)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names |= {(a.asname or a.name).split(".")[0] for a in node.names}
    return names


errors = 0
files = [*PKG.rglob("*.py"), *(ROOT / "tests").glob("*.py"), *(ROOT / "examples").rglob("*.py"),
         *(ROOT / "benchmarks").glob("*.py"), *(ROOT / "tools").glob("*.py")]
for f in files:
    if "csrc" in f.parts:
        continue
    pkg = ".".join(f.relative_to(ROOT).with_suffix("").parts)
    tree = ast.parse(f.read_text())
    modules = {}  # local name of an imported dlmrt submodule -> its file
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            base = pkg.split(".")[:-node.level]
            dotted = ".".join(base + ([node.module] if node.module else []))
        else:
            dotted = node.module or ""
        if not dotted.startswith("dlmrt"):
            continue
        target = module_file(dotted)
        if target is None:
            print(f"{f}:{node.lineno}: no module {dotted}")
            errors += 1
            continue
        names = top_names(target)
        for a in node.names:
            if a.name in names:
                continue
            if module_file(f"{dotted}.{a.name}"):  # a submodule
                modules[a.asname or a.name] = module_file(f"{dotted}.{a.name}")
                continue
            print(f"{f}:{node.lineno}: {dotted} has no {a.name}")
            errors += 1
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in modules \
                and node.attr not in top_names(modules[node.value.id]):
            print(f"{f}:{node.lineno}: {modules[node.value.id].relative_to(ROOT)} has no {node.attr}")
            errors += 1
print("import errors:", errors)

sys.exit(1 if errors else 0)
