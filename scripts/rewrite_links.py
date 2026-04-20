#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from urllib.parse import unquote

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".avif", ".pdf"}
FENCE_RE = re.compile(r"(```.*?```|~~~.*?~~~)", re.DOTALL)
MD_RE = re.compile(r'(!?)\[([^\]]*)\]\(([^)\n]+)\)')
WIKI_RE = re.compile(r'(!?)\[\[([^\]]+)\]\]')

def is_external(target: str) -> bool:
    prefixes = ("http://", "https://", "mailto:", "obsidian://", "file://", "data:", "tel:")
    return target.startswith(prefixes)

def split_anchor(target: str):
    if target.startswith("#"):
        return "", target[1:]
    if "#" in target:
        p, a = target.split("#", 1)
        return p, a
    return target, ""

def normalize_target(raw_target: str, current_file: Path, vault_root: Path):
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]

    if is_external(target) or target.startswith("#"):
        return None

    path_part, anchor = split_anchor(target)
    path_part = unquote(path_part)

    if path_part.startswith("/"):
        candidate = (vault_root / path_part.lstrip("/")).resolve()
    elif path_part.startswith("notes/"):
        candidate = (vault_root / path_part).resolve()
    else:
        candidate = (current_file.parent / path_part).resolve()
        try:
            candidate.relative_to(vault_root)
        except ValueError:
            candidate = (vault_root / path_part).resolve()

    if candidate.suffix == "":
        if candidate.with_suffix(".md").exists():
            candidate = candidate.with_suffix(".md")

    if not candidate.exists():
        fallback = (vault_root / path_part).resolve()
        if fallback.exists():
            candidate = fallback
        elif fallback.with_suffix(".md").exists():
            candidate = fallback.with_suffix(".md")
        else:
            return None

    rel = candidate.relative_to(vault_root).as_posix()
    if anchor:
        rel = f"{rel}#{anchor}"
    return rel

def markdown_rewrite(match, current_file, vault_root):
    bang, text, target = match.groups()
    new_target = normalize_target(target, current_file, vault_root)
    if not new_target:
        return match.group(0)
    return f'{bang}[{text}]({new_target})'

def wikilink_rewrite(match, current_file, vault_root):
    bang, inner = match.groups()

    if "|" in inner:
        target, alias = inner.split("|", 1)
    else:
        target, alias = inner, ""

    new_target = normalize_target(target, current_file, vault_root)
    if not new_target:
        return match.group(0)

    suffix = Path(new_target.split("#", 1)[0]).suffix.lower()

    if bang == "!":
        if suffix not in IMAGE_EXTS:
            return match.group(0)
        alt = alias if alias else Path(new_target.split("#", 1)[0]).stem
        return f'![{alt}]({new_target})'

    display = alias if alias else Path(new_target.split("#", 1)[0]).stem
    return f'[{display}]({new_target})'

def process_text(text: str, current_file: Path, vault_root: Path):
    parts = FENCE_RE.split(text)
    for i in range(0, len(parts), 2):
        chunk = parts[i]
        chunk = WIKI_RE.sub(lambda m: wikilink_rewrite(m, current_file, vault_root), chunk)
        chunk = MD_RE.sub(lambda m: markdown_rewrite(m, current_file, vault_root), chunk)
        parts[i] = chunk
    return "".join(parts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Vault root, e.g. /Users/you/Documents/YEEZUS")
    parser.add_argument("--scope", default="notes", help="Folder to process under vault root")
    parser.add_argument("--apply", action="store_true", help="Write changes to files")
    args = parser.parse_args()

    vault_root = Path(args.root).resolve()
    scope = (vault_root / args.scope).resolve()

    changed = []
    for md in scope.rglob("*.md"):
        old = md.read_text(encoding="utf-8")
        new = process_text(old, md, vault_root)
        if new != old:
            changed.append(md)
            if args.apply:
                md.write_text(new, encoding="utf-8")

    if args.apply:
        print(f"Updated {len(changed)} files.")
    else:
        print("Files that would change:")
        for f in changed:
            print(f)

if __name__ == "__main__":
    main()