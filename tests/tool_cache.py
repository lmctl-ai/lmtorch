"""Reject downloaded tools while allowing metadata cached from the host under test."""
import json
from pathlib import Path
import re
import sys


def check_local_cache(cache, schema):
    cache = Path(cache)
    if not cache.exists():
        return
    assert cache.is_dir() and not cache.is_symlink(), f"unexpected tool-cache artifact: {cache}"
    expected = json.loads(Path(schema).read_text(encoding="utf-8"))
    for path in cache.rglob("*"):
        relative = path.relative_to(cache)
        assert not path.is_symlink(), f"unexpected tool-cache artifact: {relative}"
        if path.is_dir():
            assert relative == Path("schemas"), f"unexpected tool-cache artifact: {relative}"
            continue
        assert relative.parent == Path("schemas") and re.fullmatch(r"[0-9a-f]{64}\.json", path.name), f"unexpected tool-cache artifact: {relative}"
        try:
            actual = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as error:
            raise AssertionError(f"unexpected tool-cache artifact: {relative}") from error
        assert actual == expected, f"unexpected tool-cache artifact: schema differs from the host under test: {relative}"


if __name__ == "__main__":
    check_local_cache(Path(sys.argv[1]), Path(sys.argv[2]))
    print("PASS tool cache contains only the local host's schema metadata")
