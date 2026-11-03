import json
from pathlib import Path
import tempfile
import unittest

from tool_cache import check_local_cache


class LocalToolCache(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="lmtorch-cache-control-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.schema = self.root / "host.json"
        self.expected = {"dsl.schema": 2, "name": "lmtorch", "types": {}}
        self.schema.write_text(json.dumps(self.expected), encoding="utf-8")
        self.cache = self.root / "cache"

    def metadata(self, value):
        folder = self.cache / "schemas"
        folder.mkdir(parents=True)
        (folder / ("a" * 64 + ".json")).write_text(json.dumps(value), encoding="utf-8")

    def test_absent_cache_and_local_schema_metadata_are_allowed(self):
        check_local_cache(self.cache, self.schema)
        self.metadata(self.expected)
        check_local_cache(self.cache, self.schema)

    def test_a_downloaded_executable_is_refused_even_with_valid_metadata(self):
        self.metadata(self.expected)
        executable = self.cache / "tools" / "lmtorch.exe"
        executable.parent.mkdir()
        executable.write_bytes(b"downloaded executable control")
        with self.assertRaisesRegex(AssertionError, "unexpected tool-cache artifact"):
            check_local_cache(self.cache, self.schema)

    def test_other_schema_or_disguised_binary_is_refused(self):
        self.metadata({"name": "wrong-host"})
        with self.assertRaises(AssertionError):
            check_local_cache(self.cache, self.schema)
        next((self.cache / "schemas").iterdir()).write_bytes(b"not JSON")
        with self.assertRaises(AssertionError):
            check_local_cache(self.cache, self.schema)


if __name__ == "__main__":
    unittest.main()
