import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from provenance import matches_recorded_sha256,portable_sha256


class ProvenanceTests(unittest.TestCase):
    def test_only_text_line_endings_are_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"input.csv"
            lf=b"item,value\na,1\n"
            crlf=lf.replace(b"\n",b"\r\n")
            path.write_bytes(lf)
            expected=hashlib.sha256(crlf).hexdigest()
            self.assertTrue(matches_recorded_sha256(path,expected))
            self.assertEqual(portable_sha256(path),hashlib.sha256(lf).hexdigest())
            path.write_bytes(lf.replace(b"a,1",b"a,2"))
            self.assertFalse(matches_recorded_sha256(path,expected))

    def test_binary_hashes_must_match_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"model.joblib"
            path.write_bytes(b"a\nb")
            self.assertFalse(matches_recorded_sha256(path,hashlib.sha256(b"a\r\nb").hexdigest()))


if __name__=="__main__":
    unittest.main()
