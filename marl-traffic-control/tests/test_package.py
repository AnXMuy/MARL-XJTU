import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from traffic_signal_marl import __version__


class PackageTest(unittest.TestCase):
    def test_package_version_is_set(self):
        self.assertEqual(__version__, "0.1.0")
