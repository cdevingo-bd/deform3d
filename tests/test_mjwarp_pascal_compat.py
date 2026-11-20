"""Source-transform tests; optional installed mujoco-warp 4.0.1 required."""

import ast
from importlib import metadata
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import mjwarp_pascal_compat as compat


class PascalCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            distribution = metadata.distribution("mujoco-warp")
        except metadata.PackageNotFoundError:
            raise unittest.SkipTest("mujoco-warp is not installed")
        if distribution.version != compat.EXPECTED_VERSION:
            raise unittest.SkipTest("source transform requires mujoco-warp 4.0.1")
        cls.original = Path(distribution.locate_file(compat.RELATIVE_SOURCE)).read_text()
        cls.patched = compat.transform(cls.original)

    def test_contact_math_is_identical(self):
        def kernel(source):
            factory = next(node for node in ast.parse(source).body
                           if isinstance(node, ast.FunctionDef) and node.name == "_flex_narrowphase_elem_detect")
            return next(node for node in factory.body if isinstance(node, ast.FunctionDef))

        original, patched = kernel(self.original), kernel(self.patched)
        self.assertEqual(len(original.args.args) - len(patched.args.args), 11)
        # One struct load and twelve aliases are the only new kernel statements.
        recovered = patched.body[:1] + patched.body[14:]
        self.assertEqual([ast.dump(node) for node in original.body],
                         [ast.dump(node) for node in recovered])
        self.assertEqual([a.arg for a in original.args.args if a.arg not in dict(compat.MESH_FIELDS)],
                         [a.arg for a in patched.args.args if a.arg != "mesh_descriptors"])

    def test_other_original_functions_are_unchanged(self):
        original = {node.name: node for node in ast.parse(self.original).body if isinstance(node, ast.FunctionDef)}
        patched = {node.name: node for node in ast.parse(self.patched).body if isinstance(node, ast.FunctionDef)}
        for name, function in original.items():
            if name not in {"_flex_narrowphase_elem_detect", "_detect_elem_geom_candidates"}:
                self.assertEqual(ast.dump(function), ast.dump(patched[name]), name)

    def test_unreviewed_or_already_patched_source_is_rejected(self):
        for source in (self.original + "\n", self.patched):
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                compat.transform(source)

    def test_gyro_math_and_other_functions_are_unchanged(self):
        distribution = metadata.distribution("mujoco-warp")
        original = Path(distribution.locate_file(compat.FORWARD_SOURCE)).read_text()
        patched = compat.transform_forward(original)
        before = {n.name: n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
        after = {n.name: n for n in ast.parse(patched).body if isinstance(n, ast.FunctionDef)}
        a = next(n for n in before['_discrete_free_gyro_kernel'].body if isinstance(n, ast.FunctionDef))
        b = next(n for n in after['_discrete_free_gyro_kernel'].body if isinstance(n, ast.FunctionDef))
        recovered = b.body[:1] + b.body[2 + len(compat.GYRO_FIELDS):]
        self.assertEqual([ast.dump(n) for n in a.body], [ast.dump(n) for n in recovered])
        self.assertEqual(len(a.args.args) - len(b.args.args), len(compat.GYRO_FIELDS) - 1)
        for name, function in before.items():
            if name not in {'_discrete_free_gyro_kernel', '_launch_discrete_free_gyro'}:
                self.assertEqual(ast.dump(function), ast.dump(after[name]), name)
        for invalid in (original + '\n', patched):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                compat.transform_forward(invalid)

    def test_convex_math_and_other_functions_are_unchanged(self):
        distribution = metadata.distribution('mujoco-warp')
        original = Path(distribution.locate_file(compat.CONVEX_SOURCE)).read_text()
        patched = compat.transform_convex(original)
        before = {n.name: n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
        after = {n.name: n for n in ast.parse(patched).body if isinstance(n, ast.FunctionDef)}
        a = next(n for n in before['ccd_kernel_builder'].body if isinstance(n, ast.FunctionDef) and n.name == 'ccd_kernel')
        b = next(n for n in after['ccd_kernel_builder'].body if isinstance(n, ast.FunctionDef) and n.name == 'ccd_kernel')
        self.assertEqual([ast.dump(n) for n in a.body],
                         [ast.dump(n) for n in b.body[:1] + b.body[2+len(compat.CONVEX_FIELDS):]])
        self.assertEqual(len(a.args.args)-len(b.args.args), len(compat.CONVEX_FIELDS)-1)
        for name, function in before.items():
            if name not in {'ccd_kernel_builder', 'convex_narrowphase'}:
                self.assertEqual(ast.dump(function), ast.dump(after[name]), name)
        for invalid in (original+'\n', patched):
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                compat.transform_convex(invalid)


if __name__ == "__main__":
    unittest.main()
