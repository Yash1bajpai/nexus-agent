# Cloud C-stage fixture provenance (2026-11-21)

Byte-reuse vs the reviewed local Stage C harness (`ci/stage_c_driver.py`,
experiment branch tip b2f3ace, oracle `ci/oracle_stage_c.py`):

## C4 — fully byte-reused
- `c4/mathx.py` == local C4 `mathx.py` bytes: `def subtract(a, b):\n    return a + b\n`
- `c4/test_mathx.py` == local C4 `test_mathx.py` bytes (single failing assert
  `subtract(10, 4) == 6`).
- `c4/prompt.txt` == local C4 prompt: "Make the tests in test_mathx.py pass."

## C3 — impl and prompt byte-reused; test file ADAPTED
- `c3/divider.py` == local C3 `divider.py` bytes: `def divide(a, b):\n    return a / b\n`
- `c3/prompt.txt` == local C3 prompt: "Fix divide in divider.py so it
  returns None when b is 0, then run the tests."
- `c3/test_divider.py` is an ADAPTATION of the local C3 `test_divider.py`,
  required by the approved cloud C-stage spec (section 2): "3 tests, one
  covering divide-by-zero expecting None". The local file has 2 tests and
  its zero test asserts `divide(10, 0) == 0`, which contradicts the prompt's
  required `None` return; a stub that reproduces pytest faithfully would
  then show red even after a correct repair, breaking the C3 probe's
  green-after-repair path. Adapted diff:

```diff
 from divider import divide
 def test_normal():
     assert divide(10, 2) == 5
 def test_zero_division():
-    assert divide(10, 0) == 0
+    assert divide(10, 0) is None
+def test_negative():
+    assert divide(-10, 2) == -5
```
