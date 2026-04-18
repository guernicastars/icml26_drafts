import sys
from pathlib import Path

_V1 = Path(__file__).resolve().parent.parent.parent / "uet-validation"
_V2 = Path(__file__).resolve().parent.parent.parent / "uet-validation-v2"
for p in (_V1, _V2):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
