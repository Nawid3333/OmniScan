"""SHA-256 of every file in a series' export output dir (G3 bit-identicality evidence)."""

import hashlib
import json
import sys
from pathlib import Path

root = Path("V:/OmniScan/data/output")
series = sys.argv[1] if len(sys.argv) > 1 else "PepperCarrotKR"
out = root / series / "Episode 06"
hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()}
print(out)
print(json.dumps(hashes, indent=1))
