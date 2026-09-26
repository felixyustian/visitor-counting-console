# Third-party notices

This project is MIT licensed (see [LICENSE](LICENSE)). That covers **this code
only**. It bundles no third-party source, but it depends at run time on the
packages in `requirements.txt`, each under its own license, and on model weights
downloaded during setup.

## Copyleft dependencies

Most of the stack is permissive (MIT / BSD / Apache-2.0). These are not:

| component | license | pulled in by | in the pipeline? |
| --- | --- | --- | --- |
| `ultralytics` | **AGPL-3.0-or-later** | `requirements.txt` | **yes** — the YOLO11 detector |
| `ultralytics-thop` | **AGPL-3.0-or-later** | `ultralytics` | yes (transitive) |
| `ultralytics-platform` | **AGPL-3.0-only** | `ultralytics` | yes (transitive) |
| `mutagen` | GPL-2.0-or-later | `yt-dlp` | no — only `download_demo_videos.py` |

Model weights: `google/siglip2-base-patch16-256`, used for the demographics, is
Apache-2.0.

## What this means

**The detector is AGPL, and this is a network service.** Running an AGPL
component inside a service you offer to others can oblige you to release that
service's source under the AGPL, or to buy a commercial license from
Ultralytics. The MIT license on this repository does not and cannot relax that:
a permissive license on your own code says nothing about what it imports.

If that obligation is unwanted, swapping the detector is a contained change.
`humanmonitor/detector.py` is the only file that imports `ultralytics`, behind a
`MultiStreamPersonTracker` that returns boxes and track IDs — any detector that
can produce those will do.

`mutagen` is the easy one: it arrives via `yt-dlp`, which is only used by
`download_demo_videos.py` to fetch sample footage. Drop that one line from
`requirements.txt` and the GPL-2.0 dependency goes with it; nothing in the
counting pipeline touches it.

## Checking this yourself

Licence metadata drifts as packages are upgraded, so verify against your own
environment rather than trusting this table:

```bash
.venv/bin/python - <<'PY'
import re
from importlib.metadata import distributions
for d in sorted(distributions(), key=lambda x: (x.metadata["Name"] or "").lower()):
    m = d.metadata
    lic = m.get("License-Expression") or next(
        (c.rsplit("::", 1)[-1].strip() for c in (m.get_all("Classifier") or [])
         if c.startswith("License ::")), None)
    if not lic:
        raw = (m.get("License") or "").strip()
        lic = raw.splitlines()[0][:40] if raw else "?"
    if re.search(r"\bA?GPL", lic, re.I):
        print(f"  {m['Name']:<24} {lic}")
PY
```

Note that some packages put their whole license text in the `License` field
rather than a short identifier, which is why the snippet prefers
`License-Expression` and the trove classifier, and truncates the fallback.
