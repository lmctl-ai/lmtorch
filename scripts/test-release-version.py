#!/usr/bin/env python3
import shutil, subprocess, tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as d:
 repo=Path(d)/'repo'; shutil.copytree(root,repo,ignore=shutil.ignore_patterns('.git','build','build-dev','third_party'))
 major,minor,patch=map(int,(repo/'VERSION').read_text().strip().split('.'))
 calc=repo/'scripts/derive-release-version.sh'
 assert subprocess.check_output([calc,'1'],text=True).strip()==f'{major}.{minor}.{patch+1}'
 assert subprocess.check_output([calc,'40'],text=True).strip()==f'{major}.{minor}.{patch+40}'
 expected=f'{major}.{minor}.{patch+40}'
 subprocess.check_call([repo/'scripts/stamp-version.sh',expected])
 assert (repo/'VERSION').read_text().strip()==expected
 assert subprocess.check_output([calc,'--diagnostic'],text=True).strip()==expected
 assert 'file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" LMTORCH_RELEASE_VERSION' in (repo/'CMakeLists.txt').read_text()
 print('PASS lmtorch run-derived release version and stamp')
