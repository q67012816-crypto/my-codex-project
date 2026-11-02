"""Download and restore the verified model archive; uses only Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import urllib.request
import uuid

BASE='https://raw.githubusercontent.com/q67012816-crypto/my-codex-project/main/experiments/cora-reproduction/2026-11-02/'
EXPECTED='74e9cd0c9b191ccad831d8cb566ada375215779d2b771fb31ea1429db495f263'
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--parts-dir',type=Path,help='Use already downloaded parts, without downloading them again')
args=parser.parse_args();dest=args.output_dir.resolve();dest.mkdir(parents=True,exist_ok=True)
manifest_path=Path(__file__).resolve().parent/'model-parts.json'
if manifest_path.exists():manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
else:
    with urllib.request.urlopen(BASE+'model-parts.json',timeout=120) as response:manifest=json.load(response)
assert manifest['sha256']==EXPECTED
target=dest/manifest['original_filename']
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
if target.exists():
    if sha(target)!=EXPECTED:raise RuntimeError('Existing output has a different hash; choose another output directory')
    print('Already verified:',target);raise SystemExit(0)
parts=args.parts_dir.resolve() if args.parts_dir else dest/'model-parts'
parts.mkdir(parents=True,exist_ok=True)
for part in manifest['parts']:
    name=part['name'];assert re.fullmatch(r'models\.zip\.part-\d{3}',name)
    path=parts/name
    if not path.exists():
        temporary=parts/(name+'.'+uuid.uuid4().hex+'.download')
        print('Downloading',name,flush=True)
        with urllib.request.urlopen(BASE+'models/'+name,timeout=120) as response,temporary.open('xb') as stream:
            while True:
                block=response.read(1024*1024)
                if not block:break
                stream.write(block)
        if sha(temporary)!=part['sha256']:raise RuntimeError('Downloaded part hash mismatch: '+name)
        temporary.rename(path)
    if path.stat().st_size!=part['bytes'] or sha(path)!=part['sha256']:raise RuntimeError('Part verification failed: '+name)
temporary=dest/('models-'+uuid.uuid4().hex+'.zip.tmp')
h=hashlib.sha256();count=0
with temporary.open('xb') as out:
    for part in manifest['parts']:
        with (parts/part['name']).open('rb') as stream:
            while True:
                block=stream.read(1024*1024)
                if not block:break
                out.write(block);h.update(block);count+=len(block)
assert count==manifest['bytes'] and h.hexdigest()==EXPECTED,'Combined archive verification failed'
temporary.rename(target)
print('Verified model archive:',target)
