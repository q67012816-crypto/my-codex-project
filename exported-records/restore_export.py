"""Reassemble and verify the complete research export; uses Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def restore(base, destination):
    manifest = json.loads((base / 'PARTS_MANIFEST.json').read_text())
    if destination.exists():
        if destination.stat().st_size != manifest['archive_bytes'] or sha256(destination) != manifest['archive_sha256']:
            raise ValueError(f'Existing output differs; choose another --output: {destination}')
        print(f'Existing archive verified: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    complete = hashlib.sha256()
    total = 0
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.cgt-restoring-', delete=False) as output:
            temporary = Path(output.name)
            for index, part in enumerate(manifest['parts'], 1):
                path = (base / part['file']).resolve()
                if not path.is_relative_to(base.resolve()):
                    raise ValueError('Part path escapes export directory')
                checksum = hashlib.sha256(); size = 0
                with path.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        checksum.update(block); complete.update(block); output.write(block); size += len(block)
                if size != part['bytes'] or checksum.hexdigest() != part['sha256']:
                    raise ValueError(f'Invalid or incomplete part: {part["file"]}')
                total += size
                print(f'Verified part {index}/{manifest["part_count"]}', flush=True)
        if total != manifest['archive_bytes'] or complete.hexdigest() != manifest['archive_sha256']:
            raise ValueError('Reassembled archive SHA256 does not match')
        temporary.replace(destination)
        print(f'Complete archive verified: {destination}', flush=True)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def extract(archive, destination):
    destination = destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('Extraction directory must be new or empty')
    with tarfile.open(archive, 'r:gz') as source:
        members = source.getmembers()
        for member in members:
            path = (destination / member.name).resolve()
            if not path.is_relative_to(destination) or not (member.isfile() or member.isdir()):
                raise ValueError(f'Unsupported archive member: {member.name}')
        destination.mkdir(parents=True, exist_ok=True)
        source.extractall(destination, members=members)
    print(f'Extracted into {destination}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Destination .tar.gz; default: beside this script')
    parser.add_argument('--extract', type=Path, help='Also extract into a new/empty directory')
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    output = args.output or base / 'cgt-complete-records.tar.gz'
    restore(base, output)
    if args.extract:
        extract(output, args.extract)


if __name__ == '__main__':
    main()
