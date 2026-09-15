#!/usr/bin/env python3
"""Read-only, bounded cache census. Naming candidates are NOT publication approval."""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

SEED = re.compile(r's(0|[1-9][0-9]*)_([0-9]+)\.(jpg|webp)$')
TEXT = re.compile(r'hash-([0-9a-f]{64})_([0-9]+)\.(jpg|webp)$')


def classify(relative):
    """Recognize only current source's complete outputImages shard convention."""
    parts = Path(relative).parts
    if len(parts) != 4 or parts[0] != 'outputImages':
        return 'unresolved', None
    seed = SEED.fullmatch(parts[-1])
    text = TEXT.fullmatch(parts[-1])
    if seed:
        value, dim = int(seed[1]), int(seed[2])
        if value <= 4294967295 and 10 <= dim <= 1024 and parts[1:3] == (f's{value % 100}', str(value % 10000)):
            return 'seed_path_candidate', {'identity': f's{value}', 'dimension': dim, 'format': seed[3]}
    if text:
        identity, dim = 'hash-' + text[1], int(text[2])
        if 10 <= dim <= 1024 and parts[1:3] == (identity[:7], identity[7:9]):
            return 'text_path_candidate', {'identity': identity, 'dimension': dim, 'format': text[3]}
    if parts[-1].startswith(('GUID', 'MULTILERP_', 'LERP_')):
        return 'private_or_mixed_unresolved', None
    return 'unresolved', None


def scan(roots, seconds=30, max_files=1000000, names=None):
    started = time.monotonic()
    catalogue = {f'hash-{hashlib.sha256(n.encode("utf-8")).hexdigest()}': n for n in (names or [])}
    summary = {'schemaVersion': 1, 'complete': True, 'publicationApproved': False,
               'limits': {'seconds': seconds, 'maxFiles': max_files}, 'roots': {},
               'namesCount': len(catalogue), 'namesCoverage': {},
               'note': 'Metadata only. Strict source-path matches are candidates, not proof of bytes, provenance, or safe publication. Other paths require classification; no content read or production generation.'}
    total = 0
    for label, root in roots:
        counts = collections.Counter(); sizes = collections.Counter(); coverage = collections.defaultdict(set)
        errors = collections.Counter(); stack = [Path(root)]
        while stack:
            if total >= max_files or time.monotonic() - started >= seconds:
                summary['complete'] = False
                break
            directory = stack.pop()
            try:
                if directory.is_symlink():
                    counts['symlink_skipped'] += 1
                    summary['complete'] = False
                    continue
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if total >= max_files or time.monotonic() - started >= seconds:
                            summary['complete'] = False
                            break
                        try:
                            if entry.is_symlink():
                                counts['symlink_skipped'] += 1
                                summary['complete'] = False
                            elif entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                            else:
                                info = entry.stat(follow_symlinks=False)
                                if not stat.S_ISREG(info.st_mode):
                                    counts['special_skipped'] += 1
                                    summary['complete'] = False
                                    continue
                                total += 1
                                kind, identity = classify(Path(entry.path).relative_to(root))
                                counts[kind] += 1; sizes[kind] += info.st_size
                                if identity and identity['identity'] in catalogue:
                                    coverage[catalogue[identity['identity']]].add(f"{identity['dimension']}.{identity['format']}")
                        except OSError as exc:
                            errors[type(exc).__name__] += 1; summary['complete'] = False
            except OSError as exc:
                errors[type(exc).__name__] += 1; summary['complete'] = False
        summary['roots'][label] = {'filesByClass': dict(counts), 'bytesByClass': dict(sizes), 'errors': dict(errors)}
        summary['namesCoverage'][label] = {name: sorted(variants) for name, variants in sorted(coverage.items())}
    summary['filesScanned'] = total
    summary['elapsedSeconds'] = round(time.monotonic() - started, 3)
    if catalogue:
        present = set().union(*(set(c) for c in summary['namesCoverage'].values()))
        summary['namesNotObserved'] = sorted(set(catalogue.values()) - present)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', required=True, help='Unique label=/absolute/cache/root (scan read-only)')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--max-files', type=int, default=1000000)
    parser.add_argument('--names-file', type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or args.max_files <= 0:
        parser.error('limits must be positive')
    roots = []
    for value in args.root:
        label, sep, root = value.partition('=')
        if not sep or not label or not Path(root).is_absolute() or label in [x[0] for x in roots]:
            parser.error('roots need unique labels and absolute paths')
        roots.append((label, root))
    names = None
    if args.names_file:
        data = json.loads(args.names_file.read_text())
        names = data['names'] if isinstance(data, dict) else data
        if not isinstance(names, list) or any(not isinstance(n, str) for n in names):
            parser.error('names must be a string array or object with names array')
    print(json.dumps(scan(roots, args.seconds, args.max_files, names), indent=2))


if __name__ == '__main__':
    main()
