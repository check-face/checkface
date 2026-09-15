"""Bound model parallelism independently of HTTP workers; no torch import needed."""
import logging
import os
from pathlib import Path, PurePosixPath

DEFAULT_MAX = 8
OVERRIDE_MAX = 64
_status = None


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ''


def quota_limits(read=_read):
    """Read visible cgroup v1/v2 CPU quotas, including ancestor limits.

    Resolve mount roots so both host processes and cgroup-namespaced containers
    work. CPU affinity supplies the cpuset bound separately. Hidden host ancestor
    quotas cannot be discovered from inside a container.
    """
    memberships = {}
    for line in read('/proc/self/cgroup').splitlines():
        parts = line.split(':', 2)
        if len(parts) == 3:
            for controller in parts[1].split(','):
                memberships[controller] = PurePosixPath(parts[2])
    limits = []
    for line in read('/proc/self/mountinfo').splitlines():
        fields = line.split()
        try:
            sep = fields.index('-')
            kind, options = fields[sep + 1], fields[sep + 3].split(',')
            controller = '' if kind == 'cgroup2' else 'cpu'
            if kind not in ('cgroup', 'cgroup2') or (kind == 'cgroup' and 'cpu' not in options):
                continue
            root, mount = PurePosixPath(fields[3]), PurePosixPath(fields[4])
            relative = memberships[controller].relative_to(root)
            # Some container proc views expose an out-of-namespace ../ path.
            # Only the mounted root is safely visible in that case.
            if '..' in relative.parts:
                relative = PurePosixPath('.')
            current = mount / relative
            while True:
                try:
                    if kind == 'cgroup2':
                        quota, period = read(str(current / 'cpu.max')).split()
                    else:
                        quota = read(str(current / 'cpu.cfs_quota_us'))
                        period = read(str(current / 'cpu.cfs_period_us'))
                    quota, period = int(quota), int(period)
                    if quota > 0 and period > 0:
                        limits.append(max(1, quota // period))
                except ValueError:
                    pass  # Unlimited, unavailable or malformed kernel metadata.
                if current == mount:
                    break
                current = current.parent
        except (ValueError, KeyError, IndexError):
            continue
    return limits


def resolve_threads(environ=None, cpu_count=None, affinity=None, quotas=None):
    env = os.environ if environ is None else environ
    count = (os.cpu_count() or 1) if cpu_count is None else cpu_count
    if affinity is None:
        try:
            affinity = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = count
    limits = quota_limits() if quotas is None else quotas
    available = max(1, min([max(1, count), max(1, affinity)] + list(limits)))
    raw = env.get('CHECKFACE_CPU_THREADS', 'auto')
    if raw == 'auto':
        requested = min(DEFAULT_MAX, available)
    elif not raw.isascii() or not raw.isdecimal() or not 1 <= int(raw) <= OVERRIDE_MAX:
        raise ValueError(f'CHECKFACE_CPU_THREADS must be auto or an integer from 1 to {OVERRIDE_MAX}')
    else:
        requested = int(raw)
    return {'requested': raw, 'available_cpus': available,
            'effective_threads': min(requested, available),
            'default_max': DEFAULT_MAX, 'limited_by_allocation': requested > available}


def configure_torch(torch):
    global _status
    configuration = resolve_threads()
    torch.set_num_threads(configuration['effective_threads'])
    configuration['effective_threads'] = torch.get_num_threads()
    _status = configuration
    logging.getLogger(__name__).warning('CPU model thread configuration: %s', configuration)
    return dict(configuration)


def thread_status():
    return dict(_status) if _status is not None else None
