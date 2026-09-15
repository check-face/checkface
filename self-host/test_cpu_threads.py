"""Allocation/configuration contract; these tests perform no inference."""
import unittest
from unittest.mock import Mock, patch
import cpu_threads


class CPUThreadsTests(unittest.TestCase):
    def resolve(self, value='auto', **kwargs):
        args = dict(cpu_count=32, affinity=32, quotas=[])
        args.update(kwargs)
        return cpu_threads.resolve_threads({'CHECKFACE_CPU_THREADS': value}, **args)

    def test_default_is_bounded_and_respects_affinity(self):
        self.assertEqual(self.resolve()['effective_threads'], 8)
        self.assertEqual(self.resolve(affinity=3)['effective_threads'], 3)
        self.assertEqual(self.resolve(cpu_count=1)['effective_threads'], 1)

    def test_nested_quota_is_stricter_than_affinity(self):
        result = self.resolve('12', affinity=16, quotas=[6, 2])
        self.assertEqual(result['effective_threads'], 2)
        self.assertTrue(result['limited_by_allocation'])
        self.assertEqual(result['requested'], '12')

    def test_explicit_override_can_exceed_default_cap(self):
        self.assertEqual(self.resolve('16')['effective_threads'], 16)
        self.assertEqual(self.resolve('1')['effective_threads'], 1)

    def test_invalid_configuration_fails(self):
        for value in ['', '0', '-1', '65', '1.5', ' 2', '2 ', 'AUTO', '٢', 'two']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'CHECKFACE_CPU_THREADS'):
                self.resolve(value)

    def test_no_platform_cpu_detection_uses_one(self):
        with patch.object(cpu_threads.os, 'cpu_count', return_value=None), patch.object(
                cpu_threads.os, 'sched_getaffinity', side_effect=OSError, create=True):
            self.assertEqual(cpu_threads.resolve_threads({}, quotas=[])['effective_threads'], 1)

    def quotas(self, contents):
        return cpu_threads.quota_limits(lambda path: contents.get(path, ''))

    def test_v2_walks_ancestors_and_rounds_down_fractional_quota(self):
        limits = self.quotas({
            '/proc/self/cgroup': '0::/team/job',
            '/proc/self/mountinfo': '1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw',
            '/sys/fs/cgroup/team/job/cpu.max': 'max 100000',
            '/sys/fs/cgroup/team/cpu.max': '250000 100000',
            '/sys/fs/cgroup/cpu.max': '800000 100000',
        })
        self.assertEqual(limits, [2, 8])

    def test_namespaced_v2_mount_root_and_sub_cpu_allocation(self):
        limits = self.quotas({
            '/proc/self/cgroup': '0::/docker/id',
            '/proc/self/mountinfo': '1 0 0:1 /docker/id /sys/fs/cgroup rw - cgroup2 cgroup rw',
            '/sys/fs/cgroup/cpu.max': '50000 100000',
        })
        self.assertEqual(limits, [1])

    def test_outside_namespace_path_stays_inside_visible_mount(self):
        self.assertEqual(self.quotas({
            '/proc/self/cgroup': '0::/../../docker/job',
            '/proc/self/mountinfo': '1 0 0:1 / /cg rw - cgroup2 cgroup rw',
            '/cg/cpu.max': '200000 100000',
        }), [2])

    def test_v1_cpu_mount_ignores_other_controllers(self):
        limits = self.quotas({
            '/proc/self/cgroup': '4:cpu,cpuacct:/job\n3:memory:/job',
            '/proc/self/mountinfo': '\n'.join([
                '1 0 0:1 / /cg/cpu rw - cgroup cgroup rw,cpu,cpuacct',
                '2 0 0:2 / /cg/memory rw - cgroup cgroup rw,memory']),
            '/cg/cpu/job/cpu.cfs_quota_us': '400000',
            '/cg/cpu/job/cpu.cfs_period_us': '100000',
            '/cg/cpu/cpu.cfs_quota_us': '-1',
            '/cg/cpu/cpu.cfs_period_us': '100000',
        })
        self.assertEqual(limits, [4])

    def test_missing_and_malformed_kernel_metadata_is_safe(self):
        self.assertEqual(self.quotas({}), [])
        self.assertEqual(self.quotas({'/proc/self/mountinfo': 'invalid'}), [])
        self.assertEqual(self.quotas({
            '/proc/self/cgroup': '0::/job',
            '/proc/self/mountinfo': '1 0 0:1 / /cg rw - cgroup2 cgroup rw',
            '/cg/job/cpu.max': '1000 0', '/cg/cpu.max': 'garbage'}), [])

    def test_configuration_applies_and_reports_actual_torch_setting(self):
        torch = Mock()
        torch.get_num_threads.return_value = 2
        with patch.dict(cpu_threads.os.environ, {'CHECKFACE_CPU_THREADS': '2'}), patch.object(
                cpu_threads, 'quota_limits', return_value=[]), patch.object(
                cpu_threads.os, 'cpu_count', return_value=8), patch.object(
                cpu_threads.os, 'sched_getaffinity', return_value={0, 1, 2, 3}, create=True):
            result = cpu_threads.configure_torch(torch)
        torch.set_num_threads.assert_called_once_with(2)
        self.assertEqual(cpu_threads.thread_status(), result)
        result['effective_threads'] = 99
        self.assertEqual(cpu_threads.thread_status()['effective_threads'], 2)


if __name__ == '__main__':
    unittest.main()
