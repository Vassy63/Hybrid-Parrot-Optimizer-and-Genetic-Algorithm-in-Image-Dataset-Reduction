import os
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from poga_demo.jobs import JobManager, read_json, write_json
from poga_demo.settings import DemoSettings


class JobTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows file sharing semantics')
    def test_status_update_survives_a_temporary_windows_read_lock(self):
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'status.json'
            write_json(path, {'state': 'starting'})
            # Open like a reader that permits reading/writing but denies deletion/rename.
            handle = kernel.CreateFileW(str(path), 0x80000000, 3, None, 3, 0, None)
            if handle == wintypes.HANDLE(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            release = threading.Timer(.12, kernel.CloseHandle, args=(handle,))
            release.start()
            try:
                write_json(path, {'state': 'running', 'message': 'Tiến độ'})
            finally:
                release.join()
            self.assertEqual(read_json(path), {'state': 'running', 'message': 'Tiến độ'})
            self.assertEqual(list(Path(temp).glob('*.tmp')), [])

    def test_failed_replace_preserves_status_and_cleans_temporary_files(self):
        for code in (5, 32, 33, 19):
            with self.subTest(winerror=code), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / 'status.json'
                write_json(path, {'state': 'running'})
                error = PermissionError('destination unavailable')
                error.winerror = code
                with patch('poga_demo.jobs.os.replace', side_effect=error), \
                     patch('poga_demo.jobs.time.sleep') as sleep:
                    with self.assertRaises(PermissionError):
                        write_json(path, {'state': 'succeeded'})
                self.assertEqual(read_json(path), {'state': 'running'})
                self.assertEqual(list(Path(temp).glob('*.tmp')), [])
                if code == 19:
                    sleep.assert_not_called()
                else:
                    self.assertGreater(sleep.call_count, 0)
                    self.assertLess(sleep.call_count, 20)

    def test_start_failure_does_not_leave_a_permanently_starting_run(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = JobManager(Path(temp))
            with patch('poga_demo.jobs.subprocess.Popen', side_effect=OSError('process unavailable')):
                with self.assertRaises(OSError):
                    manager.start(DemoSettings(), mode='diagnose')
            state = read_json(manager.folders()[0] / 'status.json')
            self.assertEqual(state['state'], 'failed')

    def test_diagnostics_run_in_a_real_subprocess_and_get_distinct_folders(self):
        with tempfile.TemporaryDirectory() as temp:
            manager = JobManager(Path(temp))
            ids = []
            for _ in range(2):
                folder = manager.start(DemoSettings(), mode='diagnose')
                ids.append(folder.name)
                deadline = time.monotonic() + 10
                while manager.active() and time.monotonic() < deadline:
                    time.sleep(.05)
                state = read_json(folder / 'status.json')
                self.assertEqual(state['state'], 'succeeded', state)
                capabilities = read_json(folder / 'environment.json')
                self.assertIn('python', capabilities)
                self.assertIn('dependencies', capabilities)
            self.assertNotEqual(ids[0], ids[1])


if __name__ == '__main__':
    unittest.main()
