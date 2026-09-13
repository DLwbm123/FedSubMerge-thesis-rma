"""Exercise GPU admission and failure handling without launching GPU processes."""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import entry


def check(fail=False, continue_after_failure=False):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / 'logs').mkdir()
        budgets = [12500, 12500, 12500, 37000]
        jobs = [dict(name=str(i), output=str(root / str(i)), memory_mib=b) for i, b in enumerate(budgets)]
        config = root / 'jobs.json'
        config.write_text(json.dumps(dict(gpu=3, jobs=jobs, max_concurrent=3, capacity_mib=38000,
                                          continue_after_failure=continue_after_failure)))
        running, launched, peaks = set(), [], []

        class Process:
            def __init__(self, command, **kwargs):
                assert command[-2:] == ['-u', 'entry.py']
                self.i = int(kwargs['env']['JOB_SLOT'])
                self.pid = self.i + 100
                self.polls = 0
                running.add(self.i)
                launched.append(self.i)
                assert sum(budgets[i] for i in running) <= 38000
                peaks.append(len(running))

            def poll(self):
                self.polls += 1
                if self.polls < 2:
                    return None
                running.remove(self.i)
                if fail and self.i == 0:
                    return 1
                output = Path(jobs[self.i]['output'])
                output.mkdir()
                (output / 'result.json').write_text('{"status":"complete"}')
                return 0

        with patch.dict(os.environ, {'JOB_CONFIG': str(config)}, clear=True), \
                patch.object(entry.subprocess, 'Popen', Process), \
                patch.object(entry.subprocess, 'check_output', return_value='40445'), \
                patch.object(entry.time, 'sleep'):
            try:
                entry.main()
            except SystemExit:
                assert fail
        state = json.loads((root / 'queue.json').read_text())
        assert max(peaks) == 3 and not running
        stopped = fail and not continue_after_failure
        assert launched == ([0, 1, 2] if stopped else [0, 1, 2, 3])
        assert state['jobs'][-1]['status'] == ('blocked_by_failure' if stopped else 'complete')


if __name__ == '__main__':
    check()
    check(fail=True)
    check(fail=True, continue_after_failure=True)
    print('PASS: three small jobs, exclusive large job, configured failure handling')
