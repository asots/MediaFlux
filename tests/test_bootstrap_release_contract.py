"""Execute the real candidate smoke shell with local Docker/jq substitutes only."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def smoke_script(root=ROOT):
    text = (root / ".github/workflows/docker.yml").read_text()
    step = text.split("      - name: Smoke test candidate images\n", 1)[1]
    block = step.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0]
    return "\n".join(line[10:] for line in block.splitlines()) + "\n"


STUB = r'''
import json, os, pathlib, signal, sys
args = sys.argv[1:]
name = pathlib.Path(sys.argv[0]).name
root = pathlib.Path(os.environ['STUB_ROOT'])
mode = os.environ['STUB_MODE']
def record(event):
    with (root / 'events.jsonl').open('a') as f:
        f.write(json.dumps(event) + '\n')
def opt(key):
    return args[args.index(key) + 1]
def fail(stage, code):
    if stage in mode.split('+'):
        raise SystemExit(code)
if name == 'seq':
    print('1')  # One iteration reaches the real exhaustion branch without waiting.
elif name == 'sleep':
    pass
elif name == 'jq':
    if '-r' in args:
        print('sha256:' + opt('arch') + '-fixture')
    else:
        fail('metadata', 31)
elif name == 'docker':
    event = {'args': args}
    if args[:2] == ['volume', 'create']:
        fail('volume', 32)
        count = len(list(root.glob('volume-*')))
        volume = 'owned-volume-' + str(count)
        (root / ('volume-' + str(count))).touch()
        event['volume'] = volume
        record(event)
        print(volume)
    elif args[:2] == ['volume', 'rm']:
        record(event)
        fail('cleanup', 39)
    elif args[0] == 'pull':
        record(event)
        fail('pull', 30)
    elif args[0] == 'run':
        if '--cidfile' in args:
            pathlib.Path(opt('--cidfile')).write_text('owned-' + pathlib.Path(opt('--cidfile')).stem)
        if 'initialize' in args:
            payload = sys.stdin.buffer.read(65537)
            settings = json.loads(payload)
            event.update(settings=settings, input_closed=True,
                         private_mode=os.fstat(0).st_mode & 0o777)
            record(event)
            if mode in ('interrupt', 'terminate'):
                os.kill(os.getppid(), signal.SIGINT if mode == 'interrupt' else signal.SIGTERM)
                raise SystemExit(0)
            # Even a noisy failed initializer must never reach the CI log.
            if 'initialize' in mode.split('+'):
                print(payload.decode(), file=sys.stderr)
                raise SystemExit(37)
        elif 'version' in args:
            record(event)
            print('{}')
        else:
            record(event)
            fail('start', 38)
            print('owned-start')
    elif args[0] == 'inspect':
        record(event)
        if 'Running' in opt('--format'):
            print('false' if mode == 'exited' else 'true')
        else:
            print('137' if mode == 'shutdown' else '143')
    elif args[0] == 'exec':
        record(event)
        fail('readiness', 1)
    elif args[0] == 'logs':
        record(event)
        print('not complete' if mode == 'shutdown-log' else 'Application shutdown complete')
    elif args[0] == 'stop':
        record(event)
        fail('stop', 40)
    elif args[0] == 'rm':
        record(event)
        if '--force' in args:
            fail('cleanup', 39)
    else:
        raise AssertionError('Unrecognized Docker operation')
else:
    raise AssertionError('Unrecognized stub')
'''


class BootstrapReleaseContractTests(unittest.TestCase):
    def run_smoke(self, mode="success", root=ROOT):
        with tempfile.TemporaryDirectory(prefix="bootstrap-smoke-") as raw:
            temp = Path(raw)
            tools = temp / "bin"
            tools.mkdir()
            for name in ("docker", "jq", "seq", "sleep"):
                tool = tools / name
                tool.write_text("#!" + sys.executable + "\n" + STUB)
                tool.chmod(0o700)
            # Restricted PATH: never fall through to a real Docker/network client.
            for name, target in {
                "python3": sys.executable, "mktemp": "/usr/bin/mktemp",
                "rm": "/usr/bin/rm", "grep": "/usr/bin/grep",
            }.items():
                (tools / name).symlink_to(target)
            env = {
                "PATH": str(tools), "HOME": str(temp), "RUNNER_TEMP": str(temp),
                "TMPDIR": str(temp), "STUB_ROOT": str(temp), "STUB_MODE": mode,
                "IMAGE_REPOSITORY": "invalid.example/candidate", "IMAGE_DIGEST": "fixture",
                "EXPECTED_VERSION": "fixture", "EXPECTED_SHA": "fixture",
                "UNRELATED_PRIVATE_VALUE": "must-not-enter-selected-settings",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            script = smoke_script(root)
            proc = subprocess.Popen(
                ["/usr/bin/bash", "--noprofile", "--norc", "-c", script],
                env=env, cwd=temp, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, start_new_session=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=15)
            finally:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5)
            trace = temp / "events.jsonl"
            events = [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else []
            private_leftovers = list(temp.glob("mediaflux-smoke.*"))
            self.assertFalse(private_leftovers, "private settings directory leaked")
            for event in events:
                for key, value in event.get("settings", {}).items():
                    if key in ("WEB_SECRET_KEY", "ENV_WEB_PASSWORD"):
                        self.assertNotIn(value, stdout + stderr)
                        self.assertNotIn(value, json.dumps([e["args"] for e in events]))
            return proc.returncode, events

    def assert_cleaned(self, events):
        created = [e["volume"] for e in events if "volume" in e]
        removed = [e["args"][2] for e in events if e["args"][:2] == ["volume", "rm"]]
        self.assertEqual(removed, created)
        for event in events:
            args = event["args"]
            self.assertNotIn("prune", args)
            if args[0] == "rm":
                self.assertTrue(args[-1].startswith(("owned-", "mediaflux-")))
        for event in events:
            args = event["args"]
            if args[0] == "run" and "--cidfile" in args:
                cid = "owned-" + Path(args[args.index("--cidfile") + 1]).stem
                self.assertTrue(any(e["args"][0] == "rm" and e["args"][-1] == cid for e in events))

    def test_explicit_initialize_precedes_start_on_same_private_volume(self):
        code, events = self.run_smoke()
        self.assertEqual(code, 0)
        init = [e for e in events if "initialize" in e["args"]]
        self.assertEqual(len(init), 2, "both platform starts require explicit initialization")
        volumes = []
        for platform, event in zip(("linux/amd64", "linux/arm64"), init):
            args = event["args"]
            self.assertIn("-i", args)
            self.assertNotIn("-t", args)
            expected = ["python", "mediaflux.py", "initialize", "--data-dir", "/app/db",
                        "--confirm-new-installation", "--expect-env-sha256", "absent", "--settings-fd", "0"]
            self.assertEqual(args[-len(expected):], expected)
            self.assertEqual(args[args.index("--platform") + 1], platform)
            self.assertEqual(event["private_mode"], 0o600)
            settings = event["settings"]
            self.assertEqual(set(settings), {"MEDIAFLUX_INITIALIZED", "ENV_WEB_PASSPORT", "ENV_WEB_PASSWORD", "WEB_SECRET_KEY"})
            self.assertTrue(all(isinstance(v, str) for v in settings.values()))
            starts = [e for e in events[events.index(event) + 1:] if "--detach" in e["args"]]
            self.assertTrue(starts)
            start = starts[0]["args"]
            for flag in ("--platform", "--user", "--mount"):
                self.assertEqual(args[args.index(flag) + 1], start[start.index(flag) + 1])
            selected_env = lambda a: [a[i + 1] for i, value in enumerate(a) if value == "--env"]
            self.assertEqual(selected_env(args), selected_env(start))
            self.assertEqual(set(selected_env(args)), {"MEDIAFLUX_DATA_DIR=/app/db", "MEDIAFLUX_CONFIG_DIR=/app/db"})
            self.assertEqual(start[-1], args[-len(expected) - 1])
            volumes.append(args[args.index("--mount") + 1])
        self.assertEqual(len(set(volumes)), 2)
        self.assert_cleaned(events)

    def test_fail_closed_and_cleanup_preserve_exit_codes(self):
        for mode, expected in (("pull", 30), ("metadata", 31), ("volume", 32),
                               ("initialize", 37), ("start", 38), ("exited", 1),
                               ("readiness", 1), ("stop", 40), ("shutdown", 1),
                               ("shutdown-log", 1), ("cleanup", 0), ("initialize+cleanup", 37),
                               ("interrupt", 130), ("terminate", 143)):
            with self.subTest(mode=mode):
                code, events = self.run_smoke(mode)
                self.assertEqual(code, expected)
                if mode in ("pull", "metadata", "volume", "initialize", "initialize+cleanup", "interrupt", "terminate"):
                    self.assertFalse(any("--detach" in e["args"] for e in events))
                self.assert_cleaned(events)


if __name__ == "__main__":
    unittest.main()
