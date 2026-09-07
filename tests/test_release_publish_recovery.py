"""用真实 workflow shell + fake PATH 验证发布失败恢复；绝不调用远端写入。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

import yaml


# fake 的未知命令一律失败，不能委托宿主机 docker/gh/git。
FAKE_TOOL = r"""
import tests  # 子进程同样必须先隔离运行路径。
import fnmatch
import json
import os
from pathlib import Path
import sys

state_path = Path(os.environ["FAKE_RELEASE_STATE"])
state = json.loads(state_path.read_text())
args = sys.argv[1:]
tool = Path(sys.argv[0]).name
state["events"].append([tool, *args])

def finish(status=0, stdout="", stderr=""):
    state_path.write_text(json.dumps(state))
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    raise SystemExit(status)

if tool == "git":
    if args[0] == "fetch":
        finish()
    if args[0] == "rev-list":
        finish(stdout=state["remote_sha"])
    if args[0] == "merge-base":
        finish(0 if state.get("main_contains", True) else 1)
elif tool == "docker":
    if args[:3] == ["buildx", "imagetools", "inspect"]:
        reference = args[3]
        if reference in state.get("inspect_errors", {}):
            finish(1, stderr=state["inspect_errors"][reference])
        digest = reference.split("@", 1)[1] if "@" in reference else state["tags"].get(reference)
        if not digest:
            finish(1, stderr=f"ERROR: {reference}: not found")
        if "--raw" in args:
            finish(stdout=json.dumps(state["manifests"][digest]))
        finish(stdout=f"Name: {reference}\nDigest: {digest}")
    if args[:3] == ["buildx", "imagetools", "create"]:
        state["tags"][args[args.index("--tag") + 1]] = args[-1].split("@")[1]
        finish()
    if args[0] == "run" and args[-3:] == ["mediaflux.py", "version", "--json"]:
        reference = next(arg for arg in args if arg.startswith("ghcr.io/"))
        digest = reference.split("@")[1]
        if digest in state["manifests"]:
            digest = state["manifests"][digest]["manifests"][0]["digest"]
        finish(stdout=json.dumps(state["images"][digest]))
elif tool == "gh" and args[0] == "release":
    action = args[1]
    release = state.get("release")
    if action == "view":
        if release is None:
            finish(1, stderr="release not found")
        result = {**release, "assets": [{"name": name} for name in release["assets"]]}
        finish(stdout=json.dumps(result))
    if action == "create":
        if release is not None:
            finish(1, stderr="release already exists")
        target = args[args.index("--target") + 1] if "--target" in args else "main"
        state["release"] = {
            "isDraft": "--draft" in args,
            "tagName": args[2],
            "targetCommitish": target,
            "body": Path(args[args.index("--notes-file") + 1]).read_text(),
            "assets": {},
        }
        finish()
    if action == "download":
        patterns = [args[i + 1] for i, arg in enumerate(args) if arg in ("-p", "--pattern")]
        names = [name for name in release["assets"] if any(fnmatch.fnmatchcase(name, p) for p in patterns)]
        if not names:
            finish(1, stderr="no assets match the file pattern")
        directory = Path(args[args.index("--dir") + 1])
        for name in names:
            value = state.get("corrupt_download", {}).get(name, release["assets"][name])
            (directory / name).write_text(value)
        finish()
    if action == "upload":
        fail_after = state.pop("fail_upload_after", None)
        files = args[args.index("--clobber") + 1:]
        for index, filename in enumerate(files):
            path = Path(filename)
            # --clobber 先移除旧附件；中断可以让同一 draft 暂时缺少 manifest。
            release["assets"].pop(path.name, None)
            if fail_after == index:
                finish(42, stderr="injected transient upload failure")
            release["assets"][path.name] = path.read_text()
        finish()
    if action == "edit" and "--draft=false" in args:
        release["isDraft"] = False
        finish()
finish(99, stderr=f"unexpected fake command: {tool} {args!r}")
"""


class ReleasePublishRecoveryTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]
    SHA = "a" * 40
    VERSION = "1.2.3"
    TAG = "v1.2.3"
    REPOSITORY = "ghcr.io/example/mediaflux"
    CANDIDATE = "sha256:" + "c" * 64
    OLD = "sha256:" + "b" * 64
    RELEASE_STEPS = {
        "Prepare GitHub Release",
        "Re-verify release tag before promotion",
        "Promote verified image tags",
        "Publish GitHub Release",
    }
    DRAFT_CONDITION = (
        "startsWith(github.ref, 'refs/tags/v') && "
        "steps.release.outputs.ready_draft == 'true'"
    )

    def setUp(self) -> None:
        for tool in ("bash", "jq"):
            self.assertIsNotNone(
                shutil.which(tool), f"{tool} is required for workflow shell tests"
            )
        self.tempdir = tempfile.TemporaryDirectory(prefix="mediaflux-release-recovery-")
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("docker", "git", "gh"):
            path = self.bin / name
            path.write_text(f"#!{sys.executable}\n" + textwrap.dedent(FAKE_TOOL))
            path.chmod(0o755)
        workflow = yaml.safe_load(
            (self.ROOT / ".github/workflows/docker.yml").read_text()
        )
        self.steps = [
            step
            for step in workflow["jobs"]["build"]["steps"]
            if step.get("name") in self.RELEASE_STEPS
        ]
        self.state_path = self.root / "state.json"
        self.state = {
            "tags": {
                f"{self.REPOSITORY}:1.2": self.OLD,
                f"{self.REPOSITORY}:latest": self.OLD,
            },
            "images": {},
            "manifests": {},
            "events": [],
            "release": None,
            "remote_sha": self.SHA,
        }
        self.add_image(self.CANDIDATE, self.VERSION, self.SHA)
        self.add_image(self.OLD, "1.2.2", "b" * 40)
        self.attempt = 0

    def add_image(self, digest: str, version: str, sha: str) -> None:
        manifests = []
        for arch, embedded_arch, suffix in (
            ("amd64", "x86_64", "1"),
            ("arm64", "aarch64", "2"),
        ):
            child = digest[:-1] + suffix
            manifests.append(
                {"digest": child, "platform": {"os": "linux", "architecture": arch}}
            )
            self.state["images"][child] = {
                "version": version,
                "commit": sha,
                "arch": embedded_arch,
                "package": "docker",
            }
        self.state["manifests"][digest] = {"schemaVersion": 2, "manifests": manifests}

    def build_info(self) -> dict:
        return {
            "name": "MediaFlux",
            "version": self.VERSION,
            "commit": self.SHA,
            "build_time": "1970-01-01T00:00:00Z",
            "platform": "linux",
            "arch": "multi",
            "package": "docker",
            "artifact_name": f"MediaFlux-{self.VERSION}-docker-multi",
            "prerelease": False,
        }

    def marker(self, sha: str | None = None) -> str:
        return (
            f"<!-- mediaflux-release:v1 version={self.TAG} commit={sha or self.SHA} -->"
        )

    def assets(self) -> dict[str, str]:
        assets = {
            "BUILD-INFO.json": json.dumps(self.build_info()),
            "RELEASE-NOTES.txt": "MediaFlux release notes\n",
            "PYTHON-DEPENDENCIES.spdx.json": json.dumps(
                {"spdxVersion": "SPDX-2.3", "packages": []}
            ),
        }
        assets["SHA256SUMS"] = "".join(
            f"{hashlib.sha256(value.encode()).hexdigest()}  {name}\n"
            for name, value in sorted(assets.items())
        )
        return assets

    def seed_release(self, *, draft: bool, assets: dict | None = None) -> None:
        self.state["release"] = {
            "isDraft": draft,
            "tagName": self.TAG,
            "targetCommitish": self.SHA,
            "body": f"Release notes\n\n{self.marker()}\n",
            "assets": self.assets() if assets is None else assets,
        }

    def run_pipeline(
        self, *, before_step=None, stop_after: str | None = None
    ) -> subprocess.CompletedProcess:
        """执行原样 run，按真实步骤顺序与显式 ready_draft 条件调度。"""
        self.attempt += 1
        runtime = self.root / f"attempt-{self.attempt}"
        runtime.mkdir()
        (runtime / "BUILD-INFO.json").write_text(json.dumps(self.build_info()))
        output = runtime / "github-output"
        output.touch()
        env = {
            **os.environ,
            "PATH": f"{self.bin}:{Path(sys.executable).parent}:{os.environ.get('PATH', '')}",
            "PYTHONPATH": str(self.ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "FAKE_RELEASE_STATE": str(self.state_path),
            "RUNNER_TEMP": str(runtime),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": "example/mediaflux",
            "IMAGE_REPOSITORY": self.REPOSITORY,
            "IMAGE_DIGEST": self.CANDIDATE,
            "VERSION": self.VERSION,
            "VERSION_REF": self.TAG,
            "DOCKER_VERSION": self.VERSION,
            "SERIES": "1.2",
            "STABLE": "true",
            "EXPECTED_SHA": self.SHA,
            "SOURCE_DATE_EPOCH": "0",
            "GH_TOKEN": "fake-test-token-never-sent",
            "MEDIAFLUX_REGISTRY_INSPECT_ATTEMPTS": "1",
            "MEDIAFLUX_REGISTRY_INSPECT_DELAY_SECONDS": "0",
        }
        result = None
        outputs = {}
        for step in self.steps:
            condition = step.get("if", "")
            if condition == self.DRAFT_CONDITION:
                if outputs.get("ready_draft") != "true":
                    continue
            else:
                self.assertEqual(condition, "startsWith(github.ref, 'refs/tags/v')")
            if before_step:
                before_step(step["name"])
            self.state_path.write_text(json.dumps(self.state))
            result = subprocess.run(
                ["bash", "-e", "-o", "pipefail"],
                input=step["run"],
                cwd=self.ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
            )
            self.state = json.loads(self.state_path.read_text())
            if result.returncode:
                return result
            outputs.update(
                line.split("=", 1)
                for line in output.read_text().splitlines()
                if "=" in line
            )
            if step["name"] == stop_after:
                # 模拟 runner 在两个成功的 step 之间中断，不替换任何 run 块。
                break
        assert result is not None
        return result

    def promotions(self) -> list:
        return [
            event
            for event in self.state["events"]
            if event[:4] == ["docker", "buildx", "imagetools", "create"]
        ]

    def gh_writes(self) -> list:
        return [
            event
            for event in self.state["events"]
            if event[:2] == ["gh", "release"]
            and event[2] in {"create", "upload", "edit"}
        ]

    def assert_failed_closed(self, result: subprocess.CompletedProcess) -> None:
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.promotions(), [])
        self.assertFalse(
            any(
                event[:3] == ["gh", "release", "edit"] for event in self.state["events"]
            )
        )

    def test_upload_failure_is_private_unpromoted_and_same_sha_rerun_completes(
        self,
    ) -> None:
        self.state["fail_upload_after"] = 0
        first = self.run_pipeline()
        self.assertEqual(first.returncode, 42, first.stdout + first.stderr)
        with self.subTest(boundary="no-public-release"):
            self.assertTrue(self.state["release"]["isDraft"])
        with self.subTest(boundary="no-image-promotion"):
            self.assertEqual(self.promotions(), [])
        second = self.run_pipeline()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertFalse(self.state["release"]["isDraft"])
        self.assertEqual(set(self.state["release"]["assets"]), set(self.assets()))
        self.assertEqual(
            self.state["tags"][f"{self.REPOSITORY}:latest"], self.CANDIDATE
        )

    def test_partial_upload_with_valid_manifest_can_resume(self) -> None:
        self.state["fail_upload_after"] = 1
        first = self.run_pipeline()
        self.assert_failed_closed(first)
        self.assertTrue(self.state["release"]["isDraft"])
        self.assertEqual(set(self.state["release"]["assets"]), {"BUILD-INFO.json"})
        second = self.run_pipeline()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertFalse(self.state["release"]["isDraft"])

    def test_normal_publish_verifies_assets_before_promotion_and_publishes_last(
        self,
    ) -> None:
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        events = self.state["events"]
        create = next(
            event for event in events if event[:3] == ["gh", "release", "create"]
        )
        for flag in ("--draft", "--verify-tag", "--target"):
            self.assertIn(flag, create)
        self.assertEqual(create[create.index("--target") + 1], self.SHA)
        self.assertIn(self.marker(), self.state["release"]["body"].splitlines())
        upload_index = next(
            i
            for i, event in enumerate(events)
            if event[:3] == ["gh", "release", "upload"]
        )
        download_indices = [
            i
            for i, event in enumerate(events)
            if event[:3] == ["gh", "release", "download"]
        ]
        promote_index = events.index(self.promotions()[0])
        edit_index = next(
            i
            for i, event in enumerate(events)
            if event[:3] == ["gh", "release", "edit"]
        )
        self.assertLess(upload_index, download_indices[0])
        self.assertLess(download_indices[0], promote_index)
        self.assertLess(promote_index, download_indices[-1])
        self.assertLess(download_indices[-1], edit_index)
        self.assertFalse(self.state["release"]["isDraft"])
        self.assertEqual(set(self.state["release"]["assets"]), set(self.assets()))
        self.assertEqual(len(self.promotions()), 3)

    def test_complete_published_rerun_keeps_assets_and_all_tags_even_with_new_digest(
        self,
    ) -> None:
        first = self.run_pipeline()
        self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
        before_assets = dict(self.state["release"]["assets"])
        before_tags = dict(self.state["tags"])
        self.state["events"] = []
        # 兼容旧发布的 targetCommitish=main；公开 Release 以 tag + manifest 绑定 SHA，
        # 不把它当可续传的 owned draft，也不要求公开正文保留私有 draft 标记。
        self.state["release"]["targetCommitish"] = "main"
        self.state["release"]["body"] = "Published release notes"
        self.CANDIDATE = "sha256:" + "d" * 64
        self.add_image(self.CANDIDATE, self.VERSION, self.SHA)
        second = self.run_pipeline()
        self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
        self.assertEqual(self.promotions(), [])
        self.assertEqual(self.gh_writes(), [])
        self.assertEqual(self.state["tags"], before_tags)
        self.assertEqual(self.state["release"]["assets"], before_assets)

    def test_complete_owned_draft_reuses_verified_assets_without_clobber(self) -> None:
        self.seed_release(draft=True)
        assets = dict(self.state["release"]["assets"])
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(self.state["release"]["isDraft"])
        self.assertEqual(self.state["release"]["assets"], assets)
        self.assertEqual([event[2] for event in self.gh_writes()], ["edit"])

    def test_complete_draft_interrupted_before_promotion_resumes_without_upload(
        self,
    ) -> None:
        checkpoint = self.run_pipeline(stop_after="Prepare GitHub Release")
        self.assertEqual(
            checkpoint.returncode, 0, checkpoint.stdout + checkpoint.stderr
        )
        self.assertTrue(self.state["release"]["isDraft"])
        self.assertEqual(set(self.state["release"]["assets"]), set(self.assets()))
        self.assertEqual(self.promotions(), [])
        self.state["events"] = []
        resumed = self.run_pipeline()
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertFalse(self.state["release"]["isDraft"])
        self.assertEqual([event[2] for event in self.gh_writes()], ["edit"])

    def test_complete_draft_interrupted_after_promotion_reuses_promoted_digest(
        self,
    ) -> None:
        checkpoint = self.run_pipeline(stop_after="Promote verified image tags")
        self.assertEqual(
            checkpoint.returncode, 0, checkpoint.stdout + checkpoint.stderr
        )
        self.assertTrue(self.state["release"]["isDraft"])
        self.assertEqual(set(self.state["release"]["assets"]), set(self.assets()))
        original_digest = self.CANDIDATE
        assets = dict(self.state["release"]["assets"])
        self.state["events"] = []
        self.CANDIDATE = "sha256:" + "d" * 64
        self.add_image(self.CANDIDATE, self.VERSION, self.SHA)
        resumed = self.run_pipeline()
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        self.assertFalse(self.state["release"]["isDraft"])
        self.assertEqual(self.state["release"]["assets"], assets)
        self.assertEqual(set(self.state["tags"].values()), {original_digest})
        self.assertEqual([event[2] for event in self.gh_writes()], ["edit"])

    def test_invalid_or_different_source_manifest_cannot_be_repaired(self) -> None:
        invalid_manifests = {
            "broken-json": "{not-json",
            "not-an-object": "[]",
            "different-sha": json.dumps({**self.build_info(), "commit": "e" * 40}),
            "different-version": json.dumps({**self.build_info(), "version": "1.2.4"}),
            "missing-fields": json.dumps({"commit": self.SHA}),
        }
        for label, manifest in invalid_manifests.items():
            for draft in (True, False):
                with self.subTest(manifest=label, draft=draft):
                    self.state["events"] = []
                    self.seed_release(draft=draft, assets={"BUILD-INFO.json": manifest})
                    result = self.run_pipeline()
                    self.assert_failed_closed(result)
                    self.assertEqual(self.gh_writes(), [])
                    self.assertEqual(
                        self.state["release"]["assets"], {"BUILD-INFO.json": manifest}
                    )

    def test_unknown_or_different_candidate_draft_is_never_adopted(self) -> None:
        variations = (
            {"body": "Unknown draft"},
            {"body": self.marker("e" * 40)},
            {"body": self.marker().replace(self.SHA, self.SHA[:12])},
            {"body": self.marker() + " not a standalone marker"},
            {"targetCommitish": "e" * 40},
            {"tagName": "v1.2.4"},
        )
        for variation in variations:
            with self.subTest(variation=variation):
                self.state["events"] = []
                self.seed_release(draft=True, assets={})
                self.state["release"].update(variation)
                result = self.run_pipeline()
                self.assert_failed_closed(result)
                self.assertEqual(self.gh_writes(), [])

    def test_published_release_missing_any_required_asset_is_never_repaired(
        self,
    ) -> None:
        for missing in self.assets():
            with self.subTest(missing=missing):
                self.state["events"] = []
                assets = self.assets()
                assets.pop(missing)
                self.seed_release(draft=False, assets=assets)
                result = self.run_pipeline()
                self.assert_failed_closed(result)
                self.assertEqual(self.gh_writes(), [])
                self.assertEqual(self.state["release"]["assets"], assets)

    def test_existing_download_hash_mismatch_fails_before_any_write(self) -> None:
        for draft in (True, False):
            with self.subTest(draft=draft):
                self.state["events"] = []
                self.seed_release(draft=draft)
                self.state["corrupt_download"] = {
                    "RELEASE-NOTES.txt": "corrupted download"
                }
                result = self.run_pipeline()
                self.assert_failed_closed(result)
                self.assertIn("SHA-256 mismatch", result.stderr)
                self.assertEqual(self.gh_writes(), [])

    def test_uploaded_assets_must_be_downloaded_and_hash_verified(self) -> None:
        self.state["corrupt_download"] = {"RELEASE-NOTES.txt": "corrupted download"}
        result = self.run_pipeline()
        self.assert_failed_closed(result)
        self.assertTrue(self.state["release"]["isDraft"])
        self.assertIn("SHA-256 mismatch", result.stderr)
        self.assertEqual([event[2] for event in self.gh_writes()], ["create", "upload"])

    def test_checksum_list_cannot_omit_duplicate_or_escape_asset_names(self) -> None:
        valid = self.assets()["SHA256SUMS"]
        for checksum in (
            "",
            valid + valid.splitlines()[0] + "\n",
            "a" * 64 + "  ../../outside\n",
        ):
            with self.subTest(checksum=checksum):
                self.state["events"] = []
                assets = {**self.assets(), "SHA256SUMS": checksum}
                self.seed_release(draft=True, assets=assets)
                result = self.run_pipeline()
                self.assert_failed_closed(result)
                self.assertEqual(self.gh_writes(), [])

    def test_newer_mutable_tags_are_not_downgraded(self) -> None:
        self.add_image(self.OLD, "1.2.4", "b" * 40)
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.state["tags"][f"{self.REPOSITORY}:latest"], self.OLD)
        self.assertEqual(self.state["tags"][f"{self.REPOSITORY}:1.2"], self.OLD)
        self.assertEqual(self.state["tags"][f"{self.REPOSITORY}:1.2.3"], self.CANDIDATE)
        self.assertEqual(len(self.promotions()), 1)

    def test_registry_inspection_error_leaves_verified_draft_unpromoted(self) -> None:
        self.state["inspect_errors"] = {
            f"{self.REPOSITORY}:latest": "ERROR: unauthorized: authentication required"
        }
        result = self.run_pipeline()
        self.assert_failed_closed(result)
        self.assertTrue(self.state["release"]["isDraft"])
        self.assertEqual(set(self.state["release"]["assets"]), set(self.assets()))

    def test_either_architecture_with_a_different_exact_tag_commit_refuses_promotion(
        self,
    ) -> None:
        for arch in ("amd64", "arm64"):
            with self.subTest(arch=arch):
                self.state["events"] = []
                self.seed_release(draft=True)
                self.add_image(self.OLD, self.VERSION, self.SHA)
                descriptor = next(
                    item
                    for item in self.state["manifests"][self.OLD]["manifests"]
                    if item["platform"]["architecture"] == arch
                )
                self.state["images"][descriptor["digest"]]["commit"] = "e" * 40
                self.state["tags"][f"{self.REPOSITORY}:1.2.3"] = self.OLD
                result = self.run_pipeline()
                self.assert_failed_closed(result)
                self.assertTrue(self.state["release"]["isDraft"])

    def test_verified_same_commit_exact_tag_reuses_original_digest(self) -> None:
        self.add_image(self.OLD, self.VERSION, self.SHA)
        self.state["tags"][f"{self.REPOSITORY}:1.2.3"] = self.OLD
        result = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(set(self.state["tags"].values()), {self.OLD})

    def test_tag_is_rechecked_after_upload_and_before_promotion(self) -> None:
        def move_tag(name: str) -> None:
            if name == "Re-verify release tag before promotion":
                self.state["remote_sha"] = "e" * 40

        result = self.run_pipeline(before_step=move_tag)
        self.assert_failed_closed(result)
        self.assertTrue(self.state["release"]["isDraft"])

    def test_tag_is_rechecked_before_publishing_draft(self) -> None:
        def move_tag(name: str) -> None:
            if name == "Publish GitHub Release":
                self.state["remote_sha"] = "e" * 40

        result = self.run_pipeline(before_step=move_tag)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.state["release"]["isDraft"])
        self.assertFalse(
            any(
                event[:3] == ["gh", "release", "edit"] for event in self.state["events"]
            )
        )

    def test_source_must_be_in_remote_main_before_draft_creation(self) -> None:
        self.state["main_contains"] = False
        result = self.run_pipeline()
        self.assert_failed_closed(result)
        self.assertEqual(self.gh_writes(), [])


if __name__ == "__main__":
    unittest.main()
