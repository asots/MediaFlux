from __future__ import annotations
import importlib.util,json,tempfile,unittest
from pathlib import Path
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path); assert s and s.loader; m=importlib.util.module_from_spec(s); import sys; sys.modules[name]=m; s.loader.exec_module(m); return m
C=load('checksums',Path('packaging/scripts/generate_checksums.py')); S=load('sbom',Path('packaging/scripts/generate_sbom.py')); B=load('release_build_info',Path('packaging/scripts/generate_build_info.py'))
class ReleaseMetadataTests(unittest.TestCase):
 def test_checksums_are_sorted_and_exclude_self(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d); (r/'z.bin').write_bytes(b'z'); (r/'a.bin').write_bytes(b'a'); C.generate_checksums(r); lines=(r/'SHA256SUMS').read_text().splitlines(); self.assertTrue(lines[0].endswith('a.bin')); self.assertTrue(lines[1].endswith('z.bin')); self.assertNotIn('SHA256SUMS', '\n'.join(lines))
 def test_release_metadata_uses_the_same_docker_writer_and_repeatable_checksums(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); output=root/'BUILD-INFO.json'
   info=B.generate_build_info('v1.2.3','abc','linux','multi','docker',build_time='2026-01-01T00:00:00Z')
   B.write_build_info(output,info); first=output.read_bytes(); C.generate_checksums(root); sums=(root/'SHA256SUMS').read_bytes()
   B.write_build_info(output,info); C.generate_checksums(root)
   self.assertEqual(output.read_bytes(),first); self.assertEqual((root/'SHA256SUMS').read_bytes(),sums)
   self.assertEqual(json.loads(first),info.as_dict()); self.assertEqual(info.artifact_name,'MediaFlux-1.2.3-docker-multi')
   self.assertEqual(list(root.glob('.*.tmp')),[])
 def test_release_metadata_marks_prerelease_from_semver_not_build_metadata(self):
  with tempfile.TemporaryDirectory() as d:
   output=Path(d)/'BUILD-INFO.json'
   for version,expected in (('v1.2.3-rc.1+build-foo',True),('v1.2.3+build-foo',False)):
    B.write_build_info(output,B.generate_build_info(version,'abc','linux','multi','docker'))
    self.assertEqual(json.loads(output.read_text())['prerelease'],expected)
 def test_spdx_sbom_contains_python_dependencies(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d); req=r/'requirements.txt'; req.write_text('fastapi==0.141.1 \\\n    --hash=sha256:' + 'a' * 64 + '\n# x\nuvicorn==0.52.3 \\\n    --hash=sha256:' + 'b' * 64 + '\n'); out=r/'SBOM.spdx.json'; S.generate_sbom([req],out,'https://example/sbom/1'); p=json.loads(out.read_text()); self.assertEqual(p['spdxVersion'],'SPDX-2.3'); self.assertEqual(p['name'],'MediaFlux locked Python dependencies'); self.assertEqual([(x['name'], x['versionInfo']) for x in p['packages']],[('fastapi','0.141.1'),('uvicorn','0.52.3')])
 def test_spdx_sbom_honors_source_date_epoch(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d); req=r/'requirements.txt'; req.write_text('fastapi==0.141.1\n'); first=r/'a.json'; second=r/'b.json'
   S.generate_sbom([req],first,'https://example/sbom/1',source_date_epoch=0)
   S.generate_sbom([req],second,'https://example/sbom/1',source_date_epoch='0')
   self.assertEqual(first.read_bytes(),second.read_bytes())
   self.assertEqual(json.loads(first.read_text())['creationInfo']['created'],'1970-01-01T00:00:00Z')
 def test_release_notes_include_download_table_and_changelog(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d); cl=r/'CHANGELOG.md'; cl.write_text('# Changelog\n\n## [0.1.0] - 2026-08-17\n\n### Added\n- 新功能甲\n\n### Fixed\n- 修复乙\n\n## [0.0.9] - 2026-08-01\n\n### Fixed\n- 旧版\n',encoding='utf-8')
   out=r/'RELEASE-NOTES.txt'; B.generate_release_notes('v0.1.0','li88iioo/MediaFlux',cl,out); t=out.read_text(encoding='utf-8')
   self.assertIn('docker pull ghcr.io/li88iioo/mediaflux:0.1.0',t); self.assertIn('docker compose up -d',t)
   self.assertIn('新功能甲',t); self.assertIn('修复乙',t); self.assertNotIn('旧版',t)
 def test_release_notes_without_matching_changelog_section(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d); cl=r/'CHANGELOG.md'; cl.write_text('# Changelog\n\n## [9.9.9] - 2026-01-01\n\n### Fixed\n- x\n',encoding='utf-8')
   out=r/'RELEASE-NOTES.txt'; B.generate_release_notes('v0.1.0','o/r',cl,out); t=out.read_text(encoding='utf-8')
   self.assertIn('docker pull ghcr.io/o/r:0.1.0',t); self.assertNotIn('## 本版变化',t)
 def test_changelog_section_requires_dated_nonempty_release_heading(self):
  valid='# Changelog\n\n## [0.1.0] - 2026-08-24\n\n### Fixed\n- x\n'
  self.assertIn('- x', B._changelog_section(valid,'0.1.0'))
  self.assertEqual(B._changelog_section('## [0.1.0]\n\n- x\n','0.1.0'),'')
  self.assertEqual(B._changelog_section('## [0.1.0] - 2026-08-24\n\n','0.1.0'),'')
  self.assertEqual(B._changelog_section('## [0.1.0] - 2026/08/24\n\n- x\n','0.1.0'),'')
  self.assertEqual(B._changelog_section('## [0.1.0] - 2026-13-40\n\n- x\n','0.1.0'),'')
if __name__=='__main__': unittest.main()
