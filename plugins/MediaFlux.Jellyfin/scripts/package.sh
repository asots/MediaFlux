#!/usr/bin/env bash
set -euo pipefail
plugin=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$plugin/../.." && pwd)
dotnet_bin=${DOTNET_BIN:-dotnet}
"$dotnet_bin" build "$plugin/MediaFlux.Jellyfin.csproj" -c Release --nologo
python3 - "$plugin" "$repo" <<'PY'
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
plugin, repo = map(Path, sys.argv[1:])
manifest = json.loads((plugin / 'meta.json').read_text())
project = ET.parse(plugin / 'MediaFlux.Jellyfin.csproj').getroot()
version = project.findtext('PropertyGroup/Version')
assert manifest['version'] == '.'.join((version.split('.') + ['0'] * 4)[:4]), 'manifest/assembly version mismatch'
framework = project.findtext('PropertyGroup/TargetFramework')
assert framework == 'net10.0', 'Jellyfin 12.2 requires net10.0'
controller = project.find("ItemGroup/PackageReference[@Include='Jellyfin.Controller']")
model = project.find("ItemGroup/PackageReference[@Include='Jellyfin.Model']")
assert controller.get('Version') == model.get('Version') == '12.2.0', 'Jellyfin packages must target 12.2.0'
assert manifest['targetAbi'] == controller.get('Version') + '.0', 'manifest/API baseline mismatch'
assert manifest['guid'] in (plugin / 'Plugin.cs').read_text(), 'manifest/plugin ID mismatch'
output = plugin / 'artifacts'
output.mkdir(exist_ok=True)
archive = output / f'MediaFlux.Jellyfin-{manifest["version"]}.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as package:
    package.write(plugin / 'bin' / 'Release' / framework / 'MediaFlux.Jellyfin.dll', 'MediaFlux.Jellyfin.dll')
    package.write(plugin / 'meta.json', 'meta.json')
    package.write(repo / 'docs' / 'jellyfin-plugin.md', 'README.md')
    package.write(repo / 'LICENSE', 'LICENSE')
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
checksum = archive.with_suffix('.zip.sha256')
checksum.write_text(f'{digest}  {archive.name}\n')
print(f'Package: {archive}\nSHA256: {digest}')
PY
