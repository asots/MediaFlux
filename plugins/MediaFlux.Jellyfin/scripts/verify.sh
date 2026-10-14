#!/usr/bin/env bash
# Build official v12.2 controllers, then exercise their real DELETE/config HTTP routes in TestServer.
# Does not start Jellyfin or contact a MediaFlux/cloud production instance.
set -euo pipefail
plugin=$(cd "$(dirname "$0")/.." && pwd)
repo=$(cd "$plugin/../.." && pwd)
cache=${JELLYFIN_VERIFY_CACHE:-"$repo/.superpowers/tmp/jellyfin-plugin"}
dotnet_bin=${DOTNET_BIN:-dotnet}
if [ "$#" -gt 1 ] || [ "${1-12.2}" != "12.2" ]; then
  echo "Usage: $0 [12.2] (only Jellyfin v12.2 is supported)" >&2
  exit 2
fi
source_dir="$cache/jellyfin-12.2"
mkdir -p "$cache"
if [ ! -f "$source_dir/Jellyfin.Api/Jellyfin.Api.csproj" ]; then
  archive="$cache/jellyfin-12.2.tar.gz"
  curl --fail --location --retry 2 "https://codeload.github.com/jellyfin/jellyfin/tar.gz/refs/tags/v12.2" -o "$archive"
  tar -xzf "$archive" -C "$cache"
fi
"$dotnet_bin" build "$source_dir/Jellyfin.Api/Jellyfin.Api.csproj" -c Release --nologo
"$dotnet_bin" test "$plugin/Tests/MediaFlux.Jellyfin.Tests.csproj" -c Release --nologo \
  "-p:JellyfinApiDir=$source_dir/Jellyfin.Api/bin/Release/net10.0"
