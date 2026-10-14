using System.Security.Cryptography;
using MediaBrowser.Controller.Entities;
using MediaBrowser.Controller.Entities.Movies;
using MediaBrowser.Controller.Entities.TV;
using MediaBrowser.Controller.Library;
using MediaBrowser.Model.IO;

namespace MediaFlux.Jellyfin;

public sealed record SelectedItem(Guid Id, string Path, string ItemType, FileSystemMetadata[] DeletePaths);
public sealed record DeletePlan(Dictionary<string, string> Fingerprints, SelectedItem[] Items, FileSystemMetadata[] Footprint);

public static class DeletePlanner
{
    private static readonly StringComparer PathComparer = OperatingSystem.IsWindows()
        ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;

    public static async Task<DeletePlan> PrepareAsync(IReadOnlyList<BaseItem> items, MappingSet mappings, ILibraryManager library, CancellationToken token)
    {
        var paths = new HashSet<string>(PathComparer);
        var footprint = new List<FileSystemMetadata>();
        var snapshots = new List<SelectedItem>();
        foreach (var item in items)
        {
            // Native LibraryManager uses these exact paths with DeleteFileLocation=true.
            var roots = item.GetDeletePaths().Select(r => new FileSystemMetadata { FullName = r.FullName, IsDirectory = r.IsDirectory }).ToArray();
            snapshots.Add(new SelectedItem(item.Id, item.Path ?? "", item.GetType().FullName!, roots));
            footprint.AddRange(roots);
            var itemPath = item.Path ?? "";
            var relevant = mappings.Match(itemPath) is not null || roots.Any(r => mappings.Overlaps(r.FullName));
            if (!relevant) continue;
            if (IsStrm(itemPath) && mappings.Match(itemPath) is not null)
            {
                if (item is not (Movie or Episode) || !item.IsFileProtocol)
                    throw new PluginFault("已映射 STRM 仅支持明确选择的电影/单集；此媒体类型不能安全联动删除。");
                if (IsGrouped((Video)item, library))
                    throw new PluginFault("该电影/单集含多版本、分段或关联媒体，无法确认单项原生删除范围；请先拆分关联，或在 MF 中操作。");
                _ = MappingSet.Normalize(itemPath);
                paths.Add(itemPath);
            }
        }
        if (paths.Count > 1000) throw new PluginFault("一次最多选择 1000 个 STRM 文件。");
        CheckFootprint(footprint, paths, mappings, allowMissingDirectories: false, token);
        var fingerprints = new Dictionary<string, string>(PathComparer);
        foreach (var path in paths)
        {
            CheckNoLinks(path);
            fingerprints.Add(path, await FingerprintAsync(path, token).ConfigureAwait(false));
        }
        return new DeletePlan(fingerprints, snapshots.ToArray(), footprint.ToArray());
    }

    public static async Task RecheckAsync(DeletePlan plan, IReadOnlyList<BaseItem> currentItems, MappingSet mappings, ILibraryManager library, CancellationToken token)
    {
        foreach (var old in plan.Items)
        {
            var current = currentItems.Single(i => i.Id == old.Id);
            if (!PathComparer.Equals(current.Path, old.Path) || current.GetType().FullName != old.ItemType)
                throw new PluginFault("等待期间媒体条目已改名、换路径或类型；云任务可能已成功，本地删除已取消。请刷新媒体库并核对 MF 日志。");
            if (plan.Fingerprints.ContainsKey(old.Path) && current is Video video && IsGrouped(video, library))
                throw new PluginFault("等待期间媒体关联版本发生变化；云任务已成功，本地删除已取消。请核对 MF 日志，不要盲目重试。");
            var now = current.GetDeletePaths().ToArray();
            if (now.Any(r => !old.DeletePaths.Any(p => PathComparer.Equals(p.FullName, r.FullName) && p.IsDirectory == r.IsDirectory)))
                throw new PluginFault("等待期间原生删除范围出现新的文件或目录；云任务可能已成功，本地删除已取消。请核对 MF 日志。");
        }
        // MF's confirmed-success STRM linkage may already have removed files/directories.
        // This is NOT a delete trigger: only a prior explicit selection + completed/ok authorizes continuation.
        CheckFootprint(plan.Footprint, new HashSet<string>(plan.Fingerprints.Keys, PathComparer), mappings, true, token);
        foreach (var pair in plan.Fingerprints)
        {
            if (!File.Exists(pair.Key)) continue;
            CheckNoLinks(pair.Key);
            if (await FingerprintAsync(pair.Key, token).ConfigureAwait(false) != pair.Value)
                throw new PluginFault("等待云任务期间 STRM 内容已变化，本地删除已取消。云任务可能已成功，请核对 MF 日志，勿盲目重试。");
        }
    }

    // 12.2 stores version relations in LinkedChildren, not the legacy in-memory arrays.
    private static bool IsGrouped(Video video, ILibraryManager library) =>
        video.IsStacked || video.PrimaryVersionId.HasValue || video.OwnerId != Guid.Empty
        || library.GetLocalAlternateVersionIds(video).Any()
        || library.GetLinkedAlternateVersions(video).Any();

    private static void CheckFootprint(IEnumerable<FileSystemMetadata> footprint, HashSet<string> selected,
        MappingSet mappings, bool allowMissingDirectories, CancellationToken token)
    {
        var examined = 0;
        foreach (var root in footprint)
        {
            if (!mappings.Overlaps(root.FullName)) continue;
            if (!root.IsDirectory)
            {
                CheckFile(root.FullName, selected, mappings);
                continue;
            }
            if (!Directory.Exists(root.FullName))
            {
                if (allowMissingDirectories) continue;
                throw new PluginFault("无法读取原生删除目录；不能确认其中是否存在其他已管理 STRM，本地删除已取消。");
            }
            CheckNoLinks(root.FullName);
            var directories = new Stack<string>();
            directories.Push(root.FullName);
            while (directories.TryPop(out var directory))
            {
                token.ThrowIfCancellationRequested();
                foreach (var entry in new DirectoryInfo(directory).EnumerateFileSystemInfos())
                {
                    if (++examined > 10000) throw new PluginFault("原生删除目录过大，不能安全确认 STRM 范围；请在 MF 中操作。");
                    if ((entry.Attributes & FileAttributes.ReparsePoint) != 0)
                        throw new PluginFault("原生删除目录包含符号链接或重解析点，不能安全确认 STRM 范围。");
                    if (entry is DirectoryInfo) directories.Push(entry.FullName);
                    else CheckFile(entry.FullName, selected, mappings);
                }
            }
        }
    }

    private static void CheckFile(string path, HashSet<string> selected, MappingSet mappings)
    {
        if (IsStrm(path) && mappings.Match(path) is not null && !selected.Contains(path))
            throw new PluginFault("原生删除还会移除未明确选中的已映射 STRM（整季/整剧、同目录版本、附加项或其他文件）。已拒绝整批请求；请逐个选择电影/单集、拆分目录或在 MF 中操作。");
    }

    public static bool IsStrm(string? path) => !string.IsNullOrEmpty(path)
        && path.EndsWith(".strm", StringComparison.OrdinalIgnoreCase);

    private static void CheckNoLinks(string path)
    {
        if (new FileInfo(path).LinkTarget is not null) throw new PluginFault("STRM 或删除目录不能是符号链接。");
        for (var dir = new DirectoryInfo(Path.GetDirectoryName(path)!); dir is not null; dir = dir.Parent)
            if (dir.LinkTarget is not null) throw new PluginFault("STRM 或删除目录经过符号链接，无法确认文件身份。");
    }

    private static async Task<string> FingerprintAsync(string path, CancellationToken token)
    {
        await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 4096, FileOptions.Asynchronous | FileOptions.SequentialScan);
        return "sha256:" + Convert.ToHexString(await SHA256.HashDataAsync(stream, token).ConfigureAwait(false)).ToLowerInvariant();
    }
}
