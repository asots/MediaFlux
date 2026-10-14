using System.Text.RegularExpressions;
using MediaFlux.Jellyfin.Configuration;

namespace MediaFlux.Jellyfin;

public sealed class PluginFault(string message, int statusCode = 409) : Exception(message)
{
    public int StatusCode { get; } = statusCode;
}

// Compare whole path components, not textual StartsWith (e.g. /tv vs /tv-old).
public sealed class MappingSet
{
    public PathMapping[] Mappings { get; }

    public MappingSet(IEnumerable<PathMapping> mappings)
    {
        Mappings = mappings.Select(m => new PathMapping
        {
            Local = Normalize(m.Local), Server = Normalize(m.Server)
        }).Distinct().ToArray();
        for (var i = 0; i < Mappings.Length; i++)
        {
            if (Mappings[i].Local is "/" or "//" || Mappings[i].Server is "/" or "//")
                throw new PluginFault("路径映射不能使用文件系统根目录。");
            for (var j = 0; j < i; j++)
            {
                if (Equal(Mappings[i].Local, Mappings[j].Local))
                    throw new PluginFault("同一 MF 本地前缀不能重复配置；请修正映射。");
                if (Equal(Mappings[i].Server, Mappings[j].Server))
                {
                    throw new PluginFault("多个 MF 来源映射到了同一 Jellyfin 目录，无法确认文件身份；请修正映射。");
                }
            }
        }
    }

    public PathMapping? Match(string path)
    {
        if (!IsAbsolute(path)) return null;
        var value = Normalize(path);
        return Mappings.Where(m => Within(value, m.Server))
            .OrderByDescending(m => m.Server.Length).FirstOrDefault();
    }

    public bool Overlaps(string path)
    {
        if (!IsAbsolute(path)) return false;
        var value = Normalize(path);
        return Mappings.Any(m => Within(value, m.Server) || Within(m.Server, value));
    }

    public static bool IsAbsolute(string? path) => !string.IsNullOrEmpty(path)
        && (path.StartsWith('/') || path.StartsWith("\\\\", StringComparison.Ordinal)
            || Regex.IsMatch(path, @"^[a-zA-Z]:[/\\]"));

    public static string Normalize(string path)
    {
        if (string.IsNullOrWhiteSpace(path) || path.Length > 4096 || !IsAbsolute(path)
            || path.Any(char.IsControl))
            throw new PluginFault("路径必须是可读取的绝对文件系统路径。");
        var windows = IsWindows(path);
        var normalized = windows ? path.Replace('\\', '/') : path;
        var pieces = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (pieces.Any(p => p is "." or ".."))
            throw new PluginFault("路径不能包含 . 或 .. 片段。");
        if (normalized.StartsWith("//", StringComparison.Ordinal))
        {
            if (pieces.Length < 2) throw new PluginFault("UNC 路径必须包含服务器和共享名。");
            normalized = "//" + string.Join('/', pieces);
        }
        else if (windows)
            normalized = string.Join('/', pieces);
        else
            normalized = "/" + string.Join('/', pieces);
        return normalized.Length == 2 && normalized[1] == ':' ? normalized + "/" : normalized;
    }

    private static bool IsWindows(string path) => path.StartsWith("\\\\", StringComparison.Ordinal)
        || path.StartsWith("//", StringComparison.Ordinal) || Regex.IsMatch(path, @"^[a-zA-Z]:");
    public static bool Equal(string a, string b) => string.Equals(a, b,
        IsWindows(a) ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
    public static bool Within(string path, string root) => Equal(path, root)
        || path.StartsWith(root.TrimEnd('/') + "/",
            IsWindows(root) ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
}
