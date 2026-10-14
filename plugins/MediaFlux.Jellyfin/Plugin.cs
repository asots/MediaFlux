using MediaBrowser.Common.Configuration;
using MediaBrowser.Common.Plugins;
using MediaBrowser.Controller;
using MediaBrowser.Model.Plugins;
using MediaBrowser.Model.Serialization;
using MediaFlux.Jellyfin.Configuration;

namespace MediaFlux.Jellyfin;

public sealed class Plugin : BasePlugin<PluginConfiguration>, IHasWebPages
{
    public const string CompiledBaseline = "12.2.0";
    public const string SourceCheckedVersions = "12.2";
    public static Plugin? Instance { get; private set; }
    private readonly IServerApplicationHost _host;
    private readonly object _gate = new();
    private string _lastDeleteError = "";

    public Plugin(IApplicationPaths paths, IXmlSerializer serializer, IServerApplicationHost host)
        : base(paths, serializer)
    {
        _host = host;
        Instance = this;
    }

    public override string Name => "MediaFlux";
    public override string Description => "明确删除 STRM 前先由 MediaFlux 校验并完成云端删除；默认关闭。";
    public override Guid Id => Guid.Parse("9037f29f-2ebc-4b8a-952f-960e24402bb5");
    public bool IsTargetVersion => _host.ApplicationVersion is { Major: 12, Minor: 2 };

    public IEnumerable<PluginPageInfo> GetPages() =>
    [new PluginPageInfo { Name = Name, EmbeddedResourcePath = "MediaFlux.Jellyfin.Configuration.configPage.html" }];

    public PluginConfiguration Snapshot()
    {
        lock (_gate)
        {
            var c = Configuration;
            return new PluginConfiguration
            {
                Enabled = c.Enabled, MfUrl = c.MfUrl, Username = c.Username, Password = c.Password,
                MappingMode = c.MappingMode, TaskTimeoutSeconds = c.TaskTimeoutSeconds,
                ManualMappings = c.ManualMappings.Select(m => m with { }).ToArray(),
                MfMappings = c.MfMappings.Select(m => m with { }).ToArray(),
                MfLibraries = c.MfLibraries.Select(l => l with { Locations = l.Locations.ToArray() }).ToArray(),
                SyncedMfUrl = c.SyncedMfUrl, SyncedUsername = c.SyncedUsername,
                SyncedAtUtc = c.SyncedAtUtc, Revision = c.Revision
            };
        }
    }

    public static bool CacheCurrent(PluginConfiguration c) => c.SyncedAtUtc.Length > 0
        && c.MfUrl == c.SyncedMfUrl && c.Username == c.SyncedUsername;

    public static MappingSet ActiveMappings(PluginConfiguration c) => new(
        c.MappingMode switch
        {
            "mf" => c.MfMappings,
            "manual" => c.ManualMappings,
            _ => throw new PluginFault("映射模式无效；请在插件配置中重新保存。")
        });

    // This also validates writes from Jellyfin's built-in generic config endpoint.
    // Generic JSON cannot read or replace Password, nor inject a trusted MF cache.
    public override void UpdateConfiguration(BasePluginConfiguration configuration)
    {
        lock (_gate)
        {
            var c = (PluginConfiguration)configuration;
            var old = Snapshot();
            if (c.Password.Length == 0) c.Password = old.Password;
            c.MfMappings = old.MfMappings; c.MfLibraries = old.MfLibraries;
            c.SyncedMfUrl = old.SyncedMfUrl; c.SyncedUsername = old.SyncedUsername;
            c.SyncedAtUtc = old.SyncedAtUtc;
            c.MfUrl = c.MfUrl.Trim(); c.Username = c.Username.Trim();
            if (c.MappingMode is not ("mf" or "manual")) throw new PluginFault("请选择 MF 或手动映射模式。");
            if (c.MfUrl.Length > 0) c.MfUrl = MfClient.BaseUri(c.MfUrl).AbsoluteUri.TrimEnd('/');
            if (c.Username.Length > 128 || c.Password.Length > 512) throw new PluginFault("MF 用户名或密码超出允许长度。");
            if (c.TaskTimeoutSeconds is < 5 or > 600) throw new PluginFault("任务等待时间必须为 5–600 秒。");
            c.ManualMappings = new MappingSet(c.ManualMappings).Mappings;
            if (c.Enabled)
            {
                if (!IsTargetVersion) throw new PluginFault("纳管删除目标为 Jellyfin 12.2 系列，当前服务器版本不允许启用。");
                RequireCredentials(c);
                if (c.MappingMode == "mf" && !CacheCurrent(c))
                    throw new PluginFault("请先关闭插件、保存凭据并成功同步 MF 范围，再启用。");
                if (ActiveMappings(c).Mappings.Length == 0)
                    throw new PluginFault("启用前必须至少配置一条有效映射。");
            }
            c.Revision = old.Revision + 1;
            base.UpdateConfiguration(c);
        }
    }

    public static void RequireCredentials(PluginConfiguration c)
    {
        if (c.MfUrl.Length == 0 || c.Username.Length == 0 || c.Password.Length == 0)
            throw new PluginFault("请配置 MF URL、用户名和密码；本地内容未删除。");
        _ = MfClient.BaseUri(c.MfUrl);
    }

    public void CommitSync(PluginConfiguration expected, MfOverview overview)
    {
        lock (_gate)
        {
            var current = Snapshot();
            if (current.Revision != expected.Revision || current.MappingMode != "mf")
                throw new PluginFault("同步期间配置已变化；请重新同步。");
            current.MfMappings = overview.Mappings;
            current.MfLibraries = overview.Libraries;
            current.SyncedMfUrl = current.MfUrl; current.SyncedUsername = current.Username;
            current.SyncedAtUtc = DateTimeOffset.UtcNow.ToString("O");
            current.Revision++;
            base.UpdateConfiguration(current);
        }
    }

    public void EnsureUnchanged(PluginConfiguration expected)
    {
        if (Snapshot().Revision != expected.Revision)
            throw new PluginFault("等待期间插件配置已变化；云任务可能已完成，请核对 MF 日志，勿盲目重试。本地删除已取消。");
    }

    public void RecordDeleteError(string error) { lock (_gate) _lastDeleteError = error; }
    public string LastDeleteError { get { lock (_gate) return _lastDeleteError; } }

    public override void SaveConfiguration(PluginConfiguration config)
    {
        lock (_gate)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(ConfigurationFilePath)!);
            if (!OperatingSystem.IsWindows())
            {
                using var file = new FileStream(ConfigurationFilePath, new FileStreamOptions
                {
                    Mode = FileMode.OpenOrCreate, Access = FileAccess.Write,
                    UnixCreateMode = UnixFileMode.UserRead | UnixFileMode.UserWrite
                });
                File.SetUnixFileMode(ConfigurationFilePath, UnixFileMode.UserRead | UnixFileMode.UserWrite);
            }
            base.SaveConfiguration(config);
        }
    }
}
