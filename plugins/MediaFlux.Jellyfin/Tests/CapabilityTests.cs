using System.Net;
using Jellyfin.Database.Implementations.Entities;
using MediaBrowser.Controller.Entities.TV;
using Moq;
using Xunit;

namespace MediaFlux.Jellyfin.Tests;

public sealed class CapabilityTests
{
    [Fact]
    public async Task EachDeleteHandshakesInTheSameAuthenticatedSessionBeforePost()
    {
        using var f = new TestFixture();
        using var first = await f.Delete(f.Episode("a.strm").Object);
        using var second = await f.Delete(f.Episode("b.strm").Object);
        Assert.Equal(HttpStatusCode.NoContent, first.StatusCode); Assert.Equal(HttpStatusCode.NoContent, second.StatusCode);
        Assert.Equal(2, f.MfSessions); Assert.Equal(2, f.Handler.LoginPosts);
        Assert.Equal(2, f.Handler.CapabilityGets); Assert.Equal(2, f.Handler.DeletePosts);
        var expected = new[] { "GET /login", "POST /login", "GET /", "GET /api/jellyfin/delete-capabilities", "POST /api/jellyfin/delete", "GET /api/guangya/organize/status" };
        Assert.Equal(expected.Concat(expected), f.Handler.Requests);
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("null")]
    [InlineData("[]")]
    [InlineData("bad json")]
    [InlineData("{\"scope_version\":0,\"kinds\":[\"movie\",\"episode\",\"season\",\"series\"]}")]
    [InlineData("{\"scope_version\":2,\"kinds\":[\"movie\",\"episode\",\"season\",\"series\"]}")]
    [InlineData("{\"scope_version\":\"1\",\"kinds\":[\"movie\",\"episode\",\"season\",\"series\"]}")]
    [InlineData("{\"scope_version\":1,\"kinds\":[\"movie\",\"episode\"]}")]
    [InlineData("{\"scope_version\":1,\"kinds\":null}")]
    [InlineData("{\"scope_version\":1,\"kinds\":[\"movie\",\"episode\",\"season\",null]}")]
    public async Task MissingOrIncompatibleCapabilityNeverPostsOrDeletesLocally(string capabilities)
    {
        using var f = new TestFixture(); f.Handler.Capabilities = capabilities;
        using var response = await f.Delete(f.Episode().Object);
        Assert.Equal(HttpStatusCode.BadGateway, response.StatusCode);
        Assert.Equal(1, f.Handler.LoginPosts); Assert.Equal(1, f.Handler.CapabilityGets);
        Assert.Equal(0, f.Handler.DeletePosts); Assert.Equal(0, f.NativeDeletes);
    }

    [Theory]
    [InlineData(HttpStatusCode.NotFound)]
    [InlineData(HttpStatusCode.Unauthorized)]
    [InlineData(HttpStatusCode.Forbidden)]
    [InlineData(HttpStatusCode.InternalServerError)]
    [InlineData(HttpStatusCode.Found)]
    public async Task CapabilityHttpFailuresNeverPost(HttpStatusCode status)
    {
        using var f = new TestFixture(); f.Handler.CapabilityCode = status;
        using var response = await f.Delete(f.Episode().Object);
        Assert.Equal(HttpStatusCode.BadGateway, response.StatusCode);
        Assert.Equal(0, f.Handler.DeletePosts); Assert.Equal(0, f.NativeDeletes);
    }

    [Fact]
    public async Task CapabilityResultsAreNotCachedAcrossDeletes()
    {
        using var f = new TestFixture(); var episode = f.Episode().Object;
        using var first = await f.Delete(episode);
        f.Handler.Capabilities = "{}";
        using var second = await f.Delete(episode);
        Assert.Equal(HttpStatusCode.NoContent, first.StatusCode); Assert.Equal(HttpStatusCode.BadGateway, second.StatusCode);
        Assert.Equal(2, f.Handler.CapabilityGets); Assert.Equal(1, f.Handler.DeletePosts); Assert.Equal(1, f.NativeDeletes);
    }

    [Theory]
    [InlineData("fingerprint")]
    [InlineData("permission")]
    [InlineData("new-file")]
    [InlineData("new-native-object")]
    [InlineData("user-gone")]
    [InlineData("config-changed")]
    public async Task AfterLoginAndHandshakeThePlanIsRecheckedBeforeAnyPost(string change)
    {
        using var f = new TestFixture(); var series = f.Series(); var episode = f.Episode("series/a.strm", parent: series);
        f.Handler.OnCapabilities = () =>
        {
            switch (change)
            {
                case "fingerprint": File.AppendAllText(episode.Object.Path, "changed"); break;
                case "permission": episode.Setup(i => i.CanDelete(It.IsAny<User>())).Returns(false); break;
                case "new-file": File.WriteAllText(Path.Combine(series.Path, "1.X"), "new"); break;
                case "new-native-object":
                    var added = new Season { Id = Guid.NewGuid(), Path = Path.Combine(series.Path, "missing"), ParentId = series.Id };
                    f.Items[added.Id] = added; break;
                case "config-changed":
                    var config = f.Plugin.Snapshot(); config.Enabled = false; f.Plugin.UpdateConfiguration(config); break;
                case "user-gone": f.Users.Setup(u => u.GetUserById(f.User.Id)).Returns((User?)null); break;
            }
        };
        using var response = await f.Delete(series);
        Assert.False(response.IsSuccessStatusCode);
        Assert.Equal(1, f.Handler.CapabilityGets); Assert.Equal(0, f.Handler.DeletePosts); Assert.Equal(0, f.NativeDeletes);
    }
    [Fact]
    public async Task AnInitiallyAbsentNativeCompanionCannotAppearDuringHandshake()
    {
        using var f = new TestFixture(); var episode = f.Episode(); var companion = Path.Combine(f.Root, "episode.srt");
        episode.Setup(i => i.GetDeletePaths()).Returns([
            new MediaBrowser.Model.IO.FileSystemMetadata { FullName = episode.Object.Path },
            new MediaBrowser.Model.IO.FileSystemMetadata { FullName = companion }]);
        f.Handler.OnCapabilities = () => File.WriteAllText(companion, "new companion");
        using var response = await f.Delete(episode.Object);
        Assert.Equal(HttpStatusCode.Conflict, response.StatusCode);
        Assert.Equal(1, f.Handler.CapabilityGets); Assert.Equal(0, f.Handler.DeletePosts); Assert.Equal(0, f.NativeDeletes);
    }

}
