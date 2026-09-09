# 日历元资料公开 suggest fixture

2026-09-09 使用隔离测试环境、现有 CalendarHttp 与固定主机
`movie.douban.com` 的 `/j/subject_suggest?q=…` 匿名 GET 获取。
无 Cookie/账号、无视频请求、无真实 TMDB 调用；不跟随重定向。
探测共4次、间隔至少2秒：#1/#2/#3 查询“琅琊榜”（后两次为脚本参数未生效的重复），
#4 查询“斗罗大陆”；已停止真实探测。

保留完整 suggest 数组及公开作品字段（episode/img/title/url/type/year/sub_title/id），
仅移除 subject URL 上的 suggest 跟踪 query，不保存 headers、Cookie 或其它会话数据。

- `douban-suggest-langyabang.json`：完整3条。电视剧实际 `type="movie"`，
  已播剧集的 episode 是正整数字符串；待播续作可为 `"unknow"`，不作为剧集证明。
- `douban-suggest-douluo.json`：完整5条。动画与真人剧都是 `type="movie"`，
  名称/年份不同；不能用第一个结果、删去季数或省略年份匹配。
- 图片域名实际见 `img1.doubanio.com`、`img3.doubanio.com`、`img9.doubanio.com`，
  均在 discovery_image 的现有白名单内。fixture 不下载图片。

suggest 不提供可核验评分、动画 genre 或搜索总数；本功能不编造这些字段，
只在完整有界数组中接受精确标题+可靠年份+剧集证据的唯一候选。

## TMDB 连通性样本（2026-09-09）

`tmdb-shixiong.json`、`tmdb-panlong.json` 是官方 TMDB `/3/search/tv` 对“师兄啊师兄”“盘龙”的真实完整首屏结果投影；仅保留匹配所需公开字段和真实分页计数，不包含 API key、请求头或用户配置。额外实测取得第一部作品的官方海报，HTTP 200、JPEG、54328 bytes（海报本体不纳入仓库）。

测试将上述两次结果离线回放；其他节目返回空候选，不将其当作全站真实匹配率。`test_real_tmdb_replay_and_controlled_secondary_cover_share_existing_watchlist` 中豆瓣对应数据为明确合成的兼容候选，仅用于 API 双图签名/收藏身份验证，不是假冒现场样本。
