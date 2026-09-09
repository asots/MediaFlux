# 爱奇艺官方追番表 fixture

`weekly_tracking.json` 来自 **2026-09-09 21:35:06 +08:00** 的真实官方公开 JSON：

```text
GET https://mesh.if.iqiyi.com/portal/lw/v7/channel/page/tracking
channelId=4&mode=page&page=1&v=17.091.26283
```

- HTTP 200，原始响应 611467 bytes。
- 原始 SHA-256：`4aa392ab7210167f7a90ebd39a7c67be6adb2ac6f3161e92e92b23d28001e758`。
- 保留实际频道、星期/日期分组、节目 ID/title/页面、正片/单集标记、更新时间标签、节目首播年、VIP 标识。
- 删除 session/pingback、用户/作者信息、图片、播放参数、剧情等无关字段。没有改造节目或排期值，没有补造免费数据。
- 7 个日期分组窗口为 2026-09-06（周日）至 2026-09-12（周六），另有 coming 组。测试固定 clock=2026-09-09；本周过滤后应得到 **40 节目、72 events、8 条有时刻**。
- 例：`items[0].video[4].data[0]` 为“逆天邪神年番”，`album_id=8837962497879001`；组日期 09-10/周四，`tag3lines[0].text=明日09:00更新`。
- “明日”标签可能重复出现在该节目的其他星期组；只能在日期一致时使用时刻。普通 `dq_updatestatus` 和节目 `date/showDate` 不用于推算周排期。
- VIP 图标不等于这次排期明确面向会员或免费，事件 audience 保持 unknown，免费进度保持空。
- `PREVUE` / coming 不作为已核验正片周排期。

旧 `recommend_animation.json` 与 `unverified_signals.json` 仅保留为先前探索资料，当前生产适配器及主要测试不再使用它们；任何 fixture 都不会被生产代码读取。

## 动漫单请求边界

生产每轮仅执行上述 tracking **1 次 GET**，请求主机仅 `mesh.if.iqiyi.com`；节目官方页面 URL 只用于展示，不补抓目录、baseinfo 或专辑页。`clock=` 保留 date、上海 naive datetime 和 aware datetime 转上海日期的语义。

结构/类型/日期不匹配不构造排期。某个日期组畸形时，保留其他已核验日期；请求失败、挑战或非成功状态不重试、不回显异常内的敏感文本。取消信号向上传播。旧窗口的 09-06 周日不能移到 09-13；“明日”时刻也不能复制到其他星期。

测试使用固定时钟及 `import tests` 配置/DB 隔离，阻断真实 socket/DNS；全部请求由 Mock/MockTransport 离线回放。非动漫、恶意 URL 与畸形日期组仅为合成错误边界，不宣称是额外实采。

## 2026-09-10 平台原封面独立取证

`platform_images_20260910.json` 是本轮 **2026-09-10T01:38:45+08:00** 开始的独立实采：仅上述 tracking 入口/原参数 **1 次 GET**。独立请求及图片证据在 `platform_images_20260910_provenance.json`，不覆盖历史快照。

- 正式原封面仅取 `items[].video[].data[].image_cover`。不取视频剧情 `back_image`，也不从 `image_url_normal` / `album_image_url_hover` 的尺寸版本猜图。
- 84 条已核验节目行、44 个唯一原封面；全部 HTTPS、无 query。本次字段实际出现的精确主机及次数为：`pic0.iqiyipic.com` 8、`pic1.iqiyipic.com` 8、`pic2.iqiyipic.com` 13、`pic3.iqiyipic.com` 4、`pic4.iqiyipic.com` 5、`pic5.iqiyipic.com` 20、`pic6.iqiyipic.com` 7、`pic7.iqiyipic.com` 6、`pic8.iqiyipic.com` 7、`pic9.iqiyipic.com` 6。这是实采字段集合，不是主机枚举探测。
- 观察到的原图路径为 `/image/{8位日期}/{2位十六进制}/{2位十六进制}/a_{数字}_m_601_m{数字}.webp`；不带 `_592_333` 尺寸后缀。封面文件内的资产编号不用于专辑身份、TMDB/豆瓣匹配或收藏。
- 唯一图片 GET：`2026-09-10T01:40:26+08:00`，取「灵武大陆」`pic0.iqiyipic.com` 上原样 HTTPS 无 query `image_cover`；200 / `image/webp` / **11366 bytes**，RIFF/WEBP magic 匹配。**其余九个 CDN 主机仅有官方字段证据，没有 GET 小样**；本轮未验证任何去除 transform query 的 URL。
- 排期窗口真实为 2026-09-07 至 2026-09-13；固定上海 2026-09-10 回放得到 **44 节目 / 84 事件**。原始/脱敏响应以及改动前后 provider 的非图片字段完全一致。历史 2026-09-09 fixture 仍为 40 节目 / 72 事件，不重映射旧日期。
- 脱敏只保留公开日期/身份/排期/正式封面，保留列表索引，删除会话、广告跟踪、播放参数与其他图片。排期和图片的完整 URL、ISO 时间、状态、MIME、字节、SHA-256 在 provenance；私有 tmp 中的原始响应交付前删除。
- 请求使用精确 HTTPS、公网 DNS 固定与 TLS、12s/2MiB/至少 2s 间隔、无身份 Cookie/账号/重试/跳转。只生成共享安全 `platform_poster_key`；缺图/坏图不丢节目，多张有效图冲突不任选一张，不改变原有 identity/日期拒绝规则。测试先隔离 DB 并禁 socket/DNS，所有恶意/缺失/冲突用例仅为离线合成。
