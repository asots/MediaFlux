# 腾讯真实动漫周排期快照（2026-09-09）

来源为官方频道内联 `__channel_prefetch__` 指向的匿名只读 POST：
`https://pbaccess.video.qq.com/trpc.vector_layout.page_view.PageService/getPage?video_appid=3000010&vversion_platform=2`。

- 七个 `calendar_YYYYMMDD.json` 是 2026-09-07 至 2026-09-13 的真实 `channel_play_schedule` 响应，42 个日历行、28 个节目 CID。
- 只请求动漫频道 `100119`。首次默认日响应提供本周导航，其余六日使用已核验的导航日期与模块参数；完整回放恰好 **7 次 POST**，不请求节目详情。
- `provenance.json` 保留原历史采集批次的请求、时间、HTTP 状态、响应大小/哈希及脱敏文件哈希；当前回归仅校验其中七个 `calendar_YYYYMMDD.json` 记录，不改写历史批次。
- 脱敏仅删除广告、Cookie、跟踪、图片及非必要模块内容；原始 CardList/子列表索引与排期字段值保留。
- 这些是实采快照，不是合成正例。生产适配器不读取 fixture。

## 排期语义

1. 切日参数取官方导航的 `week/page_id/un_mod_id/un_module_key`，传给两个公开 params 对象。响应 `selected=今天` 不随切日变化，不能把所有行归到今天。
2. `pay_time` 与 `free_time` 分别产生 member/free 事件。同日同刻但受众不同也不能合并。VIP 与 SVIP 细分说明保留在官方排期原文中。
3. `pay_episode/free_episode` 可能是未来日的排期目标，不直接表示已经播出的进度；`episode_updated` 不能作为免费进度。
4. 9 月 11 日「狐妖小红娘13 黄风岭篇」标有 `is_trailer=1`，但同日公开首播计划明确；只作为未来首播排期，不声称已播出正片。
5. 仅动漫类型 `3` 可成为节目；普通推荐、未核验的导航、旧周响应不能作为本周排期。

## 离线回归

固定 `TencentCalendarProvider(clock=lambda: datetime(2026, 9, 9, 21, 0))`，不依赖 CI 当天。该批真实回放得到 **28 个动漫、45 个日期事件**，2026-09-09 有 **6 个动漫**，覆盖本周 **7/7 天**。所有免费进度保持空，未将未来排期集数当作已免费内容。

部分失败使用真实动漫响应加故障注入：前两日成功、第三次请求失败时，保留 10 个节目/11 个事件并标记 2/7；分别在第 2–7 次请求失败时，仅保留此前成功日期并停止后续请求；首请求失败则 unavailable。取消信号向上传播。畸形日期响应不计入成功覆盖，不丢弃此前事件。

测试在应用导入前先 `import tests` 隔离配置/DB，并阻断真实 socket/DNS 调用。跨类型卡片与恶意导航仅是合成错误边界，不是额外实采。

## 2026-09-10 平台原封面独立取证

`platform_images_20260910.json` 是本轮 **2026-09-10T01:38:42+08:00** 开始的独立实采：仅上述动漫接口默认查询 **1 次 POST**，未调用 provider 的七日遍历，没有 `week` 切日参数。对应证据在 `platform_images_20260910_provenance.json`，不修改此前快照或 provenance。

- 正式原图字段：`channel_play_schedule.children_list.list.cards[].params.image_url`。6 条均为精确主机 `vcover-hz-pic.puui.qpic.cn`，路径 `/vcover_hz_pic/0/{15位CID}{13位数字}/750`，HTTPS、无 query；是平台原横图，不猜竖图地址。
- 默认日为 2026-09-10。脱敏保留原列表索引、公开身份/日期/排期和 `image_url`，不保留广告、会话、播放参数或其他图片。原响应与脱敏回放得到相同 **6 节目 / 7 事件**。
- 唯一图片 GET：`2026-09-10T01:40:23+08:00`，取「灵境行者」原样 HTTPS 无 query URL；200 / `image/jpeg` / **128710 bytes**，JPEG magic 匹配。图资产没有附加变换参数，不宣称验证过移除 transform 的其他 URL。
- 所有请求均精确官方 HTTPS、公网 DNS 固定 Host/SNI 与 TLS 校验，12s/2MiB 上限，前后请求间隔至少 2s；无身份 Cookie/账号、重试、跳转或详情/播放/搜索请求。排期与图片完整 URL、ISO 时间、状态、MIME、字节和 SHA-256 见独立 provenance；原始响应仅在私有 tmp 临时核对，交付前删除。
- 新测试回放此默认日后，在首次切日注入 unavailable，不把一次现场伪装成完整周采集。另用历史七日 fixture 验证旧排期仍为 28 节目 / 45 事件。
- 只写 `platform_poster_key`，由共享安全函数校验。缺图、坏图保留节目；同 CID 的多张有效图或元信息冲突只清空封面，不改变既有 CID/排期合并。所有恶意/缺失/冲突用例均为离线合成边界，不是新增实采。
