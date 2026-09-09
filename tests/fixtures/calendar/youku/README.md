# 优酷公开周历实采 fixture

## 动态修复轮：两次独立匿名会话取得真实 09.07–09.13 排期

本节补充此前仅 SSR 的局限，**不改写下文历史取证结论**。生产不读 fixture，不把 08.31–09.06 挪成本周。

- `dynamic_week_20260910.json` 来自 **2026-09-10T03:07:53.412923+08:00** 的官方动态 SUCCESS 响应（UTC 为 2026-09-09），保留 WEBCOMIC→每日更新→7日→公开节目卡的原始必要字段。47节目、92排期、91时刻、47平台原图key，已知免费进度为0。
- `dynamic_week_20260910_provenance.json` 记录原响应 520,563 bytes、SHA-256、官方 Page/module38 及同页 vendors/module80 的 URL/hash/MIME 证据。SDK仅静态分析，未执行未知脚本。
- **2026-09-10T03:41:12.652307+08:00** 完整生产 provider 类的独立冷会话复核，用正式 CalendarHttp 依次读取 SSR（仍旧周）、匿名初始化、动态成功，3 GET / 4.882秒，再得47节目92排期91时刻。不是仅手工probe或离线fixture成功，也不是对现有运行实例的部署证明。
- 第二次原响应524,511 bytes，9个官方 `img` 发生真实轮换，其余身份/标题/全部事件/会员与免费性质完全不变。`dynamic_runtime_20260910_poster_delta.json` 只保留这9个实际公开字段，不重复整份大树；`dynamic_runtime_20260910_provenance.json` 记录原hash、两fixture hash及原始完整 SourceResult 与“基准fixture+原图delta”在捕获周/下一周/跨年日的严格相等验证。不是合成日期或追加图片下载。

SSR 拒访预检区分非脚本文本（实体解码、跨标签拼接）、明确顶层 JSON 错误响应/直接验证导航与具体结构标记；正常 SDK 检查、错误factory、示例字符串和未调用回调不是当前错误响应。检测只识别已知证据，不执行JS、不推断任意脚本的执行结果；未匹配不构成无挑战证明，原HTTP/API错误停止边界仍独立强制生效。

正常匿名协议来自页面实际使用的官方 MTOP SDK：固定只读
`https://acs.youku.com/h5/mtop.youku.columbus.home.query/1.0/`，固定 WEBCOMIC、JSON响应和公开客户端参数。
新隔离调用不带账号/浏览器/生产 Cookie；只有 SDK 定义的 TOKEN_EMPTY/TOKEN_EXOIRED 且服务器确实为本逻辑域签发匿名grant时，才允许一次续接。
固定IP传输下不依赖自动CookieJar：仅解析两个已知匿名Cookie，校验域/path/失效时间/控制符/重复或冲突，不接受其它账号/挑战状态；调用结束释放状态，不入缓存或日志。
HTTP401/403/429、验证码、登录/非法访问、混合错误、无grant或第二次仍失败均停止，不循环重试、不换身份、不跟随跳转。

每次SSR/初始化/动态GET共用限频、请求配额、公网DNS固定、TLS、12秒和2MiB原始identity流边界；多主机关闭keepalive避免共享物理IP时跨Host/SNI复用。
本轮合计官方实际请求9/12：频道HTML+Page2、SDK1、动态桶6（其中含完整复核的SSR1）；未用的动态2和SDK1额度不滚动追加。
其它平台、海报、元数据来源本轮均无额外实网请求。

原始HTML/JS/API树在完成字段脱敏和严格回放核对后删除，仅保留公开字段fixture及来源/hash；Cookie/token/sign具体值从未保存。
离线测试覆盖真实动态树、SSR优先、七日/频道/类型/受众、初始化/预算/取消、拒访停止、缓存从error恢复以及平台图不伪造元资料或收藏身份。

## 历史 SSR 正向样本：webcomic.html

采集于 **2026-09-09 21:30（Asia/Shanghai）**，官方入口 `https://www.youku.com/ku/webcomic`。
原始响应 HTTP 200，1,575,645 bytes，SHA-256：
`b0fd5750f27a8a081c516a49d15e6205d70b66deeda5e43dcf4b8f2e890e4389`。

本文件从该次实采的脱敏原模块生成，保留 `window.__INITIAL_DATA__.moduleList[2].components[0]`：

- `typeName=KU_FLIX_MULTI_TAB_A`、`type=35`、`title=每日更新`。
- 七个 `tabList` 日期 `09.07..09.13`，对应七组 `itemList`；按日记录数 `13/13/14/14/12/13/13`。
- 92 条日期记录、47 个节目级 ID；91 条 `reason.text.title` 明确给出更新时间。3 条原文明确 SVIP；另外 41 条带 VIP 节目角标。
- 《师兄啊师兄》周三 `10:00 SVIP更新1话`、周四 `10:00更新1话` 是同节目不同排期，必须分别保留。
- 周六《开心锤锤Story：锤星食界》的 reason 是剧情文案，时刻未知，不能借用该节目其他日期的时刻。

节目 title、action.type/value/extra.category、reason、mark、普通进度和标签均为原始公开元数据；无关模块以空对象代替，跟踪、IP、广告、预览资源、图片及凭据字段已经删除。未生成或改写任何节目、日期、更新时间。采集年份应由测试注入 clock 固定；生产必须匹配当前上海周，不能把旧 fixture 挪到本周。

## 动漫请求边界

有效当周 SSR 仍仅 GET `https://www.youku.com/ku/webcomic` **1 次**。仅在已识别的七日模块无有效当前周时，允许下面有官方 SDK 证据的动态查询，至多追加 2 GET；只允许精确 `www.youku.com` / `acs.youku.com`，不补抓目录或详情。规范化的 `https://v.youku.com/video?s=<节目ID>` 只用于展示，不代表允许请求该主机。

仅明确 `extra.category=动漫` 且 `JUMP_TO_SHOW` 的节目可进入日历。日期严格匹配当前上海周，SSR 即使 HTTP 200 但仍返回旧周，也不挪动日期；动态结果必须独立通过相同的当前七日校验，两者均无有效当前周才返回 unavailable。“今”标签与 selectedIndex 不能覆盖真实 tab 日期；无年份的跨年周按当前完整上海周匹配十二月/一月，不改写 fixture。

会员与非会员排期独立，同日同刻但受众不同的事件不能合并；节目角标、普通或限免集数均不推定免费进度。有效日期组件不会被另一畸形组件抹掉；错误或挑战停止请求、不重试、不回显敏感异常，取消信号向上传播。

fixture 仅供离线回归，生产不读取本目录。固定 `clock=2026-09-09` 回放上述样本仍为 **47 个动漫、92 个事件**。类型/日期冲突、跨年日期、非会员受众、上限及推荐/限免卡片等变异用例是明确的合成边界，不宣称为额外现场采集。

测试在应用导入前先 `import tests` 隔离配置/DB，并阻断真实 socket/DNS；共享 HTTP 通过 MockTransport 验证 1 次匿名 GET，无详情抓取能力依赖。

## 历史补齐轮现场（01:38–01:56 +08:00）：当时当前周仍未恢复

`anime_20260910_completion_stale_webcomic.html` 与
`completion_20260910_observations.json` 是本轮脱敏负向样本和采样台账，**不是运行时备用来源**。
同一轮来源总配额6次，实际5次；独立图片总配额1次，实际1次。所有现场请求间隔超过2s，12s/2MiB，
精确HTTPS主机、公网DNS固定与TLS验证，无账号Cookie/重试/自动跳转、无身份切换、无随机或日期参数。

### 真实日期与官方入口链

1. **2026-09-10T01:38:30.666574+08:00**：GET `https://www.youku.com/ku/webcomic`，
   HTTP 200，`text/html; charset=UTF-8`，1,548,646 bytes，原始SHA-256：
   `a2e3055a2d121a130e1f8bfe3ec9cd7f0368181473d74b0526f7c3af0619ba50`。
   真实七日仍是 **08.31–09.06**，每天 `11/12/13/14/12/12/12` 条、42节目86排期。
   以当前上海周 **2026-09-07–13** 回放仍为 `unavailable`，不能改日期或套用历史正向fixture。
2. `https://comic.youku.com/` 来自本轮唯一官方范围web搜索，现场302明确指向
   `https://www.youku.com/channel/webcomic`；后者现场302又指回第一步的 `webcomic`。
   两次跳转均由固定客户端拒绝自动跟随；新目标只在单独计数、人工批准后请求。
   302响应体未消费，body大小/hash记null，不伪造空响应的实采哈希。

### 第4次检测假阳性与主线授权第5次澄清

第1次页面明确引用的业务脚本
`https://g.alicdn.com/youku-node/pc-pages-v2/4.1.822/v2/static/js/Page.chunk.js`
在来源第4次返回200、`application/javascript`、954,363 bytes，SHA-256：
`b75e77c2b96aa89df0475aa6b27889b0394c2cb0ec525955d6e9cb2a4859ae6b`。
当时仅因文本启发式命中停止，未能证明实际验证码挑战。

主线随后明确允许同轮剩余额度重取该原URL一次，仅澄清假阳性并静态提取公开调用点。
**2026-09-10T01:56:09.218779+08:00** 第5次同UA、无Cookie/参数，MIME、大小、hash与第4次完全相同。
`node --check` 在无NODE_OPTIONS/preload环境中只编译通过，未执行bundle。
静态审查确认 `FAIL_SYS_ILLEGAL_ACCESS` 唯一出现于内容发布失败的catch，提示“发布内容存在非法输入”，
**是普通业务代码，不是服务器返回的验证码页面**。

静态找到频道调用 `mtop.youku.columbus.home.query`，参数由 `getChannelPlatoParams` 构造，
动漫nodeKey为 `WEBCOMIC`，请求数据字段为 `ms_codes`、`params`、`system_info`。
调用交给 `window.lib.mtop` 或外部module 80，并启用Cookie同步；SDK实现不在已取得bundle中。
`ecode=0` 不构成免签证明；既不能断言必须账号登录，也没有证据允许拼接/请求免签匿名固定HTTPS URL。
因此**第6次不使用，未执行或猜解签名/token接口，当前周仍未恢复**。
周历组件只读取已有 `tabList`/`itemList` 并切换本地索引，没有发现独立逐日取数调用。

### 原图字段与独立图片核验

`poster_provenance.json` 保留本轮第1次旧周卡片中 `img`/`hImg` 的3个实际公开字段小样：
- `img`：`liangcang-material.alicdn.com` 84条，`m.ykimg.com` 2条。
- `hImg`：`liangcang-material.alicdn.com` 86条。
- 原值均为HTTP，无查询/凭据。主线批准第一条img逐字path升级HTTPS进行独立图片第1/1次请求。

**2026-09-10T01:55:57.499192+08:00** 请求
`https://liangcang-material.alicdn.com/prod/upload/981696b79f914f5497f78fb8b9c8d14e.webp.jpg`，
返回 **200 / image/jpeg / 236,869 bytes**，JPEG magic `FFD8FF`，SHA-256：
`2327c354be1fe81a886a117d38177a37581cf008d563e7f3dfcd9a7ab3a29a88`。
没有跟随跳转、重试或请求其它图片；`m.ykimg.com` **仅有源字段证据，没有HTTPS图片GET验证**。
主线已确认两个精确host/namespace，并负责共享白名单，provider只调用共享key校验函数。

仅接 `img`，缺失/非法不以 `hImg` 补缺；同节目两个不同安全key只清图片，不删节目/排期，
后续重复旧key不会恢复冲突图片。等价HTTP/HTTPS/协议相对URL和缩放参数归同一个安全key；
key不带查询、不作为元资料或收藏身份。图片字段不放宽当前周日期校验，也不增加provider HTTP请求。

### 脱敏与离线回归边界

脱敏HTML仅保留实际周历日期/节目身份/排期文案/会员标签及一个真实业务script src；
无关模块、跟踪/广告/账号、安全SDK和播放器代码不保留。所有原始HTML/JS/图片已删除，
只留来源/响应元数据、脱敏调用说明、字段样本与哈希。

测试将同次剥离的图字段按节目ID回接旧周fixture，仅在原始08.31–09.06周回放42节目86事件；
当前09.07–09.13仍不可用。其余图片/当周卡片组合为明确的合成边界，不代表当周现场恢复。
覆盖图片缺失/非法URL、跨主机命名空间混淆、等价URL、同ID多key冲突、非动漫同ID与独立收藏身份。
原有七日/免费性质/上限/302/错误停止边界保留。全部回归先import tests隔离DB并阻断socket/DNS；
MockTransport、node语法检查和历史回放不算当前周可用性证明。
