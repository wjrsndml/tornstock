# Torn Stock Market — 分钟级历史数据集

本仓库包含 **Torn City** 游戏中全部 35 只个股 + 1 只市场指数（共 36 个标的）的 **1 分钟精度历史行情数据**，数据来源为 [tornsy.com](https://tornsy.com/api) 公开 API。

> ⚠️ **重要说明**  
> - 数据仅供学习与研究使用。  
> - **TCSE** 是 **Torn City Stock Exchange 市场指数**，反映整体市场走势，**不是个股**。TCSE 额外包含 `marketcap`（总市值）字段。  
> - 数据源更新频率为每分钟一次，延迟约 5–10 秒。

---

## 1. 数据集概览

| 项目 | 说明 |
|------|------|
| 来源 | <https://tornsy.com/api>（免费无需 API Key） |
| 频率 | **1 分钟** (m1) |
| 时间范围 | 最近 3 年 |
| 条目数 | ~5,500 万行 |
| 标的总数 | **36**（35 只个股 + 1 只指数） |
| 文件格式 | [Apache Parquet](https://parquet.apache.org/)（zstd 压缩） |
| 数据文件 | `data/merged/{STOCK}.parquet` |
| 抓取脚本 | `fetch_m1_data.py` |

---

## 2. 文件结构

```
tornstock/
├── README.md                # 本文件
├── requirements.txt         # Python 依赖
├── fetch_m1_data.py         # 数据抓取脚本（支持断点续传、10 并发）
├── stocks.txt               # 游戏内股票背景故事的完整原文
├── .gitignore               # 已排除 .venv、chunks/、日志等
└── data/
    ├── merged/              # ★ 最终合并后的数据文件
    │   ├── ASS.parquet
    │   ├── BAG.parquet
    │   ├── ... (共 36 个文件)
    │   └── YAZ.parquet
    └── fetch.log            # 抓取日志（已 gitignore）
```

---

## 3. 数据格式

每个 Parquet 文件包含以下字段：

| 列名 | 类型 | 说明 |
|------|------|------|
| `timestamp` | int64 | Unix 时间戳（秒），已取整到分钟 |
| `price` | float64 | 该分钟收盘价（美元） |
| `total_shares` | int64 | 该股票当时的总股本 |
| `stock` | string | 股票代码（3–4 字母大写，如 `ASS`、`TCSE`） |

**TCSE（指数）** 额外包含：

| 列名 | 类型 | 说明 |
|------|------|------|
| `marketcap` | float64 | 市场总市值 |

> 注意：原始 API 返回的 `price` 为字符串，本数据集已转为 float64。

---

## 4. 使用方法

### 4.1 Python / Pandas

```python
import pandas as pd

# 读取单只股票
ass = pd.read_parquet("data/merged/ASS.parquet")
print(ass.head())
#    timestamp  price  total_shares stock
# 0 1690070400 310.82  3.375814e+09   ASS
# 1 1690070460 310.84  3.375815e+09   ASS
# ...

# 时间戳转日期
ass["datetime"] = pd.to_datetime(ass["timestamp"], unit="s", utc=True)
```

### 4.2 DuckDB（推荐大数据分析）

```sql
-- 一次性加载全部股票
SELECT stock, count(*) AS rows
FROM 'data/merged/*.parquet'
GROUP BY stock
ORDER BY stock;

-- 计算 TCSE 与各股票的 30 日滚动相关性
SELECT ...
```

### 4.3 Polars

```python
import polars as pl

df = pl.read_parquet("data/merged/LSC.parquet")
# 筛选最近一个月
recent = df.filter(pl.col("timestamp") > 1750000000)
```

### 4.4 计算常见指标

```python
import pandas as pd

df = pd.read_parquet("data/merged/ASS.parquet")
df["datetime"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
df = df.set_index("datetime")

# 日线 OHLC（从 1 分钟聚合）
daily = df["price"].resample("1D").agg(["first", "max", "min", "last"])
daily.columns = ["open", "high", "low", "close"]

# 收益率
df["return_1m"] = df["price"].pct_change()
df["return_1h"] = df["price"].pct_change(60)

# 波动率（1 小时窗口，标准差）
df["vol_1h"] = df["return_1m"].rolling(60).std()
```

---

## 5. 股票目录

### 5.1 指数 ⚠️

| 代码 | 名称 | 说明 |
|------|------|------|
| **TCSE** | TCSE Market Index | **市场总指数**，非个股。反映全部股票的综合走势，额外包含 `marketcap` 字段 |

---

### 5.2 个股一览

| # | 代码 | 名称 | 所属行业 | 当前股价 | 总股本 | 投资者数 |
|---|------|------|----------|----------|--------|----------|
| 1 | ASS | Alcoholics Synonymous | 酒类进口 | $360.21 | 192.6 亿 | 16,535 |
| 2 | BAG | Big Al's Gun Shop | 武器弹药 | $486.60 | 163.5 亿 | 12,488 |
| 3 | CBD | Herbal Releaf Co. | 药用大麻 | $404.93 | 201.5 亿 | 21,848 |
| 4 | CNC | Crude & Co | 原油 | $868.85 | 318.4 亿 | 17,920 |
| 5 | ELT | Empty Lunchbox Traders | 房地产/装修 | $321.39 | 75.1 亿 | 7,269 |
| 6 | EVL | Evil Ducks Candy Corp | 糖果/快乐值 | $659.25 | 71.9 亿 | 21,917 |
| 7 | EWM | Eaglewood Mercenary | 私人军事 | $290.32 | 184.8 亿 | 19,408 |
| 8 | FHG | Feathery Hotels Group | 酒店旅游 | $893.65 | 805.0 亿 | 38,449 |
| 9 | GRN | Grain | 农业/谷物 | $320.43 | 267.7 亿 | 33,273 |
| 10 | HRG | Home Retail Group | 房地产 | $261.19 | 1561.5 亿 | 18,837 |
| 11 | IIL | I Industries Ltd. | 老式计算机硬件 | $135.82 | 188.3 亿 | 23,871 |
| 12 | IOU | Insured On Us | 保险 | $176.89 | 839.1 亿 | 25,997 |
| 13 | IST | International School TC | 教育 | $541.00 | 13.9 亿 | 17,645 |
| 14 | LAG | Legal Authorities Group | 法律服务 | $458.90 | 115.4 亿 | 15,596 |
| 15 | LOS | Lo Squalo Waste | 废物处理 | $112.36 | 343.5 亿 | 10,067 |
| 16 | LSC | Lucky Shots Casino | 赌场/博彩 | $579.62 | 121.4 亿 | 30,131 |
| 17 | MCS | Mc Smoogle Corp | 食品加工 | $835.66 | 412.4 亿 | 35,133 |
| 18 | MSG | Messaging Inc. | 社交媒体 | $295.09 | 23.8 亿 | 9,718 |
| 19 | MUN | Munster Beverage Corp. | 能量饮料 | $554.91 | 632.6 亿 | 16,494 |
| 20 | PRN | Performance Ribaldry | 成人娱乐 | $617.98 | 407.4 亿 | 35,447 |
| 21 | PTS | PointLess | 加密货币/Points 交易 | $78.81 | 2228.6 亿 | 24,858 |
| 22 | SYM | Symbiotic Ltd. | 制药 | $735.37 | 519.3 亿 | 57,117 |
| 23 | SYS | Syscore MFG | 网络安全/杀毒 | $669.36 | 135.1 亿 | 10,875 |
| 24 | TCC | Torn City Clothing | 服装零售 | $515.01 | 104.2 亿 | 10,856 |
| 25 | TCI | Torn City Investments | 银行/投资 | $1,165.14 | 242.8 亿 | 37,853 |
| 26 | TCM | Torn City Motors | 汽车销售 | $295.06 | 121.9 亿 | 15,426 |
| 27 | TCP | TC Media Productions | 媒体广播 | $524.49 | 83.2 亿 | 14,385 |
| 28 | TCT | The Torn City Times | 报纸/新闻 | $323.46 | 127.9 亿 | 40,199 |
| 29 | TGP | Tell Group Plc. | 广告营销 | $151.10 | 454.1 亿 | 15,490 |
| 30 | THS | Torn City Health Service | 医疗保健 | $381.83 | 196.3 亿 | 47,417 |
| 31 | TMI | TC Music Industries | 音乐产业 | $229.69 | 852.4 亿 | 18,251 |
| 32 | TSB | Torn & Shanghai Banking | 银行 | $1,170.10 | 257.9 亿 | 27,332 |
| 33 | WLT | Wind Lines Travel | 航空/旅行 | $799.26 | 219.8 亿 | 14,071 |
| 34 | WSU | West Side University | 高等教育 | $108.84 | 957.8 亿 | 61,880 |
| 35 | YAZ | Yazoo | 搜索引擎/科技 | $52.51 | 618.6 亿 | 19,485 |

> 注：股价、股本、投资者数来自 2026-07-23 实时数据。

---

### 5.3 每只股票的背景故事（来自官方 Stock Market 页面）

#### ASS — Alcoholics Synonymous（酒类进口）
由 Reg Olsenderk 于 1994 年创立，是 Torn City 唯一持有酒类进口许可证的公司（口号："It comes through us before it goes through you"）。最初从欧洲进口无酒精饮料，主打神职人员、孕妇和 AA 戒酒互助会的市场。结果这些啤酒实际上是 0% 碳水、25% 酒精——这个美丽的误会奠定了 ASS 对 Torn 进口啤酒市场的垄断。

---

#### BAG — Big Al's Gun Shop（武器弹药）
由 Alan Hasselhoff 创立，四十年来遵循三条铁律：**No Background Checks, No Irish, No Refunds**。Torn City 武器、弹药和装甲的主要来源，90% 库存来自与恐怖组织和腐败军人的长期合作。

---

#### CBD — Herbal Releaf Co.（药用大麻）
大麻在 Torn City 虽已合法，但毒瘾的毁灭性后果意味着无副作用的替代品始终有市场。CBD 系列产品包括精油、软糖、电子烟、汽水和婴儿食品，提供"绿色地球订阅盒"。

---

#### CNC — Crude & Co（原油）
原油及能源相关产业。

---

#### ELT — Empty Lunchbox Traders（房地产/装修）
Torn City 快速扩张的两大支柱：廉价易得的毒品，以及 ELT 设计和建造的顶级开发项目。公司向每位新市民捐赠免费棚屋，确保 Torn 公民从抵达那一刻就踏上房产阶梯。

---

#### EVL — Evil Ducks Candy Corp（糖果/快乐值）
1984 年，EVL 的巧克力大师们发现可以用含有多巴胺的类巧克力物质替代真正的巧克力。随后拓展到无糖糖果、非薄荷薄荷糖、无松露松露巧克力。每根巧克力棒还含有七种强效泻药——**"我们给你 s**ts and giggles!"**

---

#### EWM — Eaglewood Mercenary（私人军事）
由前美国海军陆战队员 Rick Eag 和 Phillip Le-Wood 于 1995 年创立。擅长政变、暗杀和假旗恐怖袭击。作为私人军事承包商（PMC），员工不受日内瓦公约约束。

---

#### FHG — Feathery Hotels Group（酒店旅游）
自 2019 年前总监 George Gooseberry 被解雇后蒸蒸日上。新任运营总监 Jeremy Hedgemaster 承诺到 2024 年所有客房、浴室和厕所将 100% 无摄像头，所有房间加装黑光灯选项。

---

#### GRN — Grain（农业/谷物）
经济不确定时代，投资者更倾向于谷物而非黄金。合成培育的谷物具有高营养价值，转基因作物不受 Torn City 任何限制——**10 位科学家中有 4 位认为对人类免疫系统损害极小**。

---

#### HRG — Home Retail Group（房地产）
独立房地产中介，拥有遍布 Torn City 的物业组合。如果你在高价值区域买不到房，他们会在你的街道尽头制造凶杀案；如果一对年轻夫妇在价格上挤压你，他们会让女性一方怀孕来制造紧迫感——**"We make a house a homicide."**

---

#### IIL — I Industries Ltd.（老式计算机硬件）
以落后于时代、领先于潮流而自豪。主营业务是过时的系统和配件，为收藏家和爱好者服务，同时也为 Torn 的网络犯罪分子提供设备——因为有了正确的电缆和介质，没有什么比带 B 驱动器和拨号调制解调器的 IBM PC 340 更难攻破。

---

#### IOU — Insured On Us（保险）
承诺即使你的索赔是欺诈也照样赔付——事实上，他们坚持如此。每份保单本身也由另一家保险公司承保，确保有人对你的保单索赔时他们也能索赔。**"With IOU, everyone wins."**

---

#### IST — International School TC（教育）
200 多年来一直站在 Torn City 成人教育的最前线。课程面向维持意识所需的实用知识。如果你已从社会大学毕业，IST 就是夺命的大学——**"If you've graduated from the school of hard knocks, IST is the university of taking a life."**

---

#### LAG — Legal Authorities Group（法律服务）
法律的长臂在 Torn City 或许够不到多远，但一旦被抓，后果对你的银行账户和社交生活都是毁灭性的。LAG 提供证人恐吓、证据篡改、贿赂、儿童冒充和硬盘删除等服务。大多数客户在 60 分钟内获释。

---

#### LOS — Lo Squalo Waste（废物处理）
一家 **100% 合法** 的企业，专门从事市政废物的运输和处理。环保的垃圾再分配方法确保绝大多数物品可以回收再利用——**是的，包括纸巾！** 精密的垃圾追踪系统能知道上周二你晚餐吃了什么，但不会说出来，因为这"100% 是合法生意"。

---

#### LSC — Lucky Shots Casino（赌场/博彩）
Torn 唯一持牌赌场，每年从忠实的赌客中赚取数万亿。近年来还将非法街头赌场的"许可和保护"纳入业务。开发的 AI 算法 **GAMBLOR** 确保即使你作弊，赔率也永远对他们有利。

---

#### MCS — Mc Smoogle Corp（食品加工）
以使用竞争对手乐意丢弃的动物和器官而闻名。"鼻到直肠"（nose-to-rectum）的可持续理念意味着法律上无法列出每种成分。他们相信只要吃下去就是食物，产品自豪地超过了 FDA 推荐的每百万昆虫碎片上限——**"extra protein is a feature, not a bug"**。

---

#### MSG — Messaging Inc.（社交媒体）
反社交媒体领域的先驱，在信息误导时代驱动流量和参与度无出其右。如果竞品品牌走红，就让俄罗斯水军用随意种族主义内容攻击他们；如果 Twitter 被地下室宅男的负面评论淹没，就让印尼机器人农场用虚构的南亚美女的互动提升品牌形象。

---

#### MUN — Munster Beverage Corp.（能量饮料）
1935 年由 Herzklopfen 家族在德国创立。核心产品 Munster Energy 诞生于针对 Kool-Aid 的贸易禁运，很快成为国防军的首选兴奋剂。产品线扩展后包括 Red Cow Pour Femme、Goose Juice Jr 和 Taurine Elite（专为低智商人群营销）。每款产品的共同点：**严重依赖军用级安非他命**。

---

#### PRN — Performance Ribaldry（成人娱乐）
产品组合多元化至流媒体和远程性玩具（teledildonics）新兴市场。DVD 仓库覆盖所有可以想象的细分领域，为变态们提供安全的离线替代方案——**"you'll never catch VD from a DVD"**。

---

#### PTS — PointLess（加密货币/Points 交易）
Torn City Points 的数字交易平台。Points 于 1997 年由匿名程序员**鶏肉刺身**创造，最初存储在 8 英寸软盘上，当面交易。28.8k 调制解调器问世后实现了在线交易。2004 年获 99 年合同运营 Points Market，如今仍是世界上最安全的加密平台之一，仅不到 84% 的投资者遭遇过欺诈。

---

#### SYM — Symbiotic Ltd.（制药）
Torn City 领先的制药公司。首席化学家 Amanda Ravenscroft 教授确保街头药品的纯度全球最高。研发资金来自国际部门 Bitter Pill Inc，专注于收购救命药专利，通过 Shkreliball 算法识别最有利可图的价格欺诈目标。

---

#### SYS — Syscore MFG（网络安全/杀毒）
由重罪犯创始人 Stuart Bridgens 的理念驱动：产品应该同时提供问题和解决方案。Bridgens 的防火墙软件内置恶意软件，在订阅到期的瞬间感染用户系统。**"Syscore: Helping You Go Viral since 1994."**

---

#### TCC — Torn City Clothing（服装零售）
服装及时尚零售。

---

#### TCI — Torn City Investments（银行/投资）
认识到人生太短暂不适合长期投资，TCI 精心策划了一系列缩短型投资产品，适合濒死、受死亡威胁或对死亡好奇的客户。**"无你的事"**（none of your business）政策确保你永远不会知道钱投到了哪里——**"Invest with TCI today: You'll make a killing."**

---

#### TCM — Torn City Motors（汽车销售）
Torn City 排名第一的"非自愿捐赠"汽车经销商。现金收车，不管车辆状态，也不管你从哪个州偷的。经过精心的身份改装服务后，即便原车主被这辆车当街撞倒也认不出来——而且经常发生！每辆车附赠六副可互换车牌和追踪退款保证——**"Drive it like you stole it, because someone did."**

---

#### TCP — TC Media Productions（媒体广播）
Torn City 第五大受欢迎的媒体机构，平均广播时长仅 34 秒——迎合现代注意力缺陷观众。成功将真人秀格式卖给外国广播公司，包括 **"Man Vs Hydraulic Press"**、**"Chedburn Nerfed My 900lb Son"**、**"Project Hit and Runway"** 和 **"Real Housewives of Subversive Alliance"**。

---

#### TCT — The Torn City Times（报纸/新闻）
自 2015 年起读者量暴增 4,000%。2019 年推出文章弹窗提醒系统，确保《时代》无与伦比的新闻始终是头条，**不管你想不想看**。

---

#### TGP — Tell Group Plc.（广告营销）
好的广告讲故事，伟大的广告让你觉得错过了什么。TGP 的创新 B2B（Business to Bastard）营销技术针对最绝望的社会成员。通过有心理摧毁力的广告打击核心人群，**将客户的自尊心降到宁愿结束自己生命也不买下一个领先品牌的程度**。

---

#### THS — Torn City Health Service（医疗保健）
世界上最有效率的公立医疗提供者，每天有数千名枪击受害者、瘾君子和遭遇不幸的黄瓜爱好者涌入他们的门。低资格要求、无背景调查和脆弱的药品储存吸引了多样化的应聘者。

---

#### TMI — TC Music Industries（音乐产业）
T-pop（将家庭友好的歌词和电子节拍与性感表演者结合）的流行导致收入飙升。T-pop 艺人贡献了 TC Music 年收入的 58%，代表性团体包括 Hot Sausage、Grandpappy Never Loved You 和 **Steve Buscemi's Invisible Mustard Machine**。

---

#### TSB — Torn & Shanghai Banking（银行）
致力于一件事：**完全不透明**。不受国际法约束，无需配合外部调查，公司资产免受其来源后果的保护。无论你是洗钱大亨、有前途的逃税者还是黑市器官交易的主要参与者，TSB 都是你事业的核心选择。

---

#### WLT — Wind Lines Travel（航空/旅行）
Torn 排名第一的旅行运营商，五十多年来持续理解 Torn 公民的特定需求。对毒品采取**睁一只眼闭一只眼**政策——**"with Wind Lines Travel, you'll always be flying high, whether you want to or not."**

---

#### WSU — West Side University（高等教育）
Torn 唯一的高等教育和免费厕所设施提供者。课程涵盖多学科，针对 Torn 公民独特需求定制。校训：**Inebriari, Occiditis, Commercia**（醉酒、杀戮、商业）。

---

#### YAZ — Yazoo（搜索引擎/科技）
尽管与一家英国奶昔品牌经历了一系列惨烈的法律纠纷，Yazoo 终于重新成为立陶宛、乌兹别克斯坦和波兰边境小镇 Görlitz-Zgorzelec 最受欢迎的搜索引擎。收购了 **Dik-Doc**、**InstaGran** 和 **Club Pangolin** 等社交媒体品牌，仍是世界上唯一提供在线国际象棋聊天室并公开承认向中国出售用户数据的搜索引擎。

---

## 6. 股票分红 / 权益块一览

在游戏中持股达到一定数量可获得被动加成或定期分红。

### Active 型（定期分红）

| 股票 | 所需股数 | 周期 | 分红内容 |
|------|----------|------|----------|
| ASS | 1,000,000 | 7 天 | 1x Six Pack of Alcohol |
| BAG | 3,000,000 | 7 天 | 1x Ammunition Pack（特殊弹药） |
| CNC | 7,500,000 | 31 天 | $80,000,000 |
| EVL | 100,000 | 7 天 | 1,000 Happy |
| EWM | 1,000,000 | 7 天 | 1x Box of Grenades |
| FHG | 2,000,000 | 7 天 | 1x Feathery Hotel Coupon |
| GRN | 500,000 | 31 天 | $4,000,000 |
| CBD | 350,000 | 7 天 | 50 Nerve |
| HRG | 10,000,000 | 31 天 | 1x Random Property |
| IOU | 3,000,000 | 31 天 | $12,000,000 |
| LAG | 750,000 | 7 天 | 1x Lawyer Business Card |
| LSC | 500,000 | 7 天 | 1x Lottery Voucher |
| MCS | 350,000 | 7 天 | 100 Energy（最多 10 块） |
| MUN | 5,000,000 | 7 天 | 1x Six Pack of Energy Drink |
| PRN | 1,000,000 | 7 天 | 1x Erotic DVD |
| PTS | 10,000,000 | 7 天 | 100 Points |
| SYM | 500,000 | 7 天 | 1x Drug Pack |
| TCT | 100,000 | 31 天 | $1,000,000 |
| THS | 150,000 | 7 天 | 1x Box of Medical Supplies |
| TMI | 6,000,000 | 31 天 | $25,000,000 |
| TSB | 3,000,000 | 31 天 | $50,000,000 |
| TCC | 7,500,000 | 31 天 | 1x Clothing Cache |

### Passive 型（永久被动加成）

| 股票 | 所需股数 | 加成内容 |
|------|----------|----------|
| ELT | 5,000,000 | 房屋升级 10% 折扣 |
| IIL | 1,000,000 | 病毒编码时间 -50% |
| IST | 100,000 | 免费教育课程 |
| LOS | 7,500,000 | 任务奖励 +25% |
| MSG | 300,000 | 免费报纸分类广告 |
| SYS | 3,000,000 | 高级防火墙 |
| TCP | 1,000,000 | 公司销售加成 |
| TCI | 1,500,000 | 银行利息 +10% |
| TCM | 1,000,000 | 赛车技能 +10% |
| TGP | 2,500,000 | 公司广告加成 |
| WLT | 9,000,000 | 私人飞机（旅行）+ 免疫侦探社航班延误 |
| WSU | 1,000,000 | 教育时间 -10% |
| YAZ | 1,000,000 | 免费横幅广告 |

---

## 7. 价格机制说明

据 Torn 开发者 Chedburn 所述：

- 价格**每分钟**变动一次（原为每 15 分钟）
- 价格走势**参照现实世界中同行业真实股票**
- 预期年均增长约 **10%**，但不保证
- 部分股票波动极大，可能一个月翻倍或暴跌

---

## 8. 抓取脚本用法

```bash
# 安装依赖
python3 -m venv .venv && source .venv/bin/activate
pip install aiohttp tqdm pyarrow pandas

# 抓取全部数据（支持断点续传）
python fetch_m1_data.py

# 只抓特定股票
python fetch_m1_data.py --stocks ASS,LSC,TCSE

# 指定年限
python fetch_m1_data.py --years 2

# 调整并发和速率
python fetch_m1_data.py --concurrency 20 --rate 20
```

更多参数见 `python fetch_m1_data.py --help`。

---

## 9. 致谢

- 数据来源：[tornsy.com](https://tornsy.com/api) — 免费、无需 API Key 的公开 API
- 游戏：[Torn City](https://www.torn.com/)
- 股票背景故事：来自 Torn 官方 Stock Market 页面
