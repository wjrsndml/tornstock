# Torn 量化交易助手 — 浏览器插件开发计划

> 基于本仓库的量化研究成果（z-score 跨股轮动 ~31% 年化），参考 Torn Stock Analyzer v2.36.5 的架构，构建一个实战导向的股票买卖辅助插件。

---

## 1. TSA 架构分析

### 1.1 技术栈
- **运行方式**: Userscript（Tampermonkey/Greasemonkey），约 5800 行
- **匹配页面**: `https://www.torn.com/page.php?sid=stocks*`
- **数据源**: `tornsy.com` API（价格历史）+ `api.torn.com`（用户持仓、物品）
- **存储**: localStorage（价格历史缓存、用户设置、警报）
- **UI**: 浮动 overlay 面板 + 页面内 Quick Trade Bar

### 1.2 核心评分系统（Buy Score）：4+1 个指标
| 指标 | 满分 | 逻辑 |
|------|------|------|
| 1. 距周高点跌幅 | 60p | 动态阈值（股票自身波动率 × 倍数），跌越深分越高 |
| 2. 短期区间位置 | 35p | 价格在 h1-d2 区间内的位置百分位，越靠近底部分越高 |
| 3. 趋势反转 | 40p | 多时间框架确认：m30>h1>h2>h4 = 40p（活跃上涨） |
| 4. MACD 动量 | 25p | MACD 金叉 = 25p，MACD > Signal = 12p |
| 5. RSI 上下文 | 20p | 基于股票自身 RSI 百分位（非固定 30/70 阈值） |

**信号等级**: STRONG BUY(≥100p+金叉) → BUY(≥75p+反转) → CONSIDER(≥45p) → WAIT

### 1.3 TSA 的局限性（我们的优势）
- 只用了 42 天每小时数据的简单回测，缺乏系统性量化验证
- 4 个指标的经验权重（60/35/40/25）未经系统优化
- 没有跨股比较视角（每只股票独立评分）
- 没有完整的退出策略（只告诉你买什么，不告诉何时卖）
- 没有资金管理（仓位分配）

---

## 2. 我们的插件设计方案

### 2.1 核心差异化

| 维度 | TSA | 我们的插件 |
|------|------|------|
| 信号来源 | 4 个经验指标加权 | **z-score (W=32) + 资金流 + 价差信号**，基于 3 年分钟数据系统回测 |
| 回测验证 | 42 天，88% 胜率 | 3 年 walk-forward，~31% 年化，~89% 胜率，5+ 夏普 |
| 选股视角 | 单股独立评分 | **跨股比较**：按 z-score 排名，选最低的 P 只 |
| 退出策略 | 无（仅止盈止损提示） | **z ≥ 0 退出 + 90 天超时**，经过 7 种变体验证 |
| 仓位管理 | 无 | **P=2 或 P=3，资金均分**，可配置 |

### 2.2 功能模块

#### 模块 A: 实时信号面板（核心）
```
┌────────────────────────────────────────┐
│  📊 买入信号 (z-score 轮动)            │
│  ────────────────────────────────────  │
│  🟢 STRONG BUY: SYM  z=-2.1  ████████ │
│  🟢 STRONG BUY: FHG  z=-1.8  ███████  │
│  🟡 CONSIDER:   TCI  z=-1.3  █████    │
│  ⚪ WAIT:       其他 32 只...          │
│                                        │
│  📈 当前持仓                            │
│  SYM 持仓5天 z=-0.3  继续持有          │
│  FHG 持仓12天 z=-0.8 继续持有          │
│  BAG 持仓3天 z=-0.5  继续持有          │
│                                        │
│  ⚠ 卖出信号                            │
│  (无 — 所有持仓 z<0)                   │
└────────────────────────────────────────┘
```

#### 模块 B: 持仓管理
- 显示每只持仓的：入场价、当前 z、持有天数、浮盈/浮亏
- z ≥ 0 自动标记为"建议卖出"
- 超时预警：持有 > 80 天提醒即将超时

#### 模块 C: 权益块 ROI 规划器
- 继承 TSA 的 ROI Planner 功能
- 基于 `analysis/04_dividends.py` 的分红 ROI 数据
- 显示：下一个最有价值的权益块、所需成本、预计回本天数

#### 模块 D: Quick Trade Bar
- 一鍵买入信号最强的股票（预设金额）
- 一鍵卖出所有"建议卖出"的持仓
- 手动输入股数/金额交易

#### 模块 E: 策略状态监控
- 当前参数显示：W=32, k=1.0, P=3
- 资金利用率：在市资金/总资金
- 策略绩效追踪：滚动胜率、累计收益

### 2.3 技术架构

```
┌──────────────────────────────────────────────┐
│  Userscript (Tampermonkey)                    │
│  @match: torn.com/page.php?sid=stocks*        │
├──────────────────────────────────────────────┤
│                                               │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │ Data Layer│  │Signal Engine│  │UI Layer   │ │
│  │          │  │          │  │            │ │
│  │tornsy API│→│z-score   │→│Buy Panel   │ │
│  │(price    │  │(W=32)    │  │            │ │
│  │ history) │  │          │  │Hold Panel  │ │
│  │          │  │flow filter│  │            │ │
│  │Torn API  │  │          │  │Sell Alerts │ │
│  │(portfolio│  │spread-z  │  │            │ │
│  │, items)  │  │(opt)     │  │ROI Planner │ │
│  └──────────┘  └──────────┘  └────────────┘ │
│                                               │
│  ┌──────────────────────────────────────┐     │
│  │ localStorage Cache                    │     │
│  │ - priceHistory (1h bars, rolling)    │     │
│  │ - userSettings (W, k, P, capital)    │     │
│  │ - tradeLog (for performance tracking)│     │
│  └──────────────────────────────────────┘     │
└──────────────────────────────────────────────┘
```

### 2.4 数据流

1. **页面加载时**：
   - 从 localStorage 读取缓存的价格历史
   - 从 tornsy.com API 获取最新价格（增量更新）
   - 从 api.torn.com 获取用户持仓数据
   
2. **信号计算**（每次数据更新后）：
   - 对 35 只股票分别计算 z_32 (基于 32×24 小时的滚动窗口)
   - 计算资金流过滤（5 日 total_shares 变化）
   - 按 z-score 排名，筛选 z < -1.0 且 flow > 0 的股票
   - 选前 P 只（默认 P=3）作为买入候选
   
3. **持仓评估**：
   - 对每只持仓计算当前 z-score
   - z ≥ 0 → 建议卖出
   - 持有天数 ≥ 90 → 超时强制退出

4. **UI 更新**：
   - 渲染买入/卖出/持仓面板
   - 更新权益块 ROI 数据

### 2.5 开发步骤

#### Phase 1: 最小可用版本（3-5 天）
1. 搭建 Userscript 框架（manifest、CSP 适配、API 封装）
2. 实现价格历史缓存（tornsy API → localStorage）
3. 实现 z-score 计算引擎（W=32, k=1.0, flow filter）
4. 实现简易 UI：买入排名列表 + 持仓状态
5. 测试：在 Torn 页面上显示实时信号

#### Phase 2: 完整功能（1-2 周）
6. 接入 Torn API 获取用户持仓
7. 实现卖出信号（z ≥ 0 + 超时预警）
8. 实现 Quick Trade Bar
9. 实现权益块 ROI 规划器
10. 添加价格警报
11. 自动刷新 + 设置面板

#### Phase 3: 增强功能（2-4 周）
12. 日线级价差信号叠加（spread_z，权重 w=0.2）
13. 策略绩效追踪（累计收益、胜率、夏普）
14. 可视化图表（价格 + z-score 走势）
15. 交易日志导出
16. 参数自定义（W, k, P, 超时天数）

---

## 3. 关键实现细节

### 3.1 z-score 计算（JavaScript）

```javascript
function calcZScore(sym, priceHistory, W) {
    // W = 32 (days) × 24 (hours) = 768 hourly bars
    var entries = priceHistory[sym];
    if (!entries || entries.length < W * 24) return null;
    
    var sorted = entries.slice().sort((a, b) => a.ts - b.ts);
    var prices = sorted.map(e => e.price);
    
    // 取最近W天的小时数据
    var windowPrices = prices.slice(-W * 24);
    var mean = windowPrices.reduce((a, b) => a + b, 0) / windowPrices.length;
    var variance = windowPrices.reduce((s, p) => s + (p - mean) ** 2, 0) / windowPrices.length;
    var std = Math.sqrt(variance);
    
    var currentPrice = prices[prices.length - 1];
    return (currentPrice - mean) / std;
}
```

### 3.2 资金流过滤（JavaScript）

```javascript
function calcFlowFilter(sym, priceHistory) {
    // 5日total_shares变化
    var entries = priceHistory[sym]; // 需要 contain total_shares 字段
    if (!entries || entries.length < 5 * 24) return false;
    
    var sorted = entries.slice().sort((a, b) => a.ts - b.ts);
    var shares5dAgo = sorted[sorted.length - 5 * 24].total_shares;
    var sharesNow = sorted[sorted.length - 1].total_shares;
    
    return sharesNow > shares5dAgo; // 股本净流入
}
```

### 3.3 小时级 vs 日线级

两种模式可切换：
- **小时级**（推荐）：需要 ~768 条小时数据 (32天×24h)，信号更精确，CAGR ~31%
- **日线级**（备选）：需要 ~32 条日线数据，执行更简单，CAGR ~29% (含价差信号)

### 3.4 API 端点

| 数据 | API | 说明 |
|------|------|------|
| 价格历史 | `tornsy.com/api/{symbol}?interval=m1&from={ts}&to={ts}` | 分钟级，插件缓存为小时级 |
| 用户持仓 | `api.torn.com/user/?selections=stocks&key={key}` | 需要 Torn API key |
| 用户资金 | `api.torn.com/user/?selections=basic,money&key={key}` | 现金余额 |
| 物品价格 | `api.torn.com/market/{id}?selections=itemmarket&key={key}` | ROI 计算 |

---

## 4. 与 TSA 的关键区别总结

| | TSA | 我们的插件 |
|------|------|------|
| 方法 | 4 指标加权评分 | z-score (W=32) + 资金流 + 价差 |
| 验证 | 42 天回测 | 3 年 walk-forward |
| 选股 | 独立评分，买入所有高分 | 跨股比较，买最低 z 的 P 只 |
| 退出 | 止盈止损 % | z ≥ 0 退出 + 90 天超时 |
| 仓位 | 无建议 | P=2/3，资金均分 |
| 年化 | 未公布 | ~31% (小时级) / ~29% (日线) |

---

## 5. 文件结构规划

```
torn-stock-assistant/
├── torn-stock-assistant.user.js    # 主脚本（Userscript格式）
├── README.md                        # 安装和使用说明
├── CHANGELOG.md                     # 版本记录
└── screenshots/                     # 截图
    ├── buy-panel.png
    ├── hold-panel.png
    └── roi-planner.png
```
