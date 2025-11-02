# AI Trading Bot - 智能量化交易机器人

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/AI-DeepSeek-orange.svg" alt="AI Model">
  <img src="https://img.shields.io/badge/Exchange-Binance-green.svg" alt="Exchange">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</div>

---

**一个基于 DeepSeek AI 的智能加密货币期货量化交易框架，支持币安 Binance U本位永续合约自动交易。**

本项目受到 [nof1.ai](https://nof1.ai/) 的启发，旨在打造一个开源、可扩展的 AI 量化交易系统。

## ✨ 特性

### 🤖 AI 驱动
- **DeepSeek Reasoning Model**: 使用 `deepseek-reasoner` 模型进行深度推理分析
- **多维度决策**: AI 综合分析技术指标、市场情绪、资金费率等多个维度
- **思维链推理**: 展示完整的 AI 推理过程，决策更加透明可信

### 📊 多周期技术分析
- **多时间框架**: 支持 5m、15m、1h、4h、1d 等多个周期
- **丰富技术指标**: RSI、MACD、EMA、SMA、ATR、布林带
- **K线模式识别**: 分析最近 18 根 K 线走势

### 🛡️ 风险管理
- **仓位控制**: 最小/最大仓位限制（默认 10%-30%）
- **每日最大亏损限制**: 默认 10%
- **连续亏损保护**: 最大连续亏损次数限制
- **杠杆管理**: 可配置 1-100 倍杠杆
- **止盈止损**: 自动设置止盈止损订单

### 💹 双向交易
- **做多 (LONG)**: 看涨时开多仓
- **做空 (SHORT)**: 看跌时开空仓
- **灵活持仓**: 支持同时持有多个方向的仓位
- **自动平仓**: AI 决策 + 止盈止损自动平仓

### 🔄 多币种支持
- **一键分析**: 单次 API 调用分析多个币种
- **独立决策**: 每个币种独立分析和决策
- **智能优化**: 综合考虑账户状态和历史决策

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Binance 期货账户（U本位合约）
- DeepSeek API Key

### 安装步骤

1. **克隆仓库**
```bash
git clone https://github.com/vnxfsc/ai-trading-bot.git
cd ai-trading-bot
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **配置环境变量**
```bash
cp config/env.example .env
# 编辑 .env 文件，填入你的 API 凭证
```

`.env` 文件内容：
```env
# Binance API 配置
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET=your_binance_secret_here

# DeepSeek API 配置
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

4. **配置交易参数**
编辑 `config/trading_config.json`:

```json
{
  "trading": {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "default_leverage": 3,
    "max_leverage": 100,
    "min_position_percent": 10,
    "max_position_percent": 30,
    "reserve_percent": 20
  },
  "risk": {
    "max_daily_loss_percent": 10,
    "max_consecutive_losses": 5,
    "stop_loss_default_percent": 2,
    "take_profit_default_percent": 5
  },
  "ai": {
    "model": "deepseek-reasoner",
    "temperature": 0.7,
    "max_tokens": 2000
  },
  "schedule": {
    "interval_seconds": 180,
    "retry_times": 3,
    "retry_delay_seconds": 5
  }
}
```

5. **运行程序**
```bash
python src/main.py
```

## 📁 项目结构

```
ai-trading-bot/
├── config/                      # 配置文件
│   ├── env.example              # 环境变量示例
│   └── trading_config.json      # 交易配置
├── src/                         # 源代码
│   ├── main.py                  # 主程序入口
│   ├── ai/                      # AI 相关
│   │   ├── deepseek_client.py   # DeepSeek API 客户端
│   │   ├── prompt_builder.py    # 提示词构建器
│   │   └── decision_parser.py   # 决策解析器
│   ├── api/                     # 交易所 API
│   │   └── binance_client.py    # 币安客户端
│   ├── config/                  # 配置管理
│   │   ├── config_loader.py     # 配置加载器
│   │   └── env_manager.py      # 环境变量管理
│   ├── data/                    # 数据管理
│   │   ├── market_data.py       # 市场数据管理器
│   │   ├── position_data.py     # 持仓数据管理器
│   │   └── account_data.py      # 账户数据管理器
│   ├── trading/                 # 交易执行
│   │   ├── trade_executor.py    # 交易执行器
│   │   ├── position_manager.py  # 仓位管理器
│   │   └── risk_manager.py     # 风险管理器
│   └── utils/                   # 工具类
│       ├── indicators.py       # 技术指标计算
│       └── decorators.py       # 装饰器
├── requirements.txt             # Python 依赖
└── README.md                    # 项目说明
```

## 🏗️ 核心模块

### 1. AI 决策引擎 (`src/ai/`)

#### DeepSeek Client
- 调用 DeepSeek API 进行推理分析
- 展示完整的推理过程
- 支持多种 AI 模型

#### Prompt Builder
- 构建多维度市场分析提示词
- 包含技术指标、持仓、历史决策等上下文
- 支持多币种统一分析

#### Decision Parser
- 解析 AI 返回的 JSON 格式决策
- 验证决策合法性
- 应用默认值

### 2. 交易所接口 (`src/api/`)

#### Binance Client
- 完整的币安期货 API 封装
- 市场数据获取（K线、行情、资金费率等）
- 账户和持仓管理
- 交易执行（开仓、平仓、止盈止损）

### 3. 风险管理 (`src/trading/`)

#### Risk Manager
- 仓位大小限制
- 每日最大亏损检查
- 连续亏损保护

#### Position Manager
- 仓位管理
- 多币种持仓追踪

#### Trade Executor
- 开仓/平仓执行
- 止盈止损设置
- 失败重试机制

### 4. 市场数据分析 (`src/data/`)

#### Market Data Manager
- 多周期 K 线获取
- 技术指标计算（RSI、MACD、EMA、ATR、布林带）
- 实时行情数据

#### Indicators
- RSI (相对强弱指数)
- MACD (指数平滑移动平均线)
- EMA/SMA (指数/简单移动平均)
- ATR (平均真实波动范围)
- Bollinger Bands (布林带)

## 🤖 AI 决策示例

### 输入（市场数据）
```
=== BTC/USDT ===
价格: $95,000.00 | 24h: +1.23% | 15m: +0.50%
资金费率: 0.000100 (多头付费) | 持仓量: 1,000,000

【4h周期】
RSI: 44.5 | MACD: 0.0025
EMA20: 95,200 | EMA50: 94,500
最近18根K线（OHLC）: ...
```

### 输出（AI 决策）
```json
{
  "action": "BUY_OPEN",
  "reason": "4h周期上升趋势，RSI44未超买，MACD转正，短期看涨",
  "confidence": 0.75,
  "leverage": 5,
  "position_percent": 20,
  "take_profit_percent": 5.0,
  "stop_loss_percent": -2.0
}
```

## ⚙️ 配置说明

### 交易配置 (`trading_config.json`)

```json
{
  "trading": {
    "symbols": ["BTCUSDT", "ETHUSDT"],    // 交易币种
    "default_leverage": 3,                 // 默认杠杆
    "max_leverage": 100,                  // 最大杠杆
    "min_position_percent": 10,           // 最小仓位占比
    "max_position_percent": 30,          // 最大仓位占比
    "reserve_percent": 20                 // 预留资金占比
  },
  "risk": {
    "max_daily_loss_percent": 10,        // 每日最大亏损
    "max_consecutive_losses": 5,         // 最大连续亏损次数
    "stop_loss_default_percent": 2,      // 默认止损百分比
    "take_profit_default_percent": 5     // 默认止盈百分比
  },
  "ai": {
    "model": "deepseek-reasoner",        // AI 模型
    "temperature": 0.7,                   // 模型温度
    "max_tokens": 2000                    // 最大 token 数
  },
  "schedule": {
    "interval_seconds": 180,             // 交易周期（秒）
    "retry_times": 3,                    // 重试次数
    "retry_delay_seconds": 5              // 重试延迟（秒）
  }
}
```

## 🛡️ 安全建议

1. **API 权限控制**
   - 仅授予必要的权限（期货交易）
   - 不要授予提币权限
   - 定期轮换 API 密钥

2. **资金管理**
   - 使用小额资金进行测试
   - 设置合理的最大亏损限制
   - 定期检查账户状态

3. **风险管理**
   - 不要过度杠杆（建议 3-10 倍）
   - 监控市场异常波动
   - 设置止损保护

## 📝 运行日志示例

```
============================================================
🚀 AI交易机器人启动中...
============================================================
✅ 配置加载完成
✅ 环境变量加载完成
✅ API客户端初始化完成
✅ 数据管理器初始化完成
✅ 交易执行器初始化完成
✅ AI组件初始化完成
============================================================
🎉 AI交易机器人启动成功！
============================================================

💰 账户信息:
   总权益: 10,000.00 USDT
   未实现盈亏: +125.50 USDT
   保证金率: 150.25%

============================================================
📅 交易周期 #1 - 2024-01-15 10:30:00
============================================================

🤖 调用AI一次性分析所有币种...

📊 AI多币种决策总结:
   BTCUSDT: BUY_OPEN - 多周期上升趋势，RSI44未超买，4hMACD转正
   ETHUSDT: HOLD - 震荡整理，等待方向突破
   SOLUSDT: CLOSE - 4h RSI超买80，顶部信号
============================================================
```

## 🎯 使用场景

1. **趋势跟踪**: 基于多周期技术指标识别趋势
2. **反转交易**: 捕捉超买超卖区域的反弹
3. **套利交易**: 利用资金费率差异
4. **网格交易**: 配合止盈止损进行区间交易
5. **多币种组合**: 分散风险，提高收益稳定性

## 🔧 自定义开发

### 添加自定义指标

编辑 `src/utils/indicators.py`:
```python
def calculate_custom_indicator(data: pd.DataFrame) -> float:
    """你的自定义指标"""
    # ... 计算逻辑
    return indicator_value
```

### 添加新的交易策略

编辑 `src/main.py`，在 `TradingBot` 类中添加自定义逻辑。

## 📊 性能监控

- 查看运行日志中的交易记录
- 监控账户盈亏变化
- 分析 AI 决策准确率
- 调整参数优化收益

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## ⚠️ 免责声明

**本软件仅供学习和研究使用。加密货币交易存在高风险，可能导致资金损失。使用本软件进行实盘交易的风险由使用者自行承担。开发者不对任何交易损失负责。**

## 📜 License

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- [nof1.ai](https://nof1.ai/) - 项目灵感来源
- [DeepSeek](https://www.deepseek.com/) - AI 模型提供
- [Binance](https://www.binance.com/) - 交易所 API

## 📧 联系方式

如有问题或建议，欢迎通过以下方式：

- **GitHub Issues**: [提交 Issue](https://github.com/vbxfsc/ai-trading-bot/issues)

---

<div align="center">
  <p>⭐ 如果这个项目对你有帮助，欢迎 Star 和 Fork！</p>
  <p>Made with ❤️ by the community</p>
</div>

