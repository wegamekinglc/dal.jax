# dal.jax 开发交接

更新于 2026-10-07。本文写给接手 [issue #1](https://github.com/wegamekinglc/dal.jax/issues/1) 后续开发的人（或下一个 Claude Code 会话），说明三件事：现在做到哪了，有哪些已定的决定，下一步从哪里开始。总体设计以 issue #1 为准，本文只记录 issue 里没有、但在实现中确定下来的内容。

## 1. 现状

| 里程碑 | 状态 | 位置 |
|---|---|---|
| P0 基础设施、BS 模型、随机数、MC 引擎 | 完成 | PR #2 |
| 普通 Python 示例（`examples/`，8 个，均含 DAL 数值和性能对照） | 完成 | PR #2 |
| P1 Script 前端 | 完成 | PR #2（commit `8750dfd`、`e7caf50`） |
| P2 exact 降级与事件引擎 | 完成 | PR #2 |
| P3 fuzzy 降级、求导、scan 分组 | 完成 | PR #2 |
| P4 并行与 GPU | 完成：CPU 调优、实际 CUDA 验证、GPU dtype/块策略、RBG 一致性和性能报告 | PR #2 |
| P5–P7 | 未开始 | — |

- **分支**：所有工作都在 `feature/jax-mc-engine` 上，PR #2 的 base 是 `master`，还没有合并。`master` 上只有项目早期的探索性 notebook（PR #2 中已删除）。
- **测试**：`uv run pytest` 为 594 passed、26 skipped（169.57 s），跳过的是默认关闭的 GPU 测试。26 个 GPU 测试已在实际 CUDA 13 环境通过，含百万路径 autocall 对照。P2 新增 120 个测试，其中 32 个与 dal-python 对照；P3 新增 67 个；P4 新增 22 个 CPU 测试和 26 个 GPU 测试。
- **静态检查**：对应 Codacy 默认规范的 `lizard -C 8`、`pylint -E`、`bandit`、`semgrep --config p/python` 都已清零。本次采用本地等价检查；P4 commit `02018d2` 的 Python 3.13、3.14 和原 notebook CI 均成功；示例迁移后改为执行全部 Python 脚本。本地 8 个脚本均按默认 65,536 路径/8 设备/3 次热运行通过，CI 参数 4,096 路径/4 设备/1 次也通过，另在实际 GPU 上运行 08。每个脚本均有 DAL 数值和性能对照。

## 2. 代码地图

```
src/dal_jax/
  config.py            x64、虚拟 CPU 设备数、PRNG 实现、编译缓存
  errors.py            DAL 同名异常；script_error(msg) 根据消息里的 "Code: " 前缀选择 ScriptError 子类
  strings.py           DAL 的大小写不敏感字符串（ci_key/ci_eq/CIMap）、std::stod 语义、DebugNumber 格式
  index.py             EQ / FX / IR 指数名解析与规范名（Index::Parse）
  api.py               与 dal-python 同名的 Product_* / EvaluationDate_* / BSModelData_New / MonteCarlo_Value
  dates/               Date（Excel 序号，1970-01-01..2149-06-05）、增量、节假日（calendar_data.py 为导出数据）、日程、计息基准
  random/              sobol（directions.npy）、inverse_normal、bridge、prng（RBG 保持逐块 key 语义）
  models/              base（Model 协议、SampleDef、Scenario）、bs
  mc/                  settings、engine（分块、checkpoint、value/pricer）、parallel、tuning（GPU 精度/内存块策略）
  script/
    lexer.py           词法；index 字面量整体成词（EQ[x]@date、EQ[x]>3M）
    preprocessor.py    宏、常量、数值向量、日程（ParseSchedule）、PeriodBegin/PeriodEnd
    parser.py          递归下降解析，含 FOR 展开、DCF 折叠、EXERCISE、PAYS ON、FIX
    ast.py             frozen dataclass 节点；Spot/Fix 带 observation_id（P2 给 Spot 分配，Fix 留到 P5）
    product.py         ScriptProductData（不可变输入）/ ScriptProduct（解析结果、分区、变量编号、payoff_index）
    preparation.py     prepare → PreparedProduct（不可变、可哈希）；观察绑定、历史回放、串联 passes、path_product
    passes/            varindex → ifmeta → constfold → domain(+intervals) → constcond；eventgroup 规范化与分组
    debug.py           DebugNode：DebugJson、树形、旧版 s-expression 三种渲染
    diagnostics.py     debug_json（schema /1）、describe（schema /2）、debug_tree、debug_text
    lower/exact.py     标量事件降级；IF affected_vars 合并、active 掩码；NumPy 后端做历史硬回放
    lower/fuzzy.py     连续/离散比较、布尔 degree、嵌套 IF 混合；复用 exact 的安全算术
    lower/events.py    事件常量槽、相邻同模板事件 lax.scan、短组展开
    lower/smoothing.py CSpr / BFly 及其带 lb/rb 的两参数形式；可选 C1 smoothstep
tests/                 random/ models/ mc/ script/ dates/ test_index.py oracle/ gpu/（--run-gpu 可选）
examples/              8 个普通 Python 脚本 + _common.py；results/ 保存逐个执行的数值和计时 JSON
scripts/               export_sobol_directions.py、export_calendars.py（从 DAL 源码重新生成数据）
benchmarks/bench_mc.py 与 dal-python 同机计时
benchmarks/bench_script_compile.py 单路径脚本价格/参数梯度的编译和图规模；script_compile_cpu.json 为 CPU 实测
benchmarks/bench_suite.py 标量脚本端到端 CPU/GPU/DAL 基准；p4_*.json 为实测，docs/performance.md 为报告
```

模块和 issue 第 10 节的规划有几处不同，都是有意为之：

- 没有单独的 `events.py` / `schedule.py`：事件解析放在 `product.py`，日程解析在 `preprocessor.py`，日程生成在 `dates/schedule.py`。
- 没有 `index/` 包，只有一个 `index.py` 模块，因为脚本层只用到指数名解析。
- `fixprep`（FIX 准备）还没有实现，属于 P5；`eventgroup` 已在 P3 实现。

## 3. 已定的决策

### 3.1 对照基准

- **语义以 DAL `master` 为准**，不是以 dal-python 2026.9.25 为准。dal-python 2026.9.25 还不支持向量、`FOR`、`PAYS ... ON`、IR 指数和 `30U/360`。这些新功能用移植的 DAL C++ 单测来验证；与 dal-python 的对照只用它支持的子集。
- **`Product_Describe` 比较时去掉 `regression_features` 键**：这是 DAL `master` 新增的键，dal-python 2026.9.25 没有。
- **错误对照只比较错误码和核心消息**：DAL 的异常文本里带 C++ 源文件路径和 NOTICE 上下文行，见 `tests/oracle/test_dal_script_frontend.py` 中的 `_core()`。
- 以下 3 处差异来自 DAL `master` 与 2026.9.25 版本之间的行为变化，是预期内的，对照测试已排除：
  - 出现第二个 `ELSE` 时报 `DuplicateElse`；
  - `MAX(1)` 按向量归约解析；
  - `FIX(EQ[a] 2)` 被词法层的后缀检查拒绝。
- **P2 补充：多参数 `MAX/MIN` 的求值差异**：本机的 dal-python 2026.9.25 在 `compiled=None/False/True` 下都只计算前两个参数（已实际检查 `MAX(1,2,3)` 和 `MAX(SPOT(),101,102)`）。本实现遵循 DAL `master` 的树求值器，计算全部参数；价格 oracle 用两个参数，全部参数的语义由 `tests/script/test_exact.py` 验证。

### 3.2 前端语义细节（移植时踩过、容易改错的地方）

- **常量折叠**：`MAX` / `MIN` 按全部参数折叠，与 DAL 的树求值器一致；DAL 的 compiled 模式只看前两个参数。具名常量（脚本参数）永远不折叠，否则 `d_STRIKE` 会丢失。
- **DomainProcessor 的保留规则**：fuzzy 模式下，没有观察计划（`known_observations is None`）时只有常量定义域的比较保持连续平滑；有观察计划时所有 fuzzy 比较都保持连续。见 `domain.py` 中的 `_retain_fuzzy`。
- **数字词法**：`1e-7` 会被拆成 `1e`、`-`、`7` 三个词，DAL 也是这样，然后报 "Not a valid number string"；要写 `1E7` 或 `0.0000001`。`stod` 下溢或上溢时报 `"stod"`。
- **增量里的数字超出 32 位 int 时报 `"stoi"`**，与 DAL 用的 `std::stoi` 一致。
- **CIMap 保留第一次出现的拼写**，迭代时按 DAL 的 CI 排序（字母先折叠成大写，`{|}~` 排在 Z 之后）。变量名、常量名的输出顺序都依赖这个规则。
- **`FIXING: BEGIN|END` 必须写在日程最后**：DAL 解析它时不会跳过取值，这里照原样移植。
- **DAL 的 bug 不要移植**：
  - 非 ASCII 字符会触发 dal-python 的 UTF-8 解码错误，我们报 `InvalidScript`；
  - `"START: 2023-01-01 FREQ 1M"`（FREQ 后缺少冒号）会让 DAL 段错误。

### 3.3 P0 引擎中需要保持的约定

- **路径号映射**：全局路径号 n 对应 Sobol 点 n+1，任意设备都能直接计算任意路径，不需要 `SkipTo`。LSMC 的训练、验证、定价三个路径区间以后也按这个规则划分。
- **`shard_map` 中的参数转换**：复制的参数要在入口一次性 `pcast` 成 device-varying，scan 的 carry 也要 `pcast`。否则每个块都会在反向传播里插入一次跨设备 `psum`，梯度不随设备数扩展。
- **按块取样本**：模型输出通过 `Scenario.samples()` 按块取出。逐个取静态切片会让反向模式变得很慢。
- **`params` 结构是 `{"model": {...}, "script": {...}}`**：脚本常量与模型参数重名时报 `ReservedIdentifier`，避免 `d_<name>` 被覆盖。
- **到期日的障碍判断执行两次**：DAL 用 `START/END/FREQ` 生成的日程包含到期日，与到期日那行事件合并后，障碍判断会执行两次，fuzzy 模式下 `(1-δ)` 也就乘了两次。`tests/support.py` 的手写 payoff 按这个语义实现；P2/P3 降级脚本时，这个结果会由事件合并自然得到。

### 3.4 代码风格

- AST 节点都是 frozen dataclass，每个 pass 返回新对象，不在原节点上修改。
- 分派优先用"类型/kind → 处理函数"的表，而不是长 `match`：Codacy 的 Lizard 要求圈复杂度 ≤ 8，长 `match` 或 `if/elif` 很容易超标。参考 `passes/domain.py`、`debug.py`。
- 注释和文档字符串用英文，与现有代码一致；PR 描述和本文用中文。

## 4. 如何验证

```bash
uv sync                                   # 安装开发依赖，包括作为对照的 dal-python
uv run pytest                             # 全部测试；oracle 测试在没有安装 dal-python 时跳过
uv run pytest -m oracle                   # 只跑与 dal-python 的对照
uvx lizard -C 8 -w src tests benchmarks scripts examples
uv run --with pylint python -m pylint -E --disable=import-error src tests
uvx bandit -q -r src benchmarks scripts examples
uvx semgrep scan --config p/python --metrics off --error src scripts benchmarks examples
```

- **DAL 源码**：本机的 DAL 源码在与本仓库同级的 `../Derivatives-Algorithms-Lib`（`dal-cpp/` 和 `dal-python/`）。移植时以其中的源码和测试为金标准。
- **查看 dal-python 2026.9.25 的源码**：`git -C ../Derivatives-Algorithms-Lib archive dal-python-v2026.9.25 | tar -x -C <空目录>`。
- **依赖锁定**：CI 没有 `uv.lock`，`dal-python>=2026.9.25` 会装到 PyPI 上的最新版。如果新版改变了错误文本或输出，oracle 测试会失败。届时有两个选择：更新对照用例，或者把版本固定为 `==2026.9.25`。

## 5. P2–P4 实现与下一步 P5

P2 验收已完成：European、亚式（标量写法）、autocall 在相同 Sobol 点下，与 dal-python 的 PV 相对误差 ≤ 1e-10，含 Brownian bridge 开关和跨批次的尾块掩码。

### 5.1 P2 的入口和约定

- **准备**：`dal_jax.prepare(data, evaluation_date, historical_spots=...)` 返回 `PreparedProduct`；`prepared.path_product()` 交给现有 `MonteCarloEngine`。时间轴用 ACT/365F，每个未来事件请求 numeraire，与 legacy 路线一致。每个含 SPOT 的日期共用一个 `SpotObservation`；P2 从当日 `Sample.spot` 读取，完整指数输出绑定留到 P5。
- **历史回放**：`historical_spots` 为 `{Date: float}`。兼容 API 还接受 Python `datetime.date`；缺少过去 SPOT 的值时报 `UnboundHistoricalSpot`，非有限值时报 `MissingFixing`。先用 NumPy 在主机端硬回放；过去 PAYS 的 RHS 会执行，但不写入支付变量。过去依赖具名参数的赋值也保留为纯 JAX 回放函数，更新 `params["script"]` 时初始状态会更新。过去和未来 IF 都必须带 `affected_vars`。
- **分析**：constfold 以 `historical=True` 开始，进入未来事件前调用 `start_future()`。准备层的 `_ScalarDomains` 丢弃过去支付对变量定义域的影响，并对字面量 NaN/Infinity、零分母保留未知定义域，保证未选中分支可以继续到 exact 求值；P1 的原有 passes 语义未改。
- **降级**：`lower_event(event, const_names)` 的第三个参数直接是脚本参数映射（完整 PathProduct payoff 再从 `params["script"]` 取）。IF 两侧从入口状态执行，递归传递 active 掩码；LOG/SQRT/除法/幂/EXP 先替换未选中输入，再计算。常量和变量叶子也按 active 置零，避免无穷值乘到未选中路径的梯度里。
- **兼容 API**：`BSModelData_New` 返回 `BlackScholes`。`MonteCarlo_Value` 支持 `rsg` 和 dal-python 的 `method` 别名、显式估值日以及 `MonteCarloSettings` 的执行选项。`compiled=True/False` 接受并 warning，XLA 始终编译。P3 已接通 `enable_aad=True`，返回 PV 和全部 `d_<label>`。
- **边界**：完全到期的产品不分配模型计划、不模拟路径；未提供的历史 SPOT 仍在准备时校验。向量求值、FIX、跨日期 PAYS ON、EXERCISE 分别留到 P5/P6，报明确异常；PAYS ON 与事件同日时规范化为普通支付。已展开的标量 FOR 和解析成常量的固定向量下标可以运行。
- **BS 修复**：仅有估值日时 `sim_dim=0`。原 `generate` 对空时间步做 cumsum 会触发 XLA 编译段错误，现直接返回当日 spot。已覆盖 `none/shard_map/auto/pmap`、bridge 开关以及非整块路径数。

新增测试：`tests/script/test_preparation.py`、`tests/script/test_exact.py`、`tests/test_api_value.py`、`tests/oracle/test_dal_script_prices.py`；共享事件表在 `tests/script_cases.py`。

### 5.2 P3 的入口和约定

- **两套事件**：`PreparedProduct.events` 为 exact 优化结果，`fuzzy_events` 从原始解析事件单独生成，避免 exact constcond 删除 fuzzy 过渡。`max_nested_ifs` 取两者最大深度。
- **fuzzy 准备**：跟随 DAL `preparation.cpp` 的模型准备分支，只做 constfold 和 IF 元数据，保留连续平滑，不运行 legacy domain/constcond。不能给 autocall 后续 `alive = 1` 套上离散边界，否则 PV 和 Greeks 会改变。底层 fuzzy lowerer 仍完整支持 legacy domain pass 产生的 `is_discrete/lb/rb`。
- **混合和安全**：比较用 CSpr/BFly，AND 为乘积、OR 为 `a+b-a*b`、NOT 为 `1-a`。IF 两侧从同一入口状态求值，只混合 affected_vars；degree > `1-EPSILON` 或 < `EPSILON` 时直接选择一侧，与 DAL 一致。active 掩码递归传递并在危险算术之前替换输入。
- **历史梯度**：`PathProduct.initial_state(params)` 可提供共享初始状态，payoff 接收第四个参数。标准归约在每台设备的路径/块循环外计算历史；确定性归约按块回放，以保持独立的块 Jacobian 和逐位一致性。历史 IF 永远为硬判断，过去支付不累加，具名参数仍有梯度。
- **分组**：`passes/eventgroup.py` 把日期字面量和已折叠表达式移到 `EventConst`，移除源码位置/观察 ID/变量名拼写差异；ConstVar 保留参数索引。只有相邻相同模板才合并，`scan_group_threshold=4`，设为 `0` 禁用。`prepared.event_groups(fuzzy=True)` 提供静态 span/template/constants 诊断。短组按字段 unstack，长组用一个 scan body。manual shard_map 中，脚本状态仅在尚未 varying 时 pcast；读取参数的历史状态已 varying，重复转换会被 JAX 拒绝。
- **平滑核**：`smoothing_kernel="dal"` 默认逐段线性、对齐 DAL；可选 `"smoothstep"` 为三次 C1（含 butterfly 峰值和非对称 lb/rb），改变平滑价格和 Greeks。
- **验收**：European、Asian、autocall、嵌套条件、DCF、today/expired 的 fuzzy PV 与全部风险对照通过；月度 barrier 含 bridge 开关和 scan 开关。在 2²⁰ 路径下 `d_BARRIER≈0.08925165481`、`d_vol≈-7.227015535`，并通过公共路径有限差分。分组/展开的单路径价格和梯度绝对差 ≤ 1e-14；非法未选中分支的价格/梯度有限。

新增测试：`tests/script/test_fuzzy.py`、`test_eventgroup.py`、`test_script_greeks.py`、`test_smoothing.py`、`tests/oracle/test_dal_script_greeks.py`。还覆盖 JAX grad/jacrev/jacfwd/jvp/hessian/vmap、4 台虚拟 CPU、全部并行策略、确定性归约、float32 和小于域分析容差的非零分母。

`benchmarks/script_compile_cpu.json` 实测单路径脚本价格与参数梯度的编译（不含随机数和模型生成）：

| 日观察数 | 展开图方程数（含子图） | 分组图方程数 | 展开编译 | 分组编译 |
|---|---:|---:|---:|---:|
| 36 | 4537 | 343 | 2.25 s | 0.078 s |
| 64 | 8009 | 343 | 9.42 s | 0.097 s |
| 365 | 45333 | 343 | 未计时 | 0.078 s |
| 750 | 93073 | 343 | 未计时 | 0.080 s |

365 日展开编译在试跑中超过数分钟，故基准默认只编译 ≤64 日的展开版本，长日程保留 jaxpr/HLO 规模，全部分组版本均实际编译。可用 `--unrolled-limit 750` 重新测完整基线；计时依赖机器，不作为单测断言。

对应的 DAL 源码：`script/visitor/fuzzy.hpp`、`smoothing.hpp`、`visitor/domainproc.hpp`、`script/preparation.cpp`。

### 5.3 P4 的入口和实测

- **提交位置**：P3 已以 `6e5242e` 推送到 PR #2，Python 3.13/3.14 与原 notebook CI 已通过。P4 已提交到 PR #2。
- **精度**：`dtype="float64"` 默认不变；显式 `"auto"` 为 CPU float64 / GPU float32。路径数组为 float32 时，随机数仍先以 float64 生成；块内 JAX reduction 使用路径 dtype，块间累加为 float64。不能把小型测试的容差理解为所有产品的保证。
- **float32 边界修复**：`1-EPSILON` 在 float32 中会舍入为 1，原 `degree > 1-EPSILON` 因而不能识别满 degree。float32 用 `degree >= 1`，float64 保持 DAL 阈值，保证未选中 LOG(-1) 等分支不会污染值或梯度。
- **块大小**：设置默认由 `8192` 改为 `"auto"`，CPU 仍解析成 8192。GPU 以最小设备 allocator limit 的 20% 为预算，计入 float64 RNG、Scenario 槽位及 16 倍 AAD / 4 倍 price 余量，向下取 2 的幂，范围 256–32768；缺少内存统计回退 8192。显式正整数总是优先。`engine.block_size` 是解析后的上限；小路径数仍按 layout 缩小实际块。
- **设备放置**：`default_params()` 使用所选 mesh 的 replicated NamedSharding，`value` 同时放置外部参数和路径数。修复了 CUDA 为默认后端时，显式 CPU 引擎收到 GPU 默认数组造成的设备冲突。原生 JAX 组合从该引擎的默认参数开始。
- **RBG**：JAX 原生 vmap 对 `rbg/unsafe_rbg` 使用第一个 key 生成整个 batch，导致自动切分的结果改变。`prng.block_normals` 用 public `sequential_vmap` 包住整个 fold_in + normal，嵌套 key/block vmap 也保持逐块语义。固定块大小时设备数/策略不改变流；改变块大小仍会改变 PRNG 路径。RBG 没有稳定的实测优势，保留 Threefry 默认。
- **CPU 实测**：WSL2、i9-13900HX、32 个逻辑处理器。对长产品，8 个虚拟设备优于 16/32；8192 优于较小块，shard_map 优于 auto/pmap。CPU 8 月度 barrier 全风险约 0.416 s，单设备约 1.30 s。只算 European 价格时 16 设备略快，故没有自动固定设备数。
- **GPU 实测**：RTX 4060 Laptop 8188 MiB、driver 595.79；隔离环境 `/tmp/dal-jax-p4-gpu` 安装 cuda13，原 `.venv` 保持 CPU。35% allocator pool 下，自动 price 块均为 32768，Greek 块为 8192–32768。月度 barrier 全风险 float64 约 0.283 s、float32 约 0.213 s；周度 reverse scan 仍较慢。只验证了单个物理 GPU，CPU 并行验证用 4 个虚拟设备。
- **精度限制**：百万路径 autocall（smooth=.01）的 float32 PV 相对误差 <3e-7，但 `d_spot` 从 float64 的 -3.54332 变成约 -32.07422，`d_vol` 从 -87.20916 变成 +96.57239。脚本运算单独提升到 float64、同宽度 C1 核都没有消除偏差；路径舍入经连续状态条件放大。smooth=.1 仍有明显偏差；smooth=1 通过测试容差，但 PV 改为约 120.00612。因此原窄平滑 autocall 必须用 float64，不能只看 PV 选择精度。
- **报告**：[docs/performance.md](performance.md) 记录编译、同步热运行、XLA 内存估计、CPU 设备/块/策略、checkpoint、GPU 两种 dtype、PRNG 和上述精度压力测试。计时进程依次运行；JSON 同时保留最小值、median 与原始次数。allocator peak 为进程累计值，不能当作单产品峰值。
- **测试和 CI**：`tests/gpu` 默认跳过，实际 GPU 环境用 `pytest tests/gpu --run-gpu`。新增 `.github/workflows/gpu.yml`，手动触发、self-hosted Linux `gpu` runner，cuda12/cuda13 可选；未触发远程 GPU workflow。本机已通过所有 26 个 GPU 用例，含 bridge、全部策略、块大小、scan、跨 CPU/GPU 放置、统计 PRNG和百万路径 autocall float64/DAL 对照；另在 GPU 上通过 4 个 float32 安全端点测试。

### 5.4 P5 接续建议

1. 先实现定容可变向量、APPEND、下标和归约的设备端状态及错误标志，覆盖 fuzzy 长度取最大、较短分支补零混合；当前前端已完成。
2. 实现 PAYS ON 的跨日期 discount 槽位，以及 FIX 的模型感知准备、历史快照和 TodayFixingPolicy。当前只支持同日 PAYS ON 和历史 SPOT。
3. 实现 CorrelatedBlackScholes、多因子 bridge，再补局部波动率模型和分桶 vega。
4. 延续 DAL master 语义、可用 dal-python oracle 和已移植 C++ 单测；LSMC 留给 P6。

## 6. 工作约定

- **Git**：
  - 分支命名为 `feature/<描述>` 或 `fix/<描述>`。这个仓库的默认分支是 `master`，不是 `main`。
  - commit 信息：英文祈使句摘要，不超过 72 个字符，空一行后写正文，说明为什么这样改。
  - 提交者身份与已有提交一致（用 `git log -1 --format='%an <%ae>'` 查看），通过 `GIT_AUTHOR_*` / `GIT_COMMITTER_*` 环境变量设置，不改全局配置。
- **PR**：
  - 标题必须带类别前缀（`feat:`、`fix:` 等），不超过 70 个字符；
  - 描述用中文，分 `## Summary` 和 `## Test plan` 两节。
  - `gh pr edit` 在这个仓库会因为 Projects (classic) 的 GraphQL 弃用而失败，改用 `gh api -X PATCH repos/wegamekinglc/dal.jax/pulls/<n> -F body=@<文件>`。
  - review 线程用 GraphQL 的 `resolveReviewThread` 关闭。
- **Shell（zsh）**：
  - 循环变量不要命名为 `path`，在 zsh 里它和 `PATH` 绑定，会把 `PATH` 清空；
  - `echo ======` 会触发 equals 展开而报错。
- **是否提交由用户决定**：只在用户明确要求时提交或推送。

## 7. 待决问题（issue 第 14 节）

1. 包名：已定为 `dal_jax`（PyPI 包名为 `dal-jax`）。
2. 旧 notebook：已删除，由 `examples/` 取代。
3. `n_paths` 变化时：默认重新编译，并依靠编译缓存；`MonteCarloSettings.block_bucketing=True` 时把块数向上取到 2 的幂。
4. GPU 默认精度：P4 已定为 float64；显式 `dtype="auto"` 选择 GPU float32。原窄平滑 autocall 的 Greeks 仍必须用 float64。
5. 是否在 JAX 中实现 GSR 校准：未定（P7 之后）。
