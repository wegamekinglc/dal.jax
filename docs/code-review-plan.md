# 代码审查与风格统一方案

审查基线：`17bc047245baf1cd67b3fe69d782ef4656d60903`，2026-10-08。

目标：更 Pythonic 的表达、极简注释与完整用户文档、适当的并行计算、不可变数据和函数式核心。本文交付审查发现及实施方案；建议中的生产代码重构尚未实施。

## 结论与范围

优先解决配置与编译缓存不一致、隐式全局估值上下文、输入集合未落实不可变边界。随后统一格式与类型表达、整理文档，再优化 GSR/Hybrid 的批量预计算和 LSMC 的训练流程。

现有基础值得保留：冻结 AST、不可变 `PreparedProduct` 和 `VarTable`、复制输入的 `FixingSnapshot`、路径级 `vmap`、有界 block 调度、时间递推 `scan`、设备分片，以及覆盖 DAL 数值行为的测试。问题不是缺少函数式或并行设计，而是不同模块落实程度不一致。

全量 AST/token 扫描覆盖以下 139 个 Python 文件；人工审查重点覆盖 API → preparation/history → lowering → model → engine/LSMC → diagnostics，并检查随机数、日期、配置、测试、示例、基准和 CI。全量静态扫描不等于逐行证明所有代码正确。

| 范围 | Python 文件 | 物理行数 | 显式 for/while | 注释 token | docstring 行数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `src` | 63 | 10,786 | 210 | 165 | 614 |
| `tests` | 51 | 5,284 | 85 | 35 | 88 |
| `examples` | 17 | 937 | 31 | 11 | 18 |
| `benchmarks` | 5 | 552 | 18 | 0 | 40 |
| `scripts` | 3 | 148 | 2 | 0 | 20 |

计数包含生成的 `dates/calendar_data.py`；生成数据应由生成流程管理，不能按手写业务代码逐条修饰。循环数量仅用于定位，不是待消除循环的数量。

使用 Ruff 0.16.10 做只读扫描：`E4,E7,E9,F,I,UP,B,SIM` 共 229 条提示，其中导入排序 65、缺少显式 zip strict 参数 71、赋值 lambda 30、分号多语句 29、未使用导入 10。默认 formatter 报告 133 个 Python 文件需要调整。这是引入统一配置前的基线，不能解释成 229 个缺陷。

## 审查发现

### R1 · 高优先级：引擎允许修改配置，却复用旧编译结果

**位置：** [engine.py:125](../src/dal_jax/mc/engine.py#L125)、[engine.py:371](../src/dal_jax/mc/engine.py#L371)。

构造时生成 `_normals`、执行布局及类型；`_value_function` 缓存键只有 layout、`enable_aad` 和 payoff index。公开的 `settings`、`model`、`product` 等属性仍能赋值。预热后替换 `settings.smooth` 对应的新 settings，缓存中的闭包仍使用旧平滑宽度。

**已复现：** Python 3.13 / JAX 0.11.2 / CPU / float64，4,096 条 Sobol 路径、block 256；一年期 `IF SPOT() > 105 THEN pay PAYS 1 ELSE pay PAYS 0 END`，BS spot=100、vol=0.2。

| 情况 | PV | d_spot |
| --- | ---: | ---: |
| 初始 `smooth=0.01` | 0.36530616264024235 | 0.025634262798277425 |
| 原引擎改为 `smooth=60` | 0.36530616264024235 | 0.025634262798277425 |
| 用 `smooth=60` 新建引擎 | 0.414934993709029 | 0.01428667903865279 |

这证明属性赋值与实际执行行为可以分离；不表示按不可变配置方式正常使用的估值都会出错。

**修改方案：** 将配置、模型、产品和执行计划组成冻结描述，公开属性只读；改变配置时创建新引擎。可变编译缓存留在显式拥有生命周期的宿主对象中。不能仅给缓存键增加 `smooth`：随机数生成器、设备和布局也捕获了构造时信息。

**验收：** 属性替换明确报错或受支持的重配置入口返回新对象；旧对象结果稳定，新对象与独立构造结果一致；覆盖 seed、平滑、dtype、device 等有不同依赖的配置类别。

### R2 · 高优先级：估值日期和 fixing 来自两个独立全局状态

**位置：** [api.py:33](../src/dal_jax/api.py#L33)、[fixings.py:111](../src/dal_jax/script/fixings.py#L111)、[preparation.py:254](../src/dal_jax/script/preparation.py#L254)、[observation.py:122](../src/dal_jax/script/observation.py#L122)。

`_evaluation_date` 和 `_global_snapshot` 分别受锁保护。锁能保护单次读写，不能保证一个估值请求从设置日期到读取历史的完整隔离；两个请求交错设置时会互相影响。日期首次默认读取还会将当天日期保存在进程中，长期进程不会自动跨日。显式传入 date 和 snapshot 的调用路径已有良好基础。

**修改方案：** 把已有 `ValuationSettings` 扩展为在入口完全解析的不可变估值上下文，后续流程只读取该值。纯核心要求日期和历史快照明确，不再回退到全局；空历史使用显式空快照。日期与历史在同一请求边界确定，debug/explain 与估值共用同一个上下文。

保留旧 `EvaluationDate_Set/Get` 的进程全局语义与“零全局可变业务状态”不能同时实现。分阶段迁移到显式上下文或显式创建的兼容 session；过渡期旧接口是明确标注的兼容例外，最终版本再移除。改成 `ContextVar` 只能改变隔离范围，不能宣称已变成纯函数。

**验收：** 两组不同日期、不同 fixing 的请求交错执行，结果只依赖各自参数；同一上下文重复 preparation 相等；覆盖 today policy、缺失历史、完全到期产品和诊断输出。跨日默认值应有明确的新契约。

### R3 · 中优先级：冻结对象未统一快照化调用者的集合

**位置：** [base.py:30](../src/dal_jax/models/base.py#L30)、[product.py:23](../src/dal_jax/script/product.py#L23)、[gsrslv.py:43](../src/dal_jax/models/gsrslv.py#L43)、[hybrid.py:167](../src/dal_jax/models/hybrid.py#L167)。

`SampleDef.index_names`、`ScriptProductSettings.regression_features`、`GSRSLVSettings.variance_correlations` 已实测：传入列表后修改原列表，冻结对象的内容随之变化。`HybridGSRRate.factors` 也缺少边界归一化。字段标注 tuple 并不执行运行时校验；传列表不符合其注解，但当前构造器接受了它，且其他模型构造器已经采用自动转 tuple 的约定。

**修改方案：** 统一为入口接收合理的序列并复制为 tuple，或一致地拒绝不符合契约的类型；本库更适合前者。嵌套集合逐层归一化。借鉴 `LocalVolSurface`、`CorrelatedBlackScholes` 和 `FixingSnapshot` 已有做法，不重复改写已正确冻结的 `Regression`、`VarTable`。

**验收：** 修改原始列表/嵌套列表不影响对象；对象在要求可哈希的位置能稳定哈希；工厂入口与直接 Python 构造入口契约一致。

### R4 · 中优先级：可变模块常量、全局缓存和启动设置需要分别治理

**位置：** [inverse_normal.py:92](../src/dal_jax/random/inverse_normal.py#L92)、[settings.py:25](../src/dal_jax/mc/settings.py#L25)、[parser.py:39](../src/dal_jax/script/parser.py#L39)、[sobol.py:37](../src/dal_jax/random/sobol.py#L37)、[prng.py:28](../src/dal_jax/random/prng.py#L28)、[__init__.py:9](../src/dal_jax/__init__.py#L9)。

- `_SPLINE_FPP` 是模块级 NumPy 数组，已确认 `flags.writeable=True`。它参与逆正态近似，意外修改会改变以后构建的计算；不同已编译函数可能保留不同常量。
- `_CHOICES`、`_FUNCTIONS`、`_PARSERS`、`_KERNELS`、日期分发表等都是可变 dict；目前未发现运行中修改它们的正常路径，属于不可变约定未落实，不应误报为已发生的数据竞争。
- Sobol、bridge、PRNG、calendar 的 `cache` 以及 eventgroup 的 `lru_cache` 都包含进程共享的可变存储。部分缓存无容量上限；Sobol 数组已设只读、eventgroup 已限为 128，这些已有保护应保留。
- import 时启用 JAX x64 是外部框架的进程配置副作用，和日期/fixing 的业务全局状态性质不同。

**修改方案：** 数值常量发布为 tuple，静态 host 分发表使用 tuple/frozenset 或无外部可变别名的只读映射。不要把所有 JAX 参数 dict 都替换为 `MappingProxyType`，否则可能破坏 pytree 支持。缓存转为执行 session 所有，明确容量与清理；确需保留共享 memoization 时，列为可解释的过渡例外。JAX 设置移到明确的启动边界，需要单独设计 float64 默认行为的兼容迁移。

JAX 转换要求纯函数；外部值可能在 tracing 时被捕获，不能依靠之后修改全局变量更新已编译行为。[JAX 关于纯函数及全局状态的说明](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#pure-functions)

**验收：** 公共输入不能改变常量；缓存释放后结果一致；不同 session 的缓存和业务状态隔离；用独立进程验证导入/配置顺序及默认精度。

### R5 · 中优先级：LSMC 把训练结果、运行诊断和编译缓存混在对象上

**位置：** [lsmc.py:93](../src/dal_jax/mc/lsmc.py#L93)、[lsmc.py:221](../src/dal_jax/mc/lsmc.py#L221)、[lsmc.py:269](../src/dal_jax/mc/lsmc.py#L269)、[lsmc.py:352](../src/dal_jax/mc/lsmc.py#L352)、[explain.py:191](../src/dal_jax/script/explain.py#L191)。

`_collect_raw` 动态给另一个 engine 添加 `_record_functions`，并依赖多个私有方法。`train` 写 `self.regressions`，`value` 写 `self.replicate_means`，diagnostics 再从 engine 读取回归结果。调用顺序和“最近一次运行”成为隐式输入，重入或并发复用同一对象时尤其难推理。当前审查未证明有 tracer 泄漏或常规单次估值错误。

**修改方案：** 抽出稳定的路径/record 收集接口；训练函数返回冻结 `TrainingResult(policy, regressions, diagnostics)`，估值返回结果及其对应的 replica 信息。编译由独立宿主层负责，数值层只传显式参数和 policy。保留现有 DAL 风格 dict 输出的边界适配。

**验收：** A、B 两次训练结果能同时保存并独立 explain；交错估值不覆盖先前诊断；Frozen 与 RetrainedBump 的风险语义及 RQMC stream 分离不变。

### R6 · 中优先级：每次 fixing 查找都重建完整字典

**位置：** [fixings.py:68](../src/dal_jax/script/fixings.py#L68)、[observation.py:122](../src/dal_jax/script/observation.py#L122)。

`find` 每调用一次就将全部 `entries` 转成 dict。若历史快照有 N 条、准备阶段查询 H 次，仅重建索引就要 O(HN) 工作。它比许多只有几个元素的 host for 循环更值得优化。

**修改方案：** 单次查找利用已排序不可变键二分；批量 preparation 在局部建立一次索引或使用批量 resolver。目标分别为 O(log N) 查找或 O(N+H) 批量处理，不新增全局可变索引。保留 FX 反向报价、大小写归一化和精确时间戳规则。

**验收：** 直接/反向 FX、缺失值、同一天不同时间和别名行为一致；用 N、H 分别增长的基准验证扩展性，不能只测一个小样本。

### R7 · 中优先级：GSR/SLV/Hybrid 的静态规划与数值预计算混合

**位置：** [gsr.py:285](../src/dal_jax/models/gsr.py#L285)、[gsr.py:294](../src/dal_jax/models/gsr.py#L294)、[gsr.py:328](../src/dal_jax/models/gsr.py#L328)、[gsrslv.py:179](../src/dal_jax/models/gsrslv.py#L179)、[hybrid.py:467](../src/dal_jax/models/hybrid.py#L467)。

日期/knot 分段、区间积分、到期日系数和采样输出交错使用 Python 循环；部分 `integrals(0, time)` 重复计算。随着日期和 maturity 增加，tracing 展开及重复预计算值得测量。此项是源码定位的性能候选，尚未测得端到端加速比。

**修改方案：** allocate 阶段产生不可变 interval/observation plan，预计算区间 id、静态时间差、gather 索引、mask 和可复用结构；init 阶段仍以动态模型参数计算系数，沿独立 interval/maturity/factor 轴批量运算。不能把影响 Greeks 的参数相关值提前固化。价格过程沿时间仍使用 scan。

当前 `GSRCurve.log_df` / `pieces` 含 Python `math`、bisect、日期和参数名处理，不能原样套 `vmap`。先分离 host 与 array 层，再考虑共享积分表、批量矩阵操作和观测 gather。

**验收：** 零波动、零时间增量、退化相关矩阵、active g、rate/vol/correlation 风险、延迟支付与所有 FIX 类型保持现有数值契约；同时记录编译时间、warm 时间、峰值内存和图规模。

### R8 · 中优先级：LSMC 有伪批量表达，也有必须保序的训练循环

**位置：** [regression.py:218](../src/dal_jax/mc/regression.py#L218)、[regression.py:263](../src/dal_jax/mc/regression.py#L263)、[lsmc.py:159](../src/dal_jax/mc/lsmc.py#L159)。

`select_regression` 在 `vmap(lambda i: stack(all_predictions)[i])` 中，索引 i 只选择已经构造的预测栈；表达比直接 stack 复杂，并没有把不同阶数的拟合本身变成统一批量求解。不能据此断言实际重复执行 degree 次，编译器可能消除重复。

`_multi_normalization` 则逐训练路径在 Python 中做 Welford 更新。这里明确要求 DAL 路径顺序，直接替换并行 mean/variance 可能改变 sigma floor、rank 和行权决策。回溯训练对后续日期的目标有依赖，不能沿行权日期直接 vmap。

**修改方案：** 先去掉多余 vmap，保持已有拟合行为；后续统一固定形状的基函数/系数，批量计算可独立的候选预测。Welford 可以试验同顺序 device scan；回溯设备训练可以试验固定形状 carry 与 mask 的反向 scan。host QR 的 pivot、rank、重正交化不因风格统一而更换算法。

scan 用于表达递推并限制 Python tracing 展开，不代表递推各步并行；carry 的结构、shape、dtype 必须固定。[JAX scan 文档](https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html)

**验收：** normalization、pivot、rank、degree selection、fallback reason、exercise policy 与 PV/Greeks 全部比较。设备 scan 即使保持逻辑顺序，也要实测 XLA 融合后的数值差异。

### R9 · 中优先级：状态记录过度依赖位置，类型信息不均匀

**位置：** [gsrslv.py:84](../src/dal_jax/models/gsrslv.py#L84)、[gsrslv.py:232](../src/dal_jax/models/gsrslv.py#L232)、[hybrid.py:519](../src/dal_jax/models/hybrid.py#L519)、[localvol.py:99](../src/dal_jax/models/localvol.py#L99)、[state.py:8](../src/dal_jax/script/lower/state.py#L8)。

`tuple(state[1:10])` 与 `state.rate[1:10]` 把状态字段排列变成跨模块契约。多个 `NamedTuple` 字段写作 `object` 或无参数的 `tuple`，难以读出数组、静态元数据和子记录的边界。

**修改方案：** 将相关 step coefficients 提取为命名子记录，明确字段读取；长构造使用关键词或单一工厂。只在真实数组位置使用 `jax.Array`，host/JAX 双后端位置使用适当联合类型或协议；不为消除 object 而写不真实的类型。保持 pytree 结构与 scan 轴明确。

**验收：** state 的构造、pytree flatten/unflatten、vmap/scan、grad 和分片路径仍匹配；通过现有行为测试验证，不添加检查字段文本排序的脆弱测试。

### R10 · 中优先级：格式缺少统一约束，后期模块更密集

**位置：** [regression.py](../src/dal_jax/mc/regression.py)、[regression_device.py](../src/dal_jax/mc/regression_device.py)、[hybrid.py](../src/dal_jax/models/hybrid.py)、[explain.py](../src/dal_jax/script/explain.py)、[test_dal_hybrid.py:12](../tests/oracle/test_dal_hybrid.py#L12)、[pyproject.toml](../pyproject.toml)、[CI](../.github/workflows/ci.yml)。

逗号/运算符空格、多语句分号、长行和 import 顺序不一致；oracle 测试仍有星号导入。现有 CI 有复杂度、错误和安全检查，但没有统一 formatter/lint 风格门槛。

**修改方案：** 固定 Ruff 开发工具版本，明确 Python 3.13 target 和统一行宽（建议 100）；先启用 formatter 及基础 E/F/I/UP 规则，清理未使用和星号导入，再逐项选择 B/SIM 规则。生成数据交由生成器一致输出。保留 DAL 公共名称；新内部接口用 snake_case，不进行破坏兼容的全库命名替换。

需要人工判读的两类提示：相邻区间 `zip(grid, grid[1:])` 应考虑 `pairwise`，不能直接加 `strict=True`；preprocessor 和 lowering 中循环内定义、立即使用的 closure，不能仅凭 B023 宣称已有 late-binding bug。JAX 的简短数学 lambda 也不必全部机械改成 def。

**验收：** 纯格式提交可单独审阅，格式化前后 AST 等价；单独审查 import 调整，特别是 x64 启用顺序。CI 增加固定版本 formatter/lint gate，与数值回归检查并行存在。现有复杂度阈值继续满足，但不为降低计数引入难懂的函数跳转。

### R11 · 中优先级：用户功能说明分散在 docstring 与里程碑文档

**位置：** [settings.py:52](../src/dal_jax/mc/settings.py#L52)、[engine.py:1](../src/dal_jax/mc/engine.py#L1)、[base.py:1](../src/dal_jax/models/base.py#L1)、[product.py:1](../src/dal_jax/script/product.py#L1)、[README](../README.md)、[P5 文档](p5.md)、[P6/P7 文档](p6-p7.md)。

注释总体不算多，重点是职责和位置：配置教程、大段模块背景、阶段性开发说明和装饰分隔线应收敛。当前未发现源码注释引用仓库文档路径；`examples/08_gpu_and_precision.py` 输出的文档路径是运行时文本，不是注释，不能作为该规则的违规证据。

**修改方案：** 代码仅保留短契约和必要原因；功能说明按用户任务重组到文档。不在删除后的注释/docstring 中添加“见 docs/...”之类反向引用。文档可链接代码、公开接口和可执行示例。

| 文档目标 | 内容来源与职责 |
| --- | --- |
| quickstart / script valuation | 安装、事件表、prepare、engine、结果；从 README 与示例整理 |
| valuation context | 日期、fixing、today policy、历史重放、过期行为 |
| execution | CPU/CUDA、dtype、设备、block、checkpoint、随机流和复现条件 |
| models / LSMC | 模型支持范围、观测、训练/估值分离、风险模式 |
| API / settings reference | 默认值、可选值、错误条件及简短契约 |
| numerical contracts | DAL 对齐、Welford/QR 顺序、分片归约、后端限制；供维护者阅读 |

保留短原因注释，例如 RBG key batching、ordered reduction、QR/Welford 顺序、无效分支安全计算和 LocalVol `compare_all` 后端限制。不要把影响正确性的约束一并删掉。P0–P7 文档保留为历史记录，用户导航以任务组织。

**验收：** 文档覆盖公开功能和真实默认值；关键片段通过现有接口运行；代码注释/docstring 无文档路径、链接或章节引用。所有 16 个示例继续使用 script engine；解析闭式解作为参照与手写 Monte Carlo payoff 区别对待。

## 循环与并行化决策表

| 位置/工作 | 推荐处理 | 原因或边界 |
| --- | --- | --- |
| MC 路径和设备 block | 保留 vmap、分片和有界 block | 已具备批量/设备并行；全量展开会增加内存 |
| GSR interval/maturity 系数 | 拆静态计划，再批量数组计算 | 独立预计算轴；动态参数不能固化 |
| GSRSLV 每区间系数 | 计划分离后 vmap/批量矩阵运算 | 参数化协方差和步系数可按区间批处理 |
| LSMC 候选预测 | 先直接 stack；统一形状后真批处理 | 删除冗余索引 vmap，不承诺未经测量的加速 |
| 多特征 Welford | 试验按路径顺序 scan | 控制宿主开销；不是并行归约 |
| LSMC backward dates | 保序；可试验反向 scan | continuation target 依赖后续行权结果 |
| LocalVol/GSR/Hybrid 路径递推 | 保留时间 scan/checkpoint | 状态依赖和内存边界 |
| deterministic block sums | 保留规定的归约顺序 | 不能用重排的 sum 换取速度后仍声称逐位一致 |
| RBG block keys | 保留 sequential_vmap 保障 | 默认 vmap 的随机流语义特殊 |
| Sobol 32-bit XOR | 可测批量 bit 归约或树归约 | 整数 XOR 可结合；完整广播的内存可能更差 |
| fixing 历史查找 | 二分或一次建立局部批量索引 | 先消除重复 O(N) 构造，比换循环语法有效 |
| parser、日期构建、异构 lowering | 保留清晰的局部构建循环，输出冻结 | 顺序/异构 host 工作不适合盲目 vmap |
| 示例/benchmark 各配置比较 | 计时任务顺序执行 | 同时竞争同一设备会污染时延比较 |

JAX 文档明确说明 RBG 在 vmap 下可从首个 key 生成整批输出，故其现有保护不能按风格偏好删除。[JAX PRNG batching 语义](https://docs.jax.dev/en/latest/jax.random.html#advanced-rng-configuration)

独立数组轴也要控制峰值内存；可根据测量选择有界 vmap 或 `lax.map(..., batch_size=...)`。Python comprehension、普通 map 与 scan 本身都不能作为“已经并行”的证据。[JAX batched map](https://docs.jax.dev/en/latest/_autosummary/jax.lax.map.html)

## 分批实施方案

规模 S/M/L 表示相对改动与验证范围，不是工期承诺。各 PR 都要能够独立审阅和回退；重构 PR 不夹带 formatter 全库变更。

| 顺序 | 修改批次 | 对应发现 | 规模/风险 | 完成标准 |
| --- | --- | --- | --- | --- |
| 1 | 冻结 engine 的执行配置；快照化公共配置集合和数值常量 | R1、R3、R4 常量部分 | M / 中 | 新旧引擎配置一致性、alias 行为测试通过；现有 API 正常调用不变 |
| 2 | 固定 formatter/lint 配置，独立做机械整理 | R10 | M / 低 | AST 对比、人工 import 审核、CI 格式门槛；无数值算法调整 |
| 3 | 用户文档按任务重组，精简注释和 docstring | R11 | M / 低 | 可运行的 script 示例、文档覆盖表、无代码反向引用 |
| 4 | 估值上下文显式化，debug/explain 共用；设计旧 global API 迁移 | R2、R4 启动边界 | L / 高 | 并发请求隔离、历史/today policy 回归、兼容迁移说明 |
| 5 | LSMC 训练/结果/编译职责分离，session 拥有缓存，命名状态字段 | R4 缓存、R5、R9 | L / 中高 | 独立结果可重复 explain；缓存生命周期与所有风险模式验证 |
| 6 | fixing 批量索引、去掉多余 vmap | R6、R8 表达部分 | S–M / 中 | 行为一致，查找扩展性改善；无额外全局状态 |
| 7 | GSR/SLV/Hybrid 静态计划及批量系数 | R7 | L / 高 | oracle、AAD、多设备及编译/运行/内存基准通过 |
| 8 | Welford/backward scan 与 Sobol XOR 的独立实验 | R8、循环表 | L / 高 | 每项独立设数值与性能门槛；无收益或破坏契约则不合入 |

第 1 批可以先于风格整理消除已复现的风险；第 4 批的兼容 API 取舍必须明确写入实现设计。第 7、8 批必须先收集基准，不能把方案中的候选优化当作已批准替换数值算法的理由。

## 验证策略

- **行为与隔离：** 输入快照、不可变配置、显式日期/fixings、独立训练结果、跨调用诊断、缓存冷暖一致性。
- **数值契约：** 使用现有 oracle 的容差与逐位要求；不统一放宽容差。固定随机路径 id、训练/估值 stream 分离、QR pivot/rank、sigma floor、degree selection、行权决策、active g 风险都必须维持。仅对现有 deterministic 模式承诺其原有逐位一致条件，不扩展为跨所有设备后端逐位一致。
- **变换和设备：** 相关修改覆盖 jit、grad、jacfwd/jacrev、vmap、scan/checkpoint；1/4 虚拟 CPU 设备和真实 CUDA 路径分别检查；float32/float64 使用各自既有误差标准。
- **用户入口：** 保留全部 16 个 script engine 示例、DAL 风格表格输出与 wheel 安装 smoke 检查。测试底层 engine 的手写 payoff 不等于用户示例，可以保留作为对照。
- **性能：** 分别记录 preparation、compile/首次调用、同步 warm 中位数、峰值内存；固定 seeds、paths、block、设备、dtype、事件/因子规模和模型参数。增加日期、maturity、feature、历史规模的扩展性用例。被测 GPU 上不并行运行竞争性 benchmark。
- **门槛：** 纯格式交给 formatter/lint；行为变化用实际结果验证。避免只搜索源码字符串、机械验证函数写法的新增单元测试。

本次执行了静态扫描及 R1/R3/R4 的小型 CPU 复现实验。没有进行全套性能比较，也没有因只新增审查文档而重跑全套定价回归；上表描述的是后续实施时的验收要求。

## 可复现检查

只读风格基线命令（使用本次审查的版本与默认格式设置）：

```sh
uvx --from ruff==0.16.10 ruff check --select E4,E7,E9,F,I,UP,B,SIM src tests examples benchmarks scripts
uvx --from ruff==0.16.10 ruff format --check src tests examples benchmarks scripts
```

引擎配置复现（在已安装项目的 Python 环境中，以 `JAX_PLATFORMS=cpu` 运行）：

```python
from dataclasses import replace

from dal_jax import BlackScholes, MonteCarloSettings, prepare
from dal_jax.api import Product_New
from dal_jax.dates import Date

today = Date.ymd(2026, 1, 1)
data = Product_New(
    [today.add_days(365)],
    ["IF SPOT() > 105 THEN pay PAYS 1 ELSE pay PAYS 0 END"],
)
prepared = prepare(data, today)
model = BlackScholes(spot=100.0, vol=0.2)
settings = MonteCarloSettings(
    platform="cpu", parallel="none", block_size=256, enable_aad=True, smooth=0.01
)
engine = prepared.engine(model, settings)
before = engine.value(4096)
engine.settings = replace(settings, smooth=60.0)
reused = engine.value(4096)
fresh = prepared.engine(model, engine.settings).value(4096)
print(before, reused, fresh)
```

冻结边界复现：

```python
from dal_jax.script.product import ScriptProductSettings

features = ["SPOT"]
settings = ScriptProductSettings(regression_features=features)
features.append("x")
print(settings.regression_features)
```

## 可复用准则

已创建本地 skill：`pythonic-functional-style`，可用 `$pythonic-functional-style` 显式调用。skill 保存通用 Python 准则及条件适用的 JAX 指南；本仓库的缺陷位置、DAL 数值约束和分批实施方案保留在本文，避免将项目细节误变为所有 Python 项目的强制规则。

核心约束是：可读表达优先；功能说明进入文档；注释/docstring 不反向引用文档；入口快照化、核心显式传状态；原则上无全局可变状态；先识别独立轴和数值依赖再并行化；用实测支持性能结论。
