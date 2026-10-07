# dal.jax 开发交接

更新于 2026-10-07。本文写给接手 [issue #1](https://github.com/wegamekinglc/dal.jax/issues/1) 后续开发的人（或下一个 Claude Code 会话），说明三件事：现在做到哪了，有哪些已定的决定，下一步从哪里开始。总体设计以 issue #1 为准，本文只记录 issue 里没有、但在实现中确定下来的内容。

## 1. 现状

| 里程碑 | 状态 | 位置 |
|---|---|---|
| P0 基础设施、BS 模型、随机数、MC 引擎 | 完成 | PR #2 |
| 示例 notebook（`examples/`，4 个） | 完成 | PR #2 |
| P1 Script 前端 | 完成 | PR #2（commit `8750dfd`、`e7caf50`） |
| P2 exact 降级与事件引擎 | **下一步** | — |
| P3 fuzzy 降级、求导、scan 分组 | 未开始 | — |
| P4 并行与 GPU | 部分完成：`shard_map` / `auto` / `pmap` 已经在 P0 实现，GPU 未验证 | — |
| P5–P7 | 未开始 | — |

- **分支**：所有工作都在 `feature/jax-mc-engine` 上，PR #2 的 base 是 `master`，还没有合并。`master` 上只有项目早期的探索性 notebook（PR #2 中已删除）。
- **测试**：`uv run pytest` 共 385 个，约 1.5 分钟，全部通过。
- **静态检查**：对应 Codacy 默认规范的 `lizard -C 8`、`pylint -E`、`bandit`、`semgrep --config p/python` 都已清零。Codacy 本身还没有接入这个仓库（它的 API 返回 "Could not find repository"），所以 PR 上的 Codacy check 会一直处于 queued，需要仓库所有者在 Codacy 后台添加仓库。

## 2. 代码地图

```
src/dal_jax/
  config.py            x64、虚拟 CPU 设备数、PRNG 实现、编译缓存
  errors.py            DAL 同名异常；script_error(msg) 根据消息里的 "Code: " 前缀选择 ScriptError 子类
  strings.py           DAL 的大小写不敏感字符串（ci_key/ci_eq/CIMap）、std::stod 语义、DebugNumber 格式
  index.py             EQ / FX / IR 指数名解析与规范名（Index::Parse）
  api.py               与 dal-python 同名的 Product_* / EvaluationDate_*（P2 在这里加 MonteCarlo_Value）
  dates/               Date（Excel 序号，1970-01-01..2149-06-05）、增量、节假日（calendar_data.py 为导出数据）、日程、计息基准
  random/              sobol（directions.npy）、inverse_normal、bridge、prng
  models/              base（Model 协议、SampleDef、Scenario）、bs
  mc/                  settings、engine（分块、checkpoint、value/pricer）、parallel
  script/
    lexer.py           词法；index 字面量整体成词（EQ[x]@date、EQ[x]>3M）
    preprocessor.py    宏、常量、数值向量、日程（ParseSchedule）、PeriodBegin/PeriodEnd
    parser.py          递归下降解析，含 FOR 展开、DCF 折叠、EXERCISE、PAYS ON、FIX
    ast.py             frozen dataclass 节点；Spot/Fix 带 observation_id（目前没有 pass 去设置它）
    product.py         ScriptProductData（不可变输入）/ ScriptProduct（解析结果、分区、变量编号、payoff_index）
    passes/            varindex → ifmeta → constfold → domain(+intervals) → constcond
    debug.py           DebugNode：DebugJson、树形、旧版 s-expression 三种渲染
    diagnostics.py     debug_json（schema /1）、describe（schema /2）、debug_tree、debug_text
    lower/smoothing.py CSpr / BFly 及其带 lb/rb 的两参数形式
tests/                 random/ models/ mc/ script/ dates/ test_index.py oracle/
examples/              4 个已执行的 notebook + nbtools.py
scripts/               export_sobol_directions.py、export_calendars.py（从 DAL 源码重新生成数据）
benchmarks/bench_mc.py 与 dal-python 同机计时
```

模块和 issue 第 10 节的规划有几处不同，都是有意为之：

- 没有单独的 `events.py` / `schedule.py`：事件解析放在 `product.py`，日程解析在 `preprocessor.py`，日程生成在 `dates/schedule.py`。
- 没有 `index/` 包，只有一个 `index.py` 模块，因为脚本层只用到指数名解析。
- `fixprep`（FIX 准备）和 `eventgroup`（scan 分组）还没有实现，分别属于 P5 和 P3。

## 3. 已定的决策

### 3.1 对照基准

- **语义以 DAL `master` 为准**，不是以 dal-python 2026.9.25 为准。dal-python 2026.9.25 还不支持向量、`FOR`、`PAYS ... ON`、IR 指数和 `30U/360`。这些新功能用移植的 DAL C++ 单测来验证；与 dal-python 的对照只用它支持的子集。
- **`Product_Describe` 比较时去掉 `regression_features` 键**：这是 DAL `master` 新增的键，dal-python 2026.9.25 没有。
- **错误对照只比较错误码和核心消息**：DAL 的异常文本里带 C++ 源文件路径和 NOTICE 上下文行，见 `tests/oracle/test_dal_script_frontend.py` 中的 `_core()`。
- 以下 3 处差异来自 DAL `master` 与 2026.9.25 版本之间的行为变化，是预期内的，对照测试已排除：
  - 出现第二个 `ELSE` 时报 `DuplicateElse`；
  - `MAX(1)` 按向量归约解析；
  - `FIX(EQ[a] 2)` 被词法层的后缀检查拒绝。

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
uvx lizard -C 8 -w src tests benchmarks scripts examples/nbtools.py
uv run --with pylint python -m pylint -E --disable=import-error src tests
uvx bandit -q -r src benchmarks scripts examples/nbtools.py
uvx semgrep scan --config p/python --metrics off --error src scripts benchmarks examples/nbtools.py
```

- **DAL 源码**：本机的 DAL 源码在与本仓库同级的 `../Derivatives-Algorithms-Lib`（`dal-cpp/` 和 `dal-python/`）。移植时以其中的源码和测试为金标准。
- **查看 dal-python 2026.9.25 的源码**：`git -C ../Derivatives-Algorithms-Lib archive dal-python-v2026.9.25 | tar -x -C <空目录>`。
- **依赖锁定**：CI 没有 `uv.lock`，`dal-python>=2026.9.25` 会装到 PyPI 上的最新版。如果新版改变了错误文本或输出，oracle 测试会失败。届时有两个选择：更新对照用例，或者把版本固定为 `==2026.9.25`。

## 5. 下一步：P2（exact 降级与事件引擎）

issue 中 P2 的验收标准：European、亚式（标量写法）、autocall 的价格与 DAL 一致（只算价格）。建议按以下顺序推进：

1. **准备流程**：
   - 在 `product.py` 之上加一个"准备"步骤，用 `partition_events(evaluation_date)` 区分过去和未来的事件；
   - 未来事件的日期按 DAL 的规则换算成时间轴：`(date - evaluationDate) / 365`，见 DAL `script/preparation.cpp` 中的 `timeLine_`；
   - 生成 `SampleDef`（是否需要 numeraire 等）；
   - 给 `Spot` 节点分配 `observation_id`。P2 只需要支持 `SPOT()`。
   - 动手前先读 DAL 的 `script/preparation.cpp`、`observationplan.hpp`、`simulation.cpp`，确认 legacy（不含 FIX）脚本走哪条分支。
2. **pass 顺序**：对不含 FIX 的旧式脚本，按 DAL 的顺序运行 varindex → ifmeta → constfold → domain → constcond（issue 3.5 节）。各个 pass 都已实现，缺一个把它们串起来的入口函数。
3. **`script/lower/exact.py`**：把事件 AST 降级成单路径函数 `event_k(state, sample_k, params) -> state`。
   - 状态是 `float[n_vars]`。
   - IF 只对 `affected_vars` 做 `where`。
   - `PAYS e` 写成 `var += e / numeraire`。
   - 具名常量从 `params["script"][name]` 读取。
   - 最后组装成 P0 的 `PathProduct(timeline, payoff, payoff_names, sample_defs, script_params)`，交给现有的 `MonteCarloEngine`，引擎不需要改。
   - 未选中分支里的非法运算要按 issue 3.2 节的 double-where 规则处理，P3 的 fuzzy 模式也依赖这一点。
4. **过去事件回放**（只支持 SPOT）：在主机端对单条路径做硬判断回放，过去事件里的 `PAYS` 只计算、不累加。完全到期的产品直接返回 0。
5. **兼容层**：在 `api.py` 中加 `BSModelData_New`、`MonteCarlo_Value(product, model, n_paths, rsg="sobol", use_bb=False, enable_aad=False, ...)`。P2 只算价格；`enable_aad=True` 留到 P3。
6. **测试**：
   - 移植 DAL `tests/script/test_simulation.cpp` 中可用的用例；
   - 在 `tests/oracle/` 中与 dal-python 的 `MonteCarlo_Value` 逐个对照（相同 Sobol 点，PV 相对误差 ≤ 1e-10）；
   - 可以参考 dal-python 的 `tests/test_script.py`、`tests/test_value.py`。

对应的 DAL 源码：`script/visitor/evaluator.hpp`（exact 求值）、`script/visitor/pastevaluator.hpp`（历史回放）、`script/evaluation.cpp`、`script/simulation.cpp`。

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
4. GPU 默认用 float32 还是 float64：未定。已有 `dtype="float32"` 选项，但 GPU 未验证（P4）。
5. 是否在 JAX 中实现 GSR 校准：未定（P7 之后）。
