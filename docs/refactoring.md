# 代码风格与函数式边界改造

本轮已按 [审查方案](code-review-plan.md) 实施八个批次。输入和执行配置冻结、估值上下文显式化、LSMC 结果独立于引擎状态、静态计划与批量计算分离，并将功能说明集中到用户文档。候选优化经过 CPU/CUDA 测量后分别取舍；没有将消除 Python 循环本身作为完成标准。

## 交付与迁移

| 批次 | 已完成内容 | 行为边界 |
| --- | --- | --- |
| 1 · 不可变输入 | 冻结引擎配置，复制 SampleDef、回归特征、SLV 相关性和 Hybrid 因子序列，冻结样条常量 | 修改配置需新建引擎；调用者修改输入列表不影响已有对象 |
| 2 · 统一风格 | Ruff 0.16.10、100 列、导入与 Python 3.13 语法统一，CI 检查格式和 lint | 机械整理独立提交，保留复杂度、Pylint 和安全检查 |
| 3 · 文档与注释 | 新增按任务组织的用户指南、完整配置表、架构与数值约束，README 提供任务导航 | 注释保留简短的数值原因；代码/docstring 无文档反向引用 |
| 4 · 估值上下文 | `ValuationContext`、显式拥有的 `ValuationSession`，准备和诊断入口捕获完整快照 | 废弃 global setters 的兼容层仍保留；完整上下文不依赖它 |
| 5 · 结果和缓存 | `TrainingResult`、`LsmcResult`、纯训练函数工厂、`path_collector`、实例缓存清理、命名数组状态 | `value()` 与 `train()` 返回类型不变；最后一次诊断字段迁移到独立结果 |
| 6 · 查找与预测 | FixingSnapshot 二分索引，验证集候选直接 stack | 保留别名、精确时间、零值和反向 FX；移除重复构造与冗余索引 vmap |
| 7 · 批量模型准备 | GSR 曲线/积分静态计划，GSRSLV 与 Hybrid 批量系数和前缀积分 | 动态模型参数仍参与 AAD；保留 singular covariance 和 active-zero 行为 |
| 8 · 循环实验 | CPU ordered Welford scan、设备 LSMC reverse scan、Sobol XOR 对照实验 | GPU Welford 保留 host；Sobol 保留原 bit 循环 |

模块级业务状态只保留一个已弃用的 DAL 兼容 session。旧 date/fixing 两套全局状态合并成原子快照，但多个 setter 调用仍不能隔离并发请求。`ValuationContext` 默认含显式空 fixing，因此新入口与例子无需这种回退。静态表使用无外部可写 backing 的只读 mapping，模块级业务 memoization 已移除。

JAX 的 x64、设备拓扑和 JIT/持久编译缓存属于第三方进程运行时。导入时启用 x64 是保留数值默认值的兼容例外；其余设置通过启动时的 `config.configure` 处理。本轮没有声称第三方运行时或 legacy API 完全无全局状态。

引擎缓存由实例拥有，`clear_cache()` 释放本实例保存的函数引用；缓存没有自动容量上限。已返回的价格函数和结果不失效。服务应限制 path-count 配置、按需清理或丢弃实例。新的数值核心在工厂构建完成后不写入宿主缓存。

迁移代码和行为说明见 [用户指南](user-guide.md#migrate-legacy-stateful-calls)。原有 `engine.regressions`、`engine.replicate_means` 改为从同一次 `evaluate()` 返回结果读取。`EvaluationDate_Set/Get` 与 `set_global_fixings` 发出 `DeprecationWarning`；旧调用仍可运行。16 个示例使用 script engine，JAX 一侧全部传递显式上下文；原生 DAL 对照的日期设置属于外部 oracle 接口。

## 验证

- Python 3.13.9 / JAX 0.11.2，固定 DAL 源码 oracle `4feabe89b105e0a3883fd4e2d74a2d70d618d8c7`：完整 CPU 测试 911 通过，65 个可选 GPU 测试在该轮跳过；真实 CUDA 测试 65 通过。
- 输入别名、只读配置、两个交错线程的 session/context 隔离、大 fixing 快照、独立 LSMC 结果和缓存清理均有行为测试。原有 oracle 容差和 deterministic 逐位要求未放宽。
- 全部 16 个示例运行通过（4096 路径、4 个虚拟 CPU 设备、1 次热运行），包含价格、Greeks、回归与诊断对照；文档六段连续工作流实际执行通过。
- Ruff、复杂度 CCN ≤ 8、Pylint errors、Bandit 和 Semgrep 检查通过。AST/token 审计确认无代码注释或 docstring 指向文档；静态模块容器无可变 list/dict/set 字面量。
- wheel/sdist 构建及 Twine 检查通过；无 native DAL 的独立安装环境运行 BS、GSR、GSRSLV、Hybrid、LSMC 价格和风险 smoke。PyPI 发布仍延期。

撤回单区间快捷表达式后，利率／Hybrid 的 CPU 与原生 DAL 定向回归 65 通过，CUDA P6/P7 定向回归 21 通过。测试中的 16 条弃用警告来自刻意覆盖 legacy 接口的用例。

PR 审查另修复三个边界问题，新增四个回归用例：

- Sobol 方向表始终复制所需维度，避免一维转置视图保留整张 21201 × 32 表；通过弱引用检查生成器存活时原表已释放，同时验证随机点不变。
- 曲线计划在主机端检查投影行号，保留所有负数行号表示折现曲线的原有语义；检查 JIT 取值、参数梯度和越界异常，避免 JAX 索引静默截断。
- 路径收集器在缓存查找前校验路径数，避免整数键与等值浮点键共享缓存后绕过校验；分别覆盖冷缓存和热缓存。

## 基准口径

数值基线为 `6aac62a`，包含冻结配置和独立 Ruff 机械整理；此前审查基线是 `17bc047`。在独立 worktree 中加载基线代码，使用同一 [bench_style.py](../benchmarks/bench_style.py)、参数、事件表和随机流比较。各计时进程顺序运行，未与其他验证任务并行。

Intel Core i9-13900HX / RTX 4060 Laptop，Python 3.13.9、JAX 0.11.2、CUDA 13。每个模型 257 条 Sobol 路径、block 128、单设备、float64，计算 PV 和全部参数梯度。基础日程 8 个事件，扩展日程 32 个事件；模型含两个 GSR 因子，Hybrid 额外含一只股票。GPU 设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false`，前后相同。

准备、lowering、XLA compilation 分别记录。编译后先执行一次，再同步执行 20 次取 warm 中位数。内存列是 XLA 的 `temp_size_in_bytes`，不是实际 allocator 峰值或进程 RSS。这组小路径测量重点是图构建与编译扩展性，不替代已有百万路径性能报告，也不保证其他设备或产品获得相同收益。


### CPU · 8 个事件

每格为基线 → 最终版本。warm 使用毫秒，临时缓冲使用 KiB。

| 模型 | lowering（s） | XLA 编译（s） | warm（ms） | 临时缓冲（KiB） |
| --- | ---: | ---: | ---: | ---: |
| gsr | 1.170 → 0.788 | 1.961 → 0.713 | 1.147 → 0.932 | 533.648 → 529.336 |
| slv | 1.369 → 0.892 | 1.074 → 0.897 | 2.288 → 2.652 | 766.773 → 767.211 |
| hybrid | 1.470 → 1.147 | 1.648 → 0.902 | 1.465 → 1.410 | 1128.984 → 718.672 |
| hybrid_slv | 1.651 → 1.414 | 1.364 → 1.320 | 3.420 → 3.571 | 1174.281 → 1176.281 |

原始记录：[基线](../benchmarks/style_baseline_cpu.json)、[最终](../benchmarks/style_final_cpu.json)。各模型 PV 和所有风险在原有容差内一致。

### CUDA · 8 个事件

每格为基线 → 最终版本。warm 使用毫秒，临时缓冲使用 KiB。

| 模型 | lowering（s） | XLA 编译（s） | warm（ms） | 临时缓冲（KiB） |
| --- | ---: | ---: | ---: | ---: |
| gsr | 1.265 → 0.771 | 4.661 → 2.069 | 6.229 → 3.787 | 386.953 → 4111.703 |
| slv | 1.231 → 0.903 | 3.235 → 2.411 | 11.529 → 6.659 | 764.500 → 763.750 |
| hybrid | 1.342 → 1.097 | 4.607 → 2.394 | 7.219 → 4.063 | 723.273 → 718.023 |
| hybrid_slv | 1.516 → 1.357 | 3.979 → 3.296 | 14.659 → 8.461 | 1165.555 → 1165.312 |

原始记录：[基线](../benchmarks/style_baseline_gpu.json)、[最终](../benchmarks/style_final_gpu.json)。各模型 PV 和所有风险在原有容差内一致。

### CPU · 32 个事件

每格为基线 → 最终版本。warm 使用毫秒，临时缓冲使用 KiB。

| 模型 | lowering（s） | XLA 编译（s） | warm（ms） | 临时缓冲（KiB） |
| --- | ---: | ---: | ---: | ---: |
| gsr | 3.764 → 1.106 | 8.817 → 1.424 | 4.689 → 3.316 | 2102.461 → 2092.273 |
| slv | 3.320 → 1.281 | 3.334 → 1.635 | 6.239 → 7.004 | 3043.523 → 3045.461 |
| hybrid | 4.180 → 2.063 | 7.155 → 3.032 | 6.771 → 5.963 | 7589.234 → 5166.547 |
| hybrid_slv | 4.944 → 2.674 | 5.085 → 3.623 | 12.411 → 12.075 | 9076.656 → 9085.844 |

原始记录：[基线](../benchmarks/style_baseline_cpu_32.json)、[最终](../benchmarks/style_final_cpu_32.json)。各模型 PV 和所有风险在原有容差内一致。

批量准备减少了所有测量用例的图构建和编译开销，32 个事件时 GSR 的编译扩展性改善尤其明显。热运行不保证一致改善：部分 CPU SLV/Hybrid-SLV 小路径用例出现回退，具体数值保留在表中。保留这组修改的依据是可读的静态/动态分离、明显的编译收益及数值契约保持，而不是统一吞吐加速承诺。

内存也有取舍。最终 CUDA GSR 的编译临时缓冲约 4.02 MiB，高于基线约 0.38 MiB；没有将其隐藏在性能结论里。单区间积分直接表达式的额外试验让 Gaussian Hybrid 临时缓冲升至约 4.02 MiB，撤回后恢复至约 0.70 MiB；最终实现统一保留 ordered integration scan。上述估计不等于实际峰值。

### 独立循环实验

Sobol 使用 4096 条路径 × 128 维；Welford 使用 16384 条路径 × 3 个特征，全部 included；backward 使用 16 个行权日期、512 条路径和一个特征。Sobol 检查整数逐位相等，Welford 检查 moments，backward 检查 policy 数组。它们是核级实验，不是完整估值 API 耗时。backward 的 unrolled 参考实现保留在基准脚本中。

| 候选 | 采用范围与原因 |
| --- | --- |
| Welford scan | CPU 采用保序 masked scan；GPU 保留 host Welford 和 host QR |
| LSMC reverse scan | 采用：显著减少编译体积，保持日期依赖，CPU warm 改善；CUDA 按本次测量记录 |
| Sobol XOR 整批归约 | 不采用：CUDA 编译显著变慢，warm 无明确收益，临时缓冲略增 |
| 单区间直接积分 | 不采用：Gaussian Hybrid 的 CUDA 临时缓冲明显增加 |

**CPU：** Welford host 37.615 ms，scan 0.104 ms；backward XLA 编译 2.765 → 0.316 s，warm 1.124 → 0.315 ms；Sobol XLA 编译 0.065 → 0.030 s，warm 0.340 → 0.294 ms。

**GPU：** Welford host 37.982 ms，scan 347.577 ms；backward XLA 编译 8.007 → 0.839 s，warm 14.741 → 7.976 ms；Sobol XLA 编译 0.202 → 3.789 s，warm 1.328 → 1.350 ms。

Welford host 记录一次运行，device scan 为编译后的 20 次中位数；这只能说明本次核级选择依据。实际 CPU 实现保留 included mask 与原路径顺序，排除行的无效值不会污染 moments。GPU host 结果继续遵守同一 Welford/QR 数值约定。

复现当前实现：

```sh
JAX_PLATFORMS=cpu uv run python benchmarks/bench_style.py --repeat 20 --output cpu.json --experiments
XLA_PYTHON_CLIENT_PREALLOCATE=false uv run python benchmarks/bench_style.py --platform gpu --repeat 20 --output gpu.json --experiments
JAX_PLATFORMS=cpu uv run python benchmarks/bench_style.py --events 32 --repeat 20 --output cpu-32.json
```

GPU 命令使用已安装 CUDA extra 的环境。基线需在 `6aac62a` 的 worktree 中运行当前基准脚本，令 `PYTHONPATH` 指向该 worktree 的 `src`，并省略 `--experiments`；设备、路径、事件、重复次数和环境变量保持一致。

## 持续遵循的准则

本地 `$pythonic-functional-style` skill 已保存通用准则。后续修改优先可读的 Python 表达、简短原因注释、文档对代码的单向依赖、不可变输入和显式结果。局部拥有的 builder/cache 可以变化；模块级业务状态原则上禁止，兼容例外需说明迁移。先识别独立轴、时间依赖、随机流和内存边界，再选择数组操作、vmap、分片或 scan；用真实后端测量决定是否保留优化。
