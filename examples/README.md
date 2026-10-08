# Python examples

所有示例都是可直接运行的普通 Python 脚本。每个脚本都打印 JAX 与 dal-python 的数值结果、误差和性能，并自动校验可逐路径比较的 PV 与 Greeks。DAL 是必需的对照依赖。默认使用 65,536 条路径、8 个以内的虚拟 CPU 设备，热运行重复 3 次。

终端表格采用 DAL 的定宽样式：文字列左对齐、数值列右对齐，列宽按表头与内容自动计算，分隔线覆盖整张表。

| 脚本 | 内容 |
|---|---|
| [01_european_option.py](01_european_option.py) | 原生 payoff、全部 Greeks、Black–Scholes 闭式解、Sobol/PRNG 收敛 |
| [02_barrier_option.py](02_barrier_option.py) | 硬判断与 fuzzy 的障碍风险、Brownian bridge、共同路径有限差分 |
| [03_jax_transforms.py](03_jax_transforms.py) | jacfwd/jacrev/JVP、C1 Hessian 与 DAL delta bump、vmap spot ladder、多 payoff Jacobian |
| [04_parallel_and_precision.py](04_parallel_and_precision.py) | 四种并行策略、确定性规约、float32、路径数分桶 |
| [05_script_and_history.py](05_script_and_history.py) | 脚本参数、历史回放、Asian、DCF 日程、原生贴现槽位和延迟支付 |
| [06_scan_and_smoothing.py](06_scan_and_smoothing.py) | 扫描与展开、事件分组、C1 核与 DAL 等价三次多项式 |
| [07_prng_streams.py](07_prng_streams.py) | Threefry/RBG 的统计检验与并行流一致性 |
| [08_gpu_and_precision.py](08_gpu_and_precision.py) | CPU/GPU 选择、float64/float32/auto、自动块大小与 DAL 对照 |
| [09_vectors.py](09_vectors.py) | 向量 Asian、FOR/下标/归约、fuzzy 长度混合、具名错误 |
| [10_fixings_and_payments.py](10_fixings_and_payments.py) | 快照、当日定盘策略、跨日期 FIX、延迟 PAYS ON |
| [11_correlated_bs.py](11_correlated_bs.py) | 双资产 basket/worst-of、多因子 bridge、default_index |
| [12_local_vol.py](12_local_vol.py) | 平坦/非平坦曲面、全部 bucket vega、共同路径有限差分 |
| [13_bermudan.py](13_bermudan.py) | LSMC 三阶段、训练/定价耗时、固定策略 JAX transforms、回归诊断 |
| [14_lsmc_validation_rqmc.py](14_lsmc_validation_rqmc.py) | 验证集阶数选择、RQMC 副本、重训策略风险 |
| [15_gsr_rates.py](15_gsr_rates.py) | 单/多因子 GSR、投影曲线、DF/Libor/Swap、利率行权 |
| [16_gsr_slv_hybrid.py](16_gsr_slv_hybrid.py) | GSRSLV、具名因子相关性、BS/local-vol 与利率混合 |

在仓库根目录运行：

```bash
uv sync --group examples
uv run --group examples python examples/01_european_option.py
uv run --group examples python examples/06_scan_and_smoothing.py --paths 65536 --devices 8 --repeat 3 --output /tmp/scan.json
```

所有脚本接受 `--paths`、`--devices`、`--repeat`、`--platform` 和 `--output`。CPU 设备在首次 JAX 运算前配置；重新配置设备数需启动新进程。01–08 可直接使用 PyPI 的 dal-python。09–16 需要固定的 DAL 源码版本 `4feabe89b105e0a3883fd4e2d74a2d70d618d8c7`，其 Python 包版本号仍是 2026.9.25，不能仅凭版本号识别新功能。以下命令建立隔离环境，编译 C++ 并安装源码 oracle（需要 Git、CMake、C++17 编译器和 uv）：

```bash
uv venv /tmp/dal-jax-native --python 3.13
uv pip install --python /tmp/dal-jax-native/bin/python -e . pytest scipy
bash scripts/build_dal_oracle.sh /tmp/dal-jax-native-build /tmp/dal-jax-native/bin/python
/tmp/dal-jax-native/bin/python examples/09_vectors.py
/tmp/dal-jax-native/bin/python examples/12_local_vol.py --output /tmp/local-vol.json
```

使用该隔离环境运行全部脚本：

```bash
for example in examples/[0-9][0-9]_*.py; do
  /tmp/dal-jax-native/bin/python "$example" --paths 4096 --devices 4 --repeat 1
done
```

GPU 示例需要对应 CUDA extra：

```bash
uv sync --group examples --extra cuda13
XLA_PYTHON_CLIENT_MEM_FRACTION=.35 uv run --group examples python examples/08_gpu_and_precision.py --platform gpu --devices 1
```

`--platform gpu` 要求实际 GPU；不会自动退回 CPU。生产用途默认 float64。长 autocall 的窄平滑条件可能放大 float32 的路径误差，PV 接近并不保证 Greeks 接近，详见 [性能报告](../docs/performance.md)。

JAX 的编译时间单独记录；首次执行和热运行都等待设备完成。DAL 记录首次和热运行，C++ 编译已在安装时完成。产品解析、主机准备和模型分配在计时外；JAX 测量已编译的价格/梯度函数，DAL 测量其 MonteCarlo_Value API。表格保留双方实际设备数，不能把这些结果理解为等核数比较。`--output` 保存全部数值、编译时间、同步热运行最小值、median 和原始次数。

Sobol 的 float64 对照使用 PV 相对容差 1e-10、Greeks 1e-8，近零值绝对容差 1e-10。JAX 与 DAL 的 MRG32 名称使用不同随机流；01 展示收敛结果，07 按解析标准误校验统计差异。C1 示例的 DAL 脚本显式实现相同三次核；05 的延迟支付用 DAL 的“保存金额、到支付日付款”脚本对照。Gamma 对照来自 DAL delta 的有限差分。

[results/](results/) 保存本机按默认参数依次运行全部脚本的 JSON，以及 08–16 在实际 RTX 4060 Laptop / CUDA 13 上的额外报告。耗时受机器和负载影响；重新运行即可生成自己的报告。CI 分别使用 PyPI oracle 执行 01–08、固定源码 oracle 执行 09–16（4,096 路径、4 个 CPU 设备、1 次热运行），源码 oracle 任务还运行完整 CPU 测试；独立分发包任务构建 alpha 并验证无 DAL 的 wheel 安装。P5 的 CPU/GPU 实测见 [P5 报告](../docs/p5.md)。

13–16 的新报告使用 4 个 CPU 设备及 1 个实际 GPU，默认 65,536 条定价路径、3 次热运行。LSMC 比较放宽 PV 相对容差至 1e-6，回归系数相对容差为 1e-8；Greeks 保持原容差。13 额外报告训练及每次重新训练的总估值耗时。14 的重训风险示例使用最多 1,024 条训练路径、4,096 条定价路径，不能直接与它的 RQMC 时表比较。新方法、迁移约定及完整性能口径见 [P6/P7 文档](../docs/p6-p7.md) 和 [性能报告](../docs/p6-p7-performance.md)。
