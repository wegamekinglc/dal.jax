# 示例 notebook

每个 notebook 都可以从头到尾直接运行。仓库里保存的是带输出的版本，数值来自一台 32 线程的笔记本（WSL2，8 个虚拟 CPU 设备），耗时会随机器变化。

| notebook | 内容 |
|---|---|
| [01_european_option](01_european_option.ipynb) | 入门：定义产品和模型，价格与 Greeks，与 Black-Scholes 闭式解和 DAL 对照，Sobol 与伪随机数的收敛速度 |
| [02_barrier_option](02_barrier_option.ipynb) | 为什么硬判断下 `jax.grad` 给出 `d_BARRIER = 0`，CSpr 平滑如何修正，用有限差分验证，与 DAL 逐位对照（含 Brownian bridge） |
| [03_jax_transforms](03_jax_transforms.ipynb) | 把 `pricer` 当作纯函数：`jit`、`jacrev`、`jacfwd`、`hessian`（gamma）、`vmap`（整条 spot 曲线）；亚式期权、延迟支付、`path_payoffs` 逐路径诊断 |
| [04_parallel_and_precision](04_parallel_and_precision.ipynb) | 虚拟 CPU 设备的扩展性，四种并行方式，`deterministic_reduction` 的逐位一致性，`float32`，块数分桶，GPU |

## 运行

在仓库根目录执行：

```bash
uv sync --group examples                       # 安装 dal_jax 本身，以及 ipykernel、matplotlib、nbconvert、dal-python
uv run --group examples --with jupyterlab jupyter lab examples/
```

也可以不打开界面，直接执行并写回输出：

```bash
cd examples
uv run --group examples jupyter nbconvert --to notebook --execute --inplace 01_european_option.ipynb
```

说明：

- 每个 notebook 的第一个代码单元会把 CPU 拆成最多 8 个虚拟设备，这一步必须在任何 JAX 运算之前执行；重新配置设备数需要重启内核；
- 与 DAL 对照的单元需要 `dal-python`，没有安装时会自动跳过；
- `nbtools.py` 提供共享的作图样式和 Markdown 表格输出。
