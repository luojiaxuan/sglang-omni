# H100 接力交接 — thinker decode forward 优化（从 0 开始）

> **用途：** 在 **H100** 上接力 thinker decode forward 优化。承接 A6000（诊断）
> → B200（online-FP8 实测）的工作。核心待验证假设：**FP8 把 expert 权重流量减半，
> 在带宽更低、更 memory-bound 的 H100 上是否能真正提速 decode（B200 上不行）。**
>
> 本文档自包含：背景结论、git 入口、H100 环境从 0 搭建、踩坑、实验流程、对照基线。
> 所有脚本/分析工具/基线数字都在 fork 分支里，`git clone` 即可。
>
> **关联：** issue [sgl-project/sglang-omni#760](https://github.com/sgl-project/sglang-omni/issues/760)
> （B200 结果 comment: [#issuecomment-4705723359](https://github.com/sgl-project/sglang-omni/issues/760#issuecomment-4705723359)）；
> 同包 `FINDINGS_A6000.md` / `FINDINGS_B200.md`。
>
> **2026-06-16 最新状态：这个假设已在 H100 上验证完，结论是不成立。**
> generic online-FP8 只把 thinker 权重从 28.6 GB/卡降到 14.7 GB/卡，bs=32
> forward 只快约 2-3%，decode step 基本不变。当前接力方向应回到 **BF16
> baseline**，重点看 H100 BF16 forward split、scheduler/build 开销、以及 H100 BF16
> MoE kernel config。详见 `FINDINGS_H100.md`。

---

## 0. 一句话目标

已在 H100（TP=2）上用**与 B200 完全相同的方法**跑完 BF16 vs online-FP8 的 decode A/B。
结论：**FP8 没有带来有意义的 H100 decode 提速，后续回到 BF16。**

新的接力目标：

1. 以 BF16 c32 rebaseline 作为 H100 active baseline。
2. 用 nsys / scheduler phase 重新拆 H100 的 5-6 ms forward，确认 MoE/attention/dense/TP
   的真实占比。
3. 继续看 host/scheduler：c32 约 1.8 ms/step，不再是可以忽略的尾巴。

---

## 1. 背景：已完成的结论

| 阶段 | 硬件 | 结论 |
|------|------|------|
| 诊断 | A6000 (~0.77 TB/s) | decode **memory-bound**；MoE = 69% GPU-busy / **59% critical path**；18.6 ms/step。lever = 减 MoE 权重流量（FP8/int8）。 |
| 实测 | B200 (~8 TB/s) | **online-FP8 通用路径已落地**，但 **FP8 不提速（~7–10% 慢）**。原因：`fp8_w8a8` MoE kernel 在 B200 **无 tuned config** + W8A8 每步 activation 量化 + B200 带宽富余（decode forward 仅 ~5.9 ms，权重流量占比远低于 A6000）。 |
| 结构发现 | B200 | decode step ≈ 8.5 ms 中 **仅 ~70% 是 GPU forward；~30%（~2.5 ms）是 host/scheduler**（`recv` IPC-relay ~1 ms + `sched` ~0.57 ms），与量化无关。 |
| 实测 | H100 (~3.35 TB/s) | generic online-FP8 **不值得作为 latency/throughput lever**：BF16 bs=32 forward ~5.8 ms，FP8 ~5.6-5.7 ms；step 基本不变。BF16 rebaseline c16/c24/c32 显示 H100 瓶颈已变成 GPU forward + host/scheduler + kernel config 的混合问题。 |

**H100 结果为什么重要：** H100 HBM3 ≈ **3.35 TB/s**，是 B200（~8 TB/s）的 ~42%。
A6000 的结论曾预测 H100 上 FP8 会赢；实测没有赢，说明 H100 已经不再是 A6000
那种单纯 MoE 权重带宽主导的 regime。

**B200 基线数字（H100 对照用）：**

| 指标（bs=32, TP=2, decode） | B200 BF16 | B200 FP8 |
|------|------|------|
| forward GPU ms/step（`[fwd-by-bs]`, PHASE_SYNC） | 5.8–5.9 | 6.2–6.5 |
| decode step 总 ms（`[step phases]`） | 8.44 | 8.79 |
| host/scheduler 开销 | ~2.5 ms（~30%） | ~2.5 ms（~30%） |
| thinker 权重显存/卡 | 28.6 GB | 14.7 GB |
| CUDA graph 命中 | 100% | 100% |

---

## 2. Git：唯一入口

代码（引擎策略 + harness + findings + 本文档）都在 fork 分支 `perf/b200-moe-fp8`：

```bash
git clone https://github.com/luojiaxuan/sglang-omni.git
cd sglang-omni
git checkout perf/b200-moe-fp8
```

**关键文件：**

| 文件 | 作用 |
|------|------|
| `sglang_omni/model_runner/model_worker.py` | `_apply_model_worker_backend_policy`：online-FP8（无 native ckpt）→ pin triton fused-MoE FP8。**无需改动**，H100 直接生效。 |
| `benchmarks/diagnostics/thinker_decode_forward/scripts/serve_thinker.sh` | 起 thinker（TP），支持 `QUANTIZATION=` / `NSYS_PREFIX=` 旋钮 |
| `.../scripts/_common.sh` | 全部 env 默认值（无 host 硬编码路径） |
| `.../scripts/decode_load_text.py` | **纯文本** steady-decode 压测（免 simuleval；驱动 `[fwd-by-bs]`） |
| `.../scripts/prewarm_fp8_jit.sh` | online-FP8 JIT 预热（**FP8 必跑一次**，见 §4.2） |
| `.../FINDINGS_A6000.md` / `FINDINGS_B200.md` | BF16/FP8 基线对照 |
| `.../analysis/decode_split.py` / `overlap.py` | nsys decode-graph 拆分（**需先修 nsys schema**，见 §5.5） |
| `tests/unit_test/qwen3_omni/test_fp8_backend_config.py` | backend policy 单测（含 online-FP8 → triton 用例） |
| `docs/basic_usage/qwen3_omni.md` § General online FP8 | 用法 + 注意事项 |

---

## 3. H100 项目目标（分阶段）

- **Phase 0 — smoke：** 已完成。
- **Phase 1 — BF16 baseline：** 已完成；active baseline 见 `FINDINGS_H100.md`。
- **Phase 2 — FP8 A/B：** 已完成；结论是 FP8 只省显存，不作为当前提速方向。
- **Phase 3 — host/scheduler 占比：** 已完成初步量化；c32 约 1.8 ms/step（~24%）。
- **Phase 4 — nsys split：** 下一步。先修 `decode_split.py`（§5.5），再拆 GPU forward（MoE/attn/AR/dense %）。

---

## 4. 环境搭建（H100，从 0）

### 4.1 硬件 / 镜像 / 依赖

- **2× H100**（TP=2）。80 GB 足够：BF16 thinker ~28.6 GB/卡，FP8 ~14.7 GB/卡。
  为了和 FP8 A/B 及共享机器稳定性对齐，当前 H100 baseline 使用
  `thinker_memory_fraction=mem_fraction_static=0.55`。
- CUDA ≥ 12.x（Hopper sm_90）、PyTorch 支持 sm_90。
- Python 依赖（venv 放 `/data` 下，见 AGENTS.md）：`sglang==0.5.12.post1`、`sgl_kernel`、`flashinfer`、与 B200 同版本。
- **`ninja` 必装**（online-FP8 JIT 编译要用；见 §5.1）。`pip install ninja`。
- `nsys`（2025.x，可选，用于 Phase 4）。

```bash
# venv 放 /data，可复用
python3 -m venv /data/<you>/sglang-omni/.venv
source /data/<you>/sglang-omni/.venv/bin/activate
pip install -e .            # 或按仓库 README 装 sglang-omni + deps
pip install ninja           # 关键
```

### 4.2 模型权重（完整性自查！）

```bash
export HF_HOME=/data/cache/huggingface          # 共享 HF cache（AGENTS.md）
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct
```

> **坑（B200 实测）：** HF cache 里可能是**残缺 checkpoint**（`model.safetensors.index.json`
> 引用 15 个 shard，但只下了部分，缺 thinker 的 shard 9–14）。起服务前**务必校验完整**：

```bash
python - <<'PY'
import json, os, glob
snap = glob.glob(os.path.expanduser("$HF_HOME")+"/hub/models--Qwen--Qwen3-Omni-30B-A3B-Instruct/snapshots/*/")[0]
idx = json.load(open(snap+"model.safetensors.index.json"))
shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.realpath(snap+s))]
print("referenced", len(shards), "missing", missing)
PY
# 缺就补：huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct <缺的 shard 文件名...>
```

### 4.3 PATH（让 JIT 找到 ninja）

直接用 venv 的 python 起服务时，venv 的 `bin/` 不在 PATH，JIT 子进程会找不到 `ninja`：

```bash
export PATH=/data/<you>/sglang-omni/.venv/bin:$PATH
```

---

## 5. 关键 gotchas（B200 踩坑，H100 大概率一样）

### 5.1 ninja 必须在 PATH
online-FP8 在 load/首个 prefill 会 JIT 编译 `per_tensor_quant_fp8` /
`per_token_group_quant_8bit`（tvm-ffi + ninja）。缺 ninja → `FileNotFoundError: 'ninja'`，
thinker 启动即死。

### 5.2 online-FP8 首个 prefill 的 JIT 竞争（必须预热）
首个 prefill 懒编译 FP8 量化 kernel；TP 下该编译与 NCCL collective 竞争会污染 CUDA
context → `Failed to CUDA calloc async ...`，thinker crash。**解决：先跑一次预热**（用
`CUDA_LAUNCH_BLOCKING=1` 串行编译，落持久 cache `~/.cache/tvm-ffi`）：

```bash
QUANTIZATION=fp8 GPUS=0,1 THINKER_GPUS=0,1 TP_SIZE=2 PORT=8133 \
  PYTHON=/data/<you>/sglang-omni/.venv/bin/python \
  MODEL_PATH=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  bash benchmarks/diagnostics/thinker_decode_forward/scripts/prewarm_fp8_jit.sh
```
预热后，正常（非 blocking）FP8 server 即可干净启动。cache 持久，每台机/每次清 cache 后跑一次即可。

### 5.3 untuned FP8 MoE kernel
日志会告警 `Using default MoE kernel config ... Config file not found at
.../E=128,N=384,device_name=NVIDIA_H100...,dtype=fp8_w8a8.json`。H100 大概率也**没有
tuned config** → FP8 MoE 跑通用配置，可能吃掉带宽收益。这是 FP8 在 H100 若不提速的首要嫌疑。
（要榨：`sglang/benchmark/kernels/fused_moe_triton` 生成 H100 tuned config 后复测。）

### 5.4 进程清理要彻底
kill server 时，coordinator/HTTP（uvicorn）进程**不持有 GPU 显存**，`nvidia-smi
--query-compute-apps` 看不到它，但它**占着 port**——下次起服务会 fallback 到别的随机端口，
导致 health/压测连不上。按 `/proc/*/cmdline` 扫 `thinker_decode_forward` /
`spawn_main` / `stage_workers` / `sglang_omni_qwen3_text_tp_server` 全杀（别用 `pkill -f`
匹配到自己的命令行）。`prewarm_fp8_jit.sh` 末尾有现成清理逻辑可参考。

### 5.5 nsys / simuleval 工具坑
- `run_concurrency.py`（SST 音频 load）需要 `simuleval` CLI——B200 上**没装**。用
  `decode_load_text.py`（纯文本压测）替代，照样驱动 thinker prefill+decode。
- nsys 2025.6.3：`decode_split.py` 用的 sqlite 表名 `CUPTI_ACTIVITY_KIND_KERNEL` 已改名；
  且 text-driven capture 出现 “no CUDA kernel data”。Phase 4 前需修这两处。

---

## 6. 实验流程（H100，可复制粘贴）

统一 env（按机器改 `<you>` / GPU id）：

```bash
cd sglang-omni
export PATH=/data/<you>/sglang-omni/.venv/bin:$PATH
export HF_HOME=/data/cache/huggingface
COMMON="PYTHON=/data/<you>/sglang-omni/.venv/bin/python \
  MODEL_PATH=Qwen/Qwen3-Omni-30B-A3B-Instruct \
  GPUS=0,1 THINKER_GPUS=0,1 ENCODER_GPU=0 TP_SIZE=2 PORT=8133 SERVED_NAME=qwen3-omni \
  SGLANG_OMNI_PHASE_PROFILE=1 SGLANG_OMNI_PHASE_SYNC=1 \
  SGLANG_OMNI_DECODE_STATS=1 SGLANG_OMNI_DECODE_STATS_INTERVAL=2 \
  ENABLE_MIXED_CHUNK= CHUNKED_PREFILL_SIZE=0"
PKG=benchmarks/diagnostics/thinker_decode_forward
```

### Phase 0/1 — BF16 baseline

```bash
# 起 BF16 server（后台），等 health
env $COMMON nohup bash $PKG/scripts/serve_thinker.sh > /tmp/bf16.log 2>&1 &
until curl -fsS http://127.0.0.1:8133/health | grep -q healthy; do sleep 3; done

# 正确性自查（应返回 "Paris"）
curl -sS -X POST http://127.0.0.1:8133/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-omni","messages":[{"role":"user","content":"Capital of France? one word."}],"modalities":["text"],"temperature":0,"max_tokens":8}'

# 跑 45s 稳定 decode 压测
python $PKG/scripts/decode_load_text.py --base-url http://127.0.0.1:8133 \
  --concurrency 32 --max-tokens 256 --duration 45

# 读基线：forward GPU ms + host 拆分（bs=32 那行）
grep -hE "fwd-by-bs|step phases\] decode|decode stats" /tmp/bf16.log | tail
# 清理
python - <<'PY'  # 见 §5.4 的清理脚本
PY
```

### Phase 2 — FP8 A/B

```bash
# 1) 预热 JIT（见 §5.2，每机一次）
env QUANTIZATION=fp8 $COMMON bash $PKG/scripts/prewarm_fp8_jit.sh

# 2) 起 FP8 server（warm cache 后正常启动），同 load
env QUANTIZATION=fp8 $COMMON nohup bash $PKG/scripts/serve_thinker.sh > /tmp/fp8.log 2>&1 &
until curl -fsS http://127.0.0.1:8133/health | grep -q healthy; do sleep 3; done
python $PKG/scripts/decode_load_text.py --base-url http://127.0.0.1:8133 \
  --concurrency 32 --max-tokens 256 --duration 45
grep -hE "fwd-by-bs|step phases\] decode" /tmp/fp8.log | tail
```

确认日志里 `backend policy: ... effective_quantization=fp8 native_fp8_block_quant=False
moe_runner_backend=triton`，且 `Load weight end ... mem usage≈14.7 GB`（FP8 权重已生效）。

---

## 7. 测量方法（apples-to-apples）

- **Headline 指标：** `[fwd-by-bs] decode 32:X ms`（PHASE_SYNC 真 GPU forward）。两条
  TP rank 都看。**不要**把 throughput / seg-s 当绝对值（PHASE_SYNC 扰动吞吐）；throughput
  仅作交叉验证趋势。
- **host/scheduler：** `[step phases] decode ... | gpu=… host_pre=… host_post=…`，算
  `(step - gpu)/step` 即 host 占比。
- 唯一变量是 `QUANTIZATION`（其余 GPU/TP/load/flags 全一致）。

**对照表模板：**

| 指标（bs=32, TP=2, H100, decode） | H100 BF16 | H100 FP8 | Δ | B200 参考 |
|------|------|------|------|------|
| forward GPU ms/step | _填_ | _填_ | _填_ | 5.9 / 6.2 |
| step 总 ms | _填_ | _填_ | | 8.44 / 8.79 |
| host 占比 | _填_ | _填_ | | ~30% |
| 权重显存/卡 (GB) | _填_ | _填_ | | 28.6 / 14.7 |
| graph 命中 / 正确性 | _填_ | _填_ | | 100% / ✅ |

---

## 8. 当前决策

- **H100 FP8 结论：** 不作为当前 decode latency/throughput 优化方向。它省显存，但 bs=32
  forward 只快约 2-3%，step 基本不动。
- **H100 BF16 baseline：** c32 约 3775 tok/s，`[fwd-by-bs]` bs=32 约 5.8 ms，
  `[step phases]` decode step 约 7.75 ms，host/scheduler 约 1.8 ms（~24%）。
- **下一步：** 回 BF16，做 nsys forward split + scheduler/build profile。只有在 H100
  tuned `fp8_w8a8` MoE config 出来、或目标变成省显存时，再回头测 FP8。

---

## 9. H100 TODO checklist

- [ ] 2× H100 + 环境（CUDA/torch sm_90、sglang 0.5.12.post1、**ninja**、nsys 可选）
- [ ] `git clone` fork + `checkout perf/b200-moe-fp8`
- [ ] 校验 BF16 checkpoint 完整（15 shards，§4.2）
- [ ] `export PATH` 含 venv bin（§4.3）
- [x] Phase 0 smoke（BF16 起服务 + "Paris"）
- [x] Phase 1 BF16 baseline（`[fwd-by-bs]` + `[step phases]`）
- [x] Phase 2 FP8：先 `prewarm_fp8_jit.sh`，再 A/B
- [x] 填 §7 对照表 → 写 `FINDINGS_H100.md`
- [x] Phase 3 host/scheduler 占比
- [ ] 修 `decode_split.py`（nsys 2025.6.3）→ nsys GPU-forward 拆分
- [ ] 更新 #760 comment with BF16 rebaseline / final H100 decision

---

*文档版本：2026-06-16。H100 FP8 A/B 已完成；当前方向回 BF16 rebaseline。*
