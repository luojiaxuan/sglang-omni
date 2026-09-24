# Qwen3-TTS Base 声音克隆的前导静音复现

## 问题

有用户反馈:Qwen3-TTS Base 在 x-vector-only 声音克隆模式下(只给参考音频、不给参考文本),
生成音频开头有长度不固定的静音,直接加在可感知的首音延迟(audible TTFA)上。报告中的
数据是跨句子 125–460 ms、跨 seed 330–740 ms。给出的机制解释是:talker 在开头几帧有相当
概率采到"静音区"的 codebook-0 token,采几帧由采样决定。一个参照实现的做法是在前 N 个
解码帧屏蔽这些 token。

仓库现状:#1901 只处理流式 CustomVoice(Ryan/英文)确定出现的 1 帧 80 ms 静音,在
vocoder 输出端扣掉这段 PCM;Base 请求不走这条路径。#1754 的 T-PR16 探针只检查了
0.6B-Base 的第 1 帧能否直接跳过(结论:不能,首帧内容随请求变化),没有测量前导静音
会持续几帧。

## 本次要回答的问题

1. 我们的栈上,1.7B-Base 与 0.6B-Base 在 x-vector-only 模式下,前导静音的分布是什么样的
   (中位数、尾部、超过 1 帧和 2 帧的比例)?
2. 波动主要来自句子(prompt)还是来自 seed?
3. ICL 模式(给参考文本)作为对照,前导静音是否明显更短、更稳定?

## 设计

- 服务:`examples/configs/qwen3_tts_1_7b.yaml` / `qwen3_tts_0_6b.yaml` 默认配置,非流式
  `/v1/audio/speech`(`stream=false`),所以测到的是模型生成的 codec 本身,与流式分块无关。
- 采样:服务默认值(temperature 0.9、top_k 50、top_p 1.0、repetition_penalty 1.05),
  `language` 不传(默认 `auto`)。
- 网格:2 种模式(xvec、icl)× 3 条参考音频(`zhaochenyang20/seed-tts-eval-mini` 的 EN
  prompt-wavs)× 12 句(原报告 8 句,加上参照实现测试里的 4 句)× 10 个 seed(0–9),
  每个模型 720 条请求。并发 8。
- 起声检测(`onset_client.py`):主指标与原报告一致,取第一个峰值达到 0.02(-34 dBFS)
  的 5 ms 帧;辅助指标取第一个 RMS 达到 -40 dBFS 的 10 ms 窗(5 ms 步长),用来检查结论
  对阈值是否敏感。每条 WAV 都保留。
- 汇总(`summarize.py`):按 (模型, 模式) 给出汇总分布、超过 80 ms / 160 ms 的比例、方差
  在 seed 与 (参考 × 句子) 之间的拆分、逐句中位数与范围。

## 运行记录

| 项 | 值 |
|---|---|
| 主机 | eval-h100(Radix 租约 `01M38CW18G6X7XE8JP4B36DB85`,1×H100 80GB,GPU 0,2026-09-23 17:25–19:25 PT) |
| 镜像 | CI 同款 `hongccc/sglang-omni@sha256:ebe4239e…5790df`,节点上重打为 `localhost/sglang-omni:dev`,image ID `2bac261779c8` |
| 代码 | 本分支 `7bd89ae9`(基于 main `481914c3`),只推 `sglang_omni/` 与 `examples/configs/`,`PYTHONPATH` 指向它;`SGLANG_OMNI_AUTO_CLONE=0` 关掉镜像入口的自动 clone |
| 容器 | `sglang-omni-jaxan-1`,`--network host`,服务只绑 `127.0.0.1:18731` |
| 节点目录 | `/data/luojiaxuan/runs/20260924T002735Z/`(`out/<模型>/records.jsonl` 与 `wav/`,`logs/`) |
| 看护 | 本机每 60 s 轮询:容器状态、日志里的 Traceback/OOM/Killed、`records.jsonl` 行数、主机可用内存;5 分钟无新记录报 STALE |
| 收尾 | 17:50 PT 已 teardown、release(`machines mine` 为空,余额 72 未变);节点上保留镜像(39 GB)与 venv(1.4 GB)供后续 A/B 复用,输出已删 |
| 墙钟 | 第二次发射 17:41 PT,venv 约 3 min,1.7B 约 4 min,0.6B 约 2 min,17:49 PT 结束 |
| 产物 | HF 私有数据集 `gavinlaw/sglang-omni-qwen3-tts-base-onset-en-audio`,revision `9686136ea94f4888180c294160560260a6aa63ff`(1440 条 WAV 的 tar、两份 `records-*.jsonl`、汇总文本),已按文件大小对账 |

## 结果

1440/1440 条请求成功,全部越过阈值。两种起声指标(-34 dBFS 峰值、-40 dBFS RMS)的中位数相差不超过 25 ms,
下文用峰值指标。一帧 codec 是 80 ms。

| Model | Mode | Median (ms) | Mean (ms) | p90 (ms) | Max (ms) | > 1 frame (80 ms) | > 2 frames (160 ms) |
|---|---|---|---|---|---|---|---|
| 1.7B-Base | x-vector only | 450 | 430 | 605 | 895 | 91% | 90% |
| 1.7B-Base | ICL | 380 | 437 | 847 | 1730 | 85% | 74% |
| 0.6B-Base | x-vector only | 375 | 337 | 580 | 1225 | 76% | 70% |
| 0.6B-Base | ICL | 450 | 432 | 745 | 1260 | 88% | 82% |

n = 360 per row (3 references × 12 prompts × 10 seeds).

1. **x-vector-only 的前导静音在我们的栈上复现,而且比原报告更重**(原报告均值 326 ms)。
   1.7B 九成请求开头有 2 帧以上静音。1.7B 的方差多半来自 seed(seed 内 22191 ms² 对
   参考×句子间 6986 ms²),但参考之间的均值仍有差别(336 / 478 / 475 ms);0.6B 的逐句中位数
   从 40 到 468 ms 不等,句子的影响不能忽略。
2. **1.7B 的分布是双峰的**。按 80 ms 分箱,1.7B xvec 为
   `0:34 1:3 2:4 3:6 4:65 5:99 6:85 7:38 8:18 9:4 10:3 11:1`(1–3 帧只占 3.6%);0.6B 为
   `0:81 1:28 2:10 3:21 4:55 5:69 6:55 7:16 8:14 …`(1–3 帧占 16.4%,双峰没那么干净)。
   直方图本身只是描述;"第一帧的选择决定整段"由下文的 token 探针直接证实。静音段电平约
   -60 dB,是参考录音底噪的水平,不是数字静音。
3. **ICL 的起声与参考音频末尾停顿同向**。1.7B ICL 按参考拆开,中位数 105 / 342 / 755 ms,
   对应参考末尾静音 135 / 170 / 300 ms;0.6B 同方向(278 / 418 / 570 ms)。只有 3 条参考,
   这是相关不是因果;要确认需对同一参考裁掉或补长末尾静音再测。

## 修复:前 N 帧屏蔽静音 token(2026-09-23 PT 晚)

设计与外审问题见 [fix_design.md](fix_design.md),外审原文见
[docs/reviews/2026-09-23-qwen3-tts-base-leading-silence-mask.md](../../reviews/2026-09-23-qwen3-tts-base-leading-silence-mask.md)。

### 实现

- `Qwen3TTSSGLangRequestData.mask_leading_silence = state.x_vector_only_mode`,只有 x-vector-only
  请求打标;ICL、CustomVoice、VoiceDesign 不受影响。
- `Qwen3TTSModelRunner.apply_codec_suppress_tokens` 对打标且已生成帧数 `len(output_codes) < N`
  的行,把静音集合 S 的 logit 置为 -inf。屏蔽发生在温度、top-k/top-p 之前,所以 top-k 在
  屏蔽后的分布上重新截断,原本排名 50 以外的 token 可能进入候选。retraction 重新 prefill 时
  `output_codes` 保留,不会重复屏蔽。
- N 由 tts_engine factory 参数 `leading_silence_mask_frames` 给出,默认 0(关闭)。S 在
  `before_memory_pool` 里由本 checkpoint 的 codec encoder 推导,启动日志打印集合。
- 单元测试:`tests/unit_test/qwen3_tts/test_mask_logit_shaping.py` 新增一例,3 例全过(容器内)。

### S 的三版与 token 探针

| 版本 | 定义 | 1.7B 大小 | 覆盖基线开场静音帧 | 语音帧落入 |
|---|---|---|---|---|
| v1 | 数字静音 + -80/-60 dBFS 白噪声 | 11 | 45% | 0 |
| v2(已提交) | 白/粉/棕噪声,数字静音到 -50 dBFS,每 5 dB,各 8 s | 24 | 69% | 0 |
| 数据推导(侦察) | 基线开场静音重新编码后出现 ≥3 次的 id | 28 | 98% | 0.9% |

覆盖率一栏来自 token 探针(模型实际采样的 token),语音帧一栏来自重新编码。-50 dBFS 取自
覆盖率-电平前沿的拐点:1.7B 与 0.6B 都在 -50 dBFS 饱和,到 -40 dBFS 仍无语音帧落入。v2 在
5 个随机种子下交集 24、并集 29,变化的只是罕见 id,覆盖率不变。

**token 探针**(`token_probe.sh`,调试副本记录每个请求前 12 帧的 codebook-0,并发 1 逐条对齐,
1.7B,360 条):

- N=0:322 条静音开头的请求,第 0 帧**全部是 1995**;直接开口的请求第 0 帧是 404、1221、9、
  1342 等。双峰就是第 0 帧是否采到 1995。
- N=1(屏蔽 v2 S):仍静音开头的 152 条,第 0 帧**全部是 1221**;1221 共被采到 213 次,其中
  61 次直接开口。1221 不是平稳噪声探针能得到的 token,是一个"轻起音、吸气"类 token。
- 所以 v1、v2、数据推导三版 S 在 N=1 下输出逐条相同:三者都含 1995,都不含 1221,第 0 帧的
  采样胜者不变。N=2 与 N=3 同样逐条相同:第 2 帧起从未采到 S 内的 token。

### 扫描结果

hyper00 4×H200,x-vector-only。网格 = 复现用的 360 条起声网格;SeedTTS = seed-tts-eval EN 全集
1088 条(`--no-ref-text`,seed 0,并发 16),WER 用 CI 同款 Qwen3-ASR,SIM 为 WavLM 说话人相似度。
失控 = 生成到 `max_new_tokens`(163.84 s)的条数。

| job | grid median | grid p90 | grid > 160 ms | SeedTTS WER | SIM | SeedTTS median | SeedTTS p90 | SeedTTS > 160 ms | runaways |
|---|---|---|---|---|---|---|---|---|---|
| 1.7B N0 | 445 | 635 | 89% | 0.854% | 61.00 | 495 | 745 | 95% | 0 |
| 1.7B N0(重复) | 445 | 635 | 89% | 0.846% | 61.00 | 495 | 745 | 95% | 0 |
| 1.7B N1 v1 | 135 | 520 | 46% | 0.888% | 60.94 | 470 | 735 | 73% | 0 |
| 1.7B N2 v1 | 115 | 455 | 34% | 0.888% | 60.86 | 155 | 607 | 47% | 0 |
| 1.7B N3 v1 | 115 | 455 | 34% | 0.888% | 61.00 | 155 | 610 | 46% | 0 |
| 1.7B N1 v2 | 135 | 520 | 46% | 0.888% | 60.94 | 470 | 737 | 73% | 0 |
| 1.7B N2 v2 | 115 | 455 | 34% | 0.904% | 60.87 | 155 | 607 | 47% | 0 |
| 1.7B N1 v2+1221 | 35 | 65 | 1% | 1.114% | 59.92 | 35 | 60 | 0% | 0 |
| 1.7B N2 v2+1221 | 35 | 65 | 1% | 1.105% | 59.95 | 35 | 60 | 0% | 0 |
| 0.6B N0 | 375 | 580 | 69% | 1.532% | 58.41 | 480 | 837 | 86% | 9 |
| 0.6B N1 v2 | 75 | 521 | 20% | 1.089% | 58.43 | 75 | 562 | 34% | 2 |
| 0.6B N2 v2 | 75 | 236 | 12% | 1.072% | 58.10 | 75 | 490 | 23% | 3 |
| 0.6B N1 v2+1221 | 40 | 70 | 0% | 1.315% | 57.24 | 35 | 55 | 0% | 3 |
| 0.6B N2 v2+1221 | 40 | 70 | 0% | 1.356% | 57.22 | 35 | 55 | 0% | 4 |

读法:

- 运行间噪声:两次 1.7B N0 的 WER 差 0.008 个百分点,SIM 相同;同一配置网格逐条可复现。
- **v2,N=2**:1.7B 的 SeedTTS 起声中位数 495 → 155 ms,超过 2 帧 95% → 47%,代价是 WER +0.05
  个百分点(约每 1 万词多错 5 个)、SIM -0.13。0.6B 超过 2 帧 86% → 23%,WER 1.53% → 1.07%
  (失控从 9 条降到 3 条,静音开头的生成更容易停不下来),SIM -0.31。N=3 不再改善。
- **v2 + 1221**:两个模型的前导静音基本消失(> 160 ms 为 0–1%),但 SIM 降约 1.1、1.7B WER
  0.85% → 1.11%。1221 是正常的轻起音 token,禁用它等于逼每句硬起音,首音素与音色受损。
  1221 由探针在 1.7B 上找到,是写死的 id,不属于可推导的 S。

### 待定:默认值与是否处理 1221

| 方案 | 1.7B SeedTTS > 160 ms | 质量代价 | 说明 |
|---|---|---|---|
| A. 关闭(N=0,当前提交) | 95% | 无 | 外审建议默认关 |
| B. v2,N=2 | 47% | WER +0.05 pp,SIM -0.13 | 在噪声边缘;0.6B 还减少失控 |
| C. v2 + 1221,N=1 | 0% | WER +0.26 pp,SIM -1.1 | 起声最好,质量代价可见,1221 是写死 id |

我的倾向:B,默认开启(本仓库规则是性能优化在正确性允许时默认开),同时保留 N=0 的关闭方式;
C 的质量代价需要主观听感和首音素错误率再判断。外审主张默认关闭,理由是这改变了输出行为而
不只是性能;这是分歧点,留给 luojiaxuan 定。

### 尚未做(外审提出,留作后续)

主观盲听与首音素错误率、点击与起音包络、长句(30–120 s)与非英语、流式首块的可感知 TTFA、
预先声明的非劣效边界与按参考聚类的 bootstrap 置信区间、1221 的软惩罚(有限 logit 偏置)
与"检测到语音即解除"的变体、ICL 末尾停顿的因果实验。

## 错误与返工

- 第一次发射(17:37 PT)服务启动失败:最新 main 需要 sglang 0.5.20 的
  `radix_eviction_policy_config`,镜像里的 sglang 更旧。CI 在镜像之上另建 venv 补齐 pin,
  `run.sh` 改为同样的做法后重发(17:41 PT)。
- 向节点同步修改后的 `run.sh` 时,一条命令误把 tar 解到了节点 `/tmp`(`/tmp/run.sh`、
  `/tmp/.github/`,均为本账号文件),当即删除并核对已不存在。
- 扫描第一次发射:一个容器里 4 个服务各开 224 个 OpenMP 线程,撞上 podman 默认每容器 2048 的
  进程上限(`libgomp: Thread creation failed`,另两台报 `unknown parameter type`)。改为
  `--pids-limit 65536` 并设 `OMP_NUM_THREADS=32`。
- 扫描第二次:benchmark 把参考音频解压到 `/data/tmp` 并以本地路径请求,服务只允许 `refs/`。
  放宽到 `/data`。
- 0.6B 的相似度阶段 OOM:最初误判为与 ASR 服务抢显存(因此加了 `wait_gpu_free`,留着无害),
  真正原因是失控生成的 163.84 s 音频在 WavLM 批内 padding 后要 64 GB。改为逐条打分
  (`similarity_one_by_one.py`),`sweep.sh` 已默认用它。
- S 的覆盖率最初用"生成音频重新编码"近似。它对开场静音段用哪些 token 近似得不错(数据推导版
  与探针相差不到 1 个百分点),但看不到决定性的第 0 帧退路 1221,导致数据推导版的侦察
  白跑了一轮;token 探针直接记录采样 token 后才定位。

## 决策日志

1. **eval-h100 的存储**。问题:skill 记录 09-22 这台没有可写个人目录,规则要求管理员提供之前
   不部署。实测 09-23:`/data` 已是 21 TB、`1777` 的共享盘,另有用户建了自己的目录,管理员放了
   `_root_cleanup_*`,与 hyper 的 `/data01` 同一种给法。默认:个人根定为 `/data/luojiaxuan`(0700)。
   理由:用户点名用 H100,前提已变;没有退回 `/home`、`/tmp`。回滚:release 前 `rm -rf
   /data/luojiaxuan`(rootless 文件需 `podman unshare rm -rf`)。
2. **家目录属主错误的绕行**。家目录里的 `.config`、`.cache`、`.local` 属于旧 uid 1021,rootless
   podman 拒绝启动、拉镜像写签名也失败。默认:helper 对这台的每条远程命令导出
   `HOME=/data/luojiaxuan/home` 与三个 XDG 目录。副作用:map 文件在 `/data/luojiaxuan/home/`。
   根治需管理员修正家目录属主。回滚:删掉 `nodes.json` 里的 `remote_env`。
3. **fuse-overlayfs 不可用**。系统片段 `/etc/containers/storage.conf.d/00-podman-static.conf`
   强制 fuse-overlayfs,但登录会话打不开 `/dev/fuse`(`/dev/net/tun` 同样),容器启动失败。
   默认:用户片段 `storage.conf.d/99-native-overlay.conf` 置空 `mount_program`,内核 5.15 +
   ext4 原生 rootless overlay 可用;按新配置重置存储并重拉镜像(128 s)。回滚:删该片段。
4. **测量设置**。`language` 不传(`auto`);并发 8(只影响批内数值噪声,不影响起声分布的统计);
   不开 deterministic inference。
5. **修复扫描换到 hyper00**。eval-h100 上出现一个本账号的租约(18:29 PT 创建,非本 session),
   同机只能持一个租约且不能动别的任务的租约,改用 hyper00 4×H200。扫描自带 N=0 对照,
   比较都在同一台机器上;hyper00 的 N=0 网格(中位数 445 ms)与 H100 上的复现(450 ms)一致。
   hyper00 当天被重新部署(`/data01` 消失,新 `/data` 14 TB),个人根改为 `/data/luojiaxuan`,
   HOME/XDG 重定向同 eval-h100,GPU 按 UUID 对应宿主机编号。回滚:`nodes.json` 与 helper 的改动
   各自独立。
6. **S 由 v1 换成 v2**。依据:覆盖率-电平前沿与种子稳定性(见上)。回滚:把
   `SILENCE_PROBE_*` 常量改回数字静音 + -80/-60 dBFS 白噪声即可。
7. **租约续 1 小时**(19:17 PT,扣 4 积分)用于 token 探针与 1221 实验;19:50 PT 前释放。
8. **侦察用的补丁只在节点副本里**(数据推导 S、token 日志、1221),不进提交的代码。
9. **默认值暂留 0**。有分歧(见"待定"),等 luojiaxuan 决定;改默认只需改 factory 参数。

## 外审处理(GPT-6 Pro,2026-09-23 18:28–18:41 PT)

- **采纳**:双向量化 S 的覆盖率(因此发现 v1 只覆盖 45%,并最终找到 1221);逐行按模式门控
  (外审看的是改动前的 runner,改动后已按行门控);写明屏蔽在 top-k 之前;确认
  `--no-ref-text` 走 x-vector(`resolve_x_vector_only_mode` 在无参考文本时返回 True,且屏蔽后
  生成帧数下降、起声改变);修正复现里"与参考无关""双峰""ICL 因果"的过强措辞。
- **暂缓**:盲听、首音素错误率、长句、非英语、流式、置信区间、软惩罚与"检测到语音即解除"
  的对照(见"尚未做")。
- **分歧**:外审主张默认关闭;我倾向方案 B 默认开启(仓库规则:正确性允许时性能优化默认开)。
  实测 1.7B 的质量差在噪声边缘,但主观评测未做,交给 luojiaxuan 定。

## 修复扫描的运行记录

| 项 | 值 |
|---|---|
| 主机 | hyper00(租约 `01M38GHZNZ0CDHN601PC03VYG7`,4×H200,宿主机 GPU 2–5,18:30–19:50 PT,续租 1 次) |
| 代码 | 本分支,节点目录 `/data/luojiaxuan/runs/20260924T013215Z/` |
| 产物 | HF `gavinlaw/sglang-omni-qwen3-tts-base-onset-en-audio` 下 `sweep-20260924T013215Z/`,commit `ee2662c5592a95d00f48b772bc7d40f78c961bd8`,5 个 tar 分片、17 个任务目录(含 WAV、WER/SIM 逐条结果、探针 token),已按大小对账;节点副本已删 |
| 收尾 | 容器已删、map 清零、租约已释放;节点保留镜像(39 GB)与 venv(1.4 GB) |
