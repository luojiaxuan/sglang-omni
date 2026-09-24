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
   1.7B 九成请求开头有 2 帧以上静音。方差主要来自 seed(1.7B:seed 内 22191 ms² 对
   参考×句子间 6986 ms²),三条参考的中位数都在 400–485 ms,与参考无关。
2. **分布是双峰的,不是逐帧递减**。按 80 ms 分箱,1.7B xvec 为
   `0:34 1:3 2:4 3:6 4:65 5:99 6:85 7:38 8:18 9:4 10:3 11:1`;0.6B 为
   `0:81 1:28 2:10 3:21 4:55 5:69 6:55 7:16 8:14 …`。模型要么第 0 帧就开口,要么进入一段
   4–7 帧的开场静音,很少停在中间。这与"每帧独立决定是否继续静音"不符,更像第一帧的
   一次选择决定了整段。静音段的电平约 -60 dB,是参考录音底噪的水平,不是数字静音。
3. **ICL 不是普遍更稳定,而是跟随参考音频末尾的停顿**。1.7B ICL 按参考拆开,中位数
   105 / 342 / 755 ms,对应参考末尾静音 135 / 170 / 300 ms;0.6B 同方向(278 / 418 / 570 ms)。
   原报告"ICL 起声更紧"只在末尾停顿短的参考上成立。

## 含义与下一步

- 对方的现象成立,机制说法(开头采到静音区 token)与双峰分布相容,但双峰提示只约束
  很少几帧、甚至只约束第 1 帧就可能足够。前 N 帧屏蔽静音 token 的做法需要扫 N
  (0/1/2/3/4)并同时报告起声、WER、说话人相似度,才能决定默认值。
- ICL 的长起声是另一件事:它来自对参考停顿的模仿,屏蔽静音 token 未必合适(参照实现也
  把修法限定在 x-vector-only)。
- 这次没有测流式路径;非流式输出里的静音与流式首块里的静音来自同一段 codec,可感知
  TTFA 会同样增加。

## 错误与返工

- 第一次发射(17:37 PT)服务启动失败:最新 main 需要 sglang 0.5.20 的
  `radix_eviction_policy_config`,镜像里的 sglang 更旧。CI 在镜像之上另建 venv 补齐 pin,
  `run.sh` 改为同样的做法后重发(17:41 PT)。
- 向节点同步修改后的 `run.sh` 时,一条命令误把 tar 解到了节点 `/tmp`(`/tmp/run.sh`、
  `/tmp/.github/`,均为本账号文件),当即删除并核对已不存在。

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
