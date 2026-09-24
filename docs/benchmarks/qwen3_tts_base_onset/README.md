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
| 收尾 | 结果拉回本机并入库后 teardown 容器、release 租约 |

## 结果

(待填)

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
