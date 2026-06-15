# AGENTS.md

This repository is developed on shared GPU machines. Keep the rules below in
mind when working here.

## Development Rules

- Commit promptly in small, focused increments. Do not accumulate a large pile
  of unrelated changes before committing.
- Keep code, checkpoints, logs, datasets, and experiment artifacts under your
  personal persistent data directory, not inside the container filesystem.
- Treat containers as disposable. They may be removed when machine resources are
  tight, so anything important must live in the mounted `/data` directory.
- Always mount the shared HuggingFace cache:

  ```bash
  -v /data/cache/huggingface:/root/.cache/huggingface
  ```

- Use your own data directory:

  ```bash
  -v /data/<your-name>:/data
  ```

  Docker can create `/data/<your-name>` on the host when the container starts,
  even if your shell user cannot create directories directly under `/data`.
- With this mount, the host directory `/data/<your-name>` appears as `/data`
  inside the container. Inside the container, put both code and data under
  `/data`, for example `/data/sglang-omni` and `/data/runs`.
- Put virtual environments under `/data` as well, for example `/data/.venv`.
  This keeps the Python environment reusable if the container is stopped or
  re-entered.
- Do not rely on files left only in `/root`, `/workspace`, `/tmp`, or the
  container image layer.
- If a machine instead mounts the whole host `/data` directory into the
  container as `/data`, then keep your files under `/data/<your-name>`.
- Use a unique container name, usually including your name.

## SGLang Omni Docker

```bash
docker pull hongccc/sglang-omni:dev

docker run -it \
  --shm-size 32g \
  --gpus all \
  -v /data/cache/huggingface:/root/.cache/huggingface \
  -v /data/<your-name>:/data \
  --ipc=host \
  --privileged \
  --name sglang-omni-<your-name> \
  hongccc/sglang-omni:dev \
  /bin/zsh
```

B200 machines may use dedicated data/cache mounts if that is the local machine
convention:

```bash
docker run -it \
  --shm-size 32g \
  --gpus all \
  -v /data01/cache/huggingface:/root/.cache/huggingface \
  -v /data02/<your-name>:/data \
  --ipc=host \
  --privileged \
  --name sglang-omni-<your-name> \
  hongccc/sglang-omni:dev \
  /bin/zsh
```

## Miles Docker

Specify `nofile` to avoid crashes from too many open files.

```bash
docker pull radixark/miles:latest

docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/cache/huggingface:/root/.cache/huggingface \
  -v /data/<your-name>:/data \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --name sglang-miles-<your-name> \
  radixark/miles:latest \
  /bin/zsh
```

## Slime Docker

```bash
docker pull slimerl/slime:latest

docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /data/cache/huggingface:/root/.cache/huggingface \
  -v /data/<your-name>:/data \
  --ipc=host \
  --ulimit nofile=65536:65536 \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  --name sglang-rl-<your-name> \
  slimerl/slime:latest \
  /bin/zsh
```

## Re-enter An Existing Container

```bash
docker start -i <your-docker-name>
```

## Checklist

- Shared HF cache is mounted.
- Personal `/data/<your-name>` directory is mounted.
- Code, virtual environments, and experiment outputs are stored under the
  mounted `/data` path.
- Container name is unique.
- Long-running training or data-heavy workloads use the needed `--ulimit`
  settings.
