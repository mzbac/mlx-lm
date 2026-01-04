# Distributed HTTP Model Server

`mlx_lm/examples/distributed_server.py` runs an OpenAI-compatible HTTP server
across multiple MLX processes and/or machines using `mlx.launch`.

It is currently focused on pipeline-parallel inference for MiniMax models.

> [!NOTE]
> The distributed server is not recommended for production. It implements only
> basic security checks and assumes a trusted network.

## Quick Start (Single Machine, 2 Ranks)

```shell
cd <path_to_mlx_lm_repo>

mlx.launch -n 2 --backend ring -- \
  python mlx_lm/examples/distributed_server.py \
  --model <hf_repo_or_local_path> \
  --host 0.0.0.0 --port 8080
```

## Multi-Machine Setup

### 1) Create a `hosts.json`

Create a `hosts.json` file listing your machines:

```json
[
  {"ssh": "localhost", "ips": ["192.168.1.100"]},
  {"ssh": "user@remote-host", "ips": ["192.168.1.101"]}
]
```

- `ssh`: SSH target for the node (passwordless SSH recommended).
- `ips`: IPs used for inter-node communication (must be reachable from peers).

### 2) Ensure a consistent environment on all nodes

Every node must have:

- The same `mlx-lm` code (same git revision).
- A working Python environment that can import `mlx_lm`.
- Access to the model (download from Hugging Face or pre-cached locally).

In practice, the easiest path is to clone `mlx-lm` into the same location on
each node and install it in editable mode:

```shell
git clone <repo_url> ~/mlx-lm
cd ~/mlx-lm
pip install -e .
```

### 3) Choose a MiniMax pipeline split

MiniMax models use pipeline parallelism. Set `MINIMAX_PIPELINE_SPLIT` to a
comma-separated list of layer counts, one per rank, that must sum to the
model's `num_hidden_layers`.

Important details:

- Rank `0` is the last stage.
- Rank `N-1` is the first stage.

Example for a 2-rank run on a 62-layer model:

```shell
MINIMAX_PIPELINE_SPLIT=22,40
```

If rank 1 OOMs, give it fewer layers (increase the first number, decrease the
second). If rank 0 OOMs, do the opposite.

### 4) Launch the distributed server

On the machine you launch from:

```shell
conda activate <your_env>
cd ~/mlx-lm

# Make sure you are not running another server on :8080
lsof -i tcp:8080

nohup mlx.launch --backend ring --hostfile hosts.json \
  --env MLX_METAL_FAST_SYNCH=1 \
  --env MINIMAX_PIPELINE_SPLIT=22,40 \
  --python python -- \
  python ~/mlx-lm/mlx_lm/examples/distributed_server.py \
  --model mlx-community/MiniMax-M2.1-6bit \
  --host 0.0.0.0 --port 8080 \
  --temperature 0.0 --top-p 0.95 --top-k 40 \
  > distributed_server.log 2>&1 &
```

### 5) Test the server

Health check:

```shell
curl http://<host_ip>:8080/health
```

Chat completion:

```shell
curl http://<host_ip>:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Say this is a test!"}],
    "max_tokens": 64
  }'
```

For request/response fields, see `mlx_lm/SERVER.md` (the distributed server is
intended to match `mlx_lm/server.py` where possible).

## OpenCode Integration

To use the distributed server with OpenCode, set the OpenAI-compatible base URL
to your server's `/v1` endpoint.

Edit `~/.config/opencode/opencode.json`:

```shell
vim ~/.config/opencode/opencode.json
```

Example config (replace the `baseURL` host with your local server address):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX (local)",
      "options": {
        "baseURL": "http://<mlx-server-host>:8080/v1"
      },
      "models": {
        "default_model": {
          "name": "MiniMax M2.1",
          "limit": {
            "context": 60000,
            "output": 16000
          }
        }
      }
    }
  }
}
```

## Tool Calling Notes (MiniMax)

MiniMax generations may include tokenizer-spaced paths inside tool arguments
(for example, `"/ Users/.../ index. html"`). The server normalizes *path-like*
string arguments (strings starting with `/`, `~/`, `./`, or `../`) by removing
spaces around `/` and `.` before executing tools.

## Syncing Code Changes

If you change `mlx_lm/examples/distributed_server.py` (or any server/model code),
sync the updated files to all worker nodes before restarting.

Example (replace with your remote host):

```shell
scp mlx_lm/examples/distributed_server.py user@remote-host:~/mlx-lm/mlx_lm/examples/distributed_server.py
```

## Troubleshooting

- Server never becomes healthy: check `distributed_server.log` for model
  download progress or import errors on the remote node(s).
- Metal OOM during long prompts: reduce prompt length, reduce `max_tokens`, or
  adjust `MINIMAX_PIPELINE_SPLIT` to shift layers away from the OOMing rank.
- Port already in use: stop the old server (or whatever is bound to `:8080`)
  before launching a new instance.
