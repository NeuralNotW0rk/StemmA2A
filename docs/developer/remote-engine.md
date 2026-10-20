# Remote GPU engine

The desktop app and its host backend can run on a laptop while generation runs on a remote GPU
machine. The remote side is [engine_service.py](../../backend/engine_service.py). It is a Flask
server on port 5001 that wraps a `LocalEngine`, and it is normally run in Docker behind a Cloudflare
tunnel.

---

## Build the image

You can build the image on any machine, including one without a GPU:

```powershell
npm.cmd run docker:build-gpu
```

This runs [build_and_export_image.ps1](../../backend/build_and_export_image.ps1). The script builds
`execution-engine:latest` for `linux/amd64` and saves it to `execution-engine.tar`. The image is based
on `nvidia/cuda:12.6.0-devel-ubuntu22.04` with Python 3.10 and torch 2.7.1, and it builds flash-attn
for SM 7.5–9.0.

---

## Deploy

```bash
scp execution-engine.tar user@gpu-host:/path/to/
ssh user@gpu-host
docker load -i /path/to/execution-engine.tar
cd StemmA2A && docker compose up -d
```

[compose.yaml](../../compose.yaml) defines two services:

- **`execution-engine`** has these mounts:
  - `./backend` → `/app` (the code is mounted, not baked into the image)
  - `${LOCAL_DATA_PATH}` → `${CONTAINER_DATA_PATH}`
  - `../diffracture`
  - `../neutral-selection`

  It needs one NVIDIA GPU and 8 GB of `shm`, and it sets `HF_HUB_OFFLINE=1`. At startup,
  [entrypoint.sh](../../backend/entrypoint.sh) installs the sibling libraries in editable mode and runs
  `diagnose_gpu.py`.
- **`tunnel`** runs `cloudflared` with `CF_TUNNEL_TOKEN`.

> Compose doesn't publish any ports, so the engine can only be reached through the tunnel.
> Because `HF_HUB_OFFLINE=1`, the CLAP, CLIP and T5 weights must already be in
> `<data>/huggingface`.

---

## Configure the client

In the client's `.env`, set:

```ini
ENGINE_URL=https://<your-tunnel-hostname>
CF_ACCESS_CLIENT_ID=...
CF_ACCESS_CLIENT_SECRET=...
```

---

## How assets move

1. **Request.** `RemoteEngine.execute` de-anchors the parameters, keeping only asset UIDs, and POSTs
   them to `/execute`.
2. **Upload what's missing.** If the server lacks any assets, it replies
   `422 {"missing_uids": [...]}`. The client uploads those files in 4 MiB chunks through `/upload`,
   with 2 uploads at a time and 3 attempts per file. Directories are zipped first. The client then
   retries the request, for at most 3 rounds. The chunk size keeps each request under Cloudflare's
   roughly 100 s limit.
3. **Validate.** The server deletes cached files that are empty or the wrong size, so they get
   uploaded again.
4. **Download results.** When a job finishes, the client downloads each result from
   `/download_asset/<uid>` into `backend/tmp/cache/`. The host then ingests the results into the
   project.

Shared models on the server come from its own `shared_models.json`. The client lists them with
`GET /shared_models` and matches them by checkpoint or encoder UID.

---

## Engine service API (port 5001)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `GET /operations` | Adapter operations |
| `GET /shared_models` | Server-side shared models |
| `POST /execute` | Validates assets (422 with `missing_uids` if any are missing), then queues the job |
| `GET /job_status/<id>` | Job status and results |
| `POST /jobs/<id>/cancel` | Cancels a job |
| `GET /download_asset/<uid>` | Serves a CAS file |
| `POST /upload` | Chunked, atomic CAS upload |
