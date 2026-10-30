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

`ENGINE_URL` is only the default. The engine menu in the toolbar switches between the local engine
and a remote URL without a restart, and the app remembers your choice. A switch is refused while
jobs are running, unless you confirm that their results can be lost. The Cloudflare credentials
still come from `.env`.

While the remote engine is active, a dot on the engine menu shows whether its `/health` answers:
green when it does, red when it doesn't, with the reason in the tooltip. The check runs every 20
seconds and whenever the menu opens. A red dot with "Redirected to a login page" means Cloudflare
Access rejected the credentials.

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

Shared models on the server live in its model store, `<data>/models/` (see
[Architecture](architecture.md#storage-and-ids)). The client lists them with `GET /shared_models`
and matches them by checkpoint or encoder UID. **Share with Engine** sends a registry entry to
`POST /shared_models`. Files are named by UID only. If the server lacks any, it replies
`422 {"missing_uids": [...]}` and the client uploads them through the same chunked `/upload` as for
`/execute`. The server then moves them from the cache into the store.

---

## Cache eviction

The server's asset cache (`<data>/cache/`) is size-capped. Eviction is always recoverable: if an
evicted file is needed again, `/execute` reports it missing and the client uploads it again. The
policy in [cache_policy.py](../../backend/engine/cache_policy.py) only looks at each file's size and
when it was last used, never at what the file holds.

- **Last use** is the file's modified time. `/execute` and `/download_asset` update it on every hit.
- **Eviction order.** Once the cache is over its cap, files are evicted until it is under 80% of
  the cap. The file with the highest `idle_time / (1 + log2(size / 1 MiB))` goes first, so large
  files outlast small ones by a bounded factor. A 5 GiB checkpoint idle for 13 days ranks with a
  1 MiB latent idle for 1 day.
- **Never evicted:**
  - files used within the minimum age
  - inputs of jobs that are still pending or running
- **Shared models aren't in the cache.** Their files live in `<data>/models/`, which eviction never
  scans and the cap doesn't count. Deleting `cache/` is always safe.
- **Copying files in.** Last use comes from modified times, so files copied into `cache/` with
  their original times kept (for example `rsync -a`) look as old as those times. Copy without
  keeping times, or `touch` the files afterwards.
- **Upload staging.** `.part` files older than a day are left over from failed uploads and are
  deleted.

The cap covers `cache/` only, not `huggingface/` or the other folders in the data root. A sweep runs
at startup and then on an interval. These settings are read from the host's `.env` through
[compose.yaml](../../compose.yaml):

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE_CACHE_MAX_GB` | `50` | Cap in GiB. `0` turns eviction off. |
| `ENGINE_CACHE_MIN_AGE_HOURS` | `24` | Files used more recently than this are never evicted. |
| `ENGINE_CACHE_SWEEP_MINUTES` | `15` | Interval between background sweeps. |

> A job result that the client hasn't downloaded within the minimum age can be evicted. Results
> normally download within seconds, because the client polls running jobs.

---

## Engine service API (port 5001)

| Endpoint | Purpose |
|---|---|
| `GET /health` | Health check |
| `GET /operations` | Adapter operations |
| `GET /shared_models` | Server-side shared models |
| `POST /shared_models` | Registers a shared model (422 with `missing_uids` if files are missing) |
| `DELETE /shared_models/<id>` | Removes a shared model; its files return to the cache |
| `POST /execute` | Validates assets (422 with `missing_uids` if any are missing), then queues the job |
| `GET /job_status/<id>` | Job status and results |
| `POST /jobs/<id>/cancel` | Cancels a job |
| `GET /download_asset/<uid>` | Serves a CAS file |
| `POST /upload` | Chunked, atomic CAS upload |
| `GET /cache` | Cache size, and what a sweep would evict now (nothing is deleted) |
| `POST /cache/sweep` | Runs an eviction sweep now |
