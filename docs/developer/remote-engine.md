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
   with 2 uploads at a time and 3 attempts per asset. The client then retries the request, for at
   most 3 rounds. The chunk size keeps each request under Cloudflare's roughly 100 s limit.
   - **Directories** (such as encoders) are never archived. Each file is sent with its path inside
     the directory, and a retry resumes from the file that failed. `POST /upload/finalize` then
     checks the staged directory's total size and `from_directory` UID, and stores it as a
     directory at `cache/xx/<uid>`. A directory that doesn't match its UID is discarded and not
     retried.
3. **Validate.** The server deletes cached entries that are empty or the wrong size, so they get
   uploaded again. A directory's size is the total of its files, the same size the client records.
4. **Download results.** When a job finishes, the client downloads each result from
   `/download_asset/<uid>` into `backend/tmp/cache/`. The host then ingests the results into the
   project.

Shared models on the server live in its model store, `<data>/models/` (see
[Architecture](architecture.md#storage-and-ids)). The client lists them with `GET /shared_models`
and matches them by checkpoint or encoder UID. **Share with Engine** sends a registry entry to
`POST /shared_models`. Files are named by UID only. If the server lacks any, it replies
`422 {"missing_uids": [...]}` and the client uploads them through the same chunked `/upload` as for
`/execute`. The server then moves them from the cache into the store.

### Copying files onto the server instead of uploading them

When a file is already on the server, or copying it there is faster than uploading, there are two
ways to hand it to the engine service.

**Inbox.** Copy the file or directory into `<data>/inbox/` (on the host, `${LOCAL_DATA_PATH}/inbox/`)
and name it exactly by its UID, such as `f28c7dd48e02c8a6.xxh3_64`. The client's upload log and the
engine's "Requesting missing/corrupt assets" line both show the UIDs, and hovering over a transfer
in the app shows its UID too. The engine scans the inbox every 10 seconds. An entry that is unchanged
between two scans is moved into `cache/xx/<uid>` and marked as just used. See
[inbox.py](../../backend/engine/inbox.py).
- **Copying.** Copy under `<uid>.part` and rename it when the copy finishes. The inbox ignores
  `.part` names, and some copy tools set the final file size before they've written everything.
- **Checks.** A directory must hash to its name, or it stays in the inbox and the engine logs why. A
  file's UID comes from its model adapter, so the inbox can't check it. A wrong or incomplete file
  fails the size check in `/execute` and is uploaded instead.
- **Uploads already running.** An upload of the same UID stops at its next chunk, because `/upload`
  replies `already_present` once the engine has it.

The inbox only fills the cache, so the files can still be evicted. To keep a model permanently, share
it afterwards with **Share with Engine**; its files are then moved from the cache into the store.

**Ingest script.** [scripts/ingest_model.py](../../backend/scripts/ingest_model.py) computes the
UIDs itself with the model's adapter, copies the files into the store and adds the registry entry, so
no renaming is needed. It needs the backend's Python environment. Run it in the container with paths
as the container sees them, or on the host with `LOCAL_DATA_PATH` set to the host data folder. Run it
with `--dry-run` first and check that the UIDs match the model in your project. Restart the engine
service for the model to appear in its shared models list.

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
- **Directories** count as one entry, sized by the total of their files, and are evicted whole.
- **Upload staging.** `.part` files and directories with nothing written for a day are left over
  from failed uploads and are deleted.

The cap covers `cache/` only, not `huggingface/` or the other folders in the data root. A sweep runs
at startup and then on an interval. These settings are read from the host's `.env` through
[compose.yaml](../../compose.yaml):

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE_CACHE_MAX_GB` | `50` | Cap in GiB. `0` turns eviction off. |
| `ENGINE_CACHE_MIN_AGE_HOURS` | `24` | Files used more recently than this are never evicted. |
| `ENGINE_CACHE_SWEEP_MINUTES` | `15` | Interval between background sweeps. |
| `ENGINE_INBOX_SCAN_SECONDS` | `10` | Interval between scans of the inbox. |

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
