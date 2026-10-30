<script lang="ts">
  import type { CleanupPreview, PanelMessage } from '../../utils/types'

  interface Props {
    onerror: (error: PanelMessage) => void
  }

  let { onerror }: Props = $props()

  // Long lists are cut off in the panel; the move still covers every previewed file
  const MAX_LISTED_FILES = 300

  let preview = $state<CleanupPreview | null>(null)
  let isBusy = $state(false)
  let status = $state<string | null>(null)
  // The batch whose "Delete permanently" was clicked once and is waiting for confirmation
  let confirmPurgeId = $state<string | null>(null)

  let orphansByDir = $derived.by((): [string, { count: number; bytes: number }][] => {
    const totals: Record<string, { count: number; bytes: number }> = {}
    for (const orphan of preview?.orphans ?? []) {
      const dir = orphan.path.split('/')[0]
      totals[dir] ??= { count: 0, bytes: 0 }
      totals[dir].count += 1
      totals[dir].bytes += orphan.size
    }
    return Object.entries(totals)
  })

  function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`
    const units = ['KB', 'MB', 'GB', 'TB']
    let value = bytes / 1024
    let unit = 0
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024
      unit += 1
    }
    return `${value.toFixed(value < 10 ? 2 : 1)} ${units[unit]}`
  }

  function formatBatchDate(created: number | null, id: string): string {
    return created ? new Date(created * 1000).toLocaleString() : id
  }

  function reportError(title: string, e: unknown): void {
    console.error(`${title}:`, e)
    onerror({ title, message: e instanceof Error ? e.message : String(e) })
  }

  async function scan(): Promise<void> {
    isBusy = true
    try {
      preview = await window.api.getCleanupPreview()
    } catch (e: unknown) {
      reportError('Scanning Project Files Failed', e)
    } finally {
      isBusy = false
    }
  }

  async function moveToTrash(): Promise<void> {
    if (!preview || preview.orphans.length === 0) return
    isBusy = true
    try {
      const result = await window.api.quarantineFiles(preview.orphans.map((o) => o.path))
      status = result.batch
        ? `Moved ${result.batch.file_count} file(s) (${formatBytes(result.batch.bytes)}) to the project trash.`
        : 'No files were moved.'
      if (result.skipped > 0) {
        status += ` ${result.skipped} file(s) were skipped because they are referenced again or no longer exist.`
      }
    } catch (e: unknown) {
      reportError('Moving Files to Trash Failed', e)
    } finally {
      isBusy = false
    }
    await scan()
  }

  async function restoreBatch(batchId: string): Promise<void> {
    isBusy = true
    confirmPurgeId = null
    try {
      const result = await window.api.restoreTrash(batchId)
      status = `Restored ${result.restored} file(s).`
      if (result.conflicts.length > 0) {
        status += ` ${result.conflicts.length} file(s) stayed in the trash because a file now exists at their original location.`
      }
    } catch (e: unknown) {
      reportError('Restoring Files Failed', e)
    } finally {
      isBusy = false
    }
    await scan()
  }

  async function purgeBatch(batchId: string): Promise<void> {
    if (confirmPurgeId !== batchId) {
      confirmPurgeId = batchId
      return
    }
    confirmPurgeId = null
    isBusy = true
    try {
      const result = await window.api.purgeTrash(batchId)
      status = `Permanently deleted ${result.file_count} file(s), freeing ${formatBytes(result.bytes)}.`
    } catch (e: unknown) {
      reportError('Deleting Trash Failed', e)
    } finally {
      isBusy = false
    }
    await scan()
  }

  $effect(() => {
    void scan()
  })
</script>

<div class="view-content">
  {#if preview === null}
    <div class="loading">
      {#if isBusy}
        <div class="spinner"></div>
        <p>Scanning project files...</p>
      {/if}
    </div>
  {:else}
    {#if status}
      <p class="status">{status}</p>
    {/if}

    <section>
      <h3>Unreferenced Files</h3>
      {#if preview.orphans.length === 0}
        <p class="hint-text">Every file in generate/, process/ and cache/ is used by the graph.</p>
      {:else}
        <p>
          {preview.orphans.length} file(s), {formatBytes(preview.orphan_bytes)}, are no longer used
          by any element in this project or its backup graph.
        </p>
        <ul class="dir-totals">
          {#each orphansByDir as [dir, totals] (dir)}
            <li>{dir}/: {totals.count} file(s), {formatBytes(totals.bytes)}</li>
          {/each}
        </ul>
        <details>
          <summary>Show files</summary>
          <ul class="file-list">
            {#each preview.orphans.slice(0, MAX_LISTED_FILES) as orphan (orphan.path)}
              <li title={orphan.path}>
                <span class="file-path">{orphan.path}</span>
                <span class="file-size">{formatBytes(orphan.size)}</span>
              </li>
            {/each}
          </ul>
          {#if preview.orphans.length > MAX_LISTED_FILES}
            <p class="hint-text">...and {preview.orphans.length - MAX_LISTED_FILES} more.</p>
          {/if}
        </details>
        {#if preview.unfinished_jobs > 0}
          <p class="warning">
            {preview.unfinished_jobs} job(s) are still running. Wait for them to finish, then rescan.
          </p>
        {/if}
        <p class="hint-text">
          Files are moved to the project's .trash folder and can be restored until you delete them.
        </p>
        <div class="button-group">
          <button
            class="primary"
            onclick={moveToTrash}
            disabled={isBusy || preview.unfinished_jobs > 0}
          >
            Move to Trash
          </button>
          <button onclick={scan} disabled={isBusy}>Rescan</button>
        </div>
      {/if}
    </section>

    <section>
      <h3>Trash</h3>
      {#if preview.trash.length === 0}
        <p class="hint-text">The trash is empty.</p>
      {:else}
        <ul class="batch-list">
          {#each preview.trash as batch (batch.id)}
            <li>
              <div class="batch-info">
                <strong>{formatBatchDate(batch.created, batch.id)}</strong>
                <span>{batch.file_count} file(s), {formatBytes(batch.bytes)}</span>
              </div>
              <div class="button-group">
                <button onclick={() => restoreBatch(batch.id)} disabled={isBusy}>Restore</button>
                <button class="danger" onclick={() => purgeBatch(batch.id)} disabled={isBusy}>
                  {confirmPurgeId === batch.id ? 'Confirm: delete forever' : 'Delete permanently'}
                </button>
              </div>
            </li>
          {/each}
        </ul>
      {/if}
    </section>
  {/if}
</div>

<style>
  .view-content {
    padding: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }

  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
  }

  section {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  h3 {
    margin: 0;
    font-size: 1rem;
  }

  p {
    margin: 0;
  }

  .status {
    padding: 0.5rem;
    background-color: var(--color-background-glass-2);
    border-radius: 0.25rem;
  }

  .warning {
    color: var(--color-accent-danger);
    font-size: 0.85rem;
  }

  .hint-text {
    font-size: 0.85rem;
    color: var(--color-text-dimmed, #aaa);
    line-height: 1.3;
  }

  ul {
    margin: 0;
    padding: 0;
    list-style: none;
  }

  .dir-totals {
    font-size: 0.85rem;
  }

  details summary {
    cursor: pointer;
    font-size: 0.85rem;
  }

  .file-list {
    max-height: 240px;
    overflow-y: auto;
    margin-top: 0.5rem;
    font-size: 0.8rem;
  }

  .file-list li {
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
  }

  .file-path {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .file-size {
    flex-shrink: 0;
    color: var(--color-text-dimmed, #aaa);
  }

  .batch-list li {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    padding: 0.5rem;
    background-color: var(--color-background-glass-2);
    border-radius: 0.25rem;
  }

  .batch-list li + li {
    margin-top: 0.5rem;
  }

  .batch-info {
    display: flex;
    flex-direction: column;
    font-size: 0.85rem;
  }

  .button-group {
    display: flex;
    gap: 0.5rem;
  }

  button {
    padding: 0.5rem 1rem;
    border: none;
    border-radius: 0.25rem;
    cursor: pointer;
  }

  button.danger {
    background-color: var(--color-accent-danger);
    color: var(--color-text-button);
  }
</style>
