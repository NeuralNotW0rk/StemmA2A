<!-- src/renderer/src/components/views/ElementInfoView.svelte -->
<script lang="ts">
  import { ELEMENT_INFO_CONFIG } from '../../utils/app-config'
  import type { ElementData } from '../../utils/types'

  interface Props {
    selectedElementData: ElementData | null
  }

  let { selectedElementData }: Props = $props()
  let exportMessage = $state<string>('')
  let isExporting = $state<boolean>(false)

  interface GraphElementNode {
    id: string
    type: string
    [key: string]: unknown
  }

  let element = $derived(selectedElementData as GraphElementNode | null)

  let displayEntries = $derived(
    selectedElementData
      ? Object.entries(selectedElementData)
          .filter(([key]) => !(ELEMENT_INFO_CONFIG.ignoredKeys as Set<string>).has(key))
          .sort(([keyA], [keyB]) => {
            const indexA = (ELEMENT_INFO_CONFIG.priorityKeys as readonly string[]).indexOf(keyA)
            const indexB = (ELEMENT_INFO_CONFIG.priorityKeys as readonly string[]).indexOf(keyB)

            // If both are priority keys, preserve their order defined in the array
            if (indexA !== -1 && indexB !== -1) return indexA - indexB
            if (indexA !== -1) return -1 // Only A is a priority key
            if (indexB !== -1) return 1 // Only B is a priority key

            // Neither are priority keys, sort alphabetically
            return keyA.localeCompare(keyB)
          })
      : []
  )

  async function handleExport(): Promise<void> {
    if (!element) return
    isExporting = true
    exportMessage = ''
    try {
      const response = await window.api.exportSharedModel(element.id) as { message: string; success: boolean }
      exportMessage = response.message
    } catch (err: unknown) {
      exportMessage = `Export failed: ${err instanceof Error ? err.message : String(err)}`
    } finally {
      isExporting = false
    }
  }
</script>

<div class="view-content">
  {#if element}
    <div class="header-row">
      <h3 class="title">Element Details</h3>
    </div>

    <ul>
      {#each displayEntries as [key, value] (key)}
        <li>
          <strong class="key">{key}</strong>
          <span class="value">{JSON.stringify(value, null, 2)}</span>
        </li>
      {/each}
    </ul>

    {#if element.type === 'model'}
      <div class="footer-actions">
        <button class="export-btn" onclick={handleExport} disabled={isExporting} type="button">
          {isExporting ? 'Exporting...' : 'Export Shared Model Config'}
        </button>
        {#if exportMessage}
          <div class="message-banner" class:error={exportMessage.includes('failed')}>
            {exportMessage}
          </div>
        {/if}
      </div>
    {/if}
  {:else}
    <div class="content-empty">
      <p>Select an element on the graph to see its details.</p>
    </div>
  {/if}
</div>

<style>
  .view-content {
    padding: 1rem;
    height: 100%;
    overflow-y: auto;
  }
  .content-empty {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
    text-align: center;
    color: var(--color-text-overlay-secondary);
  }
  .header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }
  .title {
    margin: 0;
    font-size: 1.1rem;
    color: var(--color-text-overlay);
  }
  .footer-actions {
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid var(--color-border-overlay-subtle, rgba(255, 255, 255, 0.1));
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .export-btn {
    background-color: var(--color-action-primary, #6200ee);
    color: white;
    border: none;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 600;
    transition: background-color 0.2s;
    align-self: flex-start;
  }
  .export-btn:hover:not(:disabled) {
    background-color: var(--color-action-primary-hover, #7722ff);
  }
  .export-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }
  .message-banner {
    background-color: var(--color-background-glass-3, rgba(255, 255, 255, 0.1));
    color: var(--color-text-success, #00c853);
    padding: 0.5rem;
    border-radius: 4px;
    font-size: 0.8rem;
    border: 1px solid var(--color-border-success, rgba(0, 200, 83, 0.3));
  }
  .message-banner.error {
    color: var(--color-text-error, #ff1744);
    border-color: var(--color-border-error, rgba(255, 23, 68, 0.3));
  }

  ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }

  li {
    margin-bottom: 0.75rem;
    font-family: 'Courier New', Courier, monospace;
    font-size: 0.8rem;
    word-wrap: break-word;
    display: flex;
    flex-direction: column;
  }

  .key {
    color: var(--color-text-overlay-secondary);
    font-weight: bold;
    margin-bottom: 0.25rem;
  }

  .value {
    color: var(--color-text-overlay-tertiary);
    white-space: pre-wrap;
    background-color: var(--color-background-glass-2);
    padding: 0.25rem 0.5rem;
    border-radius: 0.25rem;
  }
</style>
