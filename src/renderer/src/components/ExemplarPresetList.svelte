<!-- src/renderer/src/components/ExemplarPresetList.svelte -->
<script lang="ts">
  import type { ExemplarPreset } from '../utils/types'

  let {
    presets,
    targetIds,
    selectedKeys = $bindable([]),
    disabled = false,
    onAddNew
  } = $props<{
    presets: ExemplarPreset[]
    // Individuals the selected presets will be generated for
    targetIds: string[]
    selectedKeys: string[]
    disabled?: boolean
    onAddNew: () => void
  }>()

  function missingCount(preset: ExemplarPreset): number {
    return targetIds.filter((id: string) => !preset.covered_ids.includes(id)).length
  }

  function toggle(key: string): void {
    selectedKeys = selectedKeys.includes(key)
      ? selectedKeys.filter((k: string) => k !== key)
      : [...selectedKeys, key]
  }

  function formatNumber(value: unknown): string {
    const n = Number(value)
    return Number.isFinite(n) ? String(Math.round(n * 100) / 100) : String(value)
  }

  // Short summary of the settings that most distinguish one exemplar from another
  function details(params: Record<string, unknown>): string[] {
    const parts: string[] = []
    if (params.seed !== undefined) parts.push(`seed ${params.seed}`)
    if (params.seconds_total !== undefined) parts.push(`${formatNumber(params.seconds_total)} s`)
    if (params.steps !== undefined) parts.push(`${params.steps} steps`)
    if (params.cfg_scale !== undefined) parts.push(`cfg ${formatNumber(params.cfg_scale)}`)
    const sampler =
      params.sampler_category === 'rectified_flow' ? params.rf_sampler_type : params.k_sampler_type
    if (sampler) parts.push(String(sampler))
    if (params.init_audio_id) parts.push('init audio')
    if (params.init_latent_id) parts.push('init latent')
    if (Array.isArray(params.gratings) && params.gratings.length > 0) {
      parts.push(`${params.gratings.length} grating${params.gratings.length > 1 ? 's' : ''}`)
    }
    return parts
  }

  function sourceLabel(preset: ExemplarPreset): string {
    const [first, ...rest] = preset.sources
    if (!first) return ''
    const relation =
      first.distance === 0
        ? 'on'
        : first.distance === 1
          ? 'from parent'
          : `from ancestor (${first.distance} gens up)`
    const more = rest.length > 0 ? ` +${rest.length} more` : ''
    return `${first.audio_name} ${relation} ${first.individual_name}${more}`
  }
</script>

<div class="exemplar-preset-list">
  <div class="header-row">
    <h4 class="list-title">Lineage Exemplars</h4>
    <button class="small-btn" onclick={onAddNew} {disabled}>Add New Exemplar…</button>
  </div>

  <div class="presets">
    {#each presets as preset (preset.key)}
      {@const missing = missingCount(preset)}
      {@const prompt = String(preset.params.prompt ?? '').trim()}
      <label class="preset-row" class:done={missing === 0}>
        <input
          type="checkbox"
          checked={selectedKeys.includes(preset.key)}
          disabled={disabled || missing === 0}
          onchange={() => toggle(preset.key)}
        />
        <div class="preset-body">
          <div class="preset-title" class:untitled={!prompt}>{prompt || '(no prompt)'}</div>
          <div class="preset-details">{details(preset.params).join(' · ')}</div>
          <div class="preset-source">{sourceLabel(preset)}</div>
        </div>
        <span class="preset-status">
          {#if missing === 0}
            Already generated
          {:else if missing < targetIds.length}
            {missing} of {targetIds.length} need it
          {:else if targetIds.length > 1}
            {targetIds.length} individuals
          {/if}
        </span>
      </label>
    {/each}
  </div>
</div>

<style>
  .exemplar-preset-list {
    margin-bottom: 1rem;
  }
  .header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }
  .header-row .list-title {
    margin: 0;
    font-weight: 500;
    color: var(--color-overlay-text);
  }
  .small-btn {
    min-width: 0;
    min-height: 0;
    padding: 0.25rem 0.75rem;
    font-size: 0.85rem;
  }
  .presets {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .preset-row {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.6rem 0.75rem;
    border: 1px solid var(--color-border-glass-1);
    border-radius: 0.375rem;
    background: rgba(255, 255, 255, 0.02);
    cursor: pointer;
  }
  .preset-row.done {
    opacity: 0.55;
    cursor: default;
  }
  .preset-row input[type='checkbox'] {
    width: auto;
    margin: 0.2rem 0 0;
    cursor: inherit;
  }
  .preset-body {
    flex-grow: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }
  .preset-title {
    color: var(--color-overlay-text);
    font-size: 0.9rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .preset-title.untitled {
    font-style: italic;
    color: var(--color-text-muted, #aaa);
  }
  .preset-details,
  .preset-source {
    font-size: 0.78rem;
    color: var(--color-text-muted, #aaa);
  }
  .preset-source {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .preset-status {
    flex-shrink: 0;
    font-size: 0.75rem;
    color: var(--color-text-muted, #aaa);
    white-space: nowrap;
  }
</style>
