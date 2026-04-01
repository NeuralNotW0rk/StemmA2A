<script lang="ts">
  import type { FormConfig, FormField, ModelData, AudioData } from '../utils/forms'
  import NodeSelector from './NodeSelector.svelte'

  let {
    config,
    formData = $bindable(),
    isFormValid = $bindable(),
    contextData = null
  } = $props<{
    config: FormConfig
    formData: Record<string, unknown>
    isFormValid: boolean
    contextData?: Record<string, unknown> | null
  }>()

  let batchFields = $state<Set<string>>(new Set())

  function toggleBatchMode(fieldName: string): void {
    const newBatchFields = new Set(batchFields)
    if (newBatchFields.has(fieldName)) {
      newBatchFields.delete(fieldName)
    } else {
      newBatchFields.add(fieldName)
    }
    batchFields = newBatchFields
  }

  $effect(() => {
    isFormValid = visibleFields.every(
      (f) =>
        !f.validation?.required ||
        (formData[f.name] != null &&
          (typeof formData[f.name] !== 'string' || String(formData[f.name]).trim() !== ''))
    )
  })

  export function getPayload(): Record<string, unknown> {
    const payload: Record<string, unknown> = {}
    for (const field of config) {
      const fieldName = field.name
      const value = formData[fieldName]

      if (value !== undefined && value !== null) {
        if (field.type === 'node' && typeof value === 'object' && value && 'id' in value) {
          payload[fieldName] = value.id
        } else {
          payload[fieldName] = value
        }
      }
    }
    return payload
  }

  export function getBatchFields(): Set<string> {
    return batchFields
  }

  async function selectFieldFile(
    fieldName: string,
    filters?: { name: string; extensions: string[] }[]
  ): Promise<void> {
    const plainFilters = filters ? JSON.parse(JSON.stringify(filters)) : undefined
    const path = await window.api.openFile({ title: 'Select File', filters: plainFilters })
    if (path) {
      formData[fieldName] = path
    }
  }
  const visibleFields = $derived(
    config.filter((field: FormField) => {
      if (!field.show_if) {
        return true
      }

      // Combine context and form data, with form data taking precedence.
      const data = { ...contextData, ...formData }

      for (const key in field.show_if) {
        const condition = field.show_if[key]
        if (condition === 'exists') {
          if (!data[key]) {
            return false
          }
        } else {
          if (data[key] !== condition) {
            return false
          }
        }
      }

      return true
    })
  )
</script>

<form
  onsubmit={(e: Event) => {
    e.preventDefault()
  }}
>
  {#each visibleFields as field (field.name)}
    {@const isBatch = batchFields.has(field.name)}
    <div class="form-field">
      {#if field.type === 'textarea'}
        <label for={field.name}>{field.label}</label>
        <textarea bind:value={formData[field.name]} id={field.name} placeholder={field.placeholder}
        ></textarea>
      {:else if field.type === 'integer'}
        <div class="label-container">
          <label for={field.name}>{field.label}</label>
          <button
            class="batch-toggle"
            onclick={() => toggleBatchMode(field.name)}
            title="Toggle sequence mode"
          >
            {isBatch ? '−' : '+'}
          </button>
        </div>
        {#if isBatch}
          <input
            type="text"
            bind:value={formData[field.name]}
            id={field.name}
            placeholder="e.g. 1, 2, 5-10:2"
          />
        {:else}
          <input
            type="number"
            step="1"
            bind:value={formData[field.name]}
            onchange={(e) => {
              const value = e.currentTarget.value
              if (value) {
                formData[field.name] = Math.round(Number(value))
              }
            }}
            id={field.name}
            placeholder={field.placeholder}
          />
        {/if}
      {:else if field.type === 'float'}
        <div class="label-container">
          <label for={field.name}>{field.label}</label>
          <button
            class="batch-toggle"
            onclick={() => toggleBatchMode(field.name)}
            title="Toggle sequence mode"
          >
            {isBatch ? '−' : '+'}
          </button>
        </div>
        {#if isBatch}
          <input
            type="text"
            bind:value={formData[field.name]}
            id={field.name}
            placeholder="e.g. 0.5, 1.2, 2.0-3.0:0.5"
          />
        {:else}
          <input
            type="number"
            step="any"
            bind:value={formData[field.name]}
            id={field.name}
            placeholder={field.placeholder}
          />
        {/if}
      {:else if field.type === 'boolean'}
        <label for={field.name}>{field.label}</label>
        <input
          type="checkbox"
          checked={formData[field.name] === 'true' || formData[field.name] === true}
          onchange={(e: Event & { currentTarget: EventTarget & HTMLInputElement }) =>
            (formData[field.name] = e.currentTarget.checked)}
          id={field.name}
        />
      {:else if field.type === 'file'}
        <label for={field.name}>{field.label}</label>
        <div class="path-input">
          <input type="text" bind:value={formData[field.name]} placeholder={field.placeholder} />
          <button onclick={() => selectFieldFile(field.name, field.filters)}>Browse</button>
        </div>
      {:else if field.type === 'select'}
        <div class="label-container">
          <label for={field.name}>{field.label}</label>
          <button
            class="batch-toggle"
            onclick={() => toggleBatchMode(field.name)}
            title="Toggle sequence mode"
          >
            {isBatch ? '−' : '+'}
          </button>
        </div>
        {#if isBatch}
          <input
            type="text"
            bind:value={formData[field.name]}
            id={field.name}
            placeholder="e.g. value1, value2, value3"
          />
        {:else}
          <select bind:value={formData[field.name]} id={field.name}>
            {#each field.options as option (option.value)}
              <option value={option.value}>{option.label}</option>
            {/each}
          </select>
        {/if}
      {:else if field.type === 'node'}
        <NodeSelector
          label={field.label}
          filter={field.filter}
          bind:node={formData[field.name] as ModelData | AudioData}
          id={field.name}
        />
      {:else}
        <div class="label-container">
          <label for={field.name}>{field.label}</label>
          <button
            class="batch-toggle"
            onclick={() => toggleBatchMode(field.name)}
            title="Toggle sequence mode"
          >
            {isBatch ? '−' : '+'}
          </button>
        </div>
        {#if isBatch}
          <input
            type="text"
            bind:value={formData[field.name]}
            id={field.name}
            placeholder="e.g. value1, value2, value3"
          />
        {:else}
          <!-- Default to text input for 'string' and others -->
          <input
            type="text"
            bind:value={formData[field.name]}
            id={field.name}
            placeholder={field.placeholder}
          />
        {/if}
      {/if}
    </div>
  {/each}
</form>

<style>
  .form-field {
    margin-bottom: 1rem;
  }

  .form-field:last-child {
    margin-bottom: 0;
  }

  .label-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
  }

  .batch-toggle {
    background: none;
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    cursor: pointer;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    font-size: 16px;
    line-height: 20px;
    padding: 0;
  }

  label {
    display: block;
    font-weight: 500;
  }

  input[type='text'],
  input[type='number'],
  textarea,
  select {
    width: 100%;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
    box-sizing: border-box;
  }

  select option {
    background: var(--color-background-mute);
    color: var(--color-overlay-text);
  }

  .path-input {
    display: flex;
    gap: 0.5rem;
  }

  .path-input input {
    flex: 1;
    background: var(--color-border-glass-1);
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    padding: 0.5rem;
    border-radius: 0.375rem;
  }

  .path-input button {
    background: var(--color-primary-t-30);
    border: 1px solid var(--color-primary-t-50);
    color: var(--color-overlay-text);
    padding: 0.5rem 1rem;
    border-radius: 0.375rem;
    cursor: pointer;
  }
</style>
