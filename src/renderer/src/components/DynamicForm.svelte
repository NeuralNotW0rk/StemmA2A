<script lang="ts">
  import type { FormConfig, FormField, ModelData, AudioData } from '../utils/forms'
  import NodeSelector from './NodeSelector.svelte'
  import { SvelteSet } from 'svelte/reactivity'

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

  const batchFields = new SvelteSet<string>()

  function toggleBatchMode(fieldName: string): void {
    if (batchFields.has(fieldName)) {
      batchFields.delete(fieldName)
    } else {
      batchFields.add(fieldName)
    }
  }

  function randomizeField(fieldName: string, type: string): void {
    if (type === 'float') {
      formData[fieldName] = Number(Math.random().toFixed(4))
    } else {
      formData[fieldName] = Math.floor(Math.random() * 4294967296)
    }
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
        <label for={field.name}>{field.label}</label>
        <div class="input-row">
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
          <div class="field-actions">
            {#if (field as FormField & { randomizable?: boolean }).randomizable && !isBatch}
              <button
                type="button"
                class="action-btn"
                onclick={() => randomizeField(field.name, field.type)}
                title="Randomize value"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  ><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle
                    cx="8.5"
                    cy="8.5"
                    r="1.5"
                  ></circle><circle cx="15.5" cy="15.5" r="1.5"></circle><circle
                    cx="15.5"
                    cy="8.5"
                    r="1.5"
                  ></circle><circle cx="8.5" cy="15.5" r="1.5"></circle><circle
                    cx="12"
                    cy="12"
                    r="1.5"
                  ></circle></svg
                >
              </button>
            {/if}
            <button
              type="button"
              class="action-btn"
              onclick={() => toggleBatchMode(field.name)}
              title="Toggle sequence mode"
            >
              {isBatch ? '−' : '+'}
            </button>
          </div>
        </div>
      {:else if field.type === 'float'}
        <label for={field.name}>{field.label}</label>
        <div class="input-row">
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
          <div class="field-actions">
            {#if (field as FormField & { randomizable?: boolean }).randomizable && !isBatch}
              <button
                type="button"
                class="action-btn"
                onclick={() => randomizeField(field.name, field.type)}
                title="Randomize value"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  ><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle
                    cx="8.5"
                    cy="8.5"
                    r="1.5"
                  ></circle><circle cx="15.5" cy="15.5" r="1.5"></circle><circle
                    cx="15.5"
                    cy="8.5"
                    r="1.5"
                  ></circle><circle cx="8.5" cy="15.5" r="1.5"></circle><circle
                    cx="12"
                    cy="12"
                    r="1.5"
                  ></circle></svg
                >
              </button>
            {/if}
            <button
              type="button"
              class="action-btn"
              onclick={() => toggleBatchMode(field.name)}
              title="Toggle sequence mode"
            >
              {isBatch ? '−' : '+'}
            </button>
          </div>
        </div>
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
        <label for={field.name}>{field.label}</label>
        <div class="input-row">
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
          <div class="field-actions">
            {#if (field as FormField & { randomizable?: boolean }).randomizable && !isBatch}
              <button
                type="button"
                class="action-btn"
                onclick={() => randomizeField(field.name, field.type)}
                title="Randomize value"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  ><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle
                    cx="8.5"
                    cy="8.5"
                    r="1.5"
                  ></circle><circle cx="15.5" cy="15.5" r="1.5"></circle><circle
                    cx="15.5"
                    cy="8.5"
                    r="1.5"
                  ></circle><circle cx="8.5" cy="15.5" r="1.5"></circle><circle
                    cx="12"
                    cy="12"
                    r="1.5"
                  ></circle></svg
                >
              </button>
            {/if}
            <button
              type="button"
              class="action-btn"
              onclick={() => toggleBatchMode(field.name)}
              title="Toggle sequence mode"
            >
              {isBatch ? '−' : '+'}
            </button>
          </div>
        </div>
      {:else if field.type === 'node'}
        <NodeSelector
          label={field.label}
          filter={field.filter}
          bind:node={formData[field.name] as ModelData | AudioData}
          id={field.name}
        />
      {:else}
        <label for={field.name}>{field.label}</label>
        <div class="input-row">
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
          <div class="field-actions">
            {#if (field as FormField & { randomizable?: boolean }).randomizable && !isBatch}
              <button
                type="button"
                class="action-btn"
                onclick={() => randomizeField(field.name, field.type)}
                title="Randomize value"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  stroke-width="2"
                  stroke-linecap="round"
                  stroke-linejoin="round"
                  ><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><circle
                    cx="8.5"
                    cy="8.5"
                    r="1.5"
                  ></circle><circle cx="15.5" cy="15.5" r="1.5"></circle><circle
                    cx="15.5"
                    cy="8.5"
                    r="1.5"
                  ></circle><circle cx="8.5" cy="15.5" r="1.5"></circle><circle
                    cx="12"
                    cy="12"
                    r="1.5"
                  ></circle></svg
                >
              </button>
            {/if}
            <button
              type="button"
              class="action-btn"
              onclick={() => toggleBatchMode(field.name)}
              title="Toggle sequence mode"
            >
              {isBatch ? '−' : '+'}
            </button>
          </div>
        </div>
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

  .input-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }

  .field-actions {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-shrink: 0;
  }

  .action-btn {
    background: none;
    border: 1px solid var(--color-overlay-border-primary);
    color: var(--color-overlay-text);
    cursor: pointer;
    width: 24px;
    height: 24px;
    min-width: 24px;
    min-height: 24px;
    flex: 0 0 24px;
    border-radius: 50%;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    box-sizing: border-box;
  }

  label {
    display: block;
    font-weight: 500;
    margin-bottom: 0.5rem;
  }

  .input-row > input[type='text'],
  .input-row > input[type='number'],
  .input-row > select {
    flex: 1;
    min-width: 0;
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
