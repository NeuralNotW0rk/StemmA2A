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

  $effect(() => {
    isFormValid = visibleFields.every(
      (f) =>
        !f.validation?.required ||
        (formData[f.name] != null &&
          (typeof formData[f.name] !== 'string' || String(formData[f.name]).trim() !== ''))
    )
  })

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
    <div class="form-field">
      {#if field.type === 'textarea'}
        <label for={field.name}>{field.label}</label>
        <textarea bind:value={formData[field.name]} id={field.name} placeholder={field.placeholder}
        ></textarea>
      {:else if field.type === 'integer'}
        <label for={field.name}>{field.label}</label>
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
      {:else if field.type === 'float'}
        <label for={field.name}>{field.label}</label>
        <input
          type="number"
          step="any"
          bind:value={formData[field.name]}
          id={field.name}
          placeholder={field.placeholder}
        />
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
        <select bind:value={formData[field.name]} id={field.name}>
          {#each field.options as option (option.value)}
            <option value={option.value}>{option.label}</option>
          {/each}
        </select>
      {:else if field.type === 'node'}
        <NodeSelector
          label={field.label}
          selectionType={field.selectionType}
          node={formData[field.name] as ModelData | AudioData}
          id={field.name}
          onSelect={(newNode) => {
            formData[field.name] = newNode
          }}
        />
      {:else}
        <label for={field.name}>{field.label}</label>
        <!-- Default to text input for 'string' and others -->
        <input
          type="text"
          bind:value={formData[field.name]}
          id={field.name}
          placeholder={field.placeholder}
        />
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

  label {
    display: block;
    margin-bottom: 0.5rem;
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
