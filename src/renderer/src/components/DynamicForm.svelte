<script lang="ts">
  import type { FormConfig, FormField } from '../utils/forms'

  let { config, formData = $bindable() } = $props<{
    config: FormConfig
    formData: Record<string, string | number | boolean | null>
  }>()

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

      for (const key in field.show_if) {
        if (formData[key] !== field.show_if[key]) {
          return false
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
      <label for={field.name}>{field.label}</label>
      {#if field.type === 'textarea'}
        <textarea bind:value={formData[field.name]} id={field.name} placeholder={field.placeholder}
        ></textarea>
      {:else if field.type === 'number'}
        <input
          type="number"
          bind:value={formData[field.name]}
          id={field.name}
          placeholder={field.placeholder}
        />
      {:else if field.type === 'boolean'}
        <input
          type="checkbox"
          checked={formData[field.name] === 'true' || formData[field.name] === true}
          onchange={(e: Event & { currentTarget: EventTarget & HTMLInputElement }) =>
            (formData[field.name] = e.currentTarget.checked)}
          id={field.name}
        />
      {:else if field.type === 'file'}
        <div class="path-input">
          <input type="text" bind:value={formData[field.name]} placeholder={field.placeholder} />
          <button onclick={() => selectFieldFile(field.name, field.filters)}>Browse</button>
        </div>
      {:else if field.type === 'select'}
        <select bind:value={formData[field.name]} id={field.name}>
          {#each field.options as option (option.value)}
            <option value={option.value}>{option.label}</option>
          {/each}
        </select>
      {:else}
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
