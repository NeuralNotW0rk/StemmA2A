<script lang="ts">
  import { onDestroy } from 'svelte'
  import {
    initiatorNodeStore,
    cyInstanceStore,
    selectionStore,
    type GraphElement
  } from '../../utils/stores'
  import NodeSelector from '../NodeSelector.svelte'
  import type { NodeData } from '../../utils/forms'

  interface Member {
    id: number
    node: NodeData | null
  }

  let {
    initiatorNode = $initiatorNodeStore,
    onclose,
    onrefresh,
    onerror
  } = $props<{
    initiatorNode?: GraphElement | null
    onclose: () => void
    onrefresh: () => void
    onerror: (error: { title: string; message: string }) => void
  }>()

  $effect(() => {
    if (!initiatorNode?.type) {
      onerror({
        title: 'Batching Error',
        message: 'The initiator node does not have a valid type for batching.'
      })
    }
  })

  let isUpdate = $derived(initiatorNode?.type === 'batch')

  let filter = $derived.by(() => {
    let refNode: NodeData | null | undefined = null

    if (isUpdate) {
      const cy = $cyInstanceStore
      if (cy && initiatorNode?.id) {
        const children = cy.$id(initiatorNode.id).children()
        if (children.length > 0) {
          refNode = children[0].data() as NodeData
        }
      }
    } else {
      refNode = initiatorNode as NodeData | null | undefined
    }

    if (!refNode) return {}

    const newFilter: Record<string, string | number | boolean | null> = {}
    if (refNode.type) newFilter.type = refNode.type

    const ctx = refNode.context as Record<string, string | number | boolean | null> | undefined
    if (ctx?.model_id) newFilter['context.model_id'] = ctx.model_id
    if (ctx?.init_audio_id) newFilter['context.init_audio_id'] = ctx.init_audio_id
    if (ctx?.prompt) newFilter['context.prompt'] = ctx.prompt

    //newFilter.parent = null // Exclude nodes that are already in a batch

    console.log('Filter:', newFilter)
    return newFilter as Record<string, string | number | boolean>
  })

  let nextMemberId = 0
  const createMember = (node: NodeData | null = null): Member => ({ id: nextMemberId++, node })

  let members: Member[] = $state([])

  $effect(() => {
    if (members.length === 0 && initiatorNode) {
      if (isUpdate) {
        const cy = $cyInstanceStore
        if (cy) {
          const children = cy.$id(initiatorNode.id).children()
          const initialMembers = children.map((child) => createMember(child.data() as NodeData))
          // Pre-populate with existing members
          members = [...initialMembers]
        }
      } else {
        members = [createMember(initiatorNode as NodeData)]
      }
    }
  })

  let isSelecting = false

  function addMember(): void {
    const newMember = createMember(null)
    members.push(newMember)
    members = members

    isSelecting = true
    selectionStore.startSelection(filter, null, (selected) => {
      isSelecting = false
      if (!selected) {
        members = members.filter((m) => m.id !== newMember.id)
        return
      }
      const member = members.find((m) => m.id === newMember.id)
      if (member) {
        member.node = selected as NodeData
      }
    })
  }

  onDestroy(() => {
    if (isSelecting) selectionStore.cancelSelection()
  })

  function removeMember(id: number): void {
    members = members.filter((m) => m.id !== id)
  }

  async function saveBatch(): Promise<void> {
    const member_ids = members.map((m) => m.node?.id).filter(Boolean)
    if (member_ids.length < 2) {
      onerror({
        title: 'Batching Error',
        message: 'You must select at least two members for a batch.'
      })
      return
    }

    try {
      if (isUpdate && initiatorNode) {
        // @ts-ignore (define in dts)
        await window.api.updateBatch(initiatorNode.id, member_ids)
      } else {
        // @ts-ignore (define in dts)
        await window.api.batchElements(member_ids)
      }
      onrefresh()
    } catch (error) {
      onerror({
        title: 'Batching Error',
        message: error instanceof Error ? error.message : String(error)
      })
    }
  }
</script>

<div class="view-container">
  <div class="view-content">
    <div class="header-row">
      <h3>Members</h3>
      <button onclick={addMember}>Add Member</button>
    </div>
    <div class="members-list">
      {#each members as member, index (member.id)}
        <div class="member-row">
          <div class="member-selector">
            <NodeSelector bind:node={members[index].node} id={`member-${member.id}`} {filter} />
          </div>
          {#if index > 0}
            <button
              class="remove-button"
              onclick={() => removeMember(member.id)}
              title="Remove member"
              aria-label="Remove member"
            >
              ✕
            </button>
          {/if}
        </div>
      {/each}
    </div>
  </div>

  <div class="panel-actions">
    <button onclick={onclose}>Cancel</button>
    <button class="primary" onclick={saveBatch}>
      {isUpdate ? 'Update Batch' : 'Create Batch'}
    </button>
  </div>
</div>

<style>
  .view-container {
    display: flex;
    flex-direction: column;
    height: 100%;
  }
  .view-content {
    flex-grow: 1;
    overflow-y: auto;
    padding: 1rem;
  }
  .header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }
  .header-row h3 {
    margin: 0;
  }
  .members-list {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }
  .member-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .member-selector {
    flex-grow: 1;
    min-width: 0;
  }
  .member-selector :global(.form-field) {
    margin-bottom: 0;
  }
  .remove-button {
    flex-shrink: 0;
    min-width: 0;
    min-height: 0;
    cursor: pointer;
    font-weight: bold;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0;
    font-size: 12px;
    background-color: transparent;
    border: 1px solid var(--color-border-glass-1);
    color: var(--color-text-muted);
    transition: all 0.2s ease;
  }
  .remove-button:hover {
    background-color: var(--color-error);
    border-color: var(--color-error);
    color: white;
    transform: scale(1.05);
  }
  .remove-button:active {
    transform: scale(0.95);
  }
  .panel-actions {
    display: flex;
    justify-content: flex-end;
    gap: 1rem;
    padding: 1rem;
    border-top: 1px solid var(--color-border-glass-1);
    background-color: var(--color-background-glass-2);
    flex-shrink: 0;
  }
</style>
