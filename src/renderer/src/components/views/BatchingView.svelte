<script lang="ts">
  import { initiatorNodeStore, cyInstanceStore, type GraphElement } from '../../utils/stores'
  import NodeSelectorList, { type NodeListItem } from '../NodeSelectorList.svelte'
  import type { NodeData } from '../../utils/forms'

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

    const newFilter: Record<string, any> = {}
    if (refNode.type) newFilter.type = refNode.type

    const ctx = refNode.context as Record<string, any> | undefined
    if (ctx?.model_id) newFilter['context.model_id'] = ctx.model_id
    if (ctx?.init_audio_id) newFilter['context.init_audio_id'] = ctx.init_audio_id
    if (ctx?.prompt) newFilter['context.prompt'] = ctx.prompt

    if (ctx?.lattice_ids && Array.isArray(ctx.lattice_ids) && ctx.lattice_ids.length > 0) {
      newFilter['context.lattice_ids'] = ctx.lattice_ids
    } else {
      newFilter['context.lattice_ids'] = null
    }

    console.log('Filter:', newFilter)
    return newFilter
  })

  let nextMemberId = 0
  const createMember = (node: NodeData | null = null): NodeListItem => ({
    id: nextMemberId++,
    node
  })

  let members: NodeListItem[] = $state([])

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

  async function saveBatch(): Promise<void> {
    const member_ids = members
      .map((m) => (typeof m.node === 'string' ? m.node : m.node?.id))
      .filter(Boolean)
    if (member_ids.length < 1) {
      onerror({
        title: 'Batching Error',
        message: 'You must select at least one member for a batch.'
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
    <NodeSelectorList
      title="Members"
      addButtonText="Add Member"
      {filter}
      bind:items={members}
      idPrefix="member"
      minItems={1}
    />
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
