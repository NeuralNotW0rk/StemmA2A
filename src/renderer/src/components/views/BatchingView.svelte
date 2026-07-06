<script lang="ts">
  import { initiatorNodeStore, cyInstanceStore, type GraphElement } from '../../utils/stores'
  import NodeSelectorList, { type NodeListItem } from '../NodeSelectorList.svelte'
  import type { NodeData, BatchData } from '../../utils/forms'
  import { BATCHING_CONFIG } from '../../utils/app-config'
  import type { ErrorInfo } from '../../utils/types'

  let {
    initiatorNode = $initiatorNodeStore,
    onclose,
    onrefresh,
    onerror
  } = $props<{
    initiatorNode?: GraphElement | null
    onclose: () => void
    onrefresh: () => void
    onerror: (error: ErrorInfo) => void
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
  let isUpdateTitle = $derived(
    initiatorNode?.type === 'batch' &&
      (initiatorNode as unknown as BatchData).member_ids?.length > 0
  )

  let filter = $derived.by(() => {
    let refNode: NodeData | null | undefined = null
    const cy = $cyInstanceStore

    // 1. Try to get reference node from first selected member in members state
    const firstMemberWithNode = members.find((m) => m.node && typeof m.node !== 'string')
    if (firstMemberWithNode && firstMemberWithNode.node) {
      refNode = firstMemberWithNode.node as NodeData
    }

    // 2. Fallback to children of initiator batch
    if (!refNode && isUpdate) {
      if (cy && initiatorNode?.id) {
        const children = cy.$id(initiatorNode.id).children()
        if (children.length > 0) {
          refNode = children[0].data() as NodeData
        }
      }
    }

    // 3. Fallback to initiatorNode itself
    if (!refNode) {
      refNode = initiatorNode as NodeData | null | undefined
    }

    if (!refNode || refNode.type === 'batch') return {}

    const newFilter: Record<string, unknown> = {}
    if (refNode.type) newFilter.type = refNode.type

    const ctx = refNode.context as Record<string, unknown> | undefined
    BATCHING_CONFIG.strictContextKeys.forEach((key) => {
      newFilter[`context.${key}`] = ctx?.[key] ?? null
    })

    // Enforce identical structural dependencies based on incoming graph edges
    if (cy && refNode.id) {
      newFilter['_incoming_node_ids'] = cy.$id(refNode.id).data('_incoming_node_ids') || []
    }

    // Ensure any node already selected in the list is excluded from being a candidate
    const memberIds = members
      .map((m) => (typeof m.node === 'string' ? m.node : m.node?.id))
      .filter(Boolean)
    if (memberIds.length > 0) {
      newFilter['_exclude_node_ids'] = memberIds
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
          if (children.length > 0) {
            const initialMembers = children.map((child) => createMember(child.data() as NodeData))
            // Pre-populate with existing members
            members = [...initialMembers]
          } else {
            // It is an empty batch, pre-populate with one empty slot
            members = [createMember(null)]
          }
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

    try {
      if (initiatorNode) {
        // @ts-ignore (define in dts)
        await window.api.updateBatch(initiatorNode.id, member_ids)
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
      minItems={0}
    />
  </div>

  <div class="panel-actions">
    <button onclick={onclose}>Cancel</button>
    <button class="primary" onclick={saveBatch}>
      {isUpdateTitle ? 'Update Batch' : 'Create Batch'}
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
