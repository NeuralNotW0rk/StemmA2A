<script lang="ts">
  import { initiatorNodeStore, cyInstanceStore, type GraphElement } from '../../utils/stores'
  import NodeSelector from '../NodeSelector.svelte'
  import type { NodeData } from '../../utils/forms'

  interface Member {
    id: number
    node: NodeData | null
  }

  let {
    initiatorNode = $initiatorNodeStore,
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
    let refNode: Record<string, any> | null | undefined = null

    if (isUpdate) {
      const cy = $cyInstanceStore
      if (cy && initiatorNode?.id) {
        const children = cy.$id(initiatorNode.id).children()
        if (children.length > 0) {
          refNode = children[0].data()
        }
      }
    } else {
      refNode = initiatorNode as Record<string, any> | null | undefined
    }

    if (!refNode) return {}

    const newFilter: Record<string, any> = {}
    if (refNode.type) newFilter.type = refNode.type
    if (refNode.context?.model_id) newFilter['context.model_id'] = refNode.context.model_id
    if (refNode.context?.init_audio_id)
      newFilter['context.init_audio_id'] = refNode.context.init_audio_id
    if (refNode.context?.prompt) newFilter['context.prompt'] = refNode.context.prompt

    newFilter.parent = null // Exclude nodes that are already in a batch

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
          // Pre-populate with existing members and an empty slot at the end
          members = [...initialMembers, createMember(null)]
        }
      } else {
        members = [createMember(initiatorNode as NodeData), createMember(null)]
      }
    }
  })

  function addMember(): void {
    members.push(createMember(null))
    members = members
  }

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

<div>
  <h3>Members</h3>
  {#each members as member, index (member.id)}
    <div>
      <NodeSelector
        bind:node={members[index].node}
        label={`Member ${index + 1}`}
        id={`member-${member.id}`}
        {filter}
      />
      {#if index > 0}
        <button onclick={() => removeMember(member.id)}>Remove</button>
      {/if}
    </div>
  {/each}

  <button onclick={addMember}>Add Member</button>
  <button onclick={saveBatch}>{isUpdate ? 'Update Batch' : 'Create Batch'}</button>
</div>
