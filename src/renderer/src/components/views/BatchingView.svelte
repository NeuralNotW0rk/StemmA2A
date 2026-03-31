<script lang="ts">
  import { initiatorNodeStore } from '../../utils/stores'
  import NodeSelector from '../NodeSelector.svelte'
  import type { NodeData } from '../../utils/forms'

  interface Member {
    id: number
    node: NodeData | null
  }

  let { onrefresh, onerror } = $props<{
    onclose: () => void
    onrefresh: () => void
    onerror: (error: { title: string; message: string }) => void
  }>()

  const initiatorNode = $initiatorNodeStore

  if (!initiatorNode?.type) {
    onerror({
      title: 'Batching Error',
      message: 'The initiator node does not have a valid type for batching.'
    })
  }

  const filter = { type: initiatorNode?.type }

  let nextMemberId = 0
  const createMember = (node: NodeData | null = null): Member => ({ id: nextMemberId++, node })

  let members: Member[] = $state([createMember(initiatorNode), createMember(null)])

  function addMember(): void {
    members.push(createMember(null))
    members = members
  }

  function removeMember(id: number): void {
    members = members.filter((m) => m.id !== id)
  }

  async function createBatch(): Promise<void> {
    const member_ids = members.map((m) => m.node?.id).filter(Boolean)
    if (member_ids.length < 2) {
      onerror({
        title: 'Batching Error',
        message: 'You must select at least two members to create a batch.'
      })
      return
    }

    try {
      // @ts-ignore (define in dts)
      await window.api.batchElements(member_ids)
      onrefresh()
    } catch (error) {
      onerror({
        title: 'Batching Error',
        message: error.message
      })
    }
  }
</script>

<div>
  <h3>Members</h3>
  {#each members as member, index (member.id)}
    <div>
      <NodeSelector
        bind:node={member.node}
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
  <button onclick={createBatch}>Create Batch</button>
</div>
