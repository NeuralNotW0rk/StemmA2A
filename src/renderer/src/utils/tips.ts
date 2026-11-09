import { writable, type Writable } from 'svelte/store'

export interface Tip {
  title: string | null
  body: string
}

// A tip from something that can't carry a data-tip attribute, such as an element drawn on the graph
// canvas. TipView shows it in preference to DOM tips; setting null clears it.
export const externalTip: Writable<Tip | null> = writable(null)
