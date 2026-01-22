/// <reference types="svelte" />
/// <reference types="vite/client" />

import 'cytoscape'

declare module 'cytoscape' {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  interface ExpandCollapseApi {
    collapse(
      nodes: cytoscape.CollectionArgument,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      options?: any
    ): Promise<cytoscape.CollectionReturnValue>
    expand(
      nodes: cytoscape.CollectionArgument,
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      options?: any
    ): Promise<cytoscape.CollectionReturnValue>
  }

  interface Core {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    expandCollapse(options?: any): ExpandCollapseApi
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    cxtmenu(options?: any): void
  }
}
