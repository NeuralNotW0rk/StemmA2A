# stemma2a

An Electron application with Svelte and TypeScript

## Recommended IDE Setup

- [VSCode](https://code.visualstudio.com/) + [ESLint](https://marketplace.visualstudio.com/items?itemName=dbaeumer.vscode-eslint) + [Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode) + [Svelte](https://marketplace.visualstudio.com/items?itemName=svelte.svelte-vscode)

## Project Setup

### Backend Setup

1. **Configure Compute Backend**:
   Before installing the dependencies, you need to configure the compute backend for PyTorch. Run the following command:
   ```bash
   $ npm run backend:configure
   ```
   This will guide you through selecting the appropriate backend for your hardware (e.g., CPU, CUDA).

2. **Install Dependencies**:
   After configuring the backend, you can set up the Python environment and install the dependencies:
   ```bash
   $ npm run backend:setup
   ```

### Install

```bash
$ npm install
```

### Development

```bash
$ npm run dev
```

### Build

```bash
# For windows
$ npm run build:win

# For macOS
$ npm run build:mac

# For Linux
$ npm run build:linux
```
