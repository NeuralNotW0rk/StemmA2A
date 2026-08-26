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

3. **Docker Deployment (CPU Host to GPU Machine)**:
   If you need to build the Docker image on a CPU machine (e.g., due to network issues on the GPU machine) and deploy it to a GPU machine, run:
   ```bash
   $ npm run docker:build-gpu
   ```
   This will build the Docker container targeting GPU architectures and export it to `execution-engine.tar` at the project root.

   Once built, deploy and run the image on the GPU machine with the following steps:

   1. **Transfer the TAR archive** to the GPU machine (replace `user@gpu-host` with your actual login and `/path/to/destination/` with the target folder):
      ```bash
      scp execution-engine.tar user@gpu-host:/path/to/destination/
      ```

   2. **SSH / Log in** to the GPU machine and import the Docker image:
      ```bash
      docker load -i /path/to/destination/execution-engine.tar
      ```

   3. **Launch the container stack** from the `StemmA2A` project root directory:
      ```bash
      docker compose up -d
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
