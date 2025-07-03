# Creator Dashboard

This repository bundles several Gradio applications:

- **Character Generator**
- **FLUX LoRA DLC**
- **FaceSwap**
- **Self‑Forcing**

The apps can be launched together from a single dashboard.

## Setup

1. Install Python 3.10 or later.
2. Run the setup script to install all dependencies:
   ```bash
   ./setup.sh
   ```
   This installs all Python requirements and the `flash-attn` package used by Self-Forcing.

You can change the Ollama host by setting the `OLLAMA_HOST` environment variable
before running the dashboard (defaults to `http://localhost:11434`).

## Running

Start the Gradio interface with:

```bash
python dashboard.py
```

The first launch may take a while because the underlying models are downloaded
automatically into the local cache directory on demand.

## License

This project is licensed under the Apache-2.0 License. See [LICENSE](LICENSE) for details.
