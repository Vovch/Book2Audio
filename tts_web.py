"""Launch the Gradio UI (shim for ``python tts_web.py``). Logic lives in ``src/`` as package ``book2audio``."""

from book2audio.web_ui import main

if __name__ == "__main__":
    main()
