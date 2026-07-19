# Upstream source

- Repository: https://github.com/tonyzhaozh/act
- Commit: `742c753c0d4a5d87076c8f69e5628c79a8cc5488`
- License: MIT (`LICENSE`)

Local modifications are intentionally limited to language conditioning and
device-independent construction:

- `language_encoder.py`: frozen DistilBERT plus trainable ACT projection
- `policy.py`: accepts instruction strings and forwards one language token
- `detr/models/detr_vae.py`: adds the language token to ACT memory inputs
- `detr/models/transformer.py`: accepts three extra tokens (latent, state, text)
- `detr/main.py`: supports construction from the project training entrypoint
