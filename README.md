# OpenRouter Style Render

A single-file Blender add-on that turns a viewport or camera render of your 3D
scene into a drawing or illustration style — retro manga, pencil sketch,
watercolor, comic book art, or anything else you can describe — using any
image-capable model on [OpenRouter](https://openrouter.ai).

Your geometry, camera framing and silhouettes are preserved; only the surface
treatment is repainted.

```
Render  ->  Base64 PNG  ->  OpenRouter  ->  Stylized artwork  ->  Blender Image Editor
```

## Features

- **N-panel workflow** — everything lives in `View3D > N > OpenRouter`, no console needed.
- **Two API modes** — multimodal Chat Completions (vision input) or the native
  Images API for true img2img with a real denoising `strength`.
- **Model-agnostic** — a curated dropdown plus a *Custom endpoint* field that
  accepts any OpenRouter model slug.
- **Non-blocking** — HTTP runs on a worker thread; results are marshalled back
  to Blender's main thread through a timer, so the UI never freezes.
- **Format-agnostic response parser** — walks arbitrary provider JSON looking for
  Base64 data URIs, raw Base64 blobs or image URLs, and polls async jobs until
  pixels appear.
- **Lossless source** — the render is written as PNG at your scene resolution.
- **Non-destructive** — render settings are snapshotted and restored in a
  `finally` block, so the operator never permanently mutates your scene.

## Requirements

- Blender **4.2 LTS or newer** (developed and tested against Blender 5.2).
- The `requests` Python package.
- An OpenRouter API key — get one at <https://openrouter.ai/keys>.

### Installing `requests`

Blender ships without `requests`. Install it into Blender's bundled Python:

1. Open the **Scripting** workspace.
2. In the Python Console, run:

   ```python
   import pip; pip.main(['install', 'requests'])
   ```

3. Restart Blender.

If the package is missing, the add-on detects it and shows the exact command
directly in the panel instead of failing silently.

## Installation

1. Download [`openrouter_style_render.py`](openrouter_style_render.py).
2. In Blender: **Edit > Preferences > Add-ons > Install from Disk…**
3. Select the `.py` file, then enable the checkbox next to
   **OpenRouter Style Render (Img2Img)**.
4. Open the 3D Viewport sidebar with `N` and switch to the **OpenRouter** tab.

## Usage

1. Paste your **API Key** and press **Test Connection** to confirm it works.
2. Pick an **AI Model**, or choose *Custom endpoint* and type a slug such as
   `black-forest-labs/flux-kontext-pro`.
3. Write a **Style Prompt** describing the target aesthetic — for example
   `retro 1980s manga ink, bold black outlines, screentone shading`.
4. Adjust **Image Influence (Denoise)**. Low values stay close to your render;
   high values allow a freer reinterpretation.
5. Choose a **Capture** source (Camera Render or Viewport Screenshot) and a size.
6. Press **Generate Stylized Render**.

The rendered source and the finished artwork are both loaded into Blender as
`OpenRouter_Source_Render` and `OpenRouter_Stylized_Output`.

### Choosing an API mode

| Mode | Endpoint | When to use it |
| --- | --- | --- |
| **Chat Completions** | `/api/v1/chat/completions` | Vision/image models that emit images in a chat response. The source render is inlined as a `data:image/png;base64,…` part. Works with the widest range of models. |
| **Images API** | `/api/v1/images/generations` | Provider-native img2img endpoints that accept an `image` field and a real `strength` value. Use this when you need precise control over how strongly the original geometry constrains the output. |

> **Note on strength.** In Chat Completions mode there is no denoising
> parameter — the model is a vision LLM, so the strength value is conveyed
> numerically inside the prompt and is therefore *advisory*. Genuine geometric
> strength control requires Images API mode against an endpoint that honours
> `strength`.

## Settings reference

| Setting | Description |
| --- | --- |
| **API Key** | Your OpenRouter key. Stored as a password field. |
| **API Mode** | Chat Completions or Images API. |
| **Model** | Curated list plus a custom slug escape hatch. |
| **Seed** | Fixed seed for reproducible output; `-1` for random. |
| **Temperature** | Sampling temperature; `-1` omits the parameter entirely. |
| **Style Prompt** | The target drawing style. Required. |
| **Avoid** | Concepts to suppress. |
| **Image Influence** | Denoise strength, `0.0`–`1.0`, default `0.65`. |
| **Capture** | Camera Render (F12) or Viewport Screenshot (OpenGL). |
| **Size** | Match scene resolution, or force 512/768/1024 px square. |
| **Full Resolution** | Ignore the scene resolution percentage. |
| **Save To** | Temp folder, alongside the `.blend`, or a custom folder. |
| **Extra JSON** | Arbitrary JSON merged into the request payload for provider-specific options. |

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `'dict' object has no attribute 'prompt'` | You are running a stale build. Restart Blender or toggle the add-on off and on — a live session keeps the old module in memory. |
| `HTTP 401 Unauthorized` | The API key is invalid or inactive. Re-copy it from <https://openrouter.ai/keys>. |
| `HTTP 402 Payment Required` | The selected model needs credits. Try a `:free` model variant or add credit. |
| `HTTP 404 Not Found` | The model slug is wrong or that model has no image output. Use the *Custom endpoint* field with a slug from the OpenRouter model page. |
| `Missing dependency: 'requests'` | Install it via the command shown in the panel. |
| `No image returned by the model` | The chosen model is text-only. Switch to an image-capable model. |
| Source render exceeds the inline limit | Lower the source size (512 or 768 px) or disable *Full Resolution*. |

## Security

Your API key is stored in the `.blend` file and in Blender's user preferences.
**Do not commit `.blend` files containing a key, and do not share screenshots of
the panel with the key visible.** `.gitignore` in this repository excludes
`.blend` files and common secret files by default.

## Project layout

```
openrouter_style_render.py   # the entire add-on, single file
verify_orsr.py               # offline test harness (stubs bpy, no Blender needed)
```

`verify_orsr.py` exercises the payload builders without launching Blender:

```bash
python verify_orsr.py
```

It validates request construction for both API modes, custom-model resolution,
strength/seed/temperature propagation, and error handling for malformed input.

## Contributing

Issues and pull requests are welcome. When reporting a bug, please include your
Blender version, the selected model, the API mode, and the exact error text —
the add-on prints full API errors to the Info editor for this reason.

## License

[MIT](LICENSE)
