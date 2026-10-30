"""Offline verification of the add-on's payload construction.

Stubs out `bpy` so the module imports outside Blender, then drives the real
_build_chat_payload / _build_images_payload / JobSettings code paths.
"""
import base64
import json
import os
import sys
import types

ADDON = os.environ.get("ORSR_ADDON") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "openrouter_style_render.py"
)
if not os.path.isfile(ADDON):
    sys.exit("add-on not found at {!r}; set ORSR_ADDON to its path".format(ADDON))

# ---------------------------------------------------------------- bpy stub
bpy = types.ModuleType("bpy")
bpy.app = types.SimpleNamespace(
    tempdir=os.environ.get("TEMP", "."),
    timers=types.SimpleNamespace(
        register=lambda *a, **k: None,
        unregister=lambda *a, **k: None,
        is_registered=lambda *a, **k: False,
    ),
    handlers=types.SimpleNamespace(load_post=[]),
)
bpy.path = types.SimpleNamespace(relpath=lambda p: p, abspath=lambda p: p)
bpy.data = types.SimpleNamespace(filepath="", images=types.SimpleNamespace(get=lambda n: None))
bpy.context = types.SimpleNamespace(preferences=types.SimpleNamespace(addons={}))

bpy_types = types.ModuleType("bpy.types")


class _Base(object):
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


for name in ("AddonPreferences", "Operator", "Panel", "PropertyGroup"):
    setattr(bpy_types, name, type(name, (_Base,), {"bl_idname": "x"}))

bpy.types = bpy_types

bpy_props = types.ModuleType("bpy.props")
for name in ("BoolProperty", "EnumProperty", "FloatProperty", "IntProperty",
             "PointerProperty", "StringProperty"):
    setattr(bpy_props, name, lambda **kw: kw)
bpy.props = bpy_props

bpy_utils = types.ModuleType("bpy.utils")
bpy_utils.register_class = lambda c: None
bpy_utils.unregister_class = lambda c: None

bpy_app_handlers = types.ModuleType("bpy.app.handlers")
bpy_app_handlers.persistent = lambda fn: fn
bpy.app.handlers = types.SimpleNamespace(load_post=[])
bpy.app.timers = types.SimpleNamespace(
    register=lambda *a, **k: None, unregister=lambda *a, **k: None,
    is_registered=lambda *a, **k: False)

sys.modules["bpy"] = bpy
sys.modules["bpy.types"] = bpy_types
sys.modules["bpy.props"] = bpy_props
sys.modules["bpy.utils"] = bpy_utils
sys.modules["bpy.app.handlers"] = bpy_app_handlers

# ------------------------------------------------------------- load module
import importlib.util

spec = importlib.util.spec_from_file_location("orsr_probe", ADDON)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print("module loaded, requests available:", mod.REQUESTS_AVAILABLE)

IMG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"0" * 64).decode()

failures = []


def check(label, condition, detail=""):
    print(("PASS  " if condition else "FAIL  ") + label + (("  -> " + detail) if detail else ""))
    if not condition:
        failures.append(label)


# ------------------------------------------------------ 1. the original bug
try:
    mod._build_chat_payload(
        {"prompt": "pencil sketch", "strength": 0.65, "temperature": -1.0,
         "seed": -1, "extra_json": "", "negative_prompt": ""},
        "black-forest-labs/flux-kontext-pro", IMG_B64)
    check("plain dict no longer crashes the builder", True)
except Exception as exc:
    check("plain dict no longer crashes the builder", False, repr(exc))

# ------------------------------------------------- 2. JobSettings from dict
js = mod.JobSettings.from_mapping(
    {"prompt": "pencil sketch", "strength": 0.4, "temperature": 0.7, "seed": 7,
     "extra_json": "", "negative_prompt": "blur", "mode": "images"})
check("dict maps to JobSettings", isinstance(js, mod.JobSettings) and js.prompt == "pencil sketch")
check("dict strength preserved", abs(js.strength - 0.4) < 1e-6, repr(js.strength))

# ---------------------------------------- 3. chat payload with a custom model
FLUX3 = "black-forest-labs/flux-3"
js_chat = mod.JobSettings(prompt="retro manga ink", strength=0.65, model=FLUX3,
                          negative_prompt="text, watermark")
payload = mod._build_chat_payload(js_chat, FLUX3, IMG_B64)
check("chat model slug is the custom one", payload["model"] == FLUX3, payload.get("model"))
check("chat declares image modality", payload["modalities"] == ["image", "text"])
check("chat system guide intact",
      payload["messages"][0]["content"] == mod.SYSTEM_GUIDE)
parts = payload["messages"][1]["content"]
check("chat carries text + image parts",
      parts[0]["type"] == "text" and parts[1]["type"] == "image_url")
check("chat image is a png data url",
      parts[1]["image_url"]["url"].startswith("data:image/png;base64,"))
check("chat prompt includes the user style", "retro manga ink" in parts[0]["text"])
check("chat prompt includes the negative prompt", "Avoid: text, watermark" in parts[0]["text"])
check("chat omits temperature when -1", "temperature" not in payload)
check("chat omits seed when -1", "seed" not in payload)

# ------------------------------------- 4. temperature + seed + extra JSON
js_full = mod.JobSettings(prompt="watercolor", strength=0.8, temperature=0.9,
                          seed=42, extra_json='{"top_p": 0.9}')
p2 = mod._build_chat_payload(js_full, "google/gemini-2.5-flash-image-preview", IMG_B64)
check("temperature forwarded", p2.get("temperature") == 0.9, repr(p2.get("temperature")))
check("seed forwarded", p2.get("seed") == 42, repr(p2.get("seed")))
check("extra json merged", p2.get("top_p") == 0.9, repr(p2.get("top_p")))

# ------------------------------------------- 5. images payload (native img2img)
js_img = mod.JobSettings(prompt="classic pencil sketch", strength=0.35, seed=3,
                         mode="images")
p3 = mod._build_images_payload(js_img, "stabilityai/stable-diffusion-xl-img2img", IMG_B64)
check("images model slug", p3["model"] == "stabilityai/stable-diffusion-xl-img2img")
check("images strength honoured", p3["strength"] == 0.35, repr(p3["strength"]))
check("images carries the source image", p3["image"].startswith("data:image/png;base64,"))
check("images requests b64_json", p3["response_format"] == "b64_json")
check("images prompt includes the style", "classic pencil sketch" in p3["prompt"])
check("images seed forwarded", p3["seed"] == 3)

# --------------------------------------------- 6. bad extra JSON is reported
try:
    mod._build_chat_payload(mod.JobSettings(prompt="x", extra_json="{not json"),
                            "m", IMG_B64)
    check("invalid extra JSON raises", False, "no exception")
except RuntimeError as exc:
    check("invalid extra JSON raises", "invalid" in str(exc).lower(), str(exc))

# ------------------------------------------- 7. empty prompt is rejected
try:
    mod._build_images_payload(mod.JobSettings(prompt="  ", mode="images"), "m", IMG_B64)
    check("empty prompt raises", False, "no exception")
except RuntimeError as exc:
    check("empty prompt raises", "empty" in str(exc).lower(), str(exc))

# ------------------------------------------- 8. custom model resolution
class _PG(_Base):
    custom_model = ""


pg = _PG(model="__custom__", custom_model=FLUX3)
check("custom model slug resolves", mod._resolve_model(pg) == FLUX3, mod._resolve_model(pg))
pg2 = _PG(model="amazon/nova-canvas-v1")
check("enum model resolves", mod._resolve_model(pg2) == "amazon/nova-canvas-v1")
try:
    mod._resolve_model(_PG(model="__custom__"))
    check("empty custom slug raises", False, "no exception")
except RuntimeError as exc:
    check("empty custom slug raises", "no model selected" in str(exc).lower(), str(exc))

# a pasted slug overrides a still-selected enum entry
check("custom slug overrides the enum",
      mod._resolve_model(_PG(model="openai/dall-e-3",
                             custom_model="black-forest-labs/flux-3"))
      == "black-forest-labs/flux-3")

# ------------------------------------------- 9. from_property_group on a PG
class _Settings(_Base):
    pass


st = _Settings(gen_mode="images", prompt="oil painting", negative_prompt="",
               strength=0.5, temperature=-1.0, seed=-1, extra_json="",
               source_mode="viewport", size_mode="1024")
snap = mod.JobSettings.from_property_group(st, api_key="k", model="m",
                                           source_path="/tmp/a.png", output_dir="/tmp")
check("from_property_group captures mode", snap.mode == "images", snap.mode)
check("from_property_group captures prompt", snap.prompt == "oil painting")
check("from_property_group captures source_mode", snap.source_mode == "viewport")
check("coerce accepts JobSettings identity", mod._coerce_settings(snap) is snap)
check("coerce accepts a PropertyGroup", isinstance(mod._coerce_settings(st), mod.JobSettings))

print()
if failures:
    print("{} CHECK(S) FAILED: {}".format(len(failures), failures))
    sys.exit(1)
print("ALL CHECKS PASSED")
