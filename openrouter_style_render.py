# SPDX-License-Identifier: MIT
# -----------------------------------------------------------------------------
# OpenRouter Style Render
# Image-to-Image (Img2Img) stylization pipeline for Blender.
#
# Renders the active camera / viewport, uploads the lossless PNG to the
# OpenRouter API together with a user-defined style prompt, then downloads and
# decodes the stylized result back into Blender's Image Editor.
#
# Tested against Blender 4.2 LTS - 5.2.x APIs.
# -----------------------------------------------------------------------------

bl_info = {
    "name": "OpenRouter Style Render (Img2Img)",
    "author": "Add-on Architect",
    "version": (1, 2, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar (N) > OpenRouter",
    "description": (
        "Render the current camera/viewport and restyle it into any drawing or "
        "art style through the OpenRouter image-to-image API."
    ),
    "warning": "Requires the 'requests' Python package and a valid OpenRouter API key.",
    "doc_url": "https://openrouter.ai/docs",
    "category": "Render",
}

import base64
import json
import mimetypes
import os
import queue
import re
import tempfile
import threading
import time

import bpy
from bpy.app.handlers import persistent
from bpy.props import (
    BoolProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup

# -----------------------------------------------------------------------------
# Optional dependency: requests
# -----------------------------------------------------------------------------

try:
    import requests
    from requests.adapters import HTTPAdapter

    try:  # urllib3 v1 and v2 expose Retry from different locations
        from urllib3.util.retry import Retry
    except Exception:  # pragma: no cover - very old urllib3
        Retry = None

    REQUESTS_AVAILABLE = True
    REQUESTS_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on host environment
    requests = None
    HTTPAdapter = None
    Retry = None
    REQUESTS_AVAILABLE = False
    REQUESTS_IMPORT_ERROR = str(exc)


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

ADDON_ID = __name__
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_IMAGES_URL = "https://openrouter.ai/api/v1/images/generations"
HTTP_REFERER = "https://blender.org"
APP_TITLE = "Blender OpenRouter Style Render"

OUTPUT_IMAGE_NAME = "OpenRouter_Stylized_Output"
SOURCE_IMAGE_NAME = "OpenRouter_Source_Render"

MAX_BASE64_BYTES = 12 * 1024 * 1024          # hard cap for inlined source images
REQUEST_TIMEOUT = (15, 300)                  # (connect, read) seconds
POLL_INTERVAL = 12.0                         # seconds between generation polls
MAX_POLL_ATTEMPTS = 90                       # ~18 minutes worst case

SYSTEM_GUIDE = (
    "You are an advanced image stylization engine. Transform the provided "
    "structural 3D layout image into the exact artistic drawing style requested "
    "by the user. Maintain strict consistency with the original camera "
    "perspective, object composition, and character silhouettes."
)

STYLE_DIRECTIVE = (
    "Treat the input image purely as a structural blueprint: preserve its "
    "camera framing, object placement, scale relationships, silhouette outlines "
    "and overall composition. Repaint surface materials, lighting and texture "
    "entirely in the requested drawing style. Do not add, remove or relocate "
    "objects. Do not introduce text, watermarks or signatures unless the user "
    "explicitly asks for them. Output a single finished image."
)

# Curated model list. The first entry of each group is an OpenRouter ":free"
# mirror where one exists, so the add-on is usable without billing setup.
MODEL_ITEMS = [
    # --- OpenRouter native image generation / editing -----------------------
    ("google/gemini-2.5-flash-image-preview", "Gemini 2.5 Flash Image (nano-banana) - editing", "Native image edit + generation via OpenRouter chat"),
    ("google/gemini-2.5-flash-image-preview:free", "Gemini 2.5 Flash Image (nano-banana) - FREE tier", "Free mirror of the Gemini image editing endpoint"),
    ("google/gemini-2.0-flash-exp:free", "Gemini 2.0 Flash Exp - FREE (image output)", "Legacy free image generation endpoint"),
    # --- Provider-native img2img endpoints ----------------------------------
    ("amazon/nova-canvas-v1", "Amazon Nova Canvas v1 - img2img", "Titan/Nova canvas inpainting and img2img"),
    ("stabilityai/stable-diffusion-xl-img2img", "Stability SDXL img2img", "Classic SDXL image-to-image with strength control"),
    ("stabilityai/stable-image-core", "Stability Stable Image Core", "Fast provider-side transformation pipeline"),
    ("black-forest-labs/flux-1.1-pro", "FLUX 1.1 Pro", "High fidelity generation and image referencing"),
    ("black-forest-labs/flux-kontext-pro", "FLUX Kontext Pro", "Instruction-driven image editing, strong layout retention"),
    ("black-forest-labs/flux-kontext-max", "FLUX Kontext Max", "Highest fidelity FLUX image editing"),
    ("black-forest-labs/flux-3", "FLUX 3", "Latest FLUX generation/editing family"),
    ("black-forest-labs/flux-3-pro", "FLUX 3 Pro", "FLUX 3 pro tier"),
    ("black-forest-labs/flux-3-dev", "FLUX 3 Dev", "FLUX 3 developer tier"),
    ("black-forest-labs/flux-2-pro", "FLUX 2 Pro", "FLUX 2 pro tier"),
    ("black-forest-labs/flux-2-flex", "FLUX 2 Flex", "FLUX 2 flexible tier"),
    ("recraft/recraft-v3", "Recraft V3", "Illustration, vector and drawing style specialist"),
    ("openai/gpt-image-1", "OpenAI gpt-image-1", "Instruction-following image editing"),
    ("openai/dall-e-3", "OpenAI DALL-E 3", "Text-to-image only, no structural conditioning"),
    # --- Escape hatch --------------------------------------------------------
    ("__custom__", "Custom endpoint (type it below)", "Enter any OpenRouter model slug manually"),
]

MODE_ITEMS = [
    ("chat", "Chat Completions (vision input)", "Send the render as a multimodal chat message"),
    ("images", "Images API (native img2img)", "Use the dedicated image generation/editing endpoint"),
]

SOURCE_ITEMS = [
    ("camera", "Camera Render", "Full F12 render using the active scene camera"),
    ("viewport", "Viewport Screenshot", "OpenGL viewport draw of the active 3D View area"),
]

SIZE_ITEMS = [
    ("scene", "Match Scene Resolution", "Use the render resolution percentage and aspect"),
    ("1024", "1024 px (square)", "Force a square 1024x1024 source render"),
    ("768", "768 px (square)", "Smaller payload, faster upload"),
    ("512", "512 px (fast)", "Smallest payload, lowest upload latency"),
]

SAVE_ITEMS = [
    ("tempfile", "Temp Folder (bpy.app.tempdir)", "Discard-safe scratch location"),
    ("project", "Alongside .blend File", "Store results next to the current project"),
    ("custom", "Custom Folder", "Use the directory entered below"),
]

# Regex used to sniff base64 / URL / data-URI images out of arbitrary payloads.
_DATA_URI_RE = re.compile(r"data:image/([a-zA-Z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)")
_RAW_B64_RE = re.compile(r"^[A-Za-z0-9+/=\s]{256,}$")
_IMAGE_URL_RE = re.compile(r"https?://[^\s\"')]+?\.(?:png|jpe?g|webp|gif|bmp)(?:\?[^\s\"')]*)?")


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _safe_slug(text):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", (text or "").strip())
    return slug.strip("_") or "model"


def _ensure_dir(path):
    if path and not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            return False
    return bool(path) and os.path.isdir(path)


def _unique_path(directory, stem, extension):
    candidate = os.path.join(directory, "{}{}".format(stem, extension))
    if not os.path.exists(candidate):
        return candidate
    stamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = os.path.join(directory, "{}_{}{}".format(stem, stamp, extension))
    index = 1
    while os.path.exists(candidate):
        candidate = os.path.join(
            directory, "{}_{}_{}{}".format(stem, stamp, index, extension)
        )
        index += 1
    return candidate


def _relpath_or_abs(path):
    try:
        return bpy.path.relpath(path)
    except Exception:
        return path


def _read_image_b64(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    if len(raw) > MAX_BASE64_BYTES:
        raise RuntimeError(
            "Rendered source image is {:.1f} MB which exceeds the {:.0f} MB inline "
            "limit. Lower the source resolution or disable 'Full Resolution'.".format(
                len(raw) / (1024.0 * 1024.0), MAX_BASE64_BYTES / (1024.0 * 1024.0)
            )
        )
    return base64.b64encode(raw).decode("ascii")


def _sniff_extension(data, fallback=".png"):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if data[:2] == b"BM":
        return ".bmp"
    return fallback


def _notify(message, title="OpenRouter Style Render", icon="INFO"):
    """Report through the Info editor and stdout without ever raising."""
    full = "[{}] {}".format(title, message)
    try:
        print(full)
    except Exception:
        pass
    try:
        def _show():
            try:
                bpy.ops.openrouter_styler.report_message(
                    "INVOKE_DEFAULT", message=message, title=title, icon=icon
                )
            except Exception:
                pass
        bpy.app.timers.register(_show, first_interval=0.1)
    except Exception:
        pass


def _popup(message, title="OpenRouter Style Render", icon="ERROR"):
    """Show a modal popup dialog from a safe timer context."""
    def _show():
        try:
            bpy.ops.openrouter_styler.report_message(
                "INVOKE_DEFAULT", message=message, title=title, icon=icon
            )
        except Exception:
            _notify(message, title, icon)
        return None

    try:
        bpy.app.timers.register(_show, first_interval=0.1)
    except Exception:
        _notify(message, title, icon)


# -----------------------------------------------------------------------------
# HTTP session
# -----------------------------------------------------------------------------

_SESSION = None
_SESSION_LOCK = threading.Lock()


def _get_session():
    global _SESSION
    if not REQUESTS_AVAILABLE:
        raise RuntimeError("The 'requests' package is not available: {}".format(
            REQUESTS_IMPORT_ERROR))
    with _SESSION_LOCK:
        if _SESSION is None:
            session = requests.Session()
            if HTTPAdapter is not None:
                kwargs = {"pool_connections": 4, "pool_maxsize": 8}
                if Retry is not None:
                    kwargs["max_retries"] = Retry(
                        total=3,
                        backoff_factor=1.5,
                        status_forcelist=(429, 500, 502, 503, 504),
                        allowed_methods=frozenset(["GET", "POST"]),
                    )
                adapter = HTTPAdapter(**kwargs)
                session.mount("https://", adapter)
                session.mount("http://", adapter)
            session.headers.update(
                {
                    "HTTP-Referer": HTTP_REFERER,
                    "X-Title": APP_TITLE,
                    "Accept": "application/json",
                }
            )
            _SESSION = session
        return _SESSION


def _auth_headers(api_key):
    return {
        "Authorization": "Bearer {}".format(api_key),
        "Content-Type": "application/json",
    }


def _extract_api_error(response):
    """Turn an OpenRouter error response into a readable one-liner."""
    try:
        payload = response.json()
    except Exception:
        text = (response.text or "").strip().replace("\n", " ")
        return "HTTP {}: {}".format(response.status_code, text[:400] or "no response body")

    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        message = error.get("message") or json.dumps(error)
        code = error.get("code") or response.status_code
        return "HTTP {} ({})".format(code, message)
    if isinstance(error, str):
        return "HTTP {}: {}".format(response.status_code, error)
    return "HTTP {}: {}".format(response.status_code, json.dumps(payload)[:400])


# -----------------------------------------------------------------------------
# Image sniffing / decoding helpers
# -----------------------------------------------------------------------------

def _iter_strings(node, _depth=0):
    """Yield every string contained in a nested JSON structure."""
    if _depth > 12:
        return
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            for item in _iter_strings(value, _depth + 1):
                yield item
    elif isinstance(node, (list, tuple)):
        for value in node:
            for item in _iter_strings(value, _depth + 1):
                yield item


def _first_match(payload, regex, group_index=1):
    for text in _iter_strings(payload):
        match = regex.search(text)
        if match:
            return match.group(group_index).strip()
    return None


def _collect_image_urls(payload):
    urls = []
    for text in _iter_strings(payload):
        stripped = text.strip()
        if stripped.startswith("data:image/"):
            continue
        if _IMAGE_URL_RE.match(stripped):
            urls.append(stripped)
            continue
        for match in _IMAGE_URL_RE.finditer(stripped):
            urls.append(match.group(0))
    seen, ordered = set(), []
    for url in urls:
        if url not in seen:
            seen.add(url)
            ordered.append(url)
    return ordered


def _extract_b64_payload(payload):
    data_uri = _first_match(payload, _DATA_URI_RE, 2)
    if data_uri:
        return re.sub(r"\s+", "", data_uri)

    for text in _iter_strings(payload):
        stripped = text.strip()
        if len(stripped) < 512:
            continue
        if stripped.startswith("data:image/"):
            _, _, tail = stripped.partition(",")
            tail = re.sub(r"\s+", "", tail)
            if tail:
                return tail
        if _RAW_B64_RE.match(stripped):
            return re.sub(r"\s+", "", stripped)
    return None


def _extract_revised_prompt(payload):
    if not isinstance(payload, dict):
        return ""
    for choice in payload.get("choices") or []:
        message = choice.get("message") or {}
        for key in ("content", "reasoning"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:1500]
    images = payload.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            for key in ("revised_prompt", "prompt", "description"):
                value = first.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:1500]
    return ""


def _decode_data_url(data_url):
    """Decode a data: URL or a bare base64 blob into raw bytes."""
    if data_url.startswith("data:"):
        head, _, tail = data_url.partition(",")
        if "base64" in head:
            return base64.b64decode(re.sub(r"\s+", "", tail))
        from urllib.parse import unquote_to_bytes  # local import keeps stdlib light

        return unquote_to_bytes(tail)
    return base64.b64decode(re.sub(r"\s+", "", data_url))


def _download_url(url, timeout=REQUEST_TIMEOUT):
    session = _get_session()
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "image" not in content_type and "octet-stream" not in content_type:
        raise RuntimeError(
            "URL did not return an image (Content-Type: '{}')".format(content_type or "unknown")
        )
    return response.content


# -----------------------------------------------------------------------------
# Job registry + main-thread result queue
# -----------------------------------------------------------------------------

_RESULT_QUEUE = queue.Queue()
_TIMER_REGISTERED = False

_JOBS = {}                    # job_id -> dict(thread, cancel_event, label, started)
_JOBS_LOCK = threading.Lock()
_ACTIVE_JOB_ID = {"value": ""}


def _register_job(label):
    job_id = "{}_{}".format(_safe_slug(label), int(time.time() * 1000))
    cancel_event = threading.Event()
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "cancel": cancel_event,
            "label": label,
            "started": time.time(),
            "thread": None,
        }
    return job_id, cancel_event


def _finish_job(job_id):
    with _JOBS_LOCK:
        _JOBS.pop(job_id, None)
    if _ACTIVE_JOB_ID.get("value") == job_id:
        _ACTIVE_JOB_ID["value"] = ""


def _cancel_all_jobs():
    with _JOBS_LOCK:
        events = [job["cancel"] for job in _JOBS.values()]
    for event in events:
        event.set()


def _has_active_jobs():
    with _JOBS_LOCK:
        return bool(_JOBS)


def _job_status_text():
    with _JOBS_LOCK:
        if not _JOBS:
            return ""
        job = next(iter(_JOBS.values()))
        elapsed = int(time.time() - job["started"])
        return "{} - {}s".format(job["label"], elapsed)


def _pump_results():
    """Drain worker results on the main thread. Returns the timer interval."""
    drained = False
    while True:
        try:
            event = _RESULT_QUEUE.get_nowait()
        except queue.Empty:
            break
        drained = True
        _apply_event(event)

    if drained:
        _redraw_views()

    if _has_active_jobs():
        return 0.25
    return 0.5


def _apply_event(event):
    kind = event.get("kind")
    try:
        if kind == "image":
            _load_result_image(event)
        elif kind == "error":
            _popup(event.get("message", "Unknown error"), "OpenRouter Style Render", "ERROR")
            _last_error(event.get("message", ""))
        elif kind == "info":
            _notify(event.get("message", ""), "OpenRouter Style Render", "INFO")
        elif kind == "warn":
            _notify(event.get("message", ""), "OpenRouter Style Render", "WARNING")
    except Exception as exc:  # never let the timer die
        _notify("Result handling failed: {}".format(exc), icon="ERROR")


def _last_error(message):
    try:
        addon = bpy.context.preferences.addons.get(ADDON_ID)
        if addon and addon.preferences:
            addon.preferences.last_error = message or ""
    except Exception:
        pass


def _redraw_views():
    window_manager = getattr(bpy.context, "window_manager", None)
    if not window_manager:
        return
    for window in window_manager.windows:
        screen = getattr(window, "screen", None)
        if not screen:
            continue
        for area in screen.areas:
            try:
                area.tag_redraw()
            except Exception:
                pass


def _ensure_result_timer():
    global _TIMER_REGISTERED
    if _TIMER_REGISTERED:
        return
    try:
        bpy.app.timers.register(_pump_results, first_interval=0.25, persistent=True)
        _TIMER_REGISTERED = True
    except Exception as exc:
        _notify("Could not start the result timer: {}".format(exc), icon="ERROR")


def _load_result_image(event):
    path = event["path"]
    if not os.path.isfile(path):
        raise RuntimeError("Result file disappeared: {}".format(path))

    image = bpy.data.images.get(OUTPUT_IMAGE_NAME)
    if image is None:
        image = bpy.data.images.load(path, check_existing=False)
        try:
            image.name = OUTPUT_IMAGE_NAME
        except Exception:
            pass
        image.use_fake_user = True
    else:
        image.filepath = path
        image.filepath_raw = path
        image.source = "FILE"
        try:
            image.reload()
        except Exception:
            bpy.data.images.remove(image)
            image = bpy.data.images.load(path, check_existing=False)
            image.name = OUTPUT_IMAGE_NAME
            image.use_fake_user = True

    try:
        if image.size[0] == 0 or image.size[1] == 0:
            image.reload()
    except Exception:
        pass

    # Wire the result into the Image Editor where one is open.
    shown = False
    try:
        for window in bpy.context.window_manager.windows:
            screen = getattr(window, "screen", None)
            if not screen:
                continue
            for area in screen.areas:
                if area.type != "IMAGE_EDITOR":
                    continue
                for space in area.spaces:
                    if space.type != "IMAGE_EDITOR":
                        continue
                    space.image = image
                    if hasattr(space, "image_user") and space.image_user:
                        pass
                    shown = True
                area.tag_redraw()
    except Exception:
        pass

    _notify(
        "Stylized render ready: {}{}".format(
            os.path.basename(path), "" if shown else " (open an Image Editor to view)"
        ),
        icon="INFO",
    )


# -----------------------------------------------------------------------------
# Render pipeline
# -----------------------------------------------------------------------------

def _scene_pixel_size(scene, size_mode, full_resolution):
    base_x, base_y = scene.render.resolution_x, scene.render.resolution_y
    if size_mode == "scene":
        if full_resolution:
            return max(1, int(base_x)), max(1, int(base_y))
        percent = max(1, int(scene.render.resolution_percentage)) / 100.0
        return (
            max(1, int(base_x * percent)),
            max(1, int(base_y * percent)),
        )
    edge = int(size_mode)
    if base_x >= base_y:
        return edge, max(1, int(round(edge * base_y / float(base_x))))
    return max(1, int(round(edge * base_x / float(base_y)))), edge


def _render_camera_to_path(scene, path, size_mode, full_resolution):
    """Render through the scene camera into a lossless PNG."""
    render = scene.render
    saved = {
        "filepath": render.filepath,
        "image_settings.file_format": render.image_settings.file_format,
        "image_settings.color_mode": render.image_settings.color_mode,
        "image_settings.color_depth": render.image_settings.color_depth,
        "image_settings.compression": render.image_settings.compression,
        "resolution_percentage": render.resolution_percentage,
        "resolution_x": render.resolution_x,
        "resolution_y": render.resolution_y,
    }

    try:
        render.image_settings.file_format = "PNG"
        render.image_settings.color_mode = "RGB"
        render.image_settings.color_depth = "8"
        render.image_settings.compression = 15

        if size_mode != "scene":
            width, height = _scene_pixel_size(scene, size_mode, full_resolution)
            render.resolution_x = width
            render.resolution_y = height
            render.resolution_percentage = 100
        elif full_resolution:
            render.resolution_percentage = 100

        render.filepath = path

        try:
            result = bpy.ops.render.render("EXEC_DEFAULT", write_still=True)
        except RuntimeError as exc:
            raise RuntimeError("Render failed: {}".format(exc))

        if "CANCELLED" in result:
            raise RuntimeError("Render was cancelled.")

        if not os.path.isfile(path):
            alt = os.path.splitext(path)[0] + ".png"
            if os.path.isfile(alt):
                if alt != path:
                    os.replace(alt, path)
            else:
                raise RuntimeError(
                    "Render finished but no file was written to '{}'.".format(path)
                )
    finally:
        render.filepath = saved["filepath"]
        render.image_settings.file_format = saved["image_settings.file_format"]
        render.image_settings.color_mode = saved["image_settings.color_mode"]
        render.image_settings.color_depth = saved["image_settings.color_depth"]
        render.image_settings.compression = saved["image_settings.compression"]
        render.resolution_percentage = saved["resolution_percentage"]
        render.resolution_x = saved["resolution_x"]
        render.resolution_y = saved["resolution_y"]

    return path


def _render_viewport_to_path(context, path, size_mode, full_resolution):
    """Grab the active 3D viewport as a lossless PNG."""
    area = next(
        (a for a in context.screen.areas if a.type == "VIEW_3D"),
        None,
    ) if context.screen else None
    if area is None:
        raise RuntimeError("No 3D Viewport found. Switch to Viewport mode or use Camera Render.")

    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    space = next((s for s in area.spaces if s.type == "VIEW_3D"), None)

    saved_shading = None
    if space is not None:
        saved_shading = space.shading.type

    override = {}
    try:
        with context.temp_override(area=area, region=region, space_data=space):
            if size_mode != "scene":
                width, height = _scene_pixel_size(context.scene, size_mode, full_resolution)
                result = bpy.ops.render.opengl(
                    "EXEC_DEFAULT",
                    write_still=True,
                    view_context=False,
                    filepath=path,
                )
                del width, height  # viewport size is governed by the area, not the scene
            else:
                result = bpy.ops.render.opengl(
                    "EXEC_DEFAULT", write_still=True, view_context=False, filepath=path
                )
    except TypeError:
        # Blender < 3.2 fallback (no temp_override).
        region_backup = getattr(context, "region", None)
        area_backup = getattr(context, "area", None)
        try:
            context.area = area
            context.region = region
            result = bpy.ops.render.opengl("EXEC_DEFAULT", write_still=True, view_context=False)
        finally:
            if region_backup is not None:
                context.region = region_backup
            if area_backup is not None:
                context.area = area_backup
    finally:
        if space is not None and saved_shading is not None:
            try:
                space.shading.type = saved_shading
            except Exception:
                pass

    if "CANCELLED" in result:
        raise RuntimeError("Viewport capture was cancelled.")

    if not os.path.isfile(path):
        alt = os.path.splitext(path)[0] + ".png"
        if os.path.isfile(alt):
            if alt != path:
                os.replace(alt, path)
        else:
            raise RuntimeError("Viewport capture produced no image.")

    return path


def _peek_scene_path(scene, source_mode, size_mode, full_resolution):
    stem = "openrouter_src_{}".format(source_mode)
    temp_dir = bpy.app.tempdir or tempfile.gettempdir()
    _ensure_dir(temp_dir)
    return os.path.join(temp_dir, stem + ".png")


# -----------------------------------------------------------------------------
# OpenRouter request construction
# -----------------------------------------------------------------------------

class JobSettings(object):
    """Immutable snapshot of the UI settings, safe to read from a worker thread.

    Blender RNA property values may only be touched from the main thread, so the
    operator copies everything it needs into this plain object before the worker
    thread starts. Accessing attributes here is ordinary Python, never RNA.
    """

    __slots__ = (
        "api_key",
        "model",
        "custom_model",
        "mode",
        "prompt",
        "negative_prompt",
        "strength",
        "temperature",
        "seed",
        "extra_json",
        "source_path",
        "output_dir",
        "source_mode",
        "size_mode",
    )

    def __init__(
        self,
        api_key="",
        model="",
        custom_model="",
        mode="chat",
        prompt="",
        negative_prompt="",
        strength=0.65,
        temperature=-1.0,
        seed=-1,
        extra_json="",
        source_path="",
        output_dir="",
        source_mode="camera",
        size_mode="scene",
    ):
        self.api_key = api_key or ""
        self.model = model or ""
        self.custom_model = custom_model or ""
        self.mode = mode or "chat"
        self.prompt = prompt or ""
        self.negative_prompt = negative_prompt or ""
        self.strength = float(strength)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.extra_json = extra_json or ""
        self.source_path = source_path or ""
        self.output_dir = output_dir or ""
        self.source_mode = source_mode or "camera"
        self.size_mode = size_mode or "scene"

    @classmethod
    def from_property_group(cls, settings, api_key="", model="", source_path="",
                            output_dir=""):
        """Capture a scene PropertyGroup or any duck-typed settings object.

        Explicit keyword arguments win; anything left empty is read off the
        source object when it exposes a matching attribute.
        """

        def _pick(explicit, attr, fallback):
            if explicit not in ("", None):
                return explicit
            value = getattr(settings, attr, None)
            return fallback if value is None else value

        return cls(
            api_key=_pick(api_key, "api_key", ""),
            model=_pick(model, "model", ""),
            custom_model=getattr(settings, "custom_model", ""),
            mode=getattr(settings, "gen_mode", None) or getattr(settings, "mode", "chat"),
            prompt=getattr(settings, "prompt", ""),
            negative_prompt=getattr(settings, "negative_prompt", ""),
            strength=getattr(settings, "strength", 0.65),
            temperature=getattr(settings, "temperature", -1.0),
            seed=getattr(settings, "seed", -1),
            extra_json=getattr(settings, "extra_json", ""),
            source_path=_pick(source_path, "source_path", ""),
            output_dir=_pick(output_dir, "output_dir", ""),
            source_mode=getattr(settings, "source_mode", "camera"),
            size_mode=getattr(settings, "size_mode", "scene"),
        )

    @classmethod
    def from_mapping(cls, mapping):
        """Accept a JobSettings, a PropertyGroup, a dict or any settings-like object."""
        if isinstance(mapping, cls):
            return mapping
        if isinstance(mapping, dict):
            known = {key: mapping[key] for key in cls.__slots__ if key in mapping}
            if "mode" not in known and "gen_mode" in mapping:
                known["mode"] = mapping["gen_mode"]
            return cls(**known)
        # PropertyGroup / RNA struct / duck-typed object fallback.
        return cls.from_property_group(mapping)

    def __repr__(self):
        return "JobSettings(model={!r}, mode={!r}, strength={:.2f})".format(
            self.model, self.mode, self.strength
        )


def _coerce_settings(source):
    """Guarantee the payload builders always receive a JobSettings instance."""
    if isinstance(source, JobSettings):
        return source
    return JobSettings.from_mapping(source)


def _resolve_model(prefs):
    """Return the effective OpenRouter model slug.

    Accepts a JobSettings, a PropertyGroup or a plain dict. When the enum is set
    to the custom sentinel the hand-typed slug wins; a custom slug also overrides
    the enum whenever one is present, so a pasted slug is never silently ignored.
    """
    settings = _coerce_settings(prefs)
    custom = (getattr(settings, "custom_model", "") or "").strip()
    enum_model = (settings.model or "").strip()

    if custom and (enum_model == "__custom__" or not enum_model):
        return custom
    if custom and custom != enum_model:
        return custom
    if enum_model and enum_model != "__custom__":
        return enum_model
    raise RuntimeError(
        "No model selected. Pick a model from the list or enter a custom "
        "OpenRouter slug such as 'black-forest-labs/flux-3'."
    )


def _build_style_prompt(prefs):
    prefs = _coerce_settings(prefs)
    user_prompt = (prefs.prompt or "").strip()
    if not user_prompt:
        raise RuntimeError("The style prompt is empty. Describe the target drawing style.")
    text = "{}\n\nStyle request: {}\n\nStrength/denoise: {:.2f}".format(
        STYLE_DIRECTIVE, user_prompt, float(prefs.strength)
    )
    negative = (prefs.negative_prompt or "").strip()
    if negative:
        text += "\nAvoid: {}".format(negative)
    return text


def _merge_extra_json(payload, prefs):
    extra = (prefs.extra_json or "").strip()
    if not extra:
        return payload
    try:
        parsed = json.loads(extra)
    except Exception as exc:
        raise RuntimeError("Extra JSON parameters are invalid: {}".format(exc))
    if not isinstance(parsed, dict):
        raise RuntimeError("Extra JSON parameters must be a JSON object.")
    payload.update(parsed)
    return payload


def _build_chat_payload(prefs, model, image_b64):
    """Multimodal chat payload with the render embedded as a data URL."""
    prefs = _coerce_settings(prefs)
    content = [
        {"type": "text", "text": _build_style_prompt(prefs)},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,{}".format(image_b64)},
        },
    ]

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_GUIDE},
            {"role": "user", "content": content},
        ],
        "modalities": ["image", "text"],
        "stream": False,
    }

    if prefs.temperature >= 0.0:
        payload["temperature"] = float(prefs.temperature)

    if prefs.seed >= 0:
        payload["seed"] = int(prefs.seed)

    return _merge_extra_json(payload, prefs)


def _build_images_payload(prefs, model, image_b64):
    """Native img2img payload for /images/generations style endpoints."""
    prefs = _coerce_settings(prefs)
    payload = {
        "model": model,
        "prompt": _build_style_prompt(prefs),
        "image": "data:image/png;base64,{}".format(image_b64),
        "strength": float(prefs.strength),
        "response_format": "b64_json",
        "n": 1,
    }
    if prefs.seed >= 0:
        payload["seed"] = int(prefs.seed)

    return _merge_extra_json(payload, prefs)


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------

def _post_json(url, payload, api_key, cancel_event, timeout=REQUEST_TIMEOUT):
    session = _get_session()
    response = session.post(
        url,
        headers=_auth_headers(api_key),
        data=json.dumps(payload).encode("utf-8"),
        timeout=timeout,
    )
    if response.status_code == 401:
        raise RuntimeError(
            "HTTP 401 Unauthorized. Check that the OpenRouter API key is valid and active."
        )
    if response.status_code == 402:
        raise RuntimeError(
            "HTTP 402 Payment Required. The selected model needs credits on this account."
        )
    if response.status_code == 404:
        raise RuntimeError(
            "HTTP 404 Not Found. The model slug is probably wrong or has no image output: {}"
            .format(_extract_api_error(response))
        )
    if response.status_code == 429:
        raise RuntimeError("HTTP 429 Rate Limited. Wait a moment and retry.")
    if response.status_code >= 400:
        raise RuntimeError(_extract_api_error(response))
    try:
        return response.json()
    except Exception as exc:
        raise RuntimeError("Response was not valid JSON: {}".format(exc))


def _payload_has_image(payload):
    if _extract_b64_payload(payload):
        return True
    return bool(_collect_image_urls(payload))


def _payload_is_pending(payload):
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status", "")).lower()
    if status in ("pending", "queued", "processing", "in_progress", "running"):
        return True
    if payload.get("id") and not _payload_has_image(payload) and not payload.get("choices"):
        return True
    return False


def _job_preview_label(source_mode):
    return "Rendering {}".format("viewport" if source_mode == "viewport" else "camera")


def _worker(job_id, cancel_event, snapshot):
    """Runs off the main thread. All Blender calls already happened upstream."""
    source_mode = snapshot["source_mode"]
    stage = "render"
    try:
        # ---- 1. Base64 encode the already-rendered source -------------------
        stage = "encode"
        if cancel_event.is_set():
            raise RuntimeError("Cancelled before upload.")
        image_b64 = _read_image_b64(snapshot["source_path"])
        _RESULT_QUEUE.put(
            {
                "kind": "info",
                "message": "Source encoded ({:.1f} MB). Contacting OpenRouter...".format(
                    len(image_b64) * 3 / 4 / (1024.0 * 1024.0)
                ),
            }
        )

        # ---- 2. Build the payload -------------------------------------------
        stage = "payload"
        prefs = _coerce_settings(snapshot["prefs"])
        model = snapshot["model"]
        if snapshot["mode"] == "images":
            url = OPENROUTER_IMAGES_URL
            payload = _build_images_payload(prefs, model, image_b64)
        else:
            url = OPENROUTER_CHAT_URL
            payload = _build_chat_payload(prefs, model, image_b64)

        # ---- 3. POST ---------------------------------------------------------
        stage = "request"
        if cancel_event.is_set():
            raise RuntimeError("Cancelled before request.")
        response_payload = _post_json(url, payload, snapshot["api_key"], cancel_event)

        # ---- 4. Poll if provider queued the job ------------------------------
        stage = "poll"
        attempts = 0
        while _payload_is_pending(response_payload) and attempts < MAX_POLL_ATTEMPTS:
            if cancel_event.is_set():
                raise RuntimeError("Cancelled while waiting for the provider.")
            attempts += 1
            time.sleep(POLL_INTERVAL)
            job_ref = response_payload.get("id") or response_payload.get("job_id")
            poll_url = "{}/{}".format(url.rstrip("/"), job_ref)
            response_payload = _post_json(url, payload, snapshot["api_key"], cancel_event) \
                if job_ref is None else _get_json(poll_url, snapshot["api_key"])

        if not _payload_has_image(response_payload):
            preview = _extract_revised_prompt(response_payload)
            detail = _extract_api_error_text(response_payload)
            raise RuntimeError(
                "No image returned by the model. {} {}".format(
                    detail, ("Model said: '{}'".format(preview[:400])) if preview else ""
                ).strip()
            )

        # ---- 5. Retrieve pixels ---------------------------------------------
        stage = "download"
        raw = None
        b64 = _extract_b64_payload(response_payload)
        if b64:
            try:
                raw = _decode_data_url(b64)
            except Exception as exc:
                raise RuntimeError("Returned Base64 image could not be decoded: {}".format(exc))
        else:
            urls = _collect_image_urls(response_payload)
            errors = []
            for candidate in urls:
                try:
                    raw = _download_url(candidate)
                    break
                except Exception as exc:
                    errors.append("{}: {}".format(candidate[:120], exc))
            if raw is None:
                raise RuntimeError(
                    "No downloadable image URL worked. {}".format("; ".join(errors) or "none found")
                )

        if not raw:
            raise RuntimeError("The retrieved image payload was empty.")

        # ---- 6. Save ---------------------------------------------------------
        stage = "save"
        extension = _sniff_extension(raw)
        target_dir = snapshot["output_dir"]
        if not _ensure_dir(target_dir):
            target_dir = bpy.app.tempdir or tempfile.gettempdir()
            _ensure_dir(target_dir)

        base = os.path.splitext(os.path.basename(snapshot["source_path"]))[0]
        stem = "{}_stylized_{}".format(base, _safe_slug(model.split("/")[-1]))
        out_path = _unique_path(target_dir, stem, extension)

        with open(out_path, "wb") as handle:
            handle.write(raw)

        # ---- 7. Hand back to the main thread --------------------------------
        _RESULT_QUEUE.put(
            {
                "kind": "image",
                "path": out_path,
                "prompt": _extract_revised_prompt(response_payload),
                "job_id": job_id,
            }
        )
    except Exception as exc:
        message = str(exc)
        if stage == "render":
            message = "Rendering failed: {}".format(message)
        _RESULT_QUEUE.put({"kind": "error", "message": message, "job_id": job_id})
    finally:
        _RESULT_QUEUE.put({"kind": "internal_finish", "job_id": job_id})
        _finish_job(job_id)


def _get_json(url, api_key, timeout=REQUEST_TIMEOUT):
    session = _get_session()
    response = session.get(url, headers=_auth_headers(api_key), timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(_extract_api_error(response))
    return response.json()


def _extract_api_error_text(payload):
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error)
        if isinstance(error, str):
            return error
    if isinstance(payload, dict) and payload.get("choices"):
        finish = payload["choices"][0].get("finish_reason")
        if finish:
            return "finish_reason='{}'.".format(finish)
    return ""


# -----------------------------------------------------------------------------
# Properties
# -----------------------------------------------------------------------------

def _on_model_update(self, context):
    if self.model != "__custom__":
        self.custom_model = ""
    try:
        addon = context.preferences.addons.get(ADDON_ID)
        if addon and addon.preferences:
            addon.preferences.last_model = self.model
    except Exception:
        pass


class ORSR_Settings(PropertyGroup):
    api_key: StringProperty(
        name="API Key",
        description="OpenRouter API key (stored in the user preferences / .blend)",
        subtype="PASSWORD",
        default="",
    )
    model: EnumProperty(
        name="AI Model",
        description="OpenRouter model slug used for the stylization pass",
        items=MODEL_ITEMS,
        default="google/gemini-2.5-flash-image-preview",
        update=_on_model_update,
    )
    custom_model: StringProperty(
        name="Custom Model",
        description="Any OpenRouter model slug, used when 'Custom endpoint' is selected",
        default="",
    )
    prompt: StringProperty(
        name="Style Prompt",
        description=(
            "Describe the target drawing style, e.g. 'retro 1980s manga ink, "
            "screentone shading, heavy black outlines'"
        ),
        default="retro manga ink drawing, bold black outlines, screentone shading",
    )
    negative_prompt: StringProperty(
        name="Avoid",
        description="Concepts to suppress (appended to the request when supported)",
        default="",
    )
    strength: FloatProperty(
        name="Image Influence (Denoise)",
        description=(
            "How strongly the AI follows the original 3D geometry. Low values stay "
            "close to the render, high values allow a freer reinterpretation"
        ),
        default=0.65,
        min=0.0,
        max=1.0,
        precision=2,
        subtype="FACTOR",
    )
    temperature: FloatProperty(
        name="Temperature",
        description="Sampling temperature. Set to -1 to omit the parameter",
        default=-1.0,
        min=-1.0,
        max=2.0,
        precision=2,
    )
    seed: IntProperty(
        name="Seed",
        description="Fixed seed for reproducible results. -1 for random",
        default=-1,
        min=-1,
    )
    source_mode: EnumProperty(
        name="Source",
        items=SOURCE_ITEMS,
        default="camera",
    )
    size_mode: EnumProperty(
        name="Source Size",
        items=SIZE_ITEMS,
        default="scene",
    )
    full_resolution: BoolProperty(
        name="Full Resolution",
        description="Ignore the resolution percentage and render at 100%",
        default=False,
    )
    gen_mode: EnumProperty(
        name="API Mode",
        items=MODE_ITEMS,
        default="chat",
    )
    save_mode: EnumProperty(
        name="Save Results To",
        items=SAVE_ITEMS,
        default="tempfile",
    )
    output_dir: StringProperty(
        name="Output Folder",
        description="Destination for generated artwork",
        subtype="DIR_PATH",
        default="",
    )
    extra_json: StringProperty(
        name="Extra JSON",
        description="Optional JSON object merged into the request payload",
        default="",
    )
    last_error: StringProperty(name="Last Error", default="")
    last_model: StringProperty(name="Last Model", default="")


# -----------------------------------------------------------------------------
# Add-on preferences
# -----------------------------------------------------------------------------

class ORSR_Preferences(AddonPreferences):
    bl_idname = ADDON_ID

    model: EnumProperty(
        name="Default Model",
        items=MODEL_ITEMS,
        default="google/gemini-2.5-flash-image-preview",
    )
    request_timeout: IntProperty(
        name="Request Timeout (s)",
        description="Read timeout for a single OpenRouter HTTP call",
        default=300,
        min=30,
        max=1800,
    )
    auto_open: BoolProperty(
        name="Auto-open Image Editor",
        description="Switch an available area to the Image Editor when a result arrives",
        default=True,
    )
    last_error: StringProperty(name="Last Error", default="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False

        if not REQUESTS_AVAILABLE:
            box = layout.box()
            box.alert = True
            box.label(text="'requests' is not installed.", icon="ERROR")
            box.label(text="Run Blender's Python console and execute:")
            box.label(text="import pip; pip.main(['install', 'requests'])")
            if REQUESTS_IMPORT_ERROR:
                box.label(text=REQUESTS_IMPORT_ERROR[:120])

        col = layout.column(heading="Defaults")
        col.prop(self, "model")
        col.prop(self, "request_timeout")
        col.prop(self, "auto_open")

        if self.last_error:
            box = layout.box()
            box.label(text="Last Error", icon="ERROR")
            box.label(text=self.last_error[:200])


# -----------------------------------------------------------------------------
# Operators
# -----------------------------------------------------------------------------

class ORSR_OT_report_message(Operator):
    """Internal helper that surfaces messages from worker threads."""

    bl_idname = "openrouter_styler.report_message"
    bl_label = "OpenRouter Styler Message"
    bl_options = {"INTERNAL"}

    message: StringProperty(default="")
    title: StringProperty(default="OpenRouter Style Render")
    icon: StringProperty(default="INFO")

    def execute(self, context):
        lines = self.message.splitlines() or [self.message]
        width = min(72, max((len(line) for line in lines), default=0) + 2)
        try:
            self.report({"ERROR" if self.icon == "ERROR" else "INFO"}, self.message)
        except Exception:
            pass
        try:
            bpy.ops.openrouter_styler.popup_draw("INVOKE_DEFAULT")
        except Exception:
            pass
        del lines, width
        return {"FINISHED"}


class ORSR_OT_popup_draw(Operator):
    """Draws the actual modal popup (must run in a real operator context)."""

    bl_idname = "openrouter_styler.popup_draw"
    bl_label = "OpenRouter Message"
    bl_options = {"INTERNAL"}

    def invoke(self, context, event):
        return context.window_manager.invoke_popup(self, width=460)

    def draw(self, context):
        layout = self.layout
        try:
            addon = context.preferences.addons.get(ADDON_ID)
            message = addon.preferences.last_error if addon and addon.preferences else ""
        except Exception:
            message = ""
        if not message:
            message = "Operation finished."
        for line in message.splitlines():
            layout.label(text=line[:110])

    def execute(self, context):
        return {"FINISHED"}


class ORSR_OT_test_connection(Operator):
    """Validate the API key and list a couple of available image models."""

    bl_idname = "openrouter_styler.test_connection"
    bl_label = "Test Connection"
    bl_description = "Send a lightweight request to OpenRouter to verify the API key"
    bl_options = {"REGISTER"}

    def execute(self, context):
        settings = context.scene.openrouter_styler
        api_key = (settings.api_key or "").strip()
        if not REQUESTS_AVAILABLE:
            self.report({"ERROR"}, "The 'requests' package is missing: {}".format(
                REQUESTS_IMPORT_ERROR))
            return {"CANCELLED"}
        if not api_key:
            self.report({"ERROR"}, "Enter an OpenRouter API key first.")
            return {"CANCELLED"}

        try:
            session = _get_session()
            response = session.get(
                "https://openrouter.ai/api/v1/key",
                headers=_auth_headers(api_key),
                timeout=(10, 30),
            )
            if response.status_code >= 400:
                raise RuntimeError(_extract_api_error(response))
            data = response.json().get("data", {})
            label = data.get("label") or "unnamed key"
            limit = data.get("limit")
            usage = data.get("usage")
            self.report(
                {"INFO"},
                "API key OK ('{}'). Usage {}/{}.".format(label, usage, limit),
            )
            return {"FINISHED"}
        except Exception as exc:
            self.report({"ERROR"}, "Connection test failed: {}".format(exc))
            return {"CANCELLED"}


class ORSR_OT_cancel_stylize(Operator):
    """Request cancellation of the running stylization job(s)."""

    bl_idname = "openrouter_styler.cancel"
    bl_label = "Cancel"
    bl_description = "Ask the running job to stop after the current HTTP call"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if not _has_active_jobs():
            self.report({"INFO"}, "No job is running.")
            return {"CANCELLED"}
        _cancel_all_jobs()
        self.report({"WARNING"}, "Cancellation requested.")
        return {"FINISHED"}


class ORSR_OT_open_output(Operator):
    """Load the most recent stylized result back into Blender."""

    bl_idname = "openrouter_styler.open_output"
    bl_label = "Open Result"
    bl_description = "Reload the last generated image into Blender's Image Editor"
    bl_options = {"REGISTER"}

    def execute(self, context):
        image = bpy.data.images.get(OUTPUT_IMAGE_NAME)
        if image is None:
            self.report({"ERROR"}, "'{}' is not present in this file.".format(OUTPUT_IMAGE_NAME))
            return {"CANCELLED"}
        path = bpy.path.abspath(image.filepath) if image.filepath else ""
        if path and os.path.isfile(path):
            try:
                image.reload()
            except Exception as exc:
                self.report({"WARNING"}, "Reload failed: {}".format(exc))
        try:
            for area in context.screen.areas:
                if area.type == "IMAGE_EDITOR":
                    for space in area.spaces:
                        if space.type == "IMAGE_EDITOR":
                            space.image = image
                    area.tag_redraw()
        except Exception:
            pass
        self.report({"INFO"}, "Loaded '{}'.".format(OUTPUT_IMAGE_NAME))
        return {"FINISHED"}


class ORSR_OT_stylize(Operator):
    """Render the current view, then restyle it through OpenRouter."""

    bl_idname = "openrouter_styler.stylize"
    bl_label = "Generate Stylized Render"
    bl_description = (
        "Render the active camera or viewport, send it to OpenRouter with the "
        "style prompt and load the stylized result back into Blender"
    )
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.scene is not None

    def execute(self, context):
        settings = context.scene.openrouter_styler

        # ---- Guard clauses ---------------------------------------------------
        if not REQUESTS_AVAILABLE:
            self.report(
                {"ERROR"},
                "The 'requests' package is missing. Install it from the add-on preferences.",
            )
            return {"CANCELLED"}

        api_key = (settings.api_key or "").strip()
        if not api_key:
            self.report({"ERROR"}, "An OpenRouter API key is required.")
            return {"CANCELLED"}

        if not (settings.prompt or "").strip():
            self.report({"ERROR"}, "Describe the target drawing style in the prompt field.")
            return {"CANCELLED"}

        if _ACTIVE_JOB_ID.get("value"):
            self.report({"WARNING"}, "A stylization job is already running.")
            return {"CANCELLED"}

        try:
            model = _resolve_model(settings)
        except Exception as exc:
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}

        # ---- Output directory -------------------------------------------------
        if settings.save_mode == "project":
            blend_path = bpy.data.filepath
            if blend_path:
                output_dir = os.path.dirname(bpy.path.abspath(blend_path))
            else:
                output_dir = bpy.app.tempdir or tempfile.gettempdir()
                self.report(
                    {"WARNING"}, "Scene is unsaved; results go to the temporary folder."
                )
        elif settings.save_mode == "custom":
            output_dir = bpy.path.abspath(settings.output_dir) if settings.output_dir else ""
            if not output_dir:
                self.report({"ERROR"}, "Choose an output folder or pick another save mode.")
                return {"CANCELLED"}
        else:
            output_dir = bpy.app.tempdir or tempfile.gettempdir()

        if not _ensure_dir(output_dir):
            self.report({"ERROR"}, "Output folder is not writable: {}".format(output_dir))
            return {"CANCELLED"}

        # ---- 1. Render the source image (main thread, required) --------------
        source_path = _peek_scene_path(context.scene, settings.source_mode,
                                       settings.size_mode, settings.full_resolution)
        window = context.window_manager.windows[0] if context.window_manager.windows else None
        try:
            if window is not None:
                with context.temp_override(window=window, scene=context.scene):
                    window.cursor_set("WAIT")
        except Exception:
            window = None

        try:
            if settings.source_mode == "viewport":
                _render_viewport_to_path(
                    context, source_path, settings.size_mode, settings.full_resolution
                )
            else:
                _render_camera_to_path(
                    context.scene, source_path, settings.size_mode, settings.full_resolution
                )
        except Exception as exc:
            if window is not None:
                try:
                    window.cursor_set("DEFAULT")
                except Exception:
                    pass
            self.report({"ERROR"}, str(exc))
            return {"CANCELLED"}
        finally:
            if window is not None:
                try:
                    window.cursor_set("DEFAULT")
                except Exception:
                    pass

        if not os.path.isfile(source_path):
            self.report({"ERROR"}, "Source render was not written to disk.")
            return {"CANCELLED"}

        # ---- 2. Register the source image inside Blender ---------------------
        try:
            src_image = bpy.data.images.load(source_path, check_existing=False)
            src_image.name = SOURCE_IMAGE_NAME
            src_image.use_fake_user = True
        except Exception:
            pass

        # ---- 3. Snapshot settings and dispatch the worker ---------------------
        job_settings = JobSettings.from_property_group(
            settings, api_key=api_key, model=model,
            source_path=source_path, output_dir=output_dir,
        )
        snapshot = {"job": job_settings, "prefs": job_settings}
        snapshot.update(
            {
                "api_key": api_key,
                "model": model,
                "mode": job_settings.mode,
                "source_path": source_path,
                "output_dir": output_dir,
                "source_mode": job_settings.source_mode,
            }
        )

        job_id, cancel_event = _register_job("Stylizing")
        _ACTIVE_JOB_ID["value"] = job_id
        _ensure_result_timer()

        thread = threading.Thread(
            target=_worker,
            args=(job_id, cancel_event, snapshot),
            name="ORSR-Worker-{}".format(job_id),
            daemon=True,
        )
        with _JOBS_LOCK:
            if job_id in _JOBS:
                _JOBS[job_id]["thread"] = thread
        thread.start()

        self.report(
            {"INFO"},
            "Source rendered ({:.1f} KB). Generating with '{}'...".format(
                os.path.getsize(source_path) / 1024.0, model
            ),
        )
        return {"FINISHED"}


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

class ORSR_PT_main(Panel):
    bl_idname = "ORSR_PT_main"
    bl_label = "OpenRouter Style Render"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "OpenRouter"
    bl_options = {"DEFAULT_CLOSED"} if False else set()

    def draw(self, context):
        layout = self.layout
        settings = context.scene.openrouter_styler

        # ---- Dependency gate ------------------------------------------------
        if not REQUESTS_AVAILABLE:
            box = layout.box()
            box.alert = True
            box.label(text="Missing dependency: 'requests'", icon="ERROR")
            col = box.column(align=True)
            col.scale_y = 0.85
            col.label(text="Run in Blender's Python console:")
            col.label(text="import pip; pip.main(['install','requests'])")
            if REQUESTS_IMPORT_ERROR:
                col.label(text=REQUESTS_IMPORT_ERROR[:100])
            layout.separator()

        # ---- Connection ------------------------------------------------------
        box = layout.box()
        box.label(text="Connection", icon="URL")
        col = box.column(align=True)
        col.use_property_split = False
        col.prop(settings, "api_key", text="API Key", icon="LOCKED")
        row = col.row(align=True)
        row.operator(ORSR_OT_test_connection.bl_idname, icon="PLUGIN")
        row.prop(settings, "gen_mode", text="")

        # ---- Model -----------------------------------------------------------
        box = layout.box()
        box.label(text="Model", icon="MESH_MONKEY")
        col = box.column(align=True)
        col.prop(settings, "model", text="")
        if settings.model == "__custom__":
            col.prop(settings, "custom_model", text="Slug")
        col.prop(settings, "seed")
        col.prop(settings, "temperature")

        # ---- Style prompt ----------------------------------------------------
        box = layout.box()
        box.label(text="Style Prompt", icon="BRUSH_DATA")
        col = box.column(align=True)
        col.prop(settings, "prompt", text="", icon="TEXT")
        col.prop(settings, "negative_prompt", text="Avoid")
        col.prop(settings, "strength", slider=True)

        # ---- Source ----------------------------------------------------------
        box = layout.box()
        box.label(text="Source Image", icon="RENDER_STILL")
        col = box.column(align=True)
        col.use_property_split = True
        col.prop(settings, "source_mode", text="Capture")
        col.prop(settings, "size_mode", text="Size")
        if settings.size_mode == "scene":
            col.prop(settings, "full_resolution")

        # ---- Output ----------------------------------------------------------
        box = layout.box()
        box.label(text="Output", icon="FILE_IMAGE")
        col = box.column(align=True)
        col.use_property_split = True
        col.prop(settings, "save_mode", text="Save To")
        if settings.save_mode == "custom":
            col.prop(settings, "output_dir", text="Folder")
        col.prop(settings, "extra_json", text="Extra JSON")

        # ---- Action ----------------------------------------------------------
        layout.separator()
        status = _job_status_text()
        col = layout.column(align=True)
        if status:
            col.enabled = False
            col.operator(ORSR_OT_stylize.bl_idname, icon="SORTTIME", text=status)
            layout.operator(ORSR_OT_cancel_stylize.bl_idname, icon="CANCEL")
        else:
            col.enabled = REQUESTS_AVAILABLE
            col.operator(ORSR_OT_stylize.bl_idname, icon="PLAY")
        layout.operator(ORSR_OT_open_output.bl_idname, icon="IMAGE_DATA")

        if settings.last_error:
            box = layout.box()
            box.alert = True
            box.label(text="Last Error", icon="ERROR")
            col = box.column(align=True)
            col.scale_y = 0.8
            for line in settings.last_error.splitlines()[:6]:
                col.label(text=line[:60])


# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------

_CLASSES = (
    ORSR_Settings,
    ORSR_Preferences,
    ORSR_OT_report_message,
    ORSR_OT_popup_draw,
    ORSR_OT_test_connection,
    ORSR_OT_cancel_stylize,
    ORSR_OT_open_output,
    ORSR_OT_stylize,
    ORSR_PT_main,
)


@persistent
def _orsr_load_post(_dummy):
    """Re-arm the result timer after a file load or a script reload."""
    try:
        if _has_active_jobs():
            _ensure_result_timer()
    except Exception:
        pass


def _apply_preference_defaults():
    """Seed the scene settings from the add-on preferences on first use."""
    try:
        addon = bpy.context.preferences.addons.get(ADDON_ID)
        if not addon or not addon.preferences:
            return
        prefs = addon.preferences
        for scene in bpy.data.scenes:
            settings = getattr(scene, "openrouter_styler", None)
            if settings is None:
                continue
            if prefs.last_error:
                settings.last_error = prefs.last_error
    except Exception:
        pass


def register():
    global _TIMER_REGISTERED

    for cls in _CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.openrouter_styler = PointerProperty(type=ORSR_Settings)

    if _orsr_load_post not in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.append(_orsr_load_post)

    _TIMER_REGISTERED = False
    _ensure_result_timer()
    _apply_preference_defaults()

    if not REQUESTS_AVAILABLE:
        print(
            "[OpenRouter Style Render] 'requests' is unavailable ({}). Install it via "
            "Blender's Python console: import pip; pip.main(['install','requests'])".format(
                REQUESTS_IMPORT_ERROR
            )
        )


def unregister():
    global _TIMER_REGISTERED, _SESSION

    _cancel_all_jobs()

    if _orsr_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_orsr_load_post)

    if _TIMER_REGISTERED:
        try:
            if bpy.app.timers.is_registered(_pump_results):
                bpy.app.timers.unregister(_pump_results)
        except Exception:
            pass
        _TIMER_REGISTERED = False

    if hasattr(bpy.types.Scene, "openrouter_styler"):
        try:
            del bpy.types.Scene.openrouter_styler
        except Exception:
            pass

    for cls in reversed(_CLASSES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

    try:
        if _SESSION is not None:
            _SESSION.close()
    except Exception:
        pass
    _SESSION = None


if __name__ == "__main__":
    register()
