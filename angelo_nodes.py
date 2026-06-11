"""AngeloRefine — single-node click-to-refine sampler.

How it works:
  - On first run (no clicks yet), the node decodes the incoming latent
    via VAE and shows the image in its own preview area. The latent is
    cached server-side keyed by node id.
  - The JS frontend attaches click handling to that preview. When the
    user clicks a region they want improved, the JS updates this node's
    click_x / click_y / click_seq widgets and re-queues the workflow.
  - On re-run, the node detects the new click_seq, builds a feathered
    circular mask centred on (click_x, click_y) with radius click_radius,
    re-noises the cached latent partially (`denoise`), and re-samples
    `steps` steps with the mask applied. ComfyUI's noise_mask handling
    re-stitches the original outside the mask at each step (the noise-
    injection inpaint pattern), so the unclicked region stays preserved.
  - The refined latent is cached for the next click, so successive clicks
    keep refining the SAME working latent rather than starting over.
  - Toggle the `reset` widget (or press the Reset button the JS adds) to
    throw away the cache and start fresh from the incoming latent.

The cache is in-process state only — restart of ComfyUI clears it.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import inspect
import io
import json
import math
import os
import threading
import time

import numpy as np
import torch
from PIL import Image as _PILImage, ImageDraw as _PILImageDraw, ImageOps as _PILImageOps

# Pillow ≥ 10.0 moved resampling filters to Image.Resampling.*
# Support both old (<10) and new (≥10) APIs without deprecation warnings.
_LANCZOS = getattr(_PILImage, "Resampling", _PILImage).LANCZOS
_BILINEAR = getattr(_PILImage, "Resampling", _PILImage).BILINEAR

import comfy.sample
import comfy.samplers
import comfy.utils
import comfy.model_management
import latent_preview
import folder_paths
import node_helpers

# Reuse PreviewImage's save machinery for the preview output.
import nodes as comfy_nodes

# Per-node state cache:
#   unique_id -> {
#     "history":   list[tuple[Tensor, Tensor | None, Tensor | None]], # stack of (latent, pixels, mask) (oldest first, current = [-1])
#     "click_seq": int,            # last processed click_seq from JS
#     "undo_seq":  int,            # last processed undo_seq from JS
#     "redo_seq":  int,            # last processed redo_seq from JS
#     "redo_stack": list,          # entries Undo popped, awaiting Redo (cleared on new edit)
#     "source_latent": Tensor,     # session base latent, for the source_image output
#     "source_pixels": Tensor|None,# decoded source base (lazy, cached)
#     "fingerprint": str,          # hash of incoming latent; mismatch = upstream changed
#     "last_mask": Tensor|None,    # most recent [1,H,W] refine mask, for the MASK output
#     # Kursat NOTE!!! 2026-06-10: gen_token is a monotonically-increasing counter
#     # stamped under the per-node lock in Step 1 of every edit run.  Step 3
#     # compares its local copy against state["gen_token"] before writing to
#     # history; if a newer run incremented the token in the meantime the stale
#     # result is discarded and history is left in the state the newer run wrote.
#     # This prevents an older slow GPU sample from overwriting a newer result
#     # when two queue items overlap on the same node (Risk 1 fix).
#     "gen_token": int,            # monotonic edit counter — guards out-of-order history writes
#   }
_STATE: dict[str, dict] = {}

# Per-node lock: prevents concurrent runs on the same node from corrupting
# the shared history stack or fingerprint. ComfyUI is typically serial, but
# rapid re-queues can overlap. Uses RLock so the same thread can re-enter
# (e.g. if ComfyUI ever calls run() from within a node callback).
_STATE_LOCK = threading.RLock()

# Per-node lock registry — protected by _STATE_LOCK itself.
_NODE_LOCKS: dict[str, threading.Lock] = {}

# Kursat NOTE!!! 2026-06-11 — Risk 2 fix (prune-during-execution): active-run counter.
# Active-run counter: tracks how many run() calls for a node are currently
# between Step 1 (lock acquired, state read) and Step 3 (results written).
# Between those two locked sections the per-node lock is NOT held, so
# _NODE_LOCKS[nid].locked() would return False even while sampling is active.
# This counter is incremented under _STATE_LOCK just before Step 1 and
# decremented under _STATE_LOCK just after Step 3, giving _prune_state() a
# reliable signal to skip nodes that are in flight.  Protected by _STATE_LOCK.
_ACTIVE_RUNS: dict[str, int] = {}


def _node_lock(node_id: str) -> threading.Lock:
    """Return (creating if needed) a per-node Lock for state mutations."""
    with _STATE_LOCK:
        if node_id not in _NODE_LOCKS:
            _NODE_LOCKS[node_id] = threading.Lock()
        return _NODE_LOCKS[node_id]


def _prune_state() -> None:
    """Kursat NOTE!!! 2026-06-11 — Risk 2 fix (prune-during-execution): safe LRU eviction.
    Evict the least-recently-used node-state entries when _STATE exceeds
    _MAX_STATE_NODES.  Must be called while holding _STATE_LOCK.

    A node is considered safe to evict only when it has no active run in
    progress (checked via _ACTIVE_RUNS) AND its per-node lock is not held.
    The per-node lock guards Step 1 and Step 3; _ACTIVE_RUNS guards the gap
    between them (Step 2 — GPU sampling — during which the lock is released).

    Stuck-counter fallback: if an unhandled exception during Step 2 prevents
    the _ACTIVE_RUNS decrement, the counter stays at >0 indefinitely.  To
    avoid keeping stale state forever, nodes whose last_touch is older than
    _PRUNE_STUCK_TIMEOUT_S are evicted regardless of their counter.
    """
    _STUCK_TIMEOUT = 300.0  # seconds — treat counter as stuck after 5 min of inactivity
    _now = time.time()

    while len(_STATE) > _MAX_STATE_NODES:
        # Prefer candidates with no active run and unlocked per-node lock.
        candidates = {
            nid: st for nid, st in _STATE.items()
            if _ACTIVE_RUNS.get(nid, 0) == 0
            and not (_NODE_LOCKS.get(nid) and _NODE_LOCKS[nid].locked())
        }
        if not candidates:
            # All nodes appear active.  Fall back to evicting any node whose
            # last_touch is old enough that its _ACTIVE_RUNS counter is likely
            # stuck (exception during Step 2 skipped the decrement).
            candidates = {
                nid: st for nid, st in _STATE.items()
                if (_now - st.get("last_touch", 0.0)) > _STUCK_TIMEOUT
            }
        if not candidates:
            # Every node is either genuinely active or recently touched — give up.
            break
        oldest = min(candidates, key=lambda nid: _STATE[nid].get("last_touch", 0.0))
        del _STATE[oldest]
        _NODE_LOCKS.pop(oldest, None)
        _ACTIVE_RUNS.pop(oldest, None)


def _active_run_inc(node_id: str) -> None:
    """Mark a node as in-flight for pruning purposes."""
    with _STATE_LOCK:
        _ACTIVE_RUNS[node_id] = _ACTIVE_RUNS.get(node_id, 0) + 1


def _active_run_dec(node_id: str) -> None:
    """Clear one in-flight mark. Safe to call after partial failures."""
    with _STATE_LOCK:
        _cnt = _ACTIVE_RUNS.get(node_id, 0) - 1
        if _cnt > 0:
            _ACTIVE_RUNS[node_id] = _cnt
        else:
            _ACTIVE_RUNS.pop(node_id, None)


def _guider_sample(
        temp_g,
        noise: torch.Tensor,
        latent: torch.Tensor,
        sampler,
        sigmas: torch.Tensor,
        *,
        denoise_mask: torch.Tensor | None = None,
        callback=None,
        disable_pbar: bool = False,
        seed: int | None = None,
) -> torch.Tensor:
    """ Kursat note!!!
    Device-safe wrapper around guider.sample().

    ComfyUI's built-in ``CFGGuider.sample()`` moves noise, latent, and
    denoise_mask to the model's load device before sampling.  Some
    third-party extensions (e.g. ComfyUI-NAG-Extended) override
    ``inner_sample`` without repeating that movement, so CPU tensors
    survive into the k-sampler's inpaint path where they collide with GPU
    tensors and raise a device-mismatch RuntimeError.

    Moving everything to ``load_device`` here is a safe no-op when the
    built-in already does it, and fixes the crash for extensions that don't.
    """
    device = temp_g.model_patcher.load_device
    noise = noise.to(device)
    latent = latent.to(device)
    if denoise_mask is not None:
        denoise_mask = denoise_mask.to(device)
    samples = temp_g.sample(
        noise, latent, sampler, sigmas,
        denoise_mask=denoise_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )

    return samples.to(
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )


def _guider_with_conds(guider, positive, negative):
    g = copy.copy(guider)
    # Kursat NOTE!!! 2026-06-11 — Risk 5 fix (TypeError masking) + variadic-signature edge case:
    # Determine arity from the method signature before calling, so a
    # TypeError raised *inside* set_conds (e.g. malformed conditioning
    # data) is never confused with a wrong-argument-count error and
    # silently swallowed.  We count required positional parameters
    # (excluding self) to decide which overload to use.
    _takes_two: bool | None = None
    try:
        _sig = inspect.signature(g.set_conds)
        _params = _sig.parameters.values()
        # A VAR_POSITIONAL (*args) or VAR_KEYWORD (**kwargs) parameter means
        # the callable accepts any number of arguments, so two-arg call is
        # always safe.  Without that, count accepted positional parameters.
        # (Previously _n_req == 0 for *args signatures was mis-classified as
        # "one-argument guider" because 0 >= 2 is False — now detected first.)
        _has_var = any(
            _p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for _p in _params
        )
        if _has_var:
            _takes_two = True  # variadic — two-arg call is always accepted
        else:
            _n_pos = sum(
                1 for _p in _params
                if _p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            )
            _takes_two = _n_pos >= 2
    except (ValueError, TypeError):
        pass  # C extension or other un-introspectable callable — fall through

    if _takes_two is True:
        g.set_conds(positive, negative)   # comfy.samplers.CFGGuider / variadic
    elif _takes_two is False:
        g.set_conds(positive)             # comfy.samplers.BaseGuider / BasicGuider
    else:
        # Signature introspection failed — last resort: try/except on arity
        # only.  Re-raise anything that isn't a call-signature mismatch.
        try:
            g.set_conds(positive, negative)
        except TypeError as _exc:
            _msg = str(_exc).lower()
            if "argument" in _msg or "positional" in _msg or "takes" in _msg:
                g.set_conds(positive)
            else:
                raise
    return g


def _truncate_sigmas_for_denoise(sigmas: torch.Tensor, denoise: float) -> torch.Tensor:
    if denoise >= 1.0:
        return sigmas
    if denoise <= 0.0:
        # Return just the terminal sigma so the sampler does nothing.
        return sigmas[-1:].new_zeros(2)
    n_total = len(sigmas) - 1  # number of timesteps in the schedule
    n_refine = max(1, round(n_total * denoise))
    return sigmas[-(n_refine + 1):]


# Max number of latents to keep in the undo stack per node. Each FLUX 2
# latent at 832x1776 is ~180 KB (bf16). However, the history also caches
# the decoded pixel tensor when available (~17 MB per entry at float32).
# 10 entries can therefore use up to ~176 MB per active node. Reduce this
# value on low-VRAM / low-RAM systems.
_HISTORY_CAP: int = 10

# Kursat NOTE!!! 2026-06-11 — Risk 4 fix (unbounded _STATE growth): LRU cap.
# Maximum number of distinct node-state entries (_STATE / _NODE_LOCKS) to
# keep in memory.  Deleted nodes, copied nodes, and reloaded workflows
# accumulate stale entries that are never explicitly freed.  When this cap
# is exceeded the least-recently-used entry is evicted.  100 covers every
# realistic workflow; raise it if you routinely use more Angelo nodes in
# one session.
_MAX_STATE_NODES: int = 100

# Valid resize methods for the Fine Upscaling crop. All routed through
# comfy.utils.common_upscale which accepts both 4D image tensors and 4D
# latent tensors. lanczos is image-quality; bislerp is latent-aware.
# Default nearest-exact preserves exact sample values (good for
# latents); for the pixel-space path lanczos / bicubic typically look
# better. The user picks one for both ops in the Fine Upscale flow.
_FINE_UPSCALE_RESIZE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp", "lanczos"]

# Smart Guided Inpaint: maps a location dropdown LABEL (set by the JS
# location selector) to a natural-language prefix that's prepended to
# the Area Prompt text before CLIP encoding. The labels here are the
# single source of truth — the JS dropdown lists exactly these keys.
# Order is preserved (py3.7+ dicts) so the JS can build its dropdown
# from the same list via the widget's tooltip / a mirrored array.
_GUIDED_LOCATION_PREFIXES = {
    "(none)": "",
    "Whole image": "Across the whole image, ",
    "Top left": "In the top left of the image, ",
    "Top middle": "In the top middle of the image, ",
    "Top right": "In the top right of the image, ",
    "Middle left": "On the left middle of the image, ",
    "Center": "In the center of the image, ",
    "Middle right": "On the right middle of the image, ",
    "Bottom left": "In the bottom left of the image, ",
    "Bottom middle": "In the bottom middle of the image, ",
    "Bottom right": "In the bottom right of the image, ",
    "Left edge": "Along the left edge of the image, ",
    "Right edge": "Along the right edge of the image, ",
    "Top edge": "Along the top edge of the image, ",
    "Bottom edge": "Along the bottom edge of the image, ",
    "Top half": "In the top half of the image, ",
    "Bottom half": "In the bottom half of the image, ",
    "Left half": "In the left half of the image, ",
    "Right half": "In the right half of the image, ",
    "Top of the image": "At the top of the image, ",
    "Bottom of the image": "At the bottom of the image, ",
}


def _latent_fingerprint(latent: torch.Tensor) -> str:
    """Quick non-cryptographic fingerprint of an incoming latent.

    Used to detect when the upstream KSampler has produced a fresh
    latent (e.g. user changed the prompt + re-queued), so we can
    automatically reset the cache instead of layering refinements
    onto a now-irrelevant base.

    Samples are spaced evenly across the *entire* tensor (not just
    the first N elements) to avoid bias toward the beginning, which
    could miss changes that only affect the tail of the flattened
    tensor (e.g. changes in image corners stored in row-major order).
    """
    flat = latent.detach().to(torch.float32).flatten()
    n = min(flat.numel(), 1024)
    if flat.numel() <= n:
        sample = flat
    else:
        indices = torch.linspace(0, flat.numel() - 1, n, dtype=torch.long, device=flat.device)
        sample = flat[indices]
    h = hashlib.sha1()
    h.update(str(tuple(latent.shape)).encode())
    h.update(sample.cpu().numpy().tobytes())
    return h.hexdigest()


def _parse_stroke_points(raw: str) -> list[tuple[float, float]]:
    """Parse the JS-set stroke_points widget. Empty / malformed → []."""
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    out = []
    if not isinstance(data, list):
        return []
    for item in data:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                out.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return out


def _stroke_mask_latent(
        latent_h: int,
        latent_w: int,
        stroke_points_pixel: list[tuple[float, float]],
        r_latent: float,
        scale_x: float,
        scale_y: float,
        device: torch.device,
) -> torch.Tensor:
    """Vectorised union of circles in latent space, one circle per
    point in stroke_points_pixel. Points come in image-pixel coords;
    we scale them to latent space here using the per-axis ratios.

    Memory: one [N_points, H, W] float tensor briefly. For N=200, a
    typical FLUX 2 latent (~52x111), that's about 4 MB peak — fine.
    """
    if not stroke_points_pixel:
        return torch.zeros((1, latent_h, latent_w), device=device, dtype=torch.float32)

    pts = torch.tensor(stroke_points_pixel, device=device, dtype=torch.float32)
    cx = pts[:, 0] * scale_x  # [N]
    cy = pts[:, 1] * scale_y  # [N]

    ys = torch.arange(latent_h, device=device, dtype=torch.float32).view(1, -1, 1)
    xs = torch.arange(latent_w, device=device, dtype=torch.float32).view(1, 1, -1)
    cxv = cx.view(-1, 1, 1)
    cyv = cy.view(-1, 1, 1)
    dist2 = (xs - cxv) ** 2 + (ys - cyv) ** 2

    circles = (dist2 <= r_latent * r_latent).to(torch.float32)  # [N, H, W]
    mask = circles.amax(dim=0)  # union via max
    return mask.unsqueeze(0)  # [1, H, W]


def _parse_rect_points(raw: str) -> tuple[float, float, float, float] | None:
    """Parse the JS-set rect_points widget — JSON list of one or more
    [x1, y1, x2, y2] entries in image-pixel coords. We use only the
    LAST rectangle (the most recent drag). In practice the JS always
    writes a single-item list; the list format is kept for forward
    compatibility. Returns None for empty / malformed input.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, list) or not data:
        return None
    last = data[-1]
    if not isinstance(last, (list, tuple)) or len(last) < 4:
        return None
    try:
        return (float(last[0]), float(last[1]), float(last[2]), float(last[3]))
    except (TypeError, ValueError):
        return None


def _rect_mask_latent(
        latent_h: int,
        latent_w: int,
        rect_pixel: tuple[float, float, float, float],
        scale_x: float,
        scale_y: float,
        device: torch.device,
) -> torch.Tensor:
    """Build a [1, H, W] filled-rectangle mask in latent space.

    rect_pixel is (x1, y1, x2, y2) in image-pixel coords; corners
    aren't required to be ordered (the user may drag in any
    direction). Output is clamped to the latent bounds.
    """
    x1, y1, x2, y2 = rect_pixel
    x_lo_p, x_hi_p = min(x1, x2), max(x1, x2)
    y_lo_p, y_hi_p = min(y1, y2), max(y1, y2)

    xlat_lo = max(0, min(latent_w, int(round(x_lo_p * scale_x))))
    xlat_hi = max(0, min(latent_w, int(round(x_hi_p * scale_x))))
    ylat_lo = max(0, min(latent_h, int(round(y_lo_p * scale_y))))
    ylat_hi = max(0, min(latent_h, int(round(y_hi_p * scale_y))))

    mask = torch.zeros((1, latent_h, latent_w), device=device, dtype=torch.float32)
    if xlat_hi > xlat_lo and ylat_hi > ylat_lo:
        mask[0, ylat_lo:ylat_hi, xlat_lo:xlat_hi] = 1.0
    return mask


def _parse_seg_polygons(raw: str):
    """Parse the seg_polygon widget — a JSON list of polygons, each a flat
    [x0,y0,x1,y1,...] coord list in image-pixel space (a SAM 3 / YOLO
    detection's silhouette). Returns the list, or None if empty/invalid."""
    if not raw or not str(raw).strip():
        return None
    try:
        polys = json.loads(raw)
    except (ValueError, TypeError):
        return None
    if not isinstance(polys, list) or not polys:
        return None
    return polys


def _polygons_mask_latent(
        latent_h: int,
        latent_w: int,
        polygons_pixel,
        scale_x: float,
        scale_y: float,
        device: torch.device,
) -> torch.Tensor:
    """Rasterise one or more silhouette polygons (image-pixel coords) into
    a filled [1, H, W] latent-space mask (their union)."""
    img = _PILImage.new("L", (latent_w, latent_h), 0)
    draw = _PILImageDraw.Draw(img)
    for poly in (polygons_pixel or []):
        if not poly or len(poly) < 6:
            continue
        pts = [
            (float(poly[i]) * scale_x, float(poly[i + 1]) * scale_y)
            for i in range(0, len(poly) - 1, 2)
        ]
        draw.polygon(pts, fill=255)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...].to(device)


def _raster_mask_latent(latent_h, latent_w, png_b64, device):
    """Decode a base64-PNG touch-up mask (image-pixel resolution, white =
    masked) into a [1, H, W] latent-space mask. The Detect Shift/Alt brush
    produces this — a raster handles brushed holes / unions that a polygon
    silhouette can't. Resized straight to the latent grid (no scale args)."""
    raw = base64.b64decode(png_b64)
    img = _PILImage.open(io.BytesIO(raw)).convert("L")
    img = img.resize((latent_w, latent_h), _BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...].to(device)


def _mask_bbox_latent(mask: torch.Tensor) -> tuple[int, int, int, int] | None:
    """Tight latent-space bbox of non-zero mask values. Returns
    (y_min, y_max, x_min, x_max) or None if the mask is empty.

    Threshold of 0.01 includes the feathered edge — bbox covers the
    full soft-edge region not just the binary interior.
    """
    m = mask[0] if mask.dim() == 3 else mask
    nz = m > 0.01
    if not nz.any():
        return None
    rows = nz.any(dim=1)
    cols = nz.any(dim=0)
    ridx = rows.nonzero(as_tuple=False).squeeze(-1)
    cidx = cols.nonzero(as_tuple=False).squeeze(-1)
    return (
        int(ridx[0].item()),
        int(ridx[-1].item()) + 1,
        int(cidx[0].item()),
        int(cidx[-1].item()) + 1,
    )


def _fine_upscale_factor(
        bbox_w_latent: int,
        bbox_h_latent: int,
        scale_x: float,
        scale_y: float,
        target_mp: float,
        max_linear: float,
) -> float:
    """Linear scale factor to apply to the cropped latent so that the
    crop is processed at ≥ target_mp (in image-pixel-equivalent terms),
    clamped to max_linear. Returns 1.0 when the crop already meets
    target — no upscale needed."""
    if scale_x <= 0 or scale_y <= 0:
        return 1.0
    bbox_w_pix = bbox_w_latent / scale_x
    bbox_h_pix = bbox_h_latent / scale_y
    current_mp = bbox_w_pix * bbox_h_pix / 1_000_000.0
    if current_mp <= 0 or current_mp >= target_mp:
        return 1.0
    needed = math.sqrt(target_mp / current_mp)
    return min(needed, max_linear)


def _resize_latent(t: torch.Tensor, target_h: int, target_w: int, method: str) -> torch.Tensor:
    """Resize the spatial dims of a latent or mask tensor using one of
    ComfyUI's standard latent-resize methods. Accepts [C,H,W], [B,C,H,W],
    or [1,H,W] (mask). Returns the same rank as input.

    `method` is one of _FINE_UPSCALE_RESIZE_METHODS. Routes through
    comfy.utils.common_upscale so bislerp + lanczos custom paths work."""
    method = method if method in _FINE_UPSCALE_RESIZE_METHODS else "nearest-exact"
    if t.dim() == 3:
        t4 = t.unsqueeze(0)
        out = comfy.utils.common_upscale(t4, target_w, target_h, method, "disabled")
        return out.squeeze(0)
    if t.dim() == 4:
        return comfy.utils.common_upscale(t, target_w, target_h, method, "disabled")
    raise ValueError(f"_resize_latent: unexpected ndim {t.dim()}")


# ----- VAE boundary -----
# Every latent->pixel and pixel->latent conversion in the node routes
# through these two functions. Centralising them means model-family
# quirks live in ONE place instead of at ~7 scattered call sites. The
# motivating case is Qwen Image Edit / Wan-derived VAEs, whose latents
# carry an extra temporal axis ([B, C, T, H, W] with T=1) that the rest
# of the node — and PIL's previewer — expect to be absent ([B, C, H, W]).
# Both helpers are thin pass-throughs today; this is the seam where that
# 5D normalisation will land without disturbing any caller.
def _vae_decode(vae, latent: torch.Tensor) -> torch.Tensor:
    """Decode a latent to pixels. Single decode chokepoint — see the
    VAE-boundary note above. Always returns a 4D image batch
    (B, H, W, C) float in [0, 1].

    Temporal/video VAEs (Qwen Image Edit, Wan) keep a frame axis: their
    latents are 5D ([B, C, T, H, W]) and `vae.decode` accordingly returns
    a 5D frame stack ([B, T, H, W, C] — ComfyUI moves channels last). The
    rest of the node, and ComfyUI's PreviewImage/PIL path, only understand
    4D image batches, so fold the frame axis into the batch dim. For image
    editing T is 1, so this is just dropping the singleton frame axis; if a
    future model ever produces T>1 the frames surface as extra batch items
    rather than crashing. The latent is passed through to `vae.decode`
    untouched — the video VAE wants its native 5D input — we only normalise
    the *pixels* it returns."""
    image = vae.decode(latent)
    if image.ndim == 5:
        b, t, h, w, c = image.shape
        image = image.reshape(b * t, h, w, c)
    return image


def _vae_encode(vae, pixels: torch.Tensor) -> torch.Tensor:
    """Encode pixels to latent samples. Single encode chokepoint —
    counterpart to _vae_decode. See the VAE-boundary note above.

    Deliberately returns the VAE's native latent shape WITHOUT collapsing
    it: a temporal/video VAE (Qwen, Wan) returns a 5D latent
    ([B, C, T, H, W]) and the sampler + model require that 5D shape to flow
    through unchanged (comfy.sample.sample is ndim-agnostic and prepare_noise
    matches the latent's shape exactly). Squeezing the frame axis here would
    break Qwen sampling — do not add a squeeze."""
    return vae.encode(pixels)


def _refine_with_fine_upscaling(
        *,
        guider,
        sampler,
        sigmas: torch.Tensor,
        vae,
        current: torch.Tensor,  # [B, C, H_lat, W_lat] cached full-res latent
        current_pixels: torch.Tensor | None,
        # [B, H_pix, W_pix, C] cached full-res pixels to avoid redundant VAE decode
        mask: torch.Tensor,  # [1, H_lat, W_lat] feathered mask, latent res
        scale_x: float,
        scale_y: float,
        target_mp: float,
        max_linear: float,
        resize_method: str,
        context_pad_pixel: int,
        inpainting_mode: str,
        seed: int,
        positive,
        negative,
        denoise: float,
        callback,
        disable_pbar: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pixel-space crop + upscale + VAE encode + refine + VAE decode +
    downscale + composite + VAE encode. The latent-space crop+upscale
    approach smears bilinearly-interpolated latents into a low-freq
    starting state that the model can't recover detail from. Going
    through pixel space (where there's an image-upscale toolkit that's
    been tuned for natural images) and re-encoding gives the model a
    "natural" latent at the higher resolution to denoise from.

    Returns a tuple of (new_latent, new_pixels). Returns (current, current_pixels)
    if the mask bbox is empty / degenerate.
    """
    bbox = _mask_bbox_latent(mask)
    if bbox is None:
        return current, current_pixels
    y0_tight, y1_tight, x0_tight, x1_tight = bbox

    # Apply context padding: grow the bbox outward by context_pad_pixel
    # in every direction (clamped to the latent boundaries). This is
    # the area the model SEES during refine. The painted-shape mask
    # stays unchanged — areas inside the padded bbox but outside the
    # painted shape have mask=0 in the cropped tensor, so the noise-
    # injection inpaint preserves them as context (the model uses them
    # to inform what to draw inside the mask, but doesn't overwrite
    # them). All downstream code uses the PADDED bbox.
    H_lat = current.shape[-2]
    W_lat = current.shape[-1]
    pad_lat_y = max(0, round(context_pad_pixel * scale_y))
    pad_lat_x = max(0, round(context_pad_pixel * scale_x))
    y0 = max(0, y0_tight - pad_lat_y)
    y1 = min(H_lat, y1_tight + pad_lat_y)
    x0 = max(0, x0_tight - pad_lat_x)
    x1 = min(W_lat, x1_tight + pad_lat_x)

    bbox_h_lat = y1 - y0
    bbox_w_lat = x1 - x0
    if bbox_h_lat <= 0 or bbox_w_lat <= 0:
        return current, current_pixels

    scale = _fine_upscale_factor(bbox_w_lat, bbox_h_lat, scale_x, scale_y, target_mp, max_linear)
    if scale <= 1.0 and inpainting_mode != "Smart Inpaint":
        # Refine with no upscale needed — fall back to the standard latent-space
        # noise-injection inpaint. Avoids unnecessary VAE round-trips when the
        # painted region already meets the MP target.
        #
        # Smart Inpaint must NOT take this shortcut. It needs the crop +
        # reference_latents + masked-zero treatment below regardless of rect
        # size; skipping it made a LARGE rectangle (already at/above the MP
        # target, so scale<=1.0 — roughly >1024px on FLUX 2) degrade to a
        # whole-latent edit with NO crop reference, so the model worked on the
        # whole image instead of the selected rect.
        print(f"[Angelo fine-upscale] scale={scale:.2f} ≤ 1.0 — using latent-space path (no VAE round-trip)")
        noise = comfy.sample.prepare_noise(current, seed, None)
        refine_sigmas = _truncate_sigmas_for_denoise(sigmas, denoise)
        temp_g = _guider_with_conds(guider, positive, negative)
        new_latent = _guider_sample(
            temp_g, noise, current, sampler, refine_sigmas,
            denoise_mask=mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        # _guider_sample already moves the result to intermediate_device().
        # Return None for pixels because the latent was modified directly;
        # this forces a fresh VAE decode for the preview in the main run() method.
        return new_latent, None

    # Smart Inpaint with a large rectangle still crops + references the selected
    # region; it just doesn't upscale (and must never downscale) — clamp the
    # factor to identity so the crop is taken at native resolution.
    if inpainting_mode == "Smart Inpaint":
        scale = max(1.0, scale)

    # ----- VAE decode the full cached latent → cached pixels -----
    # Optimization: Reuse cached pixels if available to prevent VAE degradation
    # (loss of high-frequency details) across multiple consecutive edits.
    if current_pixels is not None:
        cached_pixels = current_pixels
    else:
        cached_pixels = _vae_decode(vae, current)  # (B, H_pix, W_pix, C) float [0,1]

    H_pix = cached_pixels.shape[1]
    W_pix = cached_pixels.shape[2]
    # Pixel-per-latent ratio per axis. Use rounded float division instead of
    # integer // to get the best approximation for exotic VAEs where the ratio
    # is not exactly integer (e.g. 15.8 rounds to 16, while // gives 15).
    px_per_lat_y = max(1, round(H_pix / current.shape[-2]))
    px_per_lat_x = max(1, round(W_pix / current.shape[-1]))

    # Pixel-space bbox derived from the latent-space bbox.
    y0_p = y0 * px_per_lat_y
    y1_p = y1 * px_per_lat_y
    x0_p = x0 * px_per_lat_x
    x1_p = x1 * px_per_lat_x
    bbox_h_p = y1_p - y0_p
    bbox_w_p = x1_p - x0_p

    # Upscaled target dims in pixel space. Snap each axis independently to
    # a multiple of that axis's VAE downscale factor so the subsequent VAE
    # encode produces a clean integer-dim latent. Using the per-axis ratio
    # (instead of max of both) prevents over-snapping in the smaller axis
    # on non-square VAEs (e.g. temporal models where px_per_lat_y ≠ px_per_lat_x).
    target_h_p = max(px_per_lat_y, math.ceil(bbox_h_p * scale / px_per_lat_y) * px_per_lat_y)
    target_w_p = max(px_per_lat_x, math.ceil(bbox_w_p * scale / px_per_lat_x) * px_per_lat_x)

    print(f"[Angelo fine-upscale] bbox_lat=(h={bbox_h_lat}, w={bbox_w_lat}) "
          f"bbox_px=(h={bbox_h_p}, w={bbox_w_p}) scale={scale:.2f} "
          f"target_px=(h={target_h_p}, w={target_w_p}) "
          f"resize={resize_method} max_linear={max_linear} "
          f"vae_ratio=(x={px_per_lat_x}, y={px_per_lat_y})")

    # ----- Crop pixel image + upscale in pixel space -----
    pixel_crop = cached_pixels[:, y0_p:y1_p, x0_p:x1_p, :]  # (B, h, w, C)
    # common_upscale expects (B, C, H, W) — permute, upscale, permute back.
    pixel_crop_chw = pixel_crop.movedim(-1, 1)
    pixel_crop_up_chw = comfy.utils.common_upscale(
        pixel_crop_chw, target_w_p, target_h_p, resize_method, "disabled",
    )
    pixel_crop_up = pixel_crop_up_chw.movedim(1, -1)  # back to (B, H, W, C)

    # ----- VAE encode the upscaled pixel crop → latent at high res -----
    latent_up = _vae_encode(vae, pixel_crop_up)
    target_h_lat = latent_up.shape[-2]
    target_w_lat = latent_up.shape[-1]

    # ----- Build mask at the upscaled latent resolution -----
    # Mask resizing always uses bilinear regardless of the user's choice.
    # The user's resize_method is for the IMAGE content upscale (where
    # lanczos / bicubic / etc. have real quality differences). The mask
    # is a 1-channel feathered alpha where we just want smooth values;
    # lanczos's grayscale-branch returns a transposed 3D tensor (PIL
    # quirk) and bislerp's spherical-vector math is semantically wrong
    # on a single channel.
    mask_crop = mask[..., y0:y1, x0:x1].contiguous()
    mask_crop_up = _resize_latent(mask_crop, target_h_lat, target_w_lat, "bilinear").clamp(0.0, 1.0)

    # ===== Smart Inpaint pre-processing on the upscaled patch =====
    # Klein 9B's edit branch only activates when reference_latents is
    # present on the conditioning. We then zero the masked area so the
    # sampler regenerates that region from full noise at sigma_max
    # (the denoise=1.0 lock makes this clean: every pixel in the
    # painted rect is brand-new content, with the surrounding context
    # band restored each step by the noise_mask compositing). The
    # reference uses the PRE-ZERO upscaled patch so Klein still sees
    # what was there before we blanked it.
    # POSITIVE ONLY — putting reference_latents on negative would tell
    # CFG>1 samplers to steer AWAY from the reference scene. Non-edit
    # models ignore the field, so this is harmless on any checkpoint.
    #
    # append=False (REPLACE, not append): the reference must be ONLY this
    # upscaled crop. When the Area Prompt is empty, refine_positive falls back
    # to the node's `positive` input, which in a Klein edit workflow already
    # carries reference_latents = the WHOLE source image (from an upstream
    # ReferenceLatent node). append=True stacked the crop onto that whole-image
    # reference, and the whole-image one dominated — so the patch reproduced
    # the entire original scene instead of editing the selected region.
    # Replacing guarantees Klein sees the crop and nothing else.
    # Mask used for the SAMPLING step (both the masked-zero and the noise_mask).
    # Defaults to the feathered mask; hardened to binary for Smart Inpaint on
    # 5D temporal latents — see the note in the Smart Inpaint block below.
    sample_mask = mask_crop_up
    if inpainting_mode == "Smart Inpaint":
        reference_latent = latent_up.clone()
        positive = node_helpers.conditioning_set_values(
            positive, {"reference_latents": [reference_latent]}, append=False,
        )
        # Feather goes in the COMPOSITE, not the denoise mask — for Qwen/Wan.
        #
        # FLUX/Klein (4D, ~zero-mean latents): a soft mask works directly as
        # both the masked-zero and the noise_mask; their inpaint blend handles a
        # soft boundary cleanly, so keep the feathered mask.
        #
        # Qwen Image Edit / Wan (5D temporal latents): sample with a HARD
        # (binary) mask. These models have no clean soft-mask-during-denoise
        # behaviour — the community-standard Qwen-edit inpaint recipe is "blank
        # the region, regenerate, then composite back with a feathered blend",
        # NOT feathering the denoise mask. A soft noise_mask at denoise=1.0 on a
        # non-zero-mean latent space distorts exactly the feather band (the
        # symptom: artifacts only where the feathering happens). The smooth
        # visible edge is still produced downstream by the feathered PIXEL
        # composite + final latent blend (both use the full-res feathered
        # `mask`), so the feather is preserved without the sampling artifacts.
        if latent_up.ndim == 5:
            sample_mask = (mask_crop_up >= 0.5).to(mask_crop_up.dtype)
        # Zero the masked region. For 5D temporal latents [B,C,T,H,W] the
        # mask is [1,H,W] and needs two extra leading dims to broadcast
        # correctly; for standard 4D latents [B,C,H,W] one unsqueeze suffices.
        if latent_up.ndim == 5:
            latent_up = (1.0 - sample_mask.unsqueeze(0).unsqueeze(0)) * latent_up
        else:
            latent_up = (1.0 - sample_mask.unsqueeze(0)) * latent_up

    # ----- Refine via noise-injection inpaint on the upscaled latent -----
    noise = comfy.sample.prepare_noise(latent_up, seed, None)
    refine_sigmas = _truncate_sigmas_for_denoise(sigmas, denoise)
    temp_g = _guider_with_conds(guider, positive, negative)
    refined_latent_up = _guider_sample(
        temp_g, noise, latent_up, sampler, refine_sigmas,
        denoise_mask=sample_mask,
        callback=callback,
        disable_pbar=disable_pbar,
        seed=seed,
    )
    # _guider_sample already moves the result to intermediate_device().

    # ----- VAE decode refined latent → high-res pixel patch -----
    refined_pixel_up = _vae_decode(vae, refined_latent_up)  # (B, target_h_p, target_w_p, C)

    # ----- Downscale refined patch back to original bbox pixel size -----
    refined_pixel_up_chw = refined_pixel_up.movedim(-1, 1)
    refined_pixel_chw = comfy.utils.common_upscale(
        refined_pixel_up_chw, bbox_w_p, bbox_h_p, resize_method, "disabled",
    )
    refined_pixel = refined_pixel_chw.movedim(1, -1)  # (B, bbox_h_p, bbox_w_p, C)

    # ----- Composite refined patch into the cached pixel image -----
    # Build a pixel-space alpha by resizing the latent feathered mask to
    # full pixel resolution, cropping to the bbox. Always bilinear for
    # the same reasons as the mask upscale above — lanczos's grayscale
    # path is broken, bislerp doesn't apply to 1-channel.
    mask_4d = mask.unsqueeze(0)  # [1, 1, H_lat, W_lat]
    pixel_mask = comfy.utils.common_upscale(
        mask_4d, W_pix, H_pix, "bilinear", "disabled",
    ).clamp(0.0, 1.0)  # [1, 1, H_pix, W_pix]
    pixel_alpha_crop = pixel_mask[0, 0, y0_p:y1_p, x0_p:x1_p]  # [bbox_h_p, bbox_w_p]
    pixel_alpha_crop = pixel_alpha_crop.unsqueeze(0).unsqueeze(-1)  # [1, h, w, 1]

    new_pixels = cached_pixels.clone()
    pixel_orig_crop = cached_pixels[:, y0_p:y1_p, x0_p:x1_p, :]
    composited = refined_pixel * pixel_alpha_crop + pixel_orig_crop * (1.0 - pixel_alpha_crop)
    new_pixels[:, y0_p:y1_p, x0_p:x1_p, :] = composited

    # ----- VAE encode the composited full image → encoded latent -----
    encoded_latent = _vae_encode(vae, new_pixels)

    # ----- Blend in LATENT space using the feathered mask as alpha -----
    # The VAE encode is lossy, so naively returning encoded_latent would
    # mean the *unaltered* regions of the image drift slightly with every
    # Fine Upscale click. Avoidable: keep the original cached latent
    # outside the mask, take the encoded latent inside the mask. Mask is
    # already feathered, so the transition is smooth. Now unaltered
    # regions stay bit-exact across successive clicks; only the masked
    # area accumulates any VAE-roundtrip cost (and it gets a fresh
    # refine each click anyway, so any drift there is overwritten).
    #
    # For 5D temporal latents [B,C,T,H,W] the mask [1,H,W] needs two
    # extra leading dims to broadcast correctly; for 4D [B,C,H,W] one
    # unsqueeze is enough.
    if encoded_latent.ndim == 5:
        alpha_lat = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H_lat, W_lat]
    else:
        alpha_lat = mask.unsqueeze(0)  # [1, 1, H_lat, W_lat]
    new_latent = encoded_latent * alpha_lat + current * (1.0 - alpha_lat)

    return new_latent, new_pixels


def _circle_mask_latent_direct(
        latent_h: int,
        latent_w: int,
        cx_latent: float,
        cy_latent: float,
        r_latent: float,
        device: torch.device,
) -> torch.Tensor:
    """Build a binary [1, latent_h, latent_w] mask with a filled circle
    centred on (cx_latent, cy_latent) of radius `r_latent`, all in
    latent-space coordinates. The caller is responsible for converting
    pixel-space click coords to latent space using the correct per-axis
    scale (image_dim_pixel → latent_dim) so this function stays VAE-
    agnostic.
    """
    ys = torch.arange(latent_h, device=device, dtype=torch.float32).view(-1, 1)
    xs = torch.arange(latent_w, device=device, dtype=torch.float32).view(1, -1)
    dist2 = (xs - cx_latent) ** 2 + (ys - cy_latent) ** 2
    mask = (dist2 <= r_latent * r_latent).to(torch.float32)
    return mask.unsqueeze(0)  # [1, H, W]


def _gaussian_blur_2d(mask: torch.Tensor, sigma_latent: float) -> torch.Tensor:
    """Separable gaussian blur on a [B, H, W] or [H, W] mask tensor.

    sigma_latent is in latent-space units.
    """
    if sigma_latent <= 0:
        return mask
    # Kernel covers ~±3σ. Force odd size.
    ksize = int(2 * math.ceil(3 * sigma_latent) + 1)
    half = ksize // 2
    x = torch.arange(ksize, device=mask.device, dtype=torch.float32) - half
    k1d = torch.exp(-0.5 * (x / sigma_latent) ** 2)
    k1d = k1d / k1d.sum()

    # Reshape mask to [N, 1, H, W]
    orig_ndim = mask.dim()
    if orig_ndim == 2:
        m = mask.unsqueeze(0).unsqueeze(0)
    elif orig_ndim == 3:
        m = mask.unsqueeze(1)
    elif orig_ndim == 4:
        m = mask
    else:
        raise ValueError(f"gaussian_blur_2d: unexpected mask ndim {orig_ndim}")

    kh = k1d.view(1, 1, 1, ksize)
    kv = k1d.view(1, 1, ksize, 1)

    m = torch.nn.functional.pad(m, (half, half, 0, 0), mode="replicate")
    m = torch.nn.functional.conv2d(m, kh)
    m = torch.nn.functional.pad(m, (0, 0, half, half), mode="replicate")
    m = torch.nn.functional.conv2d(m, kv)

    if orig_ndim == 2:
        return m.squeeze(0).squeeze(0)
    if orig_ndim == 3:
        return m.squeeze(1)
    return m


def _encode_loaded_image(vae, ref_json: str, resize_mode: str, target_mp: float):
    """Load an image the user picked via the Load Image button and
    VAE-encode it into a base latent.

    `ref_json` is a JSON {name, subfolder, type} ref returned by
    ComfyUI's /upload/image (falls back to treating the string as a bare
    input-dir filename). resize_mode is "keep" (native res) or "mp"
    (scaled to ~target_mp megapixels). In both cases dimensions are
    rounded to a multiple of 16 so any supported VAE (8x or 16x) is
    happy. Returns the latent samples tensor.
    """
    # Resolve the image reference to a path.
    name, subfolder, type_ = ref_json, "", "input"
    try:
        ref = json.loads(ref_json)
        name = ref.get("name") or ref.get("filename") or ""
        subfolder = ref.get("subfolder", "") or ""
        type_ = ref.get("type", "input") or "input"
    except (ValueError, TypeError):
        pass

    if type_ == "output":
        base_dir = folder_paths.get_output_directory()
    elif type_ == "temp":
        base_dir = folder_paths.get_temp_directory()
    else:
        base_dir = folder_paths.get_input_directory()
    path = os.path.normpath(os.path.join(base_dir, subfolder, name))
    # Guard against path-traversal: require path to be equal to or strictly
    # inside base_dir. str.startswith without the sep suffix would wrongly
    # pass e.g. "/tmp/comfy_input_evil" when base is "/tmp/comfy_input".
    safe_base = os.path.normpath(base_dir)
    if not (path == safe_base or path.startswith(safe_base + os.sep)):
        raise ValueError("Angelo: invalid loaded-image path")
    if not os.path.exists(path):
        raise ValueError(f"Angelo: loaded image not found: {name}")

    img = _PILImage.open(path)
    img = _PILImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size

    if resize_mode == "mp" and target_mp > 0:
        cur_mp = (w * h) / 1.0e6
        if cur_mp > 0:
            s = math.sqrt(target_mp / cur_mp)
            w = int(round(w * s))
            h = int(round(h * s))

    # Round down to a multiple of 16 (divisible by both 8x and 16x VAEs).
    w = max(16, (w // 16) * 16)
    h = max(16, (h // 16) * 16)
    if (w, h) != img.size:
        img = img.resize((w, h), _LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0  # (H, W, 3)
    pixels = torch.from_numpy(arr)[None, ...]  # (1, H, W, 3)
    samples = _vae_encode(vae, pixels[:, :, :, :3])
    return samples


def _decode_to_preview(vae, latent_samples: torch.Tensor):
    """Decode the latent and save to the temp directory in the same
    format PreviewImage uses, so we can return the same ui dict shape.

    Returns (image_tensor, list_of_image_refs). Each image ref is a
    {filename, subfolder, type} dict.
    """
    image = _vae_decode(vae, latent_samples)  # (B, H, W, C) float in [0, 1]

    # Reuse PreviewImage's save logic via a transient instance.
    previewer = comfy_nodes.PreviewImage()
    ui = previewer.save_images(image, filename_prefix="Angelo_preview")
    return image, ui["ui"]["images"]


class AngeloRefine:
    """Click-to-refine sampler. See module docstring."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                #   Kursat Note!!!
                #   BasicScheduler          → sigmas   (use denoise=1.0 here;
                #                             Angelo truncates internally for
                #                             the per-click refine denoise)
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),

                # ===== Sampler Mode controls (top of native widget area) =====
                # Visible in the node body. When `mode == "Sampler Mode"`, the
                # toolbar is greyed and canvas clicks do nothing — this widget
                # group is the active one, producing the base latent from the
                # incoming latent using guider + sampler + sigmas above. When
                # mode flips to Edit Mode, sampler_seed_control is auto-forced
                # to "fixed" so subsequent Queue presses don't regenerate the
                # base.
                "mode": (["Sampler Mode", "Edit Mode"], {"default": "Sampler Mode",
                                                         "tooltip": "Sampler Mode: AngeloRefine acts "
                                                                    "like a KSampler — generates the "
                                                                    "base latent from the incoming "
                                                                    "(usually empty) latent. Toolbar "
                                                                    "and canvas clicks are inert. "
                                                                    "Edit Mode: click / paint / drag "
                                                                    "to refine or inpaint the cached "
                                                                    "base. Switching to Edit Mode auto-"
                                                                    "locks sampler_seed_control to "
                                                                    "'fixed' so the base stays stable."}),
                "sampler_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                         "control_after_generate": False,
                                         "tooltip": "[Sampler Mode] Seed for the base generation. "
                                                    "Controlled by sampler_seed_control after each run."}),
                "sampler_seed_control": (["fixed", "increment", "decrement", "randomize"],
                                         {"default": "randomize",
                                          "tooltip": "[Sampler Mode] After-generation seed behaviour. "
                                                     "Defaults to randomize so each base generation gets "
                                                     "a fresh seed. Auto-forced to 'fixed' (locked to the "
                                                     "seed that produced the base) when you switch to Edit "
                                                     "Mode so refine clicks stay stable."}),

                # ===== Edit Mode controls (driven from toolbar) =====
                # Hidden from the native widget area; the toolbar above the
                # preview canvas drives these. seed_control follows the same
                # fixed/random/increment/decrement pattern as sampler_seed.
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                                 "control_after_generate": False,
                                 "tooltip": "[Edit Mode] Seed for the refine pass. Hidden — "
                                            "controlled by the Seed input on the toolbar."}),
                "seed_control": (["fixed", "increment", "decrement", "randomize"],
                                 {"default": "randomize",
                                  "tooltip": "[Edit Mode] After-click seed behaviour. Hidden — "
                                             "controlled by the Seed Ctrl dropdown on the toolbar. "
                                             "Defaults to randomize so each refine click produces a "
                                             "fresh variation rather than repeating the same result."}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                                      "tooltip": "How much of the sigma schedule to use for each "
                                                 "refine click (Edit Mode). The provided sigmas are "
                                                 "truncated to the last round(N×denoise)+1 values so "
                                                 "the sampler starts at the right noise level. "
                                                 "0.3 = subtle touch-up, 0.6 = real redo, "
                                                 "0.9+ = essentially regenerate that region. "
                                                 "Connect BasicScheduler(denoise=1.0) for best "
                                                 "truncation accuracy."}),

                "click_radius": ("INT", {"default": 96, "min": 8, "max": 1024, "step": 4,
                                         "tooltip": "Pixel-space radius of the refinement region. "
                                                    "Updated automatically by the JS click widget."}),
                "feather_radius": ("INT", {"default": 24, "min": 0, "max": 256, "step": 4,
                                           "tooltip": "Pixel-space gaussian blur applied to the mask "
                                                      "before sampling. Smooths the seam between the "
                                                      "refined region and the preserved surroundings. "
                                                      "Roughly half of click_radius is a good default."}),

                # JS-driven widgets. click_x/y in pixel space; -1 = no click yet.
                # click_seq increments per click to force ComfyUI to detect a change
                # and re-execute this node (otherwise identical inputs would skip).
                # image_w/h are the actual decoded image dimensions in pixels —
                # set by the JS whenever a fresh preview is loaded into the canvas,
                # so we can compute the correct pixel→latent scale without
                # hardcoding the VAE downscale factor (FLUX 2 is 16, FLUX 1 / SDXL
                # are 8, exotic VAEs vary).
                "click_x": ("INT", {"default": -1, "min": -1, "max": 16384}),
                "click_y": ("INT", {"default": -1, "min": -1, "max": 16384}),
                "click_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),
                "image_w": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "image_h": ("INT", {"default": 0, "min": 0, "max": 16384}),
                # Undo bookkeeping: undo_seq increments when the user clicks
                # the Undo button. Python pops the last refined latent off
                # the history stack on each new undo_seq value.
                "undo_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),

                "reset": ("BOOLEAN", {"default": False,
                                      "tooltip": "Tick + re-queue to discard the cached refined "
                                                 "latent and start over from the incoming latent."}),
                # DEPRECATED as a control — always-on now. Kept declared
                # (not removed) so its slot in widgets_values stays put;
                # deleting it would shift every later widget's position
                # and drift old saved workflows on load. Hidden in the UI;
                # run() ignores the value and always decodes the preview.
                "auto_decode": ("BOOLEAN", {"default": True,
                                            "tooltip": "(Deprecated — preview always decodes now.)"}),

                # STEP 2 of the Fine Upscaling re-introduction: JS toggle
                # bar drives these. Both widgets are hidden from the node
                # UI; the green "Fine Upscale" toggle and the small "MP"
                # numeric input on the bar above the preview canvas are
                # the user-facing surface. Python's run() still prints the
                # received values so we can verify the JS bar correctly
                # drives them. SAMPLING IS STILL UNCHANGED — neither value
                # is read by any code path below the print. Step 3 wires
                # the actual crop+upscale+refine.
                # Kursat NOTE!!! 2026-06-10 — Misleading tooltip fix: tooltip previously said
                # "bilinear-upscaled in latent space" which was wrong.  The actual
                # path crops the painted region in PIXEL space, upscales it with
                # the chosen PIL filter (lanczos by default), VAE-encodes the
                # result, refines, then composites back — deliberately avoiding
                # latent-space interpolation which smears low-frequency artefacts
                # the model cannot recover from (see _refine_with_fine_upscaling
                # docstring).  Tooltip updated to match reality.
                "fine_upscaling": ("BOOLEAN", {"default": False,
                                               "tooltip": "When ON, the painted region is cropped, "
                                                          "upscaled in PIXEL space (using the chosen "
                                                          "resize method) to hit min_megapixels, "
                                                          "VAE-encoded, refined, downscaled, and "
                                                          "composited back. Gives the model more "
                                                          "effective resolution on small painted "
                                                          "regions without the low-frequency smearing "
                                                          "of latent-space interpolation. Upscale "
                                                          "capped at max_upscale× linear internally. "
                                                          "Set via the Fine Upscale toggle above the "
                                                          "preview."}),
                "min_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1,
                                             "tooltip": "Target megapixels for the refine pass when "
                                                        "Fine Upscaling is on. Only used in that mode. "
                                                        "Set via the MP input above the preview."}),
                "max_upscale": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 16.0, "step": 0.5,
                                          "tooltip": "Hard cap on the linear upscale factor in Fine "
                                                     "Upscaling. 8.0 was the original internal default. "
                                                     "Lower values reduce smoothing artifacts from "
                                                     "extreme upscales at the cost of less effective "
                                                     "resolution gain. Set via the Max input above "
                                                     "the preview."}),
                "resize_method": (_FINE_UPSCALE_RESIZE_METHODS, {"default": "lanczos",
                                                                 "tooltip": "Pixel-space upscale method for "
                                                                            "Fine Upscale. lanczos is the "
                                                                            "default — sharpest detail recovery "
                                                                            "for natural images. bilinear is "
                                                                            "smooth (good for skin/faces); "
                                                                            "bicubic middle ground; nearest-"
                                                                            "exact preserves exact sample values; "
                                                                            "area/bislerp niche."}),
                "inpainting_mode": (["Refine", "Smart Inpaint", "Smart Guided Inpaint"], {"default": "Refine",
                                                                                          "tooltip": "How the painted region is treated.\n\n"
                                                                                                     "Refine — the painted region is partially "
                                                                                                     "denoised from the existing content. Best "
                                                                                                     "for refining what's already there (faces, "
                                                                                                     "hands, textures). Paint/click as normal.\n\n"
                                                                                                     "Smart Inpaint — adds NEW content where you "
                                                                                                     "drag a rectangle. Click+hold one corner, "
                                                                                                     "release at the opposite corner. Locks "
                                                                                                     "denoise=1.0, Fine Upscale=ON, Ctx Pad=0 "
                                                                                                     "(the right defaults for adding new subjects "
                                                                                                     "with an edit model — Klein 9B etc.). "
                                                                                                     "Reference_latents are auto-injected so the "
                                                                                                     "model's edit branch activates.\n\n"
                                                                                                     "Smart Guided Inpaint — no painting or boxes. "
                                                                                                     "Pick a location from the dropdown above the "
                                                                                                     "Area Prompt; it's prepended to your prompt "
                                                                                                     "(e.g. 'In the top left of the image, ...') "
                                                                                                     "and the edit model places the content there "
                                                                                                     "across the whole image. Locks denoise=1.0, "
                                                                                                     "Fine Upscale=OFF, Area Prompt=ON; press "
                                                                                                     "'Generate Guided Edit' to run."}),

                # Hidden — DOM location dropdown (Smart Guided Inpaint)
                # drives this. Holds a LABEL key from
                # _GUIDED_LOCATION_PREFIXES; run() maps it to the prompt
                # prefix at encode time.
                "guided_location": ("STRING", {"default": "(none)", "multiline": False}),

                "fine_context_pad": ("INT", {"default": 64, "min": 0, "max": 512, "step": 8,
                                             "tooltip": "[Fine Upscale] Pixel-space padding around the "
                                                        "painted-shape bbox before cropping. Gives the "
                                                        "model surrounding context (skin, hair, "
                                                        "background) so a tight face mask at high denoise "
                                                        "still produces a coherent face that matches its "
                                                        "surroundings. The painted shape is unchanged — "
                                                        "only the area the model SEES grows. Outside the "
                                                        "painted shape, the surrounding pixels are "
                                                        "preserved (not refined). Larger pad = more "
                                                        "context + less effective resolution on the "
                                                        "painted area (bump MP to compensate)."}),

                "persistent_mask": ("BOOLEAN", {"default": False,
                                                "tooltip": "When ON, the last mask used (click point or paint "
                                                           "stroke) is held in the node. Pressing the standard "
                                                           "ComfyUI Queue button re-runs that same mask + a fresh "
                                                           "seed on the LATEST result, so each press builds further "
                                                           "on that region — use it to gradually morph an area over "
                                                           "several presses while the rest of the image stays "
                                                           "unchanged. (To re-roll the same edit on the ORIGINAL "
                                                           "image instead of building on it, use the Re-roll "
                                                           "button.) Toggled via the Persistent Mask button above "
                                                           "the preview."}),

                "paint_mode": ("BOOLEAN", {"default": False,
                                           "tooltip": "When ON, hold + drag on the preview paints a "
                                                      "freeform brush stroke (each dragged point is the "
                                                      "centre of a circle of click_radius; the union is "
                                                      "the refine mask). Release to submit. Single clicks "
                                                      "in paint mode work as one-point strokes. When OFF, "
                                                      "clicks behave as before (single-circle refine)."}),

                # Hidden — JS-set JSON list of [x_pixel, y_pixel] points
                # captured during a paint drag. Empty when no paint stroke
                # is pending (e.g. the user just single-clicked instead).
                "stroke_points": ("STRING", {"default": "", "multiline": False}),

                # Hidden — JS-set JSON list of [x1, y1, x2, y2] rectangles
                # in image-pixel coords, captured during Smart Inpaint
                # rectangle drags. The backend uses only the most recent
                # entry as the active mask.
                "rect_points": ("STRING", {"default": "", "multiline": False}),

                # Kursat NOTE!!! 2026-06-10 — Misleading tooltip fix: tooltip previously said
                # "if area text is empty, the main positive is used" which is
                # WRONG when CLIP is connected.  The code intentionally encodes
                # the empty string (→ empty conditioning) to prevent whole-image
                # reference_latents in the main positive from leaking into the
                # inpaint region (see area-conditioning block in run()).  The main
                # positive is only used when CLIP is NOT connected.  Tooltip
                # updated to describe the actual behaviour.
                "area_prompt": ("BOOLEAN", {"default": False,
                                            "tooltip": "When ON, refinements encode the Area Prompt text "
                                                       "(typed in the box below the canvas) with the "
                                                       "connected CLIP and use it instead of the main "
                                                       "positive/negative — useful for reshaping a region "
                                                       "with a different prompt (e.g. main prompt = wide "
                                                       "shot, area prompt = \"detailed face, photorealistic "
                                                       "eyes\"). The click-and-mask behaviour is unchanged; "
                                                       "only the prompt differs. If CLIP isn't connected, "
                                                       "the main positive/negative are used as a fallback. "
                                                       "NOTE: when CLIP IS connected, an empty area text "
                                                       "encodes as empty conditioning (not the main positive) "
                                                       "— this is intentional to block whole-image reference "
                                                       "latents from leaking into the inpaint region. "
                                                       "Forced ON in Smart Inpaint."}),

                # Hidden — DOM text box below the canvas drives these.
                # The Area Prompt input has a Pos/Neg toggle that decides
                # which of these two the textarea is editing. Encoded
                # with the connected CLIP at run time when area_prompt is
                # on. Negative is usually unused (Klein / CFG=1) but kept
                # for non-distilled models.
                "area_text_positive": ("STRING", {"default": "", "multiline": True}),
                "area_text_negative": ("STRING", {"default": "", "multiline": True}),

                # Hidden — the Load Image button drives these. loaded_image
                # is a JSON ref {name,subfolder,type} from /upload/image;
                # loaded_image_seq bumps on each load so run() knows a new
                # image arrived; resize_mode/target_mp come from the popup.
                "loaded_image": ("STRING", {"default": "", "multiline": False}),
                "loaded_image_seq": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "loaded_resize_mode": (["keep", "mp"], {"default": "keep"}),
                "loaded_target_mp": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 8.0, "step": 0.1}),

                # Hidden — the Detect (SAM 3 / YOLO) flow drives this. A JSON
                # list of silhouette polygons (image-pixel coords) for the
                # confirmed detection; in Refine mode it becomes the mask.
                # Cleared by the JS after the confirm run.
                "seg_polygon": ("STRING", {"default": "", "multiline": False}),

                # Hidden — the Re-roll button drives this. Bumps on each
                # press; run() pops the most recent edit to expose its
                # pre-edit base, rebuilds the SAME mask from the widgets
                # above, and re-samples with a fresh seed — swapping the
                # new variation in place of the last attempt. Declared LAST
                # so older saved workflows (which lack it) don't shift their
                # positional widgets_values; it just defaults to 0.
                "reroll_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),

                # Hidden — the Detect Shift/Alt touch-up brush drives this. A
                # base64-PNG mask at image resolution (white = masked) that, in
                # Refine, takes priority over seg_polygon. Lets the user add to
                # or subtract from (incl. holes) a SAM mask before committing.
                # Declared LAST so older saved workflows don't shift their
                # positional widgets_values; defaults to "".
                "seg_mask_png": ("STRING", {"default": "", "multiline": False}),

                # Redo (#6): the Redo button / Ctrl-Y bumps this. Python pushes
                # back onto the history stack the entry that Undo most recently
                # popped off (held on a per-node redo stack). Declared LAST so
                # older saved workflows don't shift their positional
                # widgets_values; defaults to 0.
                "redo_seq": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFF}),

                # Restore brush: when ON, clicks / paint strokes / detect masks
                # restore the painted region back to the session base image via
                # a feathered latent blend. Refine mode only; Smart modes ignore
                # it. Declared LAST so older saved workflows do not shift their
                # positional widgets_values.
                "restore_mode": ("BOOLEAN", {"default": False,
                                             "tooltip": "When ON, clicks and paint strokes restore "
                                                        "the painted region back to the session base "
                                                        "image (a feathered latent blend, no sampling). "
                                                        "Refine mode only. Toggled via the Restore "
                                                        "button on the toolbar."}),
            },
            "optional": {
                # CLIP / text encoder for the Area Prompt. Optional —
                # without it the area text is ignored and the main
                # positive/negative are used.
                "clip": ("CLIP",),
                # Base latent. OPTIONAL now: if the Load Image button is
                # used, the loaded photo becomes the base and no latent
                # input is needed. Sampler Mode still needs one (it defines
                # the output dimensions for a fresh generation).
                "latent": ("LATENT",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "latent", "source_image", "mask")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "sampling/Angelo"
    DESCRIPTION = (
        "Click-to-refine sampler. Click a region in the preview to do "
        "extra denoising on just that area while the rest of the latent "
        "is preserved. Repeated clicks layer refinements on top of each other."
    )

    def run(
            self,
            guider,
            sampler,
            sigmas,
            positive,
            negative,
            vae,
            mode,
            sampler_seed,
            sampler_seed_control,
            seed,
            seed_control,
            denoise,
            click_radius,
            feather_radius,
            click_x,
            click_y,
            click_seq,
            image_w,
            image_h,
            undo_seq,
            reset,
            auto_decode,
            fine_upscaling,
            min_megapixels,
            max_upscale,
            resize_method,
            inpainting_mode,
            fine_context_pad,
            persistent_mask,
            paint_mode,
            stroke_points,
            rect_points,
            area_prompt,
            guided_location="(none)",
            area_text_positive="",
            area_text_negative="",
            loaded_image="",
            loaded_image_seq=0,
            loaded_resize_mode="keep",
            loaded_target_mp=1.5,
            seg_polygon="",
            reroll_seq=0,
            seg_mask_png="",
            redo_seq=0,
            restore_mode=False,
            latent=None,
            clip=None,
            unique_id=None,
    ):
        node_id = str(unique_id)
        with _STATE_LOCK:
            state = _STATE.get(node_id)

        # ===== Base latent selection =====
        # While an image is LOADED (loaded_image non-empty), it owns the
        # base and the wired `latent` input is ignored — until the user
        # hits Unload (which clears loaded_image). This stops the wired
        # latent from re-winning after a load (which made undo snap back
        # to the latent image). Priority:
        #   1. a freshly loaded image (encode it, force reset)
        #   2. an already-loaded image still active → keep the cached base
        #   3. the wired `latent` input
        #   4. the cached base (no latent, no load)
        loaded_ref = str(loaded_image).strip()
        loaded_active = bool(loaded_ref)
        loaded_seq = int(loaded_image_seq)
        new_loaded = loaded_active and (
                state is None or state.get("loaded_seq") != loaded_seq
        )
        forced_base = None
        if new_loaded:
            forced_base = _encode_loaded_image(
                vae, loaded_ref, str(loaded_resize_mode), float(loaded_target_mp)
            )

        # base_from_wired_latent: only the wired-`latent` path carries a
        # meaningful fingerprint for upstream-change detection. The cache /
        # loaded-image paths take incoming from history[0], which MUTATES as
        # the undo stack is capped (the original base is evicted after
        # _HISTORY_CAP refines), so a fingerprint comparison there is bogus
        # and would spuriously reset mid-session. See the need_reset note.
        base_from_wired_latent = False
        if forced_base is not None:
            incoming = forced_base
        elif loaded_active and state is not None and state.get("history"):
            # Loaded image still active — keep its base, ignore `latent`.
            hist_0 = state["history"][0]
            incoming = hist_0[0] if isinstance(hist_0, tuple) else hist_0
        elif latent is not None:
            incoming = latent["samples"]
            base_from_wired_latent = True
        elif state is not None and state.get("history"):
            hist_0 = state["history"][0]
            incoming = hist_0[0] if isinstance(hist_0, tuple) else hist_0
        else:
            raise ValueError(
                "Angelo: no base image — connect a `latent` input or use the "
                "Load Image button."
            )

        # Normalise the base latent to the dimensionality the MODEL expects.
        # ComfyUI's stock KSampler runs the latent through
        # fix_empty_latent_channels before sampling; Angelo calls
        # comfy.sample.sample directly, so it must do this itself. The
        # load-bearing case is video/temporal VAEs (Qwen Image Edit, Wan):
        # their diffusion model wants a 5D latent [B, C, T, H, W], and a
        # 4D [B, C, H, W] base (e.g. from a plain EmptyLatentImage, or any path
        # that didn't go through a Qwen-aware latent node) makes process_img
        # fail with "not enough values to unpack (expected 5, got 4)". This
        # unsqueezes the temporal axis (and channel-pads an empty latent) for
        # those models; for ordinary 4D models (FLUX/SDXL/SD) it is a no-op,
        # and an already-5D latent is returned unchanged. Done once here so
        # every downstream path — Sampler Mode, the Edit-Mode history seed,
        # the fingerprint, and the refine round-trips — sees the right shape.
        incoming = comfy.sample.fix_empty_latent_channels(guider.model_patcher, incoming)
        incoming_fp = _latent_fingerprint(incoming)

        # ===== Smart Inpaint locks =====
        # The whole point of Smart Inpaint as a mode is to opinionate
        # the params that matter for "add new content in a rectangle"
        # workflows. Override the widget values up front so every code
        # path downstream sees the locked values regardless of what
        # the user set the toolbar to.
        # area_prompt ON makes Smart Inpaint encode the Area Prompt text
        # (typed below the canvas) with the connected CLIP and use it — a
        # separate "what to insert" prompt, kept independent of the main scene
        # prompt. While area_prompt is on the refine uses the Area text ONLY
        # (empty Area text → empty conditioning), never the main positive — see
        # the area-conditioning block further down in run().
        if inpainting_mode == "Smart Inpaint":
            denoise = 1.0
            fine_context_pad = 0
            fine_upscaling = True
            area_prompt = True
            # feather_radius is left under user control — a soft edge can
            # help blend the inserted content into the surroundings.
        elif inpainting_mode == "Smart Guided Inpaint":
            # No painting / boxes — the whole image is edited and the
            # location dropdown supplies a prompt prefix that tells the
            # edit model where to place the content. No region to crop
            # (Fine Upscale OFF), no mask edge to feather, and no mask to
            # persist (Persistent Mask is meaningless here).
            denoise = 1.0
            fine_context_pad = 0
            fine_upscaling = False
            area_prompt = True
            feather_radius = 0
            persistent_mask = False

        # ===== Sampler Mode branch =====
        # Acts like a KSampler: take the incoming latent (typically empty),
        # run a fresh denoise pass with sampler_seed + sampler_denoise, cache
        # the result as the new base. All toolbar / canvas / refine logic is
        # skipped — those are Edit Mode concerns.
        if mode == "Sampler Mode":
            base_steps = max(1, len(sigmas) - 1)
            callback = latent_preview.prepare_callback(guider.model_patcher, base_steps)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
            noise = comfy.sample.prepare_noise(incoming, sampler_seed, None)
            base_g = _guider_with_conds(guider, positive, negative)
            new_latent = _guider_sample(
                base_g, noise, incoming, sampler, sigmas,
                callback=callback,
                disable_pbar=disable_pbar,
                seed=sampler_seed,
            )
            # _guider_sample already moves the result to intermediate_device().
            # Replace the cache with the freshly-sampled base. Drops the
            # undo history (it's irrelevant — we have a brand-new image).
            #
            # CRITICAL: store the fingerprint of the INCOMING latent, not
            # the freshly-generated new_latent. The fingerprint is the
            # "did upstream change?" signal used by Edit Mode's
            # auto-reset check. If we stored new_latent's fingerprint,
            # Edit Mode would always see (cached_fp != incoming_fp)
            # → think upstream changed → wipe cache → refine the empty
            # latent. Storing incoming_fp means Refinement only resets
            # when the user actually rewires upstream.
            #
            # sampler_seed_at_run = the seed Python used. Sent back to JS
            # via the ui message so JS can (a) apply after-gen control
            # next, and (b) restore this value if the user later switches
            # the control to "fixed".
            with _STATE_LOCK:
                _STATE[node_id] = {
                    "history": [(new_latent, None, None)],
                    "click_seq": click_seq,
                    "undo_seq": undo_seq,
                    "redo_seq": redo_seq,
                    "reroll_seq": reroll_seq,
                    "redo_stack": [],
                    "fingerprint": incoming_fp,
                    "sampler_seed_at_run": int(sampler_seed),
                    "loaded_seq": loaded_seq,
                    # Source image (#3/#9): the session base, captured once so the
                    # source_image output survives _HISTORY_CAP eviction of history[0].
                    "source_latent": new_latent,
                    "source_pixels": None,
                    # Kursat NOTE!!! 2026-06-10: Generation token — starts at 0 on
                    # every fresh session; incremented under the per-node lock in
                    # Step 1 of each edit run so Step 3 can detect stale writes
                    # (see Risk 1 fix).
                    "gen_token": 0,
                    # Kursat NOTE!!! 2026-06-11 — Risk 4 fix (LRU): timestamp for _prune_state().
                    "last_touch": time.time(),
                }
                _prune_state()  # Kursat NOTE!!! 2026-06-11 — Risk 4 fix: evict oldest if over cap.
            out_latent = {"samples": new_latent}
            ui_msg = {
                "Angelo_preview": [],
                "Angelo_mode": ["Sampler Mode"],
                "Angelo_sampler_seed_at_run": [int(sampler_seed)],
            }
            # Preview always decodes now (auto_decode deprecated).
            image, image_refs = _decode_to_preview(vae, new_latent)
            ui_msg["Angelo_preview"] = image_refs
            # Freshly-generated base IS the source image — cache + emit it.
            _STATE[node_id]["source_pixels"] = image
            # No refine mask in Sampler Mode — return a zero mask at latent resolution.
            empty_mask = torch.zeros(
                (new_latent.shape[-2], new_latent.shape[-1]),
                device=new_latent.device, dtype=torch.float32,
            )
            return {"ui": ui_msg, "result": (image, out_latent, image, empty_mask)}

        # ===== Edit Mode branch (existing behaviour) =====

        # Decide whether to (re)seed the cache from the incoming latent.
        # Reset on explicit toggle, first run, or fingerprint change —
        # but NOT on fingerprint change while persistent_mask is on, because
        # the whole point of persistent_mask is to keep refining the cached
        # latent across upstream re-rolls (the user's pressing Queue
        # specifically because they want a variation of the held region,
        # not a fresh image).
        #
        # The fingerprint check ONLY applies when the base is the wired
        # `latent` input (where a change really does mean "upstream produced
        # a fresh latent"). When the base comes from the cache / loaded
        # image, `incoming` is history[0] — which mutates as the undo stack
        # is capped (_HISTORY_CAP). Comparing against it there would
        # spuriously reset to a mid-stage latent after enough refines (the
        # "suddenly reverts to an earlier stage while painting" bug). Those
        # bases only change via explicit Load / Unload / Reset, all handled
        # by `reset` / `new_loaded` above — so the fingerprint isn't needed.
        with _STATE_LOCK:
            state = _STATE.get(node_id)
            if state is not None:
                # Kursat NOTE!!! 2026-06-11 — Risk 4 fix (LRU touch): update last_touch
                # on every Edit Mode run so the LRU eviction in _prune_state() always
                # favours genuinely idle / orphaned nodes over active ones.
                state["last_touch"] = time.time()
        fingerprint_changed = (
                base_from_wired_latent
                and state is not None
                and state.get("fingerprint") != incoming_fp
        )
        need_reset = (
                reset
                or state is None
                or new_loaded
                or (fingerprint_changed and not persistent_mask)
        )

        if need_reset:
            # Any reset means the base just changed (load, unload, upstream
            # rewire). Anchor click_seq/undo_seq to the CURRENT widget
            # values so a click that was meant for the OLD base can't trip
            # the new-click gate and replay a stale inpaint onto the new
            # base. The user's next genuine click bumps click_seq and fires
            # normally.
            with _STATE_LOCK:
                _STATE[node_id] = {
                    "history": [(incoming.clone(), None, None)],
                    "click_seq": click_seq,
                    "undo_seq": undo_seq,
                    "redo_seq": redo_seq,
                    "reroll_seq": reroll_seq,
                    "redo_stack": [],
                    "fingerprint": incoming_fp,
                    "loaded_seq": loaded_seq,
                    # Source image (#3/#9): capture the base once, independent of
                    # history[0] (which mutates under _HISTORY_CAP eviction).
                    "source_latent": incoming.clone(),
                    "source_pixels": None,
                    # Kursat NOTE!!! 2026-06-10: Generation token reset alongside
                    # every explicit state reset so the counter stays consistent
                    # with the new session (see Risk 1 fix).
                    "gen_token": 0,
                    # Kursat NOTE!!! 2026-06-11 — Risk 4 fix (LRU): timestamp for _prune_state().
                    "last_touch": time.time(),
                }
                _prune_state()  # Kursat NOTE!!! 2026-06-11 — Risk 4 fix: evict oldest if over cap.
            state = _STATE[node_id]

        # ===== Edit Mode: per-node lock around all state mutations =====
        # ComfyUI is typically serial, but rapid re-queues can overlap.
        # _node_lock(node_id) is a per-node threading.Lock that prevents
        # concurrent runs from corrupting the shared history/undo/redo stacks.
        # Pattern:
        #   1. Acquire lock → pure state mutations (undo/redo/reroll, current read)
        #   2. Release lock → slow GPU sampling
        #   3. Acquire lock → write results back to history

        node_lock = _node_lock(node_id)

        # --- Step 1: state mutations under per-node lock ---
        new_click = False
        reroll_now = False
        current = None
        current_pixels = None
        # Kursat NOTE!!! 2026-06-10: Local copies of the per-run generation token
        # and the backend-saved last mask, both read under the lock in Step 1 and
        # used in Step 2/3 outside it.  Initialised to safe defaults so Step 3
        # comparisons work even on code paths that never enter the lock (e.g. a
        # very early return).
        _my_gen_token = -1        # will be overwritten inside the lock below
        _persistent_last_mask = None  # captured only when persistent_mask=True
        _restore_source_latent = None  # captured under lock for Restore brush

        with node_lock:
            state = _STATE.get(node_id)

            # Undo: if undo_seq advanced and we have history to pop, pop it.
            # We always keep at least one latent (the base / earliest refine)
            # so the preview stays valid.
            new_undo = undo_seq > 0 and undo_seq != state.get("undo_seq", -1)
            if new_undo:
                if len(state["history"]) > 1:
                    # Move the popped entry onto the redo stack so Redo (#6) can
                    # restore it. Bounded like the history stack.
                    popped = state["history"].pop()
                    redo = state.setdefault("redo_stack", [])
                    redo.append(popped)
                    if len(redo) > _HISTORY_CAP:
                        state["redo_stack"] = redo[-_HISTORY_CAP:]
                state["undo_seq"] = undo_seq
                # Undo is a PURE restore — pop the cached latent and decode it.
                # It must NEVER re-sample, or it would produce a different image
                # than the one being restored. Absorb the current click_seq so
                # the new-click gate below stays False on this run. This is
                # load-bearing because the Persistent Mask queue hook bumps
                # click_seq on EVERY queue, including the Undo button's — without
                # this, an undo while Persistent Mask is on would look like a new
                # click and re-run the last mask with the (since-randomized) seed,
                # restoring the WRONG result. (Harmless no-op when the hook didn't
                # bump it.)
                state["click_seq"] = click_seq
                # Kursat NOTE!!! 2026-06-11 — Risk 3 fix (Undo/Redo mask sync): restore
                # last_mask to match the latent we're restoring to, so the
                # MASK output socket stays consistent with the current latent.
                _h = state["history"][-1]
                state["last_mask"] = _h[2] if isinstance(_h, tuple) and len(_h) >= 3 else None

            # ===== Redo (#6): restore an entry Undo moved to the redo stack =====
            # A PURE restore — never re-samples — and runs BEFORE `current` is read
            # below, so the restored entry becomes history[-1]. Absorbs click_seq
            # for the same reason Undo does (the Persistent Mask queue hook bumps
            # it on every queue, including the Redo button's). No-op if the redo
            # stack is empty. A genuine new edit clears the redo stack (below).
            new_redo = redo_seq > 0 and redo_seq != state.get("redo_seq", -1)
            if new_redo:
                redo = state.get("redo_stack") or []
                if redo:
                    state["history"].append(redo.pop())
                    if len(state["history"]) > _HISTORY_CAP:
                        state["history"] = state["history"][-_HISTORY_CAP:]
                state["redo_seq"] = redo_seq
                state["click_seq"] = click_seq
                # Kursat NOTE!!! 2026-06-11 — Risk 3 fix (Undo/Redo mask sync): restore
                # last_mask from the entry we just re-applied.
                _h = state["history"][-1]
                state["last_mask"] = _h[2] if isinstance(_h, tuple) and len(_h) >= 3 else None

            # ===== Re-roll: redo the most recent edit with a fresh seed =====
            # The Re-roll button bumps reroll_seq (and sets a new seed) without
            # touching the mask widgets. Re-run the SAME mask on the SAME pre-
            # edit base and swap the result in place of the last attempt — so
            # the user can cycle seeds on one edit without reset → re-mask →
            # rerun. Implemented as "pop the last refine to expose its pre-edit
            # base as history[-1]"; the edit block below then re-runs from there
            # and appends the fresh variation, restoring the stack depth (net
            # effect: replace, not stack). No-op if there's no prior edit yet.
            new_reroll = reroll_seq > 0 and reroll_seq != state.get("reroll_seq", -1)
            reroll_now = new_reroll and len(state["history"]) > 1
            if reroll_now:
                state["history"].pop()
            state["reroll_seq"] = reroll_seq

            hist_last = state["history"][-1]
            if isinstance(hist_last, tuple):
                # Kursat NOTE!!! 2026-06-11 — Risk 3 fix (history tuple format): handle
                # both old 2-tuple (latent, pixels) and new 3-tuple
                # (latent, pixels, mask) history entries transparently.
                current = hist_last[0]
                current_pixels = hist_last[1] if len(hist_last) > 1 else None
            else:
                current = hist_last
                current_pixels = None

            # Has the user clicked since our last execution for this node?
            new_click = (
                    click_x >= 0
                    and click_y >= 0
                    and click_seq != state["click_seq"]
            )

            _my_gen_token = state.get("gen_token", 0)

            # Kursat NOTE!!! 2026-06-10 — Risk 3 fix: backend persistent-mask
            # capture.  Snapshot state["last_mask"] under the lock so the mask-
            # building code in Step 2 can fall back to it when persistent_mask
            # is ON and all widget mask sources (stroke_points, seg_polygon,
            # seg_mask_png) are empty — guards against JS clearing those values
            # between queue presses.
            _persistent_last_mask = state.get("last_mask") if persistent_mask else None
            if bool(restore_mode) and inpainting_mode == "Refine":
                _restore_source_latent = state.get("source_latent")
                if _restore_source_latent is None and state.get("history"):
                    hist_0 = state["history"][0]
                    _restore_source_latent = hist_0[0] if isinstance(hist_0, tuple) else hist_0

        def _claim_generation_token() -> bool:
            """Reserve a generation token only once a real edit will run."""
            nonlocal _my_gen_token
            with node_lock:
                state_for_token = _STATE.get(node_id)
                if state_for_token is None:
                    return False
                state_for_token["gen_token"] = state_for_token.get("gen_token", 0) + 1
                _my_gen_token = state_for_token["gen_token"]
                return True

        def _release_generation_token_if_current() -> None:
            """Undo this run's token claim when it fails before Step 3."""
            with node_lock:
                state_for_token = _STATE.get(node_id)
                if (
                        state_for_token is not None
                        and _my_gen_token > 0
                        and state_for_token.get("gen_token") == _my_gen_token
                ):
                    state_for_token["gen_token"] = _my_gen_token - 1

        # --- Step 2: GPU sampling outside the lock ---
        # Source latent for every edit is the current cached latent, so all
        # paths build ON TOP of the previous result:
        #   - normal clicks / paints iterate on the latest image
        #   - Persistent Mask Queue presses re-run the held mask on the
        #     latest result with a fresh seed, so the region gradually
        #     morphs further with each press (change something into
        #     something else)
        # Re-roll is the ONE exception, and it gets there for free: it
        # popped the last attempt above, so `current` is now that attempt's
        # PRE-edit base — re-running from here gives a fresh variation on the
        # ORIGINAL image, swapped in for the last attempt instead of stacked.

        _refined = None
        _refined_pixels = None
        _this_mask = None
        _active_claimed = False

        if new_click or reroll_now:
            # A re-roll re-runs the same mask widgets from the popped pre-
            # edit base (current), so it flows through this same block.
            # Pixel → latent conversion. The JS sends us the actual image
            # dimensions (image_w, image_h) so we can derive the per-axis
            # scale dynamically rather than hardcoding a VAE downscale
            # factor — that breaks for FLUX 2 (16×) vs FLUX 1 / SDXL (8×).
            # Fall back to 8× only if image dims weren't provided (shouldn't
            # happen in normal use).
            latent_h = current.shape[-2]
            latent_w = current.shape[-1]
            if image_w > 0 and image_h > 0:
                scale_x = latent_w / image_w
                scale_y = latent_h / image_h
            else:
                # Fallback for headless tests / direct node use without the
                # JS widget populating image_w/h. 8× is the most common VAE
                # downscale (FLUX 1, SDXL, SD1.5) but breaks for FLUX 2 (16×).
                scale_x = scale_y = 1.0 / 8.0
                print("[Angelo] warning: image_w/h not set by JS; "
                      "falling back to 8x VAE assumption — may be wrong for FLUX 2")

            # Use the geometric mean of both axes for isotropic radius/feather
            # in latent space. For square images scale_x == scale_y so there is
            # no change; for portrait/landscape the geometric mean gives the
            # closest-area approximation and keeps the circle visually round.
            scale_geom = math.sqrt(scale_x * scale_y)
            r_latent = max(1.0, click_radius * scale_geom)
            sigma_latent = (feather_radius * scale_geom) if feather_radius > 0 else 0.0

            # Build the mask. Sources of mask shape, in priority:
            #   1. Smart Guided Inpaint: full-image (no region — the whole
            #      image is edited, location comes from the prompt prefix).
            #   2. Smart Inpaint: a single rectangle from rect_points.
            #   3. Refine + paint_mode + stroke points: union of brush
            #      circles along the drag path.
            #   4. Refine single-click: one circle at (click_x, click_y).
            if inpainting_mode == "Smart Guided Inpaint":
                mask = torch.ones((1, latent_h, latent_w),
                                  device=current.device, dtype=torch.float32)
            elif inpainting_mode == "Smart Inpaint":
                rect = _parse_rect_points(rect_points)
                if rect is not None:
                    mask = _rect_mask_latent(
                        latent_h, latent_w, rect,
                        scale_x, scale_y, current.device,
                    )
                else:
                    # Kursat NOTE!!! 2026-06-10 — Risk 2 fix: no rectangle drawn yet.
                    # Previously this built a torch.zeros mask and let execution fall
                    # through.  _refine_with_fine_upscaling detected bbox=None and
                    # returned (current, current_pixels) unchanged, but the caller
                    # still saw _refined is not None and appended that unchanged
                    # latent to history as a phantom no-op edit — corrupting
                    # Undo/Redo with entries that contain no visible change.
                    # Setting mask=None is the sentinel that the guard below
                    # uses to skip sampling AND the history append entirely.
                    mask = None
            else:
                # Refine mask sources, in priority:
                #   1. a brushed touch-up raster mask (Detect Shift/Alt brush)
                #   2. a confirmed segmentation silhouette (SAM 3 / YOLO)
                #   3. a paint stroke (union of brush circles)
                #   4. a single click circle
                raster_png = (seg_mask_png or "").strip()
                seg_polys = _parse_seg_polygons(seg_polygon)
                stroke_pts = _parse_stroke_points(stroke_points) if paint_mode else []
                if raster_png:
                    mask = _raster_mask_latent(
                        latent_h, latent_w, raster_png, current.device,
                    )
                elif seg_polys:
                    mask = _polygons_mask_latent(
                        latent_h, latent_w, seg_polys,
                        scale_x, scale_y, current.device,
                    )
                elif stroke_pts:
                    mask = _stroke_mask_latent(
                        latent_h, latent_w,
                        stroke_pts, r_latent,
                        scale_x, scale_y, current.device,
                    )
                elif _persistent_last_mask is not None:
                    # Kursat NOTE!!! 2026-06-10 — Risk 3 fix: backend persistent-mask
                    # reuse.  All widget mask sources (raster, seg, stroke) are empty
                    # AND persistent_mask is ON, so we fall back to the last confirmed
                    # backend mask instead of a click circle.  This guards against JS
                    # clearing stroke_points / seg_polygon / seg_mask_png between
                    # queue presses, which would otherwise silently replace the
                    # intended persistent mask with a tiny circle at the last click
                    # coordinates.  Clone to avoid aliasing state["last_mask"] during
                    # feathering.  Resize when the latent dimensions changed (e.g.
                    # after a Load Image that picked a different resolution).
                    mask = _persistent_last_mask.clone()
                    if mask.shape[-2] != latent_h or mask.shape[-1] != latent_w:
                        mask = torch.nn.functional.interpolate(
                            mask.unsqueeze(0),
                            size=(latent_h, latent_w),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze(0).clamp(0.0, 1.0)
                else:
                    cx_latent = click_x * scale_x
                    cy_latent = click_y * scale_y
                    mask = _circle_mask_latent_direct(
                        latent_h, latent_w,
                        cx_latent, cy_latent, r_latent,
                        current.device,
                    )
            # Kursat NOTE!!! 2026-06-10 — Risk 2 fix: mask=None guard.
            # Smart Inpaint with no rectangle sets mask=None above.  Skipping
            # the block below keeps _refined=None and _this_mask=None so Step 3
            # performs no history append and no click_seq advance — the run is
            # treated as a pure no-op and Undo/Redo are not polluted.
            if mask is not None:
                if sigma_latent > 0:
                    mask = _gaussian_blur_2d(mask, max(0.5, sigma_latent))
                    mask = mask.clamp(0.0, 1.0)

                if _mask_bbox_latent(mask) is None:
                    # Empty masks are no-op edits. Do not claim a generation
                    # token and do not update history/last_mask.
                    mask = None

            if mask is not None:
                _this_mask = mask  # capture for state write + MASK output

                restore_now = bool(restore_mode) and inpainting_mode == "Refine"

                # Area-prompt conditioning selection. When area_prompt is on AND a
                # CLIP is connected, the refine uses the AREA text ONLY and NEVER
                # the main prompt — even when the Area text is empty, in which case
                # we encode the empty string (→ an empty conditioning) rather than
                # falling back to `positive`. This is load-bearing for the edit
                # modes: the main positive can carry whole-image reference_latents
                # (a Klein edit workflow's ReferenceLatent), and letting it leak in
                # made an empty-Area-Prompt Smart Inpaint reproduce the whole scene.
                # Negative area text is optional — falls back to the main negative
                # when empty (fine for CFG=1 / distilled models that ignore it).
                #
                # Smart Guided Inpaint prepends a location prefix to the positive
                # text (e.g. "In the top left of the image, ") so the edit model
                # places the content at the chosen spot.
                #
                # Without a CLIP we can't encode anything, so we must use the
                # already-encoded main conditioning (an unavoidable degenerate case
                # — area prompts need a CLIP connected).
                area_pos_text = str(area_text_positive)
                if inpainting_mode == "Smart Guided Inpaint":
                    prefix = _GUIDED_LOCATION_PREFIXES.get(str(guided_location), "")
                    area_pos_text = prefix + area_pos_text
                if area_prompt and clip is not None and not restore_now:
                    tokens_p = clip.tokenize(area_pos_text)
                    refine_positive = clip.encode_from_tokens_scheduled(tokens_p)
                    if str(area_text_negative).strip():
                        tokens_n = clip.tokenize(str(area_text_negative))
                        refine_negative = clip.encode_from_tokens_scheduled(tokens_n)
                    else:
                        refine_negative = negative
                else:
                    refine_positive = positive
                    refine_negative = negative

                # Sample with the mask. Use the seed widget value as-is —
                # NO click_seq offset. Per-Queue variation (when persistent_mask
                # is on) and per-click variation (when user wants different
                # attempts on the same spot) are now controlled by seed_control:
                #   fixed     → same seed each run, repeatable result
                #   randomize → seed changes after each run (via JS after-gen),
                #               so each Queue / click produces a different result
                #   increment/decrement → +1/-1 each run
                # An older version of this code did `(seed + click_seq) & mask`
                # to fake per-click variation in the absence of after-gen
                # control. That broke "fixed means fixed" — even with the seed
                # widget locked, click_seq's increment still moved the effective
                # sampling seed. Now the user has explicit control.
                this_seed = int(seed)
                # Derive the actual step count from the sigmas + denoise so the
                # progress bar matches reality. The sampler runs
                # len(refine_sigmas)-1 steps, where refine_sigmas is the truncated
                # schedule. Using the `steps` widget here was wrong: it only
                # controls the progress bar, not the actual step count, so a
                # mismatch produced a jumpy / incorrect progress display.
                _refine_steps = max(1, round((len(sigmas) - 1) * denoise))
                callback = latent_preview.prepare_callback(guider.model_patcher, _refine_steps)
                disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

                # Source latent = the current cached latent for every path (see
                # the note above). Re-roll already exposed the pre-edit base as
                # `current` via its pop, so it needs no special-casing here.
                refine_source = current

                # ===== Inpainting Mode pre-processing =====
                # Refine: no pre-processing — partial denoise from the
                #   existing latent does the work.
                # Smart Inpaint: forces fine_upscaling=True up top, so its
                #   pre-processing (latent zero + reference_latents) lives
                #   inside _refine_with_fine_upscaling.
                # Smart Guided Inpaint: whole-image edit through the non-
                #   fine-upscale path below — inject the scene as
                #   reference_latents so Klein's edit branch keeps the rest
                #   of the image faithful while applying the location-guided
                #   change. POSITIVE ONLY (negative reference would steer
                #   CFG>1 samplers away from the scene).
                if inpainting_mode == "Smart Guided Inpaint":
                    reference_latent = refine_source.clone()
                    refine_positive = node_helpers.conditioning_set_values(
                        refine_positive, {"reference_latents": [reference_latent]}, append=True,
                    )

                if restore_now:
                    # Restore brush: heal the masked area back to the session
                    # base, without sampling. Outside the feathered mask, keep
                    # the current latent bit-exact. Shape mismatch is treated as
                    # a no-op so Undo/Redo history is not polluted.
                    src = _restore_source_latent
                    if src is None or tuple(src.shape) != tuple(current.shape):
                        src_shape = None if src is None else tuple(src.shape)
                        print("[Angelo restore] base/current latent shape mismatch "
                              f"({src_shape} vs {tuple(current.shape)}) - restore skipped")
                    else:
                        if _claim_generation_token():
                            _active_run_inc(node_id)
                            _active_claimed = True
                            try:
                                src = src.to(device=current.device, dtype=current.dtype)
                                alpha = mask.to(device=current.device, dtype=current.dtype)
                                while alpha.dim() < current.dim():
                                    alpha = alpha.unsqueeze(0)
                                _refined = src * alpha + current * (1.0 - alpha)
                                _refined_pixels = None
                            except Exception:
                                _release_generation_token_if_current()
                                _active_run_dec(node_id)
                                _active_claimed = False
                                raise
                elif fine_upscaling:
                    if _claim_generation_token():
                        _active_run_inc(node_id)
                        _active_claimed = True
                        try:
                            _refined, _refined_pixels = _refine_with_fine_upscaling(
                                guider=guider, sampler=sampler, sigmas=sigmas,
                                vae=vae, current=refine_source, current_pixels=current_pixels, mask=mask,
                                scale_x=scale_x, scale_y=scale_y,
                                target_mp=float(min_megapixels),
                                max_linear=float(max_upscale),
                                resize_method=str(resize_method),
                                context_pad_pixel=int(fine_context_pad),
                                inpainting_mode=str(inpainting_mode),
                                seed=this_seed,
                                positive=refine_positive, negative=refine_negative,
                                denoise=denoise, callback=callback, disable_pbar=disable_pbar,
                            )
                        except Exception:
                            _release_generation_token_if_current()
                            _active_run_dec(node_id)
                            _active_claimed = False
                            raise
                else:
                    noise = comfy.sample.prepare_noise(refine_source, this_seed, None)
                    refine_sigmas = _truncate_sigmas_for_denoise(sigmas, denoise)
                    temp_g = _guider_with_conds(guider, refine_positive, refine_negative)
                    if _claim_generation_token():
                        _active_run_inc(node_id)
                        _active_claimed = True
                        try:
                            _refined = _guider_sample(
                                temp_g, noise, refine_source, sampler, refine_sigmas,
                                denoise_mask=mask,
                                callback=callback,
                                disable_pbar=disable_pbar,
                                seed=this_seed,
                            )
                            # _guider_sample already moves the result to intermediate_device().
                            _refined_pixels = None
                        except Exception:
                            _release_generation_token_if_current()
                            _active_run_dec(node_id)
                            _active_claimed = False
                            raise

        # --- Step 3: write results back to state under per-node lock ---
        with node_lock:
            state = _STATE.get(node_id)
            # Generation token guard. Tokens are claimed only after the mask
            # has been proven non-empty and just before the run performs a real
            # restore/sample. No-op clicks never advance gen_token, so they
            # cannot invalidate an older in-flight real edit.
            _is_current_gen = (state is not None and state.get("gen_token") == _my_gen_token)

            if _refined is not None:
                if _is_current_gen:
                    # Kursat NOTE!!! 2026-06-11 — Risk 3 fix (history tuple format): store
                    # (latent, pixels, mask) so Undo/Redo can restore the
                    # mask that was active when this edit was made.
                    state["history"].append((_refined, _refined_pixels, _this_mask))
                    if len(state["history"]) > _HISTORY_CAP:
                        state["history"] = state["history"][-_HISTORY_CAP:]
                    # A genuine new edit (click or re-roll) invalidates the redo branch.
                    state["redo_stack"] = []
                    state["click_seq"] = click_seq
                    state["refine_seed_at_run"] = int(seed)
                    state["last_mask"] = _this_mask
                # Always update local variables for this run's output socket,
                # even when the history write was skipped for a stale-gen run.
                current = _refined
                current_pixels = _refined_pixels
            elif _this_mask is not None:
                # Kursat NOTE!!! 2026-06-10 — Misleading comment fix: corrected comment.
                # This branch fires when _this_mask was set (mask was built and
                # passed the None-guard) but _refined stayed None — meaning
                # _refine_with_fine_upscaling returned the SAME latent object as
                # `current` with an identity bbox, rather than a new tensor.
                # It does NOT fire when sampling raises an exception: an uncaught
                # exception from the GPU sampling calls above propagates out of
                # run() entirely and never reaches this with-block.
                # We still persist the mask so the MASK output socket reflects
                # the shape that was used, but we do NOT append to history
                # (nothing actually changed in the latent).
                if _is_current_gen:
                    state["last_mask"] = _this_mask

        with _STATE_LOCK:
            if _active_claimed:
                _active_run_dec(node_id)
                _active_claimed = False
            # Kursat NOTE!!! 2026-06-11 — Risk 2 fix (prune-during-execution): state=None guard.
            # Re-fetch state under _STATE_LOCK in case _prune_state() removed it
            # during Step 2 (should not happen with _ACTIVE_RUNS, but guards
            # against future refactors or stuck-counter fallback eviction).
            state = _STATE.get(node_id) or state

        out_latent = {"samples": current}
        ui_msg = {
            "Angelo_preview": [],
            "Angelo_mode": ["Edit Mode"],
            "Angelo_refine_seed_at_run": [int((state or {}).get("refine_seed_at_run", seed))],
        }

        if current_pixels is not None:
            image = current_pixels
            previewer = comfy_nodes.PreviewImage()
            ui = previewer.save_images(image, filename_prefix="Angelo_preview")
            ui_msg["Angelo_preview"] = ui["ui"]["images"]
        else:
            image, image_refs = _decode_to_preview(vae, current)
            ui_msg["Angelo_preview"] = image_refs

        # Source image (#3/#9): the session base, decoded once and cached so
        # repeated edits don't re-decode it (and so it survives history[0]
        # eviction under _HISTORY_CAP). source_latent is set at every base
        # (re)establishment; the history[0] fallback only covers pre-existing
        # in-memory state from before this feature.
        source_image = (state or {}).get("source_pixels")
        if source_image is None:
            src_latent = (state or {}).get("source_latent")
            if src_latent is None and state is not None and state.get("history"):
                h0 = state["history"][0]
                src_latent = h0[0] if isinstance(h0, tuple) else h0
            if src_latent is not None:
                source_image = _vae_decode(vae, src_latent)
                if state is not None:
                    state["source_pixels"] = source_image
            else:
                # Kursat NOTE!!! 2026-06-11 — Risk 2 fix (prune-during-execution): state=None guard.
                # Fallback: decode the current latent as the source image when
                # state was evicted during Step 2 and src_latent is unavailable.
                source_image, _ = _decode_to_preview(vae, current)

        # MASK output: return the most recently computed refine mask so
        # downstream nodes (compositing, secondary inpaint, etc.) can use it.
        # last_mask is [1, H, W] float; squeeze to [H, W] for the MASK type.
        out_mask = (state or {}).get("last_mask")
        if out_mask is None:
            out_mask = torch.zeros(
                (current.shape[-2], current.shape[-1]),
                device=current.device, dtype=torch.float32,
            )
        elif out_mask.dim() == 3:
            out_mask = out_mask.squeeze(0)

        return {"ui": ui_msg, "result": (image, out_latent, source_image, out_mask)}


NODE_CLASS_MAPPINGS = {
    "AngeloRefine": AngeloRefine,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AngeloRefine": "Angelo - Sampler Custom",
}

