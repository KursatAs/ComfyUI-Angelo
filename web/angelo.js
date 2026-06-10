// ComfyUI-Angelo — click-to-refine UI extension.
//
// Strategy: each AngeloRefine node gets its own canvas DOM widget
// attached at the bottom of the node. We draw the refined preview into
// that canvas ourselves (instead of using ComfyUI's auto-preview), and
// the canvas has a real DOM click listener. This sidesteps the issue
// where DOM image elements swallow clicks before LiteGraph sees them.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAME = "AngeloRefine";

// Flip to true if something stops working and you want click → pixel-coord
// → widget-update → queue traced in the browser console.
const Angelo_DEBUG = false;
function dbg(...args) {
    if (Angelo_DEBUG) console.log("[Angelo]", ...args);
}


// --- Module-level hover tracking for the keyboard shortcuts. Set by
//     the canvas mouseenter / mouseleave handlers in attachPreviewCanvas.
let _AngeloHoveredNode = null;

// --- Global queuePrompt hook: bumps click_seq on every AngeloRefine
//     node where persistent_mask is on, so the standard ComfyUI Queue
//     button re-runs the refine on the held mask with a fresh seed.
//     Installed once per extension load; no-op for graphs without
//     AngeloRefine.
function installQueueHook() {
    if (app._AngeloQueueHookInstalled) return;
    if (typeof app.queuePrompt !== "function") return;
    const orig = app.queuePrompt.bind(app);
    app.queuePrompt = function (...args) {
        try {
            const nodes = (app.graph && app.graph._nodes) || [];
            for (const n of nodes) {
                if (n?.type !== NODE_NAME) continue;
                const persistW = findWidget(n, "persistent_mask");
                if (!persistW || !persistW.value) continue;
                const seqW = findWidget(n, "click_seq");
                if (!seqW) continue;
                setWidget(seqW, ((seqW.value || 0) + 1) & 0x7FFFFFFF);
                // Also clear `reset` so a stale tick doesn't blow away
                // the cached latent we want to refine on top of.
                const resetW = findWidget(n, "reset");
                if (resetW) setWidget(resetW, false);
                dbg("queueHook: bumped click_seq on persistent-mask node", n.id, "→", seqW.value);
            }
        } catch (e) {
            dbg("queueHook error (passing through)", e);
        }
        return orig(...args);
    };
    app._AngeloQueueHookInstalled = true;
    dbg("installed app.queuePrompt hook for persistent_mask");
}

// --- Global keyboard shortcuts. Active only when the cursor is hovering
//     a Angelo canvas AND that node is in Edit Mode. Mirrors
//     creative-tool conventions:
//       [ ]   → click_radius     (universal brush-size pattern)
//       { }   → feather_radius   (Photoshop brush hardness/softness)
//       , .   → denoise          (< > ordering on the same keys)
//     Captured in the capture phase so they beat ComfyUI's own
//     keybindings if any happen to overlap. preventDefault on a match
//     so the key event doesn't propagate further.
function installKeyboardShortcuts() {
    if (app._AngeloKeysInstalled) return;
    app._AngeloKeysInstalled = true;

    const bindings = [
        // [key, widget_name, delta, min, max, asInt, syncFn]
        ["[",  "click_radius",     -4,    8,   1024, true,  "syncClickRadiusInput"],
        ["]",  "click_radius",      4,    8,   1024, true,  "syncClickRadiusInput"],
        ["{",  "feather_radius",   -4,    0,    256, true,  "syncFeatherInput"],
        ["}",  "feather_radius",    4,    0,    256, true,  "syncFeatherInput"],
        [",",  "denoise",          -0.05, 0.05, 1.0, false, "syncDenoiseInput"],
        [".",  "denoise",           0.05, 0.05, 1.0, false, "syncDenoiseInput"],
    ];

    const handlers = {};
    for (const b of bindings) handlers[b[0]] = b;

    document.addEventListener("keydown", (event) => {
        const node = _AngeloHoveredNode;
        if (!node) return;

        // Esc dismisses pending detection candidates (even from an input,
        // so it works right after typing a concept + Detect).
        if (event.key === "Escape" && node._AngeloDetections && node._AngeloDetections.length) {
            clearDetections(node);
            return;
        }

        // Don't intercept when the user is typing in an input or textarea
        // (e.g., the toolbar Seed input, or any other DOM widget).
        const t = event.target;
        if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) {
            return;
        }

        // Space also exits detect mode (only when not typing — handled
        // above — so a space in the concept box is unaffected).
        if (event.key === " " && node._AngeloDetections && node._AngeloDetections.length) {
            clearDetections(node);
            event.preventDefault();
            return;
        }

        // 'F' fits the image to the panel (zoom=1, centred). Works in any
        // mode — it mirrors the double-middle-click reset — so it sits before
        // the Edit-Mode gate below. No modifiers, so Ctrl-F etc. stay with the
        // browser.
        if ((event.key === "f" || event.key === "F")
            && !event.ctrlKey && !event.metaKey && !event.altKey) {
            if (node._AngeloImg) {
                resetView(node);
                redrawCanvasWithOverlays(node);
                event.preventDefault();
                event.stopPropagation();
            }
            return;
        }

        // Only active in Edit Mode. Sampler Mode has the toolbar
        // greyed; the keys would feel inert.
        const modeW = findWidget(node, "mode");
        if (!modeW || String(modeW.value) !== "Edit Mode") return;

        // NOTE: Undo/Redo via Ctrl-Z / Ctrl-Y are handled below. They are
        // scoped to the Angelo canvas hover (_AngeloHoveredNode must be set),
        // which prevents them from firing when the cursor is on the ComfyUI
        // graph and conflicting with graph-level undo/redo.

        const binding = handlers[event.key];
        if (!binding) {
            // Ctrl-Z / Ctrl-Y / Ctrl-Shift-Z — scoped Undo / Redo.
            // Only fires when the cursor is hovering the Angelo canvas AND
            // the node is in Edit Mode, so it won't fight ComfyUI's graph-
            // level undo/redo when the cursor is on the graph background.
            if ((event.ctrlKey || event.metaKey) && !event.altKey
                    && modeW && String(modeW.value) === "Edit Mode") {
                const isUndo = (event.key === "z" || event.key === "Z") && !event.shiftKey;
                const isRedo = (event.key === "y" || event.key === "Y")
                    || ((event.key === "z" || event.key === "Z") && event.shiftKey);
                if (isUndo) {
                    event.preventDefault();
                    event.stopPropagation();
                    triggerUndo(node);
                    return;
                }
                if (isRedo) {
                    event.preventDefault();
                    event.stopPropagation();
                    triggerRedo(node);
                    return;
                }
            }
            return;
        }

        const [, name, delta, min, max, asInt, syncName] = binding;
        const w = findWidget(node, name);
        if (!w) return;
        let v = Number(w.value || 0) + delta;
        v = Math.max(min, Math.min(max, v));
        if (asInt) v = Math.round(v);
        else v = Math.round(v * 100) / 100;  // avoid float drift accumulating
        setWidget(w, v);

        // Sync the corresponding toolbar input so the visible value
        // tracks the keyboard adjustment.
        const syncFn = ({
            syncClickRadiusInput, syncFeatherInput,
            syncDenoiseInput,
        })[syncName];
        if (syncFn) syncFn(node);

        // For click_radius, also redraw the canvas so the hover ring
        // resizes immediately on the visible image.
        if (name === "click_radius" && typeof redrawCanvasWithOverlays === "function") {
            redrawCanvasWithOverlays(node);
        }

        dbg("key", event.key, "→", name, "=", v);
        event.preventDefault();
        event.stopPropagation();
    }, true);  // capture phase

    // Handle image paste (Ctrl+V / Cmd+V) from the OS clipboard.
    window.addEventListener("paste", (event) => {
        const node = _AngeloHoveredNode;
        if (!node) return; // Only active when hovering over an Angelo canvas.

        // Do not intercept if the user is pasting text into an input or textarea.
        const activeTag = document.activeElement ? document.activeElement.tagName : "";
        const isEditable = document.activeElement && document.activeElement.isContentEditable;
        if (activeTag === "INPUT" || activeTag === "TEXTAREA" || isEditable) {
            return;
        }

        if (event.clipboardData && event.clipboardData.files && event.clipboardData.files.length > 0) {
            const imageFiles = Array.from(event.clipboardData.files).filter(f => f.type.startsWith("image/"));

            if (imageFiles.length > 0) {
                // Strictly prevent ComfyUI from intercepting this and spawning a LoadImage node
                event.preventDefault();
                event.stopPropagation();
                event.stopImmediatePropagation();

                let file = imageFiles[0];

                // Browsers often name pasted files generically like "image.png".
                // Append a timestamp to make it distinct in uploads and logs.
                if (file.name === "image.png" || !file.name) {
                    const ext = file.type.split('/')[1] || "png";
                    const fakeName = `pasted_${Date.now()}.${ext}`;
                    file = new File([file], fakeName, { type: file.type });
                }

                // Route it through the same resolution popup as Drag & Drop / Load Image.
                showLoadImagePopup(node, file);
            }
        }
    }, true);  // capture phase on window guarantees it fires before ComfyUI's document listener

    dbg("installed keyboard shortcuts");
}

app.registerExtension({
    name: "Angelo.ClickToRefine",

    async setup() {
        installQueueHook();
        installKeyboardShortcuts();
    },

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (nodeData.name !== NODE_NAME) return;

        // --- Node setup: attach the preview canvas + hide mechanical widgets ---
        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
            hideMechanicalWidgets(this);
            attachPreviewCanvas(this);
            // Force LiteGraph to recompute layout now that hidden widgets
            // claim zero height — otherwise the node keeps its initial
            // tall size and the (now-hidden) widget slots show as gaps.
            if (typeof this.setSize === "function" && this.computeSize) {
                const min = this.computeSize();
                if (this.size[1] < min[1]) this.size[1] = min[1];
            }
            // Reflect persisted widget state in every toolbar control.
            // NOTE: on an existing-workflow load this runs BEFORE the
            // serialized widget values are restored, so the toggles
            // may show defaults here. The onConfigure hook below re-runs
            // the same sync after the restore to correct any mismatch.
            syncAllToolbarControls(this);

            // Mode widget: grey toolbar in Sampler Mode, un-grey in
            // Refinement; auto-force sampler_seed_control = fixed when
            // flipping into Refinement (and restore sampler_seed to its
            // at-run value via lockSeedToAtRun).
            const modeW = findWidget(this, "mode");
            if (modeW) {
                const origCb = modeW.callback;
                modeW.callback = (value, ...args) => {
                    const prevValue = modeW._AngeloPrevValue;
                    if (origCb) {
                        try { origCb.call(modeW, value, ...args); }
                        catch (e) { dbg("mode callback orig threw", e); }
                    }
                    modeW._AngeloPrevValue = value;
                    syncModeSwitchToFixed(this, prevValue);
                    syncModeState(this);
                };
                modeW._AngeloPrevValue = modeW.value;
            }
            syncModeState(this);   // initial state reflects persisted widget

            // Seed-control widgets: when value transitions TO "fixed"
            // (either by user click or programmatic set), restore the
            // corresponding seed widget to the seed_at_run value. This
            // ensures "fixed" always means "the seed that produced the
            // current canvas", not whatever value after-gen left in the
            // widget. Wrap each widget's callback to detect the transition.
            const samplerCtrlW = findWidget(this, "sampler_seed_control");
            if (samplerCtrlW) {
                const origCb = samplerCtrlW.callback;
                samplerCtrlW.callback = (value, ...args) => {
                    const prevValue = samplerCtrlW._AngeloPrevValue;
                    if (origCb) {
                        try { origCb.call(samplerCtrlW, value, ...args); }
                        catch (e) { dbg("sampler_seed_control callback orig threw", e); }
                    }
                    samplerCtrlW._AngeloPrevValue = value;
                    if (value === "fixed" && prevValue !== "fixed") {
                        lockSeedToAtRun(this, "sampler_seed", "sampler_seed_control");
                    }
                };
                samplerCtrlW._AngeloPrevValue = samplerCtrlW.value;
            }

            const seedCtrlW = findWidget(this, "seed_control");
            if (seedCtrlW) {
                const origCb = seedCtrlW.callback;
                seedCtrlW.callback = (value, ...args) => {
                    const prevValue = seedCtrlW._AngeloPrevValue;
                    if (origCb) {
                        try { origCb.call(seedCtrlW, value, ...args); }
                        catch (e) { dbg("seed_control callback orig threw", e); }
                    }
                    seedCtrlW._AngeloPrevValue = value;
                    if (value === "fixed" && prevValue !== "fixed") {
                        lockSeedToAtRun(this, "seed", "seed_control");
                    }
                };
                seedCtrlW._AngeloPrevValue = seedCtrlW.value;
            }
            const min = [340, 680];
            if (this.size[0] < min[0]) this.size[0] = min[0];
            if (this.size[1] < min[1]) this.size[1] = min[1];
        };

        // --- Right-click node menu: Open / Copy / Paste image (#7). Covers
        //     right-clicks on the litegraph-rendered node body / title; the
        //     DOM preview canvas has its own contextmenu handler too. ---
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (canvas, options) {
            if (origGetExtraMenuOptions) origGetExtraMenuOptions.apply(this, arguments);
            const node = this;
            const hasImg = !!node._AngeloImg;
            options.unshift(
                { content: "Angelo: open image in new tab", disabled: !hasImg,
                  callback: () => _angeloOpenImageInTab(node) },
                { content: "Angelo: copy image", disabled: !hasImg,
                  callback: () => _angeloCopyImageToClipboard(node) },
                null,
            );
            return options;
        };

        // --- onExecuted: receive the new preview URL, draw into our canvas,
        //     and run after-gen seed control + record seed_at_run for the
        //     lock-on-fixed semantics. ---
        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (origOnExecuted) origOnExecuted.apply(this, arguments);
            dbg("onExecuted", message);

            // Preview image (Angelo_preview)
            const refs = message?.Angelo_preview;
            if (refs && refs.length > 0) {
                const ref = refs[0];
                this._AngeloPreviewRef = ref;   // for the SAM 3 / YOLO detect route
                const url = makeViewUrl(ref);
                dbg("loading preview", url);
                loadIntoCanvas(this, url);
            }

            // NOTE: seg_polygon is intentionally NOT cleared here. It must
            // persist across runs like stroke_points / rect_points so a
            // Persistent Mask re-roll keeps using the SAM-detected
            // silhouette. It's cleared instead by the manual-mask triggers
            // (a click / paint / rect resets it) — see triggerRefine etc.

            // Seed_at_run capture — used by the lock-on-fixed code. ComfyUI's
            // ui message values arrive as 1-element lists (their convention).
            const lastMode = message?.Angelo_mode?.[0];
            const samplerSeedAtRun = message?.Angelo_sampler_seed_at_run?.[0];
            const refineSeedAtRun = message?.Angelo_refine_seed_at_run?.[0];
            if (samplerSeedAtRun != null) {
                this._AngeloSamplerSeedAtRun = Number(samplerSeedAtRun);
            }
            if (refineSeedAtRun != null) {
                this._AngeloRefineSeedAtRun = Number(refineSeedAtRun);
            }

            // After-gen seed control. ComfyUI's standard "seed widgets" have
            // an auto-added control_after_generate dropdown that does this
            // for them; ours are explicit ENUM widgets, so we apply the
            // logic ourselves. Runs AFTER seed_at_run is captured so a
            // subsequent lock-on-fixed restores the pre-modification value.
            if (lastMode === "Sampler Mode") {
                applyAfterGenControl(this, "sampler_seed", "sampler_seed_control");
            } else if (lastMode === "Edit Mode") {
                applyAfterGenControl(this, "seed", "seed_control");
            }

            // Auto-clear the `reset` toggle once the server confirms it has
            // processed the queue. This is more reliable than the 1-second
            // setTimeout in triggerReset (which can fire before a slow first-
            // run finishes loading models).
            const resetW = findWidget(this, "reset");
            if (resetW && resetW.value === true) setWidget(resetW, false);
        };

        // Reset and Undo now live on the DOM toggle bar above the canvas
        // (see attachPreviewCanvas). Removed the canvas-title-bar hooks
        // that used to draw + hit-test them — all interactive controls
        // sit on one horizontal line.

        // --- onConfigure: fires when a saved workflow is loaded and the
        //     node's serialized widget values are restored. onNodeCreated
        //     runs BEFORE that restore, so the toolbar would otherwise
        //     reflect defaults (e.g. Paint Mode shows OFF when the saved
        //     value was ON, and vice-versa). Re-run the full toolbar sync
        //     here so the DOM controls match the restored widget state. ---
        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (origOnConfigure) origOnConfigure.apply(this, arguments);
            // Defer one tick — some ComfyUI versions finish applying
            // widget values immediately after onConfigure returns, so a
            // microtask-delayed sync sees the final restored state.
            const node = this;
            queueMicrotask(() => {
                try { syncAllToolbarControls(node); }
                catch (e) { dbg("onConfigure sync threw", e); }
            });
        };
    },
});


// ============================================================
// Preview canvas (the DOM widget that displays + handles clicks)
// ============================================================

function attachPreviewCanvas(node) {
    if (node._AngeloWidget) return;  // already attached

    const container = document.createElement("div");
    container.style.width = "100%";
    container.style.position = "relative";
    container.style.background = "#1a1a1a";
    container.style.border = "1px solid #333";
    container.style.borderRadius = "4px";
    container.style.overflow = "hidden";
    container.style.display = "flex";
    container.style.flexDirection = "column";

    // Toolbar above the canvas, top to bottom:
    //   modeRow     — Mode dropdown, centred (the master Sampler/Edit switch)
    //   row3        — shared generation config (steps/cfg/sampler/sched), always active
    //   row4        — Sampler-Mode base-gen seed group, greyed in Edit Mode
    //   refineRows  — edit actions + refine values, greyed in Sampler Mode
    const toggleBarWrap = document.createElement("div");
    toggleBarWrap.style.display = "flex";
    toggleBarWrap.style.flexDirection = "column";
    toggleBarWrap.style.background = "#222";
    toggleBarWrap.style.borderBottom = "1px solid #333";
    toggleBarWrap.style.transition = "opacity 0.15s ease";  // smooth grey/un-grey on mode switch
    node._AngeloToolbarWrap = toggleBarWrap;

    // Mode row — centred at the top of the node, visually separated from
    // the rest of the toolbar by a bottom border.
    const modeRow = makeToolbarRow();
    modeRow.style.justifyContent = "center";
    modeRow.style.borderBottom = "1px solid #333";
    modeRow.style.position = "relative";  // anchor for the floating Cancel Detect button

    // Generation / sampler-seed rows (set-once-ish, sit under Mode).
    const row3 = makeToolbarRow();
    const row4 = makeToolbarRow();
    row4.style.transition = "opacity 0.15s ease";

    // Edit control rows — greyed in Sampler Mode (wrapped so the grey
    // target is just these rows, not the generation rows). A top border
    // separates the "edit" group from the "generation" group above.
    const refineRowsWrap = document.createElement("div");
    refineRowsWrap.style.transition = "opacity 0.15s ease";
    refineRowsWrap.style.borderTop = "1px solid #333";
    const row1 = makeToolbarRow();
    const row2 = makeToolbarRow();
    const detectRow = makeToolbarRow();   // SAM 3 detect (Refine + Smart Inpaint)
    detectRow.style.flexWrap = "nowrap";  // keep it one line; the text box flexes
    refineRowsWrap.appendChild(row1);
    refineRowsWrap.appendChild(row2);
    refineRowsWrap.appendChild(detectRow);
    node._AngeloDetectRow = detectRow;

    toggleBarWrap.appendChild(modeRow);
    toggleBarWrap.appendChild(row3);
    toggleBarWrap.appendChild(row4);
    toggleBarWrap.appendChild(refineRowsWrap);

    node._AngeloModeRow = modeRow;
    node._AngeloRefineRowsWrap = refineRowsWrap;
    node._AngeloSamplerSeedRow = row4;

    // ===== ROW 1: actions + mode toggles =====
    const resetBtn = makeActionButton("Reset", () => triggerReset(node), "reset");
    resetBtn.title = "Throw away the cached refined latent + history and start fresh from the upstream latent.";
    row1.appendChild(resetBtn);

    const undoBtn = makeActionButton("Undo", () => triggerUndo(node), "undo");
    undoBtn.title = "Pop the most recent refine off the history stack. Restores the cached latent from before the last click.";
    row1.appendChild(undoBtn);

    const redoBtn = makeActionButton("Redo", () => triggerRedo(node), "redo");
    redoBtn.title = "Re-apply the most recent edit that Undo removed. A new edit clears the redo history.";
    row1.appendChild(redoBtn);

    const rerollBtn = makeActionButton("Re-roll", () => triggerReroll(node), "reroll");
    rerollBtn.title = "Try the most recent edit again with a fresh seed — SAME mask, SAME starting image. Each press replaces the last attempt with a new variation (it doesn't stack on top). Make an edit first, then Re-roll to cycle seeds without re-painting or resetting. Works for clicks, brush strokes, rectangles and detected masks.";
    row1.appendChild(rerollBtn);

    row1.appendChild(makeSeparator());

    const persistentMaskToggle = makeToggleButton("Persistent Mask", () => {
        const w = findWidget(node, "persistent_mask");
        if (!w) return;
        setWidget(w, !w.value);
        syncPersistentMaskToggle(node);
    });
    persistentMaskToggle.title = "When ON, the last mask is held. Pressing the standard ComfyUI Queue button re-runs that region on the LATEST result with a fresh seed each time, so each press builds further — gradually morph an area over several presses without re-painting. (To re-roll the same edit on the ORIGINAL image instead, use the Re-roll button.)";
    row1.appendChild(persistentMaskToggle);
    node._AngeloPersistentMaskToggle = persistentMaskToggle;

    const areaPromptToggle = makeToggleButton("Area Prompt", () => {
        const w = findWidget(node, "area_prompt");
        if (!w) return;
        setWidget(w, !w.value);
        syncAreaPromptToggle(node);
        syncAreaPromptVisibility(node);
    });
    areaPromptToggle.title = "When ON, a text box appears between the toolbar and the canvas. Refines encode that text with the connected CLIP and use it instead of the main prompt — paint/drag a region and reshape it with a different prompt. Requires a CLIP input wired + non-empty area text. Forced ON in Smart Inpaint.";
    row1.appendChild(areaPromptToggle);
    node._AngeloAreaPromptToggle = areaPromptToggle;

    const paintModeToggle = makeToggleButton("Paint Mode", () => {
        const w = findWidget(node, "paint_mode");
        if (!w) return;
        setWidget(w, !w.value);
        syncPaintModeToggle(node);
    });
    paintModeToggle.title = "When ON, hold + drag on the preview paints a freeform brush stroke (each dragged point becomes a circle of click_radius; the union is the refine mask). Release to submit. When OFF, clicks behave as single-circle refines.";
    row1.appendChild(paintModeToggle);
    node._AngeloPaintModeToggle = paintModeToggle;

    const restoreToggle = makeToggleButton("Restore", () => {
        const w = findWidget(node, "restore_mode");
        if (!w) return;
        setWidget(w, !w.value);
        syncRestoreToggle(node);
    });
    restoreToggle.title = "Restore brush — when ON, clicks and paint strokes restore the painted "
        + "region back to the session's original base image instead of refining it. No sampling is "
        + "run. Feather applies for a soft blend; works with clicks, Paint Mode strokes, and Detect "
        + "masks. Refine mode only — Smart modes ignore it.";
    row1.appendChild(restoreToggle);
    node._AngeloRestoreToggle = restoreToggle;

    const fineUpscaleToggle = makeToggleButton("Xtra-Fine", () => {
        const w = findWidget(node, "fine_upscaling");
        if (!w) return;
        setWidget(w, !w.value);
        syncFineUpscaleToggle(node);
    });
    fineUpscaleToggle.title = "Xtra-Fine — refine the painted region at much higher effective resolution (ADetailer-style). The region is cropped, enlarged in pixel space to the MP target, re-encoded, refined, and composited back, so the model has room to render fine detail (faces, hands, eyes). Capped at Max scale.\n\nTip: pair it with Area Prompt — describe exactly what that region should be (e.g. \"detailed photorealistic face, sharp eyes\") for the strongest result.";
    row1.appendChild(fineUpscaleToggle);
    node._AngeloFineUpscaleToggle = fineUpscaleToggle;

    row1.appendChild(makeSeparator());

    // Inpainting Mode dropdown — Refine / Insert V1 / Insert V2.
    // Refine = current behaviour (best for refining existing content)
    // Smart Inpaint = drag a rectangle; locks denoise=1.0 / Fine Upscale=ON /
    //                 Ctx Pad=0; adds reference_latents so an edit model
    //                 (Klein 9B) sees the scene through its edit branch.
    const inpaintModeWidget = findWidget(node, "inpainting_mode");
    const inpaintModeOptions = (inpaintModeWidget && inpaintModeWidget.options && inpaintModeWidget.options.values)
        ? inpaintModeWidget.options.values
        : ["Refine", "Smart Inpaint", "Smart Guided Inpaint"];
    const inpaintModeSelect = makeDropdown("Inpaint",
        inpaintModeOptions,
        (val) => {
            const w = findWidget(node, "inpainting_mode");
            if (!w) return;
            setWidget(w, val);
            // Switching INTO Smart Inpaint: default feather to 15 (a soft
            // rectangle edge that blends the inserted content into the
            // surroundings — a useful default for this mode). It stays
            // user-adjustable afterwards — this only fires on the user's mode
            // pick, not on workflow load, so a saved feather value is
            // preserved across reloads.
            if (val === "Smart Inpaint") {
                const fw = findWidget(node, "feather_radius");
                if (fw) {
                    setWidget(fw, 15);
                    syncFeatherInput(node);
                }
            }
            // Mode change rewires the canvas interaction model — redraw
            // so the cursor / toolbar / overlays reflect it immediately.
            syncSmartInpaintLockedWidgets(node);
            redrawCanvasWithOverlays(node);
        }
    );
    inpaintModeSelect.title = "Inpainting Mode.\n\n"
        + "Refine — paint/click on the canvas to refine an existing region (faces, hands, textures). Partial-denoise from existing content.\n\n"
        + "Smart Inpaint — drag a rectangle on the canvas (click and hold one corner, release at the opposite). Adds NEW content in that region. Locks denoise=1.0 + Xtra-Fine=ON + Area Prompt=ON; injects reference_latents so an edit model's (FLUX 2 Klein 9B etc.) edit branch activates. Feather defaults to 15 (soft blend) but stays adjustable.\n\n"
        + "Smart Guided Inpaint — no painting or boxes. Pick a LOCATION from the dropdown above the Area Prompt (top left, center, bottom half, …); it's prepended to your prompt at run time (e.g. 'In the top left of the image, a red car') and the edit model places the content there across the whole image. Locks denoise=1.0 + Xtra-Fine=OFF + Area Prompt=ON; Feather and Persistent Mask disabled (no mask). Press 'Generate Guided Edit' to run. Coarse regions land most reliably.";
    row1.appendChild(inpaintModeSelect);
    node._AngeloInpaintModeSelect = inpaintModeSelect;

    // ===== ROW 2: numeric values =====
    const clickRadiusInput = makeNumberInput("Click R", { min: 8, max: 1024, step: 4, width: 56 }, (val) => {
        const w = findWidget(node, "click_radius");
        if (!w) return;
        setWidget(w, Math.round(val));
    });
    clickRadiusInput.title = "Pixel-space radius of the refinement region for a single click. Also the brush size in paint mode.\n\nKeyboard (cursor over canvas): [ to shrink, ] to grow.";
    row2.appendChild(clickRadiusInput);
    node._AngeloClickRadiusInput = clickRadiusInput;

    const featherInput = makeNumberInput("Feather", { min: 0, max: 256, step: 4, width: 56 }, (val) => {
        const w = findWidget(node, "feather_radius");
        if (!w) return;
        setWidget(w, Math.round(val));
    });
    featherInput.title = "Pixel-space gaussian blur on the mask edge. Smooths the seam between refined region and preserved surroundings. Roughly click_radius / 4 is a good default.\n\nKeyboard (cursor over canvas): { (shift+[) to shrink, } (shift+]) to grow.";
    row2.appendChild(featherInput);
    node._AngeloFeatherInput = featherInput;

    const denoiseInput = makeNumberInput("Denoise", { min: 0.05, max: 1.0, step: 0.05, width: 56 }, (val) => {
        const w = findWidget(node, "denoise");
        if (!w) return;
        setWidget(w, val);
    });
    denoiseInput.title = "How much of the sampler trajectory to run on the refine. 0.3 = subtle touch-up, 0.6 = real redo, 0.9+ = essentially regenerate that region.\n\nKeyboard (cursor over canvas): , to decrease, . to increase.";
    row2.appendChild(denoiseInput);
    node._AngeloDenoiseInput = denoiseInput;

    const seedInput = makeNumberInput("Seed", { min: 0, max: 0xFFFFFFFFFFFFFFFF, step: 1, width: 120 }, (val) => {
        const w = findWidget(node, "seed");
        if (!w) return;
        setWidget(w, Math.round(val));
    });
    seedInput.title = "[Edit Mode] Seed for the refine pass. After each click the Seed Ctrl dropdown decides what happens — fixed (leave alone), randomize (new random), increment (+1), decrement (-1).";
    row2.appendChild(seedInput);
    node._AngeloSeedInput = seedInput;

    const seedCtrlSelect = makeDropdown("Ctrl",
        ["fixed", "increment", "decrement", "randomize"],
        (val) => {
            const w = findWidget(node, "seed_control");
            if (!w) return;
            setWidget(w, val);
        }
    );
    seedCtrlSelect.title = "[Edit Mode] After-click seed behaviour. Mirrors ComfyUI's standard seed control_after_generate dropdown.";
    row2.appendChild(seedCtrlSelect);
    node._AngeloSeedCtrlSelect = seedCtrlSelect;

    // Xtra-Fine value group (separator + MP/Max/Ctx Pad/Method). These
    // values only affect sampling while Xtra-Fine is effectively ON.
    const fineSep = makeSeparator();
    row2.appendChild(fineSep);
    node._AngeloFineSep = fineSep;

    const mpInput = makeNumberInput("MP", { min: 0.1, max: 4.0, step: 0.1, width: 50 }, (val) => {
        const w = findWidget(node, "min_megapixels");
        if (!w) return;
        setWidget(w, val);
    });
    mpInput.title = "Xtra-Fine: target megapixels for the refine pass. Higher = bigger compute per click but sharper detail. Only used when Xtra-Fine is ON.";
    row2.appendChild(mpInput);
    node._AngeloMpInput = mpInput;

    const maxInput = makeNumberInput("Max", { min: 1.0, max: 16.0, step: 0.5, width: 50 }, (val) => {
        const w = findWidget(node, "max_upscale");
        if (!w) return;
        setWidget(w, val);
    });
    maxInput.title = "Xtra-Fine: hard cap on linear enlarge factor (8× = 64× area). Prevents pathological blow-up on tiny paints. Only used when Xtra-Fine is ON.";
    row2.appendChild(maxInput);
    node._AngeloMaxInput = maxInput;

    const ctxPadInput = makeNumberInput("Ctx Pad", { min: 0, max: 512, step: 8, width: 50 }, (val) => {
        const w = findWidget(node, "fine_context_pad");
        if (!w) return;
        setWidget(w, val);
    });
    ctxPadInput.title = "Xtra-Fine: pixel-space padding around the painted shape bbox before cropping. Gives the model surrounding context. Only used when Xtra-Fine is ON.";
    row2.appendChild(ctxPadInput);
    node._AngeloCtxPadInput = ctxPadInput;

    // Read the resize-method options from the underlying widget enum.
    const methodWidget = findWidget(node, "resize_method");
    const methodOptions = (methodWidget && methodWidget.options && methodWidget.options.values)
        ? methodWidget.options.values
        : ["nearest-exact", "bilinear", "area", "bicubic", "bislerp", "lanczos"];
    const methodSelect = makeDropdown("Method",
        methodOptions,
        (val) => {
            const w = findWidget(node, "resize_method");
            if (!w) return;
            setWidget(w, val);
        }
    );
    methodSelect.title = "Xtra-Fine: pixel-space enlarge method. lanczos = sharpest with mild ringing; bilinear = smooth (great for skin/faces); bicubic = middle; nearest-exact = blocky preserves exact values; bislerp/area = niche. Only used when Xtra-Fine is ON.";
    row2.appendChild(methodSelect);
    node._AngeloMethodSelect = methodSelect;

    // ===== DETECT ROW: SAM 3 auto-segment (Refine + Smart Inpaint) =====
    const detLabel = document.createElement("span");
    detLabel.textContent = "🔍 Detect:";
    detLabel.style.cssText = "font-size:11px; color:#bbb; padding:0 2px 0 4px;";
    detectRow.appendChild(detLabel);

    const detText = document.createElement("input");
    detText.type = "text";
    detText.placeholder = "what to segment (e.g. the face)";
    detText.style.cssText = "flex:1 1 0; min-width:0; background:#1a1a1a; color:#ddd; "
        + "border:1px solid #555; border-radius:3px; padding:2px 6px; font-size:11px;";
    detText.title = "SAM 3 concept prompt — a noun phrase describing what to find. "
        + "Hit Detect to highlight every match; click the one you want to refine/inpaint it.";
    for (const ev of ["pointerdown", "mousedown", "keydown", "keyup"]) {
        detText.addEventListener(ev, (e) => e.stopPropagation());
    }
    detText.addEventListener("keydown", (e) => { if (e.key === "Enter") runDetect(node); });
    detectRow.appendChild(detText);
    node._AngeloDetectText = detText;

    const detConf = makeNumberInput("Conf", { min: 0.05, max: 0.95, step: 0.05, width: 48 }, (val) => {
        // Persist the chosen confidence so it survives workflow reloads.
        try { localStorage.setItem("angelo_detect_conf", String(val)); } catch (e) { /* noop */ }
    });
    detConf.title = "Detection confidence threshold. Lower (≈0.2–0.3) finds more / fainter matches.";
    // Restore previously saved confidence, falling back to 0.3.
    if (detConf._AngeloInput) {
        try {
            const saved = parseFloat(localStorage.getItem("angelo_detect_conf") || "0.3");
            detConf._AngeloInput.value = String(Math.max(0.05, Math.min(0.95, isFinite(saved) ? saved : 0.3)));
        } catch (e) {
            detConf._AngeloInput.value = "0.3";
        }
    }
    detectRow.appendChild(detConf);
    node._AngeloDetectConf = detConf;

    const detBtn = makeActionButton("Detect", () => runDetect(node), "neutral");
    detBtn.title = "Run SAM 3 on the current preview and highlight matches. Click a highlight to confirm; Esc / click empty space to dismiss.";
    detectRow.appendChild(detBtn);
    node._AngeloDetectBtn = detBtn;

    // Space + separator between Detect and the quick-presets dropdown.
    detectRow.appendChild(makeSeparator());

    // Quick-detect presets — selecting one runs SAM 3 immediately with
    // that concept (does NOT change the text box). Resets to the
    // placeholder after each pick so the same item can be re-run.
    const quickSel = document.createElement("select");
    quickSel.style.cssText = "font-size:11px; padding:2px 4px; border:1px solid #555; "
        + "border-radius:3px; background:#1a1a1a; color:#ddd; margin-left:4px; min-width:104px;";
    quickSel.title = "Quick-detect a common subject — runs SAM 3 immediately. Doesn't touch the text box.";
    // Placeholder first (index 0 — reset target), then grouped presets.
    const _ph = document.createElement("option");
    _ph.value = "Quick Detect…"; _ph.textContent = "Quick Detect…";
    quickSel.appendChild(_ph);
    const _QUICK_GROUPS = {
        "People": ["Person", "Face", "Eyes", "Mouth", "Teeth", "Nose", "Ears", "Hair", "Skin", "Beard"],
        "Body": ["Hands", "Fingers", "Arms", "Legs", "Feet", "Torso"],
        "Clothing": ["Clothing", "Dress", "Shirt", "Jacket", "Pants", "Shoes", "Hat", "Glasses", "Jewelry", "Bag"],
        "Animals": ["Animal", "Dog", "Cat", "Bird", "Horse"],
        "Scene": ["Background", "Sky", "Clouds", "Sun", "Moon", "Water", "Tree", "Grass", "Flowers", "Mountains", "Road"],
        "Objects": ["Building", "Window", "Door", "Car", "Furniture", "Food", "Bottle", "Phone", "Text", "Logo"],
    };
    for (const [group, items] of Object.entries(_QUICK_GROUPS)) {
        const og = document.createElement("optgroup");
        og.label = group;
        for (const it of items) {
            const opt = document.createElement("option");
            opt.value = it;
            opt.textContent = it;
            og.appendChild(opt);
        }
        quickSel.appendChild(og);
    }
    quickSel.addEventListener("change", () => {
        const v = quickSel.value;
        quickSel.selectedIndex = 0;            // reset to "Quick Detect…"
        if (v && v !== "Quick Detect…") runDetect(node, v.toLowerCase());
    });
    quickSel.addEventListener("pointerdown", (e) => e.stopPropagation());
    quickSel.addEventListener("mousedown", (e) => e.stopPropagation());
    detectRow.appendChild(quickSel);
    node._AngeloDetectQuick = quickSel;

    // Mask grow / shrink — nudges ALL detected masks in/out together, 2px
    // at a time, so a tight SAM silhouette can be loosened (or a loose one
    // tightened) before committing. Pure-frontend: offsets the polygons /
    // bbox the JS already holds; the backend rasterises whatever it gets.
    detectRow.appendChild(makeSeparator());
    const maskLabel = document.createElement("span");
    maskLabel.textContent = "Mask:";
    maskLabel.style.cssText = "font-size:11px; color:#bbb; padding:0 1px; white-space:nowrap;";
    detectRow.appendChild(maskLabel);

    const mkGrowBtn = (txt, delta, tip) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = txt;
        b.title = tip;
        b.style.cssText = "cursor:pointer; padding:2px 7px; font-size:13px; font-weight:bold; "
            + "border:1px solid #555; border-radius:3px; background:#2a2a2a; color:#ddd; "
            + "line-height:1; user-select:none; flex:0 0 auto;";
        b.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); adjustMaskGrow(node, delta); });
        b.addEventListener("pointerdown", (e) => e.stopPropagation());
        return b;
    };
    detectRow.appendChild(mkGrowBtn("−", -2, "Shrink all detected masks by 2px"));
    const growReadout = document.createElement("span");
    growReadout.textContent = "0px";
    growReadout.style.cssText = "font-size:11px; color:#9cf; min-width:34px; text-align:center; white-space:nowrap;";
    detectRow.appendChild(growReadout);
    node._AngeloMaskGrowReadout = growReadout;
    detectRow.appendChild(mkGrowBtn("+", 2, "Grow all detected masks by 2px"));

    // "Unload SAM3" button — frees CPU RAM used by the cached model.
    // Shown only after the first successful Detect. Re-clicking Detect
    // will reload the model automatically on demand.
    detectRow.appendChild(makeSeparator());
    const unloadSam3Btn = makeActionButton("Unload SAM3", () => {
        api.fetchApi("/angelo/unload_sam3", { method: "POST" })
            .then(r => r.json())
            .then(d => _angeloToast(d.message || "SAM 3 unloaded"))
            .catch(() => _angeloToast("Unload SAM3 failed — see console"));
    }, "neutral");
    unloadSam3Btn.title = "Release the SAM 3 model from CPU RAM. "
        + "The next Detect will reload it automatically. "
        + "Use this to free memory on low-RAM systems between detect sessions.";
    detectRow.appendChild(unloadSam3Btn);

    // ===== MODE ROW: the master Sampler/Edit switch, centred up top =====
    const modeWidget = findWidget(node, "mode");
    const modeOptions = (modeWidget && modeWidget.options && modeWidget.options.values)
        ? modeWidget.options.values
        : ["Edit Mode", "Sampler Mode"];
    const modeSelect = makeDropdown("Mode",
        modeOptions,
        (val) => {
            const w = findWidget(node, "mode");
            if (!w) return;
            setWidget(w, val);   // fires the wrapped mode callback (lock + grey sync)
        }
    );
    modeSelect.title = "Sampler Mode = generate a fresh base latent from the inputs (acts like a KSampler). Edit Mode = click/drag the preview to refine or inpaint the cached latent. Switching to Edit Mode auto-locks the sampler seed to the value that produced the base.";
    modeRow.appendChild(modeSelect);
    node._AngeloModeSelect = modeSelect;

    // Floating detect-mode panel — pinned top-right of the Mode row, shown
    // only while candidates are active. Holds the red Cancel button + an
    // opacity slider for the highlight overlay (drag down to peek at the
    // edges of a just-generated region). Styled as one neat cluster.
    node._AngeloDetOpacity = 1.0;
    const detectPanel = document.createElement("div");
    detectPanel.style.cssText = "position:absolute; right:6px; top:4px; z-index:6; display:none; "
        + "flex-direction:column; align-items:stretch; gap:4px; padding:5px 6px; "
        + "background:rgba(20,20,20,0.82); border:1px solid #555; border-radius:5px;";

    const cancelDetectBtn = document.createElement("button");
    cancelDetectBtn.type = "button";
    cancelDetectBtn.textContent = "✕ Cancel Detect";
    cancelDetectBtn.title = "Leave detect mode (you can also press Esc or Space). "
        + "The highlighted candidates stay up so you can edit each one until you cancel.";
    cancelDetectBtn.style.cssText = "font-size:11px; font-weight:bold; padding:3px 10px; "
        + "border:1px solid #e66; border-radius:3px; background:rgba(200,40,40,0.95); "
        + "color:#fff; cursor:pointer;";
    for (const ev of ["pointerdown", "mousedown"]) {
        cancelDetectBtn.addEventListener(ev, (e) => e.stopPropagation());
    }
    cancelDetectBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        clearDetections(node);
    });

    const opRow = document.createElement("div");
    opRow.style.cssText = "display:flex; align-items:center; gap:6px; font-size:10px; color:#ccd;";
    const opLabel = document.createElement("span");
    opLabel.textContent = "Highlight";
    const opSlider = document.createElement("input");
    opSlider.type = "range";
    opSlider.min = "0"; opSlider.max = "1"; opSlider.step = "0.05"; opSlider.value = "1";
    opSlider.style.cssText = "flex:1 1 auto; width:96px; cursor:pointer;";
    opSlider.title = "Selection-highlight opacity — drag down to peek at the edges of what you just generated. Candidates stay clickable.";
    for (const ev of ["pointerdown", "mousedown"]) {
        opSlider.addEventListener(ev, (e) => e.stopPropagation());
    }
    opSlider.addEventListener("input", () => {
        node._AngeloDetOpacity = parseFloat(opSlider.value);
        redrawCanvasWithOverlays(node);
    });
    opRow.appendChild(opLabel);
    opRow.appendChild(opSlider);

    // Touch-up brush hint (the brush is modifier-driven, so flag it here).
    const brushHint = document.createElement("div");
    brushHint.textContent = "Refine: Shift-drag = +mask · Alt-drag = −mask";
    brushHint.style.cssText = "font-size:9px; color:#9aa; white-space:nowrap; text-align:center;";
    brushHint.title = "In Refine, hold Shift and drag on the preview to grow the mask you start over, "
        + "or Alt-drag to carve it back (holes allowed). Brush size = Click R. Then click the candidate to apply.";

    detectPanel.appendChild(cancelDetectBtn);
    detectPanel.appendChild(opRow);
    detectPanel.appendChild(brushHint);
    modeRow.appendChild(detectPanel);
    node._AngeloCancelDetectBtn = cancelDetectBtn;
    node._AngeloDetectPanel = detectPanel;
    node._AngeloDetOpacitySlider = opSlider;

    // ===== ROW 3: Load Image (always active) =====
    // Steps widget removed — step count is determined entirely by the
    // connected Flux2Scheduler (len(sigmas)-1 for Sampler Mode,
    // round((len(sigmas)-1) × denoise) for Edit Mode).

    // Load Image — bring an external photo in as the base to edit. Always
    // active (both modes); no Empty Latent needed.
    row3.appendChild(makeSeparator());
    const loadImgBtn = makeActionButton("🖼 Load Image", () => triggerLoadImage(node), "neutral");
    loadImgBtn.title = "Load an external image as the base to edit / refine. "
        + "You'll be asked to keep its resolution or resize to a target "
        + "megapixel (both rounded to a /16 multiple). The image becomes the "
        + "base — Reset and Undo return to it. While loaded, the latent input "
        + "is ignored (hit Unload to go back to it). No Empty Latent needed.";
    row3.appendChild(loadImgBtn);
    node._AngeloLoadImageBtn = loadImgBtn;

    const unloadImgBtn = makeActionButton("✕ Unload", () => unloadImage(node), "neutral");
    unloadImgBtn.title = "Clear the loaded image and return to the wired latent input "
        + "as the base. Shown only while an image is loaded.";
    unloadImgBtn.style.display = "none";
    row3.appendChild(unloadImgBtn);
    node._AngeloUnloadImageBtn = unloadImgBtn;

    // ===== ROW 4: Sampler-Mode seed group (greyed in Edit Mode) =====
    const samplerSeedInput = makeNumberInput("Smpl Seed", { min: 0, max: 0xFFFFFFFFFFFFFFFF, step: 1, width: 120 }, (val) => {
        const w = findWidget(node, "sampler_seed");
        if (!w) return;
        setWidget(w, Math.round(val));
    });
    samplerSeedInput.title = "[Sampler Mode] Seed for the base generation. After each run the Sampler Ctrl dropdown decides what happens to it.";
    row4.appendChild(samplerSeedInput);
    node._AngeloSamplerSeedInput = samplerSeedInput;

    const samplerSeedCtrlSelect = makeDropdown("Smpl Ctrl",
        ["fixed", "increment", "decrement", "randomize"],
        (val) => {
            const w = findWidget(node, "sampler_seed_control");
            if (!w) return;
            setWidget(w, val);
        }
    );
    samplerSeedCtrlSelect.title = "[Sampler Mode] After-generate seed behaviour for the base. Auto-forced to 'fixed' when you switch to Edit Mode so re-queues don't regenerate the base.";
    row4.appendChild(samplerSeedCtrlSelect);
    node._AngeloSamplerSeedCtrlSelect = samplerSeedCtrlSelect;


    container.appendChild(toggleBarWrap);

    // --- Area Prompt input — sits BETWEEN the toolbar and the canvas,
    //     and only shows when Area Prompt is ON (or Smart Inpaint forces
    //     it on). One textarea with a Pos/Neg toggle deciding which
    //     underlying widget it edits (area_text_positive /
    //     area_text_negative). Hiding it never clears the text — that
    //     lives in the widgets, and syncAreaPromptBox reloads it. ---
    attachAreaPromptBox(node, container);

    // Canvas + placeholder live in their own relative-positioned wrap
    // so the absolutely-positioned placeholder overlays ONLY the canvas.
    // The wrap flex-grows to fill whatever vertical space is left after
    // the toolbar + area-prompt box, and centres the canvas inside it.
    // The canvas's DISPLAY size is computed in JS (fitCanvasDisplaySize)
    // to fit this wrap while preserving aspect ratio — letterboxed by
    // the empty space around it, NOT by object-fit (which would break
    // the click-coordinate mapping that reads canvas.getBoundingClientRect).
    const canvasWrap = document.createElement("div");
    canvasWrap.style.position = "relative";
    canvasWrap.style.width = "100%";
    canvasWrap.style.flex = "1 1 auto";
    canvasWrap.style.minHeight = "0";
    canvasWrap.style.display = "flex";
    canvasWrap.style.alignItems = "center";
    canvasWrap.style.justifyContent = "center";
    canvasWrap.style.overflow = "hidden";

    // The canvas is ABSOLUTELY positioned inside the wrap — applyView()
    // sets its width/height (= fit size × zoom) and left/top (centre +
    // pan). This lets zoom>1 overflow the wrap (clipped by overflow:
    // hidden) while click/paint/overlay mapping stays correct, because
    // they all read canvas.getBoundingClientRect() which reflects the
    // live size+position. No max-width/height (would block zoom>1).
    const canvas = document.createElement("canvas");
    canvas.style.display = "block";
    canvas.style.position = "absolute";
    canvas.style.cursor = "crosshair";
    canvas.width = 512;
    canvas.height = 512;
    canvasWrap.appendChild(canvas);

    // Placeholder text shown until the first image arrives
    const placeholder = document.createElement("div");
    placeholder.textContent = "Queue the workflow to generate a preview.\nClick a region in the preview to refine it.";
    placeholder.style.position = "absolute";
    placeholder.style.inset = "0";
    placeholder.style.display = "flex";
    placeholder.style.alignItems = "center";
    placeholder.style.justifyContent = "center";
    placeholder.style.textAlign = "center";
    placeholder.style.color = "#888";
    placeholder.style.padding = "20px";
    placeholder.style.whiteSpace = "pre-line";
    placeholder.style.pointerEvents = "none";
    placeholder.style.fontSize = "12px";
    canvasWrap.appendChild(placeholder);

    // Corner minimap navigator — shown only when zoomed in (zoom > 1).
    const minimap = document.createElement("canvas");
    minimap.style.position = "absolute";
    minimap.style.right = "6px";
    minimap.style.bottom = "6px";
    minimap.style.border = "1px solid rgba(255,255,255,0.35)";
    minimap.style.borderRadius = "2px";
    minimap.style.background = "rgba(0,0,0,0.4)";
    minimap.style.pointerEvents = "none";
    minimap.style.display = "none";
    minimap.style.zIndex = "5";
    canvasWrap.appendChild(minimap);
    node._AngeloMinimap = minimap;

    // Persistent in-app notice bar — overlays the top of the preview for
    // actionable messages (e.g. "SAM 3 Detect isn't installed — run the
    // installer"). Stays until dismissed (✕) or a detect succeeds; this is
    // NOT a transient toast, because the message needs reading + acting on.
    const notice = document.createElement("div");
    notice.style.cssText = "position:absolute; left:0; right:0; top:0; z-index:7; display:none; "
        + "padding:8px 28px 8px 10px; background:rgba(150,40,40,0.96); color:#fff; "
        + "font:12px/1.45 Arial,sans-serif; white-space:pre-line; "
        + "border-bottom:1px solid rgba(255,255,255,0.25);";
    const noticeText = document.createElement("span");
    notice.appendChild(noticeText);
    const noticeClose = document.createElement("button");
    noticeClose.type = "button";
    noticeClose.textContent = "✕";
    noticeClose.title = "Dismiss";
    noticeClose.style.cssText = "position:absolute; right:5px; top:5px; background:transparent; "
        + "border:none; color:#fff; font-size:14px; line-height:1; cursor:pointer;";
    for (const ev of ["pointerdown", "mousedown"]) {
        noticeClose.addEventListener(ev, (e) => e.stopPropagation());
    }
    noticeClose.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); notice.style.display = "none"; });
    notice.appendChild(noticeClose);
    canvasWrap.appendChild(notice);
    node._AngeloNotice = notice;
    node._AngeloNoticeText = noticeText;

    // Loading overlay — shown while a Detect request is in flight. SAM 3's
    // FIRST detect has to build + cache the model (several seconds), during
    // which the canvas would otherwise look frozen before the first outline
    // appears. Auto-dismisses when the detect resolves; the ✕ is a manual
    // escape hatch so a hung/crashed request can never trap the overlay up.
    if (!document.getElementById("angelo-spin-style")) {
        const s = document.createElement("style");
        s.id = "angelo-spin-style";
        s.textContent = "@keyframes angelo-spin{to{transform:rotate(360deg)}}";
        document.head.appendChild(s);
    }
    const loading = document.createElement("div");
    loading.style.cssText = "position:absolute; inset:0; z-index:8; display:none; "
        + "align-items:center; justify-content:center; background:rgba(0,0,0,0.55);";
    const loadingBox = document.createElement("div");
    loadingBox.style.cssText = "position:relative; display:flex; align-items:center; gap:11px; "
        + "padding:15px 36px 15px 18px; background:rgba(28,28,28,0.97); color:#fff; "
        + "font:13px/1.4 Arial,sans-serif; border-radius:8px; "
        + "border:1px solid rgba(255,255,255,0.18); box-shadow:0 4px 16px rgba(0,0,0,0.55);";
    const spinner = document.createElement("div");
    spinner.style.cssText = "flex:0 0 auto; width:16px; height:16px; "
        + "border:2px solid rgba(255,255,255,0.25); border-top-color:#fff; "
        + "border-radius:50%; animation:angelo-spin 0.8s linear infinite;";
    const loadingText = document.createElement("span");
    loadingText.textContent = "Loading SAM 3…";
    loadingBox.appendChild(spinner);
    loadingBox.appendChild(loadingText);
    const loadingClose = document.createElement("button");
    loadingClose.type = "button";
    loadingClose.textContent = "✕";
    loadingClose.title = "Dismiss";
    loadingClose.style.cssText = "position:absolute; right:6px; top:4px; background:transparent; "
        + "border:none; color:#fff; font-size:13px; line-height:1; cursor:pointer;";
    for (const ev of ["pointerdown", "mousedown"]) {
        loadingClose.addEventListener(ev, (e) => e.stopPropagation());
    }
    loadingClose.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); loading.style.display = "none"; });
    loadingBox.appendChild(loadingClose);
    loading.appendChild(loadingBox);
    canvasWrap.appendChild(loading);
    node._AngeloLoading = loading;
    node._AngeloLoadingText = loadingText;

    // Per-node view state (zoom/pan). zoom 1 = fit; pan in CSS px.
    node._AngeloZoom = 1;
    node._AngeloPanX = 0;
    node._AngeloPanY = 0;
    node._AngeloBaseW = 0;
    node._AngeloBaseH = 0;

    container.appendChild(canvasWrap);

    // --- Zoom / pan (view layer, independent of the refine pipeline) ---
    //   • Wheel       → zoom toward the cursor (clamped 0.25–8×).
    //   • Middle-drag → pan.
    //   • Double-middle-click, or 'F' (cursor over node) → reset to fit.
    // While zoomed/panned the auto-fit (fitCanvasDisplaySize) is suppressed
    // so it never stomps the manual view; reset / new image restore fit.

    canvasWrap.addEventListener("wheel", (event) => {
        if (!node._AngeloImg || !node._AngeloBaseW) return;
        event.preventDefault();
        event.stopPropagation();   // don't also zoom the ComfyUI graph
        const wrapW = canvasWrap.clientWidth, wrapH = canvasWrap.clientHeight;
        const wrapRect = canvasWrap.getBoundingClientRect();
        // Convert cursor from viewport/visual pixels to layout pixels.
        // ComfyUI applies a CSS transform (scale) on the graph container;
        // getBoundingClientRect() reflects that scale while clientWidth does
        // not. Without this correction the cursor coordinate is wrong when
        // the graph is zoomed, causing the image to drift sideways on wheel.
        const graphScaleX = wrapRect.width > 0 ? wrapRect.width / wrapW : 1;
        const graphScaleY = wrapRect.height > 0 ? wrapRect.height / wrapH : 1;
        const cx = (event.clientX - wrapRect.left) / graphScaleX;
        const cy = (event.clientY - wrapRect.top) / graphScaleY;
        const oldZoom = node._AngeloZoom || 1;
        const factor = event.deltaY < 0 ? 1.15 : (1 / 1.15);
        const newZoom = Math.max(0.25, Math.min(8, oldZoom * factor));
        if (newZoom === oldZoom) return;
        // Wheeling back through ~1× snaps to a clean fit (zoom=1, no pan).
        // Without this, float drift leaves zoom at e.g. 1.0000002, which
        // keeps the minimap up and the auto-fit suppressed at what looks
        // like fit.
        if (Math.abs(newZoom - 1) < 0.01) {
            resetView(node);
            redrawCanvasWithOverlays(node);
            return;
        }
        const baseW = node._AngeloBaseW, baseH = node._AngeloBaseH;
        const oldW = baseW * oldZoom, oldH = baseH * oldZoom;
        const oldLeft = (wrapW - oldW) / 2 + (node._AngeloPanX || 0);
        const oldTop = (wrapH - oldH) / 2 + (node._AngeloPanY || 0);
        // Normalised image point currently under the cursor.
        const nx = (cx - oldLeft) / oldW;
        const ny = (cy - oldTop) / oldH;
        const newW = baseW * newZoom, newH = baseH * newZoom;
        // Solve pan so that same normalised point stays under the cursor.
        node._AngeloZoom = newZoom;
        node._AngeloPanX = cx - nx * newW - (wrapW - newW) / 2;
        node._AngeloPanY = cy - ny * newH - (wrapH - newH) / 2;
        applyView(node);
        redrawCanvasWithOverlays(node);
    }, { passive: false });

    // Suppress the Windows middle-click autoscroll cursor.
    canvasWrap.addEventListener("mousedown", (event) => {
        if (event.button === 1) event.preventDefault();
    });

    canvasWrap.addEventListener("pointerdown", (event) => {
        if (event.button !== 1) return;   // middle button = pan / reset
        event.preventDefault();
        event.stopPropagation();   // don't let litegraph treat it as a node drag
        const now = performance.now();
        if (node._AngeloLastMiddleDown && (now - node._AngeloLastMiddleDown) < 350) {
            // Double middle-click → reset to fit.
            node._AngeloLastMiddleDown = 0;
            node._AngeloPanning = null;
            resetView(node);
            redrawCanvasWithOverlays(node);
            return;
        }
        node._AngeloLastMiddleDown = now;
        try { canvasWrap.setPointerCapture(event.pointerId); } catch (e) { /* noop */ }
        node._AngeloPanning = {
            startX: event.clientX, startY: event.clientY,
            startPanX: node._AngeloPanX || 0, startPanY: node._AngeloPanY || 0,
            pointerId: event.pointerId,
        };
        if (node._AngeloCanvas) node._AngeloCanvas.style.cursor = "grabbing";
    });

    canvasWrap.addEventListener("pointermove", (event) => {
        const p = node._AngeloPanning;
        if (!p) return;
        node._AngeloPanX = p.startPanX + (event.clientX - p.startX);
        node._AngeloPanY = p.startPanY + (event.clientY - p.startY);
        applyView(node);
    });

    function endAngeloPan() {
        const p = node._AngeloPanning;
        if (!p) return;
        node._AngeloPanning = null;
        try { canvasWrap.releasePointerCapture(p.pointerId); } catch (e) { /* noop */ }
        redrawCanvasWithOverlays(node);   // restores the mode-appropriate cursor
    }
    canvasWrap.addEventListener("pointerup", (event) => {
        if (event.button === 1) endAngeloPan();
    });
    canvasWrap.addEventListener("pointercancel", endAngeloPan);

    // --- Right-click menu + drag-drop image loading (#7) ---
    // Right-click the preview → Open / Copy / Paste. The preview is a DOM
    // <canvas>, so we show our own LiteGraph menu here (the node's
    // getExtraMenuOptions only fires on the litegraph-rendered node body).
    canvasWrap.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();
        _angeloShowImageContextMenu(node, event);
    });

    // Drag-drop an OS image file onto the node → load it as the base, through
    // the SAME resolution popup + upload path as the Load Image button
    // (showLoadImagePopup). stopPropagation so the drop doesn't bubble to the
    // graph canvas (which would otherwise spawn a LoadImage node).
    const _dropHi = () => { container.style.outline = "2px dashed rgba(120,170,220,0.9)"; container.style.outlineOffset = "-2px"; };
    const _dropLo = () => { container.style.outline = ""; container.style.outlineOffset = ""; };
    container.addEventListener("dragover", (event) => {
        const dt = event.dataTransfer;
        if (dt && Array.from(dt.items || []).some((i) => i.kind === "file")) {
            event.preventDefault();
            event.stopPropagation();
            dt.dropEffect = "copy";
            _dropHi();
        }
    });
    container.addEventListener("dragleave", () => {
        // Clear unconditionally; a still-active drag re-adds it via dragover.
        _dropLo();
    });
    container.addEventListener("drop", (event) => {
        const dt = event.dataTransfer;
        const file = dt && dt.files && Array.from(dt.files).find((f) => f.type && f.type.startsWith("image/"));
        _dropLo();
        if (!file) return;
        event.preventDefault();
        event.stopPropagation();
        showLoadImagePopup(node, file);
    });

    // --- Pointer events (instead of mouse*) + pointer capture so long
    //     drags don't get cancelled when the cursor leaves the canvas
    //     boundary briefly. Pointer capture routes all subsequent move/
    //     up events to the canvas until the drag ends. ---

    // Helper: clamp event coordinates to the canvas's CSS rect, then
    // convert to canvas-intrinsic (= image-pixel) coordinates.
    function eventToImagePixel(event) {
        const rect = canvas.getBoundingClientRect();
        let cx = event.clientX - rect.left;
        let cy = event.clientY - rect.top;
        // Clamp so off-canvas drag positions still produce valid points
        // at the canvas edge rather than negative / oversized values.
        cx = Math.max(0, Math.min(rect.width, cx));
        cy = Math.max(0, Math.min(rect.height, cy));
        const img = node._AngeloImg;
        if (!img || !img.naturalWidth || rect.width === 0 || rect.height === 0) {
            return null;
        }
        return {
            cssX: cx,
            cssY: cy,
            pixelX: (cx / rect.width) * img.naturalWidth,
            pixelY: (cy / rect.height) * img.naturalHeight,
        };
    }

    canvas.addEventListener("pointermove", (event) => {
        const p = eventToImagePixel(event);
        if (!p) return;
        node._AngeloHover = { x: p.cssX, y: p.cssY };

        // Detection select mode.
        if (node._AngeloDetections && node._AngeloDetections.length) {
            // Active touch-up stroke → extend the brush along the drag.
            if (node._AngeloTouchup) {
                const tu = node._AngeloTouchup;
                _brushLine(tu.det, tu.last, [p.pixelX, p.pixelY], _brushRadius(node), tu.subtract);
                tu.last = [p.pixelX, p.pixelY];
                redrawCanvasWithOverlays(node);
                return;
            }
            // Brush preview while Shift/Alt is held (Refine only).
            const brushKey = (event.shiftKey || event.altKey)
                && !isSmartInpaintMode(node) && !isSmartGuidedInpaintMode(node);
            if (brushKey) {
                node._AngeloBrushPreview = { x: p.pixelX, y: p.pixelY, r: _brushRadius(node), subtract: event.altKey };
                canvas.style.cursor = "crosshair";
                node._AngeloHoverDet = -1;
                redrawCanvasWithOverlays(node);
                return;
            }
            if (node._AngeloBrushPreview) node._AngeloBrushPreview = null;
            // Otherwise highlight the candidate under the cursor.
            const det = _detAtPoint(node, p.pixelX, p.pixelY);
            const idx = det ? node._AngeloDetections.indexOf(det) : -1;
            canvas.style.cursor = idx >= 0 ? "pointer" : "default";
            if (idx !== node._AngeloHoverDet) {
                node._AngeloHoverDet = idx;
                redrawCanvasWithOverlays(node);
            }
            return;
        }

        if (node._AngeloDraggingRect) {
            // Smart Inpaint: update the live opposite-corner of the
            // drag-out rectangle as the user moves the cursor.
            node._AngeloDraggingRect.x2 = p.pixelX;
            node._AngeloDraggingRect.y2 = p.pixelY;
            node._AngeloDraggingRect.cssX2 = p.cssX;
            node._AngeloDraggingRect.cssY2 = p.cssY;
        } else if (node._AngeloPainting) {
            const stroke = node._AngeloStroke;
            const last = stroke[stroke.length - 1];
            // Dedup at 2px (image-pixel space) — saves bandwidth.
            if (!last || Math.hypot(last[0] - p.pixelX, last[1] - p.pixelY) > 2) {
                stroke.push([p.pixelX, p.pixelY]);
            }
        }
        redrawCanvasWithOverlays(node);
    });

    canvas.addEventListener("pointerleave", () => {
        node._AngeloHover = null;
        if (node._AngeloBrushPreview) node._AngeloBrushPreview = null;
        // IMPORTANT: do NOT cancel an active paint stroke here. With
        // pointer capture set on pointerdown, we keep receiving move/up
        // events even when the cursor leaves the canvas — long strokes
        // can briefly cross the boundary without breaking.
        redrawCanvasWithOverlays(node);
        if (_AngeloHoveredNode === node) _AngeloHoveredNode = null;
    });

    // Track keyboard-shortcut hover ownership separately from the
    // paint-stroke pointer events above. pointerenter fires when the
    // cursor crosses INTO the canvas — at that point this node owns the
    // keyboard shortcuts.
    canvas.addEventListener("pointerenter", () => {
        _AngeloHoveredNode = node;
    });

    canvas.addEventListener("pointerdown", (event) => {
        if (event.button !== 0) return;
        // Detection select mode owns the canvas — let the click confirm a
        // candidate; don't start a paint stroke or rectangle drag. EXCEPT a
        // Shift/Alt drag, which is the touch-up brush (Refine only): Shift
        // adds to the first-overlapped candidate's mask, Alt subtracts.
        if (node._AngeloDetections && node._AngeloDetections.length) {
            const brushKey = (event.shiftKey || event.altKey)
                && !isSmartInpaintMode(node) && !isSmartGuidedInpaintMode(node);
            if (brushKey) {
                const pp = eventToImagePixel(event);
                if (!pp) return;
                const target = _pickTouchupTarget(node, pp.pixelX, pp.pixelY);
                if (!target) return;
                try { canvas.setPointerCapture(event.pointerId); node._AngeloPointerId = event.pointerId; } catch (e) { /* noop */ }
                const subtract = event.altKey;
                _ensureEditMask(node, target);
                _brushStamp(target, pp.pixelX, pp.pixelY, _brushRadius(node), subtract);
                node._AngeloTouchup = { det: target, subtract, last: [pp.pixelX, pp.pixelY] };
                node._AngeloBrushPreview = null;
                redrawCanvasWithOverlays(node);
            }
            return;
        }
        // Smart Guided Inpaint has no canvas interaction at all — the
        // location comes from the dropdown, the run from the button.
        if (isSmartGuidedInpaintMode(node)) return;
        const smartInpaint = isSmartInpaintMode(node);
        const paintOn = isPaintModeOn(node);
        dbg("pointerdown", { smartInpaint, paintModeOn: paintOn });

        // Smart Inpaint always owns the canvas — drag-out rectangle.
        // It takes priority over paint_mode.
        if (!smartInpaint && !paintOn) return;
        const p = eventToImagePixel(event);
        if (!p) return;

        // Pointer capture: route all subsequent pointer events to this
        // element until pointerup, regardless of cursor position.
        try {
            canvas.setPointerCapture(event.pointerId);
            node._AngeloPointerId = event.pointerId;
        } catch (e) {
            dbg("setPointerCapture failed", e);
        }

        if (smartInpaint) {
            node._AngeloDraggingRect = {
                x1: p.pixelX, y1: p.pixelY,
                x2: p.pixelX, y2: p.pixelY,
                cssX1: p.cssX, cssY1: p.cssY,
                cssX2: p.cssX, cssY2: p.cssY,
            };
        } else {
            node._AngeloPainting = true;
            node._AngeloStroke = [[p.pixelX, p.pixelY]];
        }
        redrawCanvasWithOverlays(node);
        // Don't preventDefault on pointerdown — we still want pointermove
        // to fire normally. The subsequent "click" event won't fire if
        // any meaningful drag occurred (browser behaviour), and our
        // click handler also short-circuits when paint_mode is on.
    });

    function endPaintStroke(event) {
        if (!node._AngeloPainting) return;
        node._AngeloPainting = false;
        if (node._AngeloPointerId !== undefined) {
            try { canvas.releasePointerCapture(node._AngeloPointerId); }
            catch (e) { /* already released */ }
            node._AngeloPointerId = undefined;
        }
        const stroke = node._AngeloStroke || [];
        node._AngeloStroke = null;
        if (stroke.length === 0) {
            redrawCanvasWithOverlays(node);
            return;
        }
        dbg("paint stroke submitted", { points: stroke.length });
        triggerPaintRefine(node, stroke);
        redrawCanvasWithOverlays(node);
    }

    function endRectDrag() {
        if (!node._AngeloDraggingRect) return;
        const r = node._AngeloDraggingRect;
        node._AngeloDraggingRect = null;
        if (node._AngeloPointerId !== undefined) {
            try { canvas.releasePointerCapture(node._AngeloPointerId); }
            catch (e) { /* already released */ }
            node._AngeloPointerId = undefined;
        }
        const dx = Math.abs(r.x2 - r.x1);
        const dy = Math.abs(r.y2 - r.y1);
        // Reject degenerate (single-click) drags — Smart Inpaint needs
        // an actual rectangle. Threshold in image-pixel space.
        if (dx < 8 || dy < 8) {
            dbg("rect drag too small — ignored", { dx, dy });
            redrawCanvasWithOverlays(node);
            return;
        }
        dbg("smart inpaint rect submitted", r);
        triggerRectRefine(node, [r.x1, r.y1, r.x2, r.y2]);
        redrawCanvasWithOverlays(node);
    }

    function endTouchup() {
        if (!node._AngeloTouchup) return;
        node._AngeloTouchup = null;
        if (node._AngeloPointerId !== undefined) {
            try { canvas.releasePointerCapture(node._AngeloPointerId); }
            catch (e) { /* already released */ }
            node._AngeloPointerId = undefined;
        }
        redrawCanvasWithOverlays(node);
    }

    canvas.addEventListener("pointerup", (event) => {
        if (event.button !== 0) return;
        if (node._AngeloTouchup) endTouchup();
        else if (node._AngeloDraggingRect) endRectDrag();
        else endPaintStroke(event);
    });
    canvas.addEventListener("pointercancel", (event) => {
        if (node._AngeloTouchup) endTouchup();
        else if (node._AngeloDraggingRect) endRectDrag();
        else endPaintStroke(event);
    });

    // --- Single-click refine (click mode only — paint mode and
    //     Smart Inpaint handle the canvas via pointer drag above). ---
    canvas.addEventListener("click", (event) => {
        // Detection select mode owns the click (Refine + Smart Inpaint):
        // clicking a candidate edits it and keeps the rest up for more
        // edits. Empty-space clicks do nothing (so you can't accidentally
        // exit mid-batch) — leave via Cancel Detect / Esc / Space.
        if (node._AngeloDetections && node._AngeloDetections.length) {
            // Shift/Alt click is the touch-up brush, not a confirm.
            if (event.shiftKey || event.altKey) return;
            const p = eventToImagePixel(event);
            const det = p ? _detAtPoint(node, p.pixelX, p.pixelY) : null;
            if (det) confirmDetection(node, det);
            return;
        }
        if (isSmartGuidedInpaintMode(node)) return; // no canvas interaction
        if (isSmartInpaintMode(node)) return; // rectangle-drag owns it
        if (isPaintModeOn(node)) return; // paint mode owns the interaction
        const rect = canvas.getBoundingClientRect();
        const cx = event.clientX - rect.left;
        const cy = event.clientY - rect.top;
        const img = node._AngeloImg;
        if (!img || !img.naturalWidth) {
            dbg("click ignored — no image loaded yet");
            return;
        }
        const pixelX = Math.floor((cx / rect.width) * img.naturalWidth);
        const pixelY = Math.floor((cy / rect.height) * img.naturalHeight);
        dbg("click", { cx, cy, pixelX, pixelY });
        triggerRefine(node, pixelX, pixelY, cx, cy);
        flashClickOverlay(node, cx, cy);
    });

    // Add as a DOM widget on the node so LiteGraph manages its layout.
    // getMinHeight is now aspect-ratio aware: it returns enough height for
    // the toolbar PLUS a canvas that fits the current image at the node's
    // width, clamped between _PREVIEW_MIN_CANVAS_H and _PREVIEW_MAX_H.
    // When no image is loaded yet a sensible placeholder height is used.
    const widget = node.addDOMWidget("Angelo_preview_canvas", "Angelo_canvas", container, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => {
            const img = node._AngeloImg;
            if (!img || !img.naturalWidth || !img.naturalHeight) {
                return _TOOLBAR_H_APPROX + _PREVIEW_MIN_CANVAS_H;
            }
            // Derive ideal canvas height from image AR and current node width.
            const nodeW = Math.max(1, (node.size ? node.size[0] : 380) - 12);
            const ar = img.naturalHeight / img.naturalWidth;
            const idealCanvasH = Math.round(Math.min(nodeW, _PREVIEW_MAX_W) * ar);
            const canvasH = Math.max(_PREVIEW_MIN_CANVAS_H, Math.min(_PREVIEW_MAX_H, idealCanvasH));
            return _TOOLBAR_H_APPROX + canvasH;
        },
    });

    // Make the preview widget fill the node's full width.
    //
    // LiteGraph computes a DOM widget's wrapper-element width as
    //     (widget.width ?? node.width) - 2 * margin
    // (see the Vue node renderer). The `?? node.width` fallback gives full
    // width ONLY while `widget.width` is unset; once LiteGraph's widget-draw
    // path assigns `widget.width` (it sets it to the narrow content-min when
    // the node's hidden widgets report ~340), that pinned value wins. So a
    // node dragged wide shows a wide frame but the preview collapses to ~340
    // on the left, and any relayout (e.g. clicking a control) re-pins it.
    // Defining `width` as a getter that always returns the live node width
    // keeps the wrapper at full width and ignores the narrow re-pin. Setter is
    // a no-op so LiteGraph's assignment can't shrink it again.
    try {
        Object.defineProperty(widget, "width", {
            configurable: true,
            enumerable: true,
            get() { return node.size ? node.size[0] : undefined; },
            set(_v) { /* ignore — width is derived from the node, see above */ },
        });
    } catch (e) {
        dbg("could not pin widget.width getter", e);
    }

    node._AngeloWidget = widget;
    node._AngeloCanvas = canvas;
    node._AngeloCanvasWrap = canvasWrap;
    node._AngeloPlaceholder = placeholder;
    node._AngeloContainer = container;

    // Re-fit the canvas whenever its available area changes — node
    // resize, area-prompt box show/hide, etc. ResizeObserver fires on
    // the wrap's rendered-size changes, which covers all of them.
    if (typeof ResizeObserver !== "undefined") {
        const ro = new ResizeObserver(() => fitCanvasDisplaySize(node));
        ro.observe(canvasWrap);
        node._AngeloCanvasResizeObserver = ro;
        const onRemoved = node.onRemoved;
        node.onRemoved = function () {
            try { ro.disconnect(); } catch (e) { /* noop */ }
            if (onRemoved) onRemoved.apply(this, arguments);
        };
    }
    fitCanvasDisplaySize(node);

    dbg("attached preview canvas to node", node.id);
}

function _angeloIsZoomed(node) {
    // Epsilon so float drift near 1.0 doesn't read as "zoomed".
    return Math.abs((node._AngeloZoom || 1) - 1) > 1e-3
        || (node._AngeloPanX || 0) !== 0
        || (node._AngeloPanY || 0) !== 0;
}

// ── Preview size limits ─────────────────────────────────────────────────
// _PREVIEW_MAX_W / _PREVIEW_MAX_H: the largest the canvas is ever displayed
// (display pixels, not intrinsic). Prevents the preview from dominating the
// screen when the node is made large or the latent is very high-resolution.
//
// _PREVIEW_MIN_CANVAS_H: minimum canvas height used in getMinHeight(). Keeps
// the painting area usable even for very wide (landscape) latents.
//
// _TOOLBAR_H_APPROX: rough height of the toolbar rows that sit above the
// canvas. Used by getMinHeight() to compute a correct total widget height.
// ────────────────────────────────────────────────────────────────────────
const _PREVIEW_MAX_W       = 768;
const _PREVIEW_MAX_H       = 560;
const _PREVIEW_MIN_CANVAS_H = 260;
const _TOOLBAR_H_APPROX    = 190;

// Compute the BASE (fit) display size: the canvas size at zoom=1, fitting
// the wrap while preserving aspect ratio. Stored as _AngeloBaseW/H; the
// live size is base × zoom, applied by applyView().
//
// IMPORTANT: when the user has zoomed/panned (zoom != 1 or pan != 0), this
// is a NO-OP — the auto-fit must not stomp on a manual zoom. The view only
// re-fits at the neutral state (e.g. after resetView or a new image).
//
// A hard cap (_PREVIEW_MAX_W × _PREVIEW_MAX_H) prevents the preview from
// growing huge when the node is made very large, while still respecting
// the available wrap space (whichever limit is smaller wins).
function fitCanvasDisplaySize(node) {
    if (_angeloIsZoomed(node)) return;   // never auto-fit while zoomed/panned
    const canvas = node._AngeloCanvas;
    const wrap = node._AngeloCanvasWrap;
    if (!canvas || !wrap) return;
    const availW = wrap.clientWidth;
    const availH = wrap.clientHeight;
    if (availW <= 0 || availH <= 0) return;
    const img = node._AngeloImg;
    const natW = (img && img.naturalWidth) ? img.naturalWidth : canvas.width;
    const natH = (img && img.naturalHeight) ? img.naturalHeight : canvas.height;
    if (natW <= 0 || natH <= 0) return;
    // Scale to fit the available wrap space, then additionally cap at the
    // max display dimensions so a large node never shows an enormous preview.
    const scaleToFit = Math.min(availW / natW, availH / natH);
    const scaleToCap = Math.min(_PREVIEW_MAX_W / natW, _PREVIEW_MAX_H / natH);
    const scale = Math.min(scaleToFit, scaleToCap);
    node._AngeloBaseW = Math.max(1, Math.floor(natW * scale));
    node._AngeloBaseH = Math.max(1, Math.floor(natH * scale));
    applyView(node);
}

// Apply the current zoom/pan: size + position the absolutely-placed
// canvas. Display size = base × zoom; positioned centred in the wrap with
// the pan offset added. Then refresh the minimap.
function applyView(node) {
    const canvas = node._AngeloCanvas;
    const wrap = node._AngeloCanvasWrap;
    if (!canvas || !wrap) return;
    const baseW = node._AngeloBaseW, baseH = node._AngeloBaseH;
    if (!baseW || !baseH) return;
    const z = node._AngeloZoom || 1;
    const dispW = baseW * z, dispH = baseH * z;
    const wrapW = wrap.clientWidth, wrapH = wrap.clientHeight;
    const left = (wrapW - dispW) / 2 + (node._AngeloPanX || 0);
    const top = (wrapH - dispH) / 2 + (node._AngeloPanY || 0);
    canvas.style.width = dispW + "px";
    canvas.style.height = dispH + "px";
    canvas.style.left = left + "px";
    canvas.style.top = top + "px";
    updateMinimap(node);
}

// Reset to the neutral fit view (zoom=1, no pan) and re-fit to the node.
function resetView(node) {
    node._AngeloZoom = 1;
    node._AngeloPanX = 0;
    node._AngeloPanY = 0;
    fitCanvasDisplaySize(node);  // recomputes base + applyView (now neutral)
}

// Draw the corner minimap: full-image thumbnail + a rectangle marking the
// currently-visible viewport. Shown only when zoomed in (zoom > 1).
function updateMinimap(node) {
    const mm = node._AngeloMinimap;
    const wrap = node._AngeloCanvasWrap;
    const img = node._AngeloImg;
    if (!mm || !wrap) return;
    const z = node._AngeloZoom || 1;
    // Epsilon so a near-1.0 (drifted) zoom doesn't flash the minimap.
    if (z <= 1.001 || !img || !img.naturalWidth) {
        mm.style.display = "none";
        return;
    }
    // Thumbnail size: cap the long edge at 140 px, preserve aspect.
    const natW = img.naturalWidth, natH = img.naturalHeight;
    const cap = 140;
    const mmScale = Math.min(cap / natW, cap / natH);
    const mmW = Math.max(1, Math.round(natW * mmScale));
    const mmH = Math.max(1, Math.round(natH * mmScale));
    if (mm.width !== mmW) mm.width = mmW;
    if (mm.height !== mmH) mm.height = mmH;
    mm.style.width = mmW + "px";
    mm.style.height = mmH + "px";
    mm.style.display = "block";

    const ctx = mm.getContext("2d");
    ctx.clearRect(0, 0, mmW, mmH);
    ctx.drawImage(img, 0, 0, mmW, mmH);

    // Visible image region (normalised 0..1) = which part of the canvas
    // currently lands inside the wrap viewport.
    const baseW = node._AngeloBaseW, baseH = node._AngeloBaseH;
    const dispW = baseW * z, dispH = baseH * z;
    const wrapW = wrap.clientWidth, wrapH = wrap.clientHeight;
    const left = (wrapW - dispW) / 2 + (node._AngeloPanX || 0);
    const top = (wrapH - dispH) / 2 + (node._AngeloPanY || 0);
    const vx0 = Math.max(0, Math.min(1, (0 - left) / dispW));
    const vy0 = Math.max(0, Math.min(1, (0 - top) / dispH));
    const vx1 = Math.max(0, Math.min(1, (wrapW - left) / dispW));
    const vy1 = Math.max(0, Math.min(1, (wrapH - top) / dispH));

    ctx.strokeStyle = "rgba(255, 220, 80, 0.95)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(vx0 * mmW, vy0 * mmH, (vx1 - vx0) * mmW, (vy1 - vy0) * mmH);
    ctx.fillStyle = "rgba(255, 220, 80, 0.12)";
    ctx.fillRect(vx0 * mmW, vy0 * mmH, (vx1 - vx0) * mmW, (vy1 - vy0) * mmH);
}

// Area Prompt box: a single textarea below the canvas plus a Pos/Neg
// toggle. The toggle decides which underlying widget the textarea is
// bound to (area_text_positive / area_text_negative). We keep the
// non-edited prompt's value in its widget untouched, so flipping the
// toggle just re-points the textarea at the other widget's text.
function attachAreaPromptBox(node, container) {
    node._AngeloAreaPromptTarget = "positive";

    const wrap = document.createElement("div");
    wrap.style.display = "flex";  // toggled to "none" by syncAreaPromptVisibility
    wrap.style.flexDirection = "column";
    wrap.style.background = "#222";
    wrap.style.borderTop = "1px solid #333";
    wrap.style.borderBottom = "1px solid #333";
    wrap.style.padding = "4px";
    wrap.style.gap = "4px";

    // Header row: label + Pos/Neg toggle.
    const header = document.createElement("div");
    header.style.display = "flex";
    header.style.alignItems = "center";
    header.style.justifyContent = "space-between";
    header.style.gap = "6px";

    const label = document.createElement("span");
    label.textContent = "Area Prompt";
    label.style.color = "#bbb";
    label.style.fontSize = "11px";
    label.style.userSelect = "none";
    header.appendChild(label);

    const posNegBtn = document.createElement("button");
    posNegBtn.textContent = "Positive";
    posNegBtn.style.fontSize = "11px";
    posNegBtn.style.padding = "2px 8px";
    posNegBtn.style.borderRadius = "3px";
    posNegBtn.style.border = "1px solid #555";
    posNegBtn.style.background = "rgba(30, 120, 80, 0.95)";
    posNegBtn.style.color = "#fff";
    posNegBtn.style.cursor = "pointer";
    posNegBtn.style.userSelect = "none";
    posNegBtn.title = "Switch which Area Prompt this box edits. Positive = what to draw; "
        + "Negative = what to avoid (ignored by CFG=1 / distilled models like Klein, kept for others).";
    header.appendChild(posNegBtn);

    const textarea = document.createElement("textarea");
    textarea.placeholder = "Area prompt — describes the masked region (used when Area Prompt is ON).";
    textarea.style.width = "100%";
    textarea.style.boxSizing = "border-box";
    textarea.style.minHeight = "48px";
    textarea.style.resize = "vertical";
    textarea.style.fontSize = "12px";
    textarea.style.fontFamily = "inherit";
    textarea.style.padding = "4px";
    textarea.style.border = "1px solid #555";
    textarea.style.borderRadius = "3px";
    textarea.style.background = "#1a1a1a";
    textarea.style.color = "#ddd";

    // Stop pointer/key events from bubbling to the graph canvas (node
    // drag / delete / canvas shortcuts) while editing.
    for (const ev of ["pointerdown", "mousedown", "keydown", "keyup", "wheel"]) {
        textarea.addEventListener(ev, (e) => e.stopPropagation());
    }

    const targetWidgetName = () =>
        node._AngeloAreaPromptTarget === "negative" ? "area_text_negative" : "area_text_positive";

    textarea.addEventListener("input", () => {
        const w = findWidget(node, targetWidgetName());
        if (w) setWidget(w, textarea.value);
        // Auto-grow to fit content so the user doesn't have to drag-resize.
        textarea.style.height = "auto";
        textarea.style.height = textarea.scrollHeight + "px";
    });

    posNegBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        // Persist current text before switching (input handler already
        // did, but be safe), then flip the target and reload.
        const curW = findWidget(node, targetWidgetName());
        if (curW) setWidget(curW, textarea.value);

        node._AngeloAreaPromptTarget =
            node._AngeloAreaPromptTarget === "negative" ? "positive" : "negative";
        syncAreaPromptBox(node);
    });

    // "Insert Smart Phrasing" — opens a popup of edit-preservation
    // constraints; ticked ones get appended to the active Area Prompt.
    const smartBtn = document.createElement("button");
    smartBtn.type = "button";
    smartBtn.textContent = "Insert Smart Phrasing";
    smartBtn.style.cssText = "align-self:flex-start; font-size:11px; padding:3px 10px; "
        + "border:1px solid #555; border-radius:3px; background:#2a2a2a; color:#bbb; cursor:pointer;";
    smartBtn.title = "Append edit-preservation phrases (keep lighting / pose / clothes / faces the same) to the Area Prompt above.";
    for (const ev of ["pointerdown", "mousedown"]) {
        smartBtn.addEventListener(ev, (e) => e.stopPropagation());
    }
    smartBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        showSmartPhrasingPopup(node);
    });

    // Location dropdown (Smart Guided Inpaint only) — sits below the
    // "Area Prompt" heading and above the textarea. Picks the spatial
    // prefix prepended to the prompt at run time.
    const locationSelect = makeDropdown("Location", _Angelo_GUIDED_LOCATIONS, (val) => {
        const w = findWidget(node, "guided_location");
        if (w) setWidget(w, val);
    });
    locationSelect.title = "Where to place the new content. Prepended to your prompt at run time "
        + "(e.g. 'In the top left of the image, ...'). Smart Guided Inpaint only.";
    locationSelect.style.padding = "0";

    // "Generate Guided Edit" run button (Smart Guided Inpaint only) —
    // there's no click/drag to trigger a run in this mode.
    const runBtn = document.createElement("button");
    runBtn.type = "button";
    runBtn.textContent = "Generate Guided Edit";
    runBtn.style.cssText = "align-self:center; font-size:11px; padding:4px 12px; "
        + "border:1px solid #4a7; border-radius:3px; background:rgba(30,120,80,0.95); "
        + "color:#fff; font-weight:bold; cursor:pointer;";
    runBtn.title = "Run the whole-image guided edit using the Location + Area Prompt.";
    for (const ev of ["pointerdown", "mousedown"]) {
        runBtn.addEventListener(ev, (e) => e.stopPropagation());
    }
    runBtn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        triggerGuidedRefine(node);
    });

    wrap.appendChild(header);
    wrap.appendChild(locationSelect);
    wrap.appendChild(textarea);
    wrap.appendChild(smartBtn);
    wrap.appendChild(runBtn);
    container.appendChild(wrap);

    node._AngeloAreaPromptWrap = wrap;
    node._AngeloAreaPromptTextarea = textarea;
    node._AngeloAreaPromptPosNegBtn = posNegBtn;
    node._AngeloAreaPromptLabel = label;
    node._AngeloSmartPhrasingBtn = smartBtn;
    node._AngeloGuidedLocationSelect = locationSelect;
    node._AngeloGuidedRunBtn = runBtn;

    syncAreaPromptBox(node);
    syncAreaPromptVisibility(node);
}

// Edit-preservation phrases for the Smart Phrasing popup. Adding more
// here automatically adds a checkbox.
const _Angelo_SMART_PHRASES = [
    "Keep the lighting the same",
    "Keep the pose the same",
    "Keep the clothes the same",
    "Keep the faces the same",
];

// Append the chosen phrases to the currently-active Area Prompt textarea
// (positive or negative per the Pos/Neg toggle), comma-joined onto any
// existing text. Skips phrases already present (case-insensitive) so
// re-opening the popup doesn't duplicate them. Persists to the widget.
function appendSmartPhrases(node, phrases) {
    const textarea = node._AngeloAreaPromptTextarea;
    if (!textarea || !phrases.length) return;
    const cur = textarea.value.trim();
    const lower = cur.toLowerCase();
    const fresh = phrases.filter((p) => !lower.includes(p.toLowerCase()));
    if (!fresh.length) return;
    const addition = fresh.join(", ");
    const next = cur ? (cur.replace(/,\s*$/, "") + ", " + addition) : addition;
    textarea.value = next;
    const wname = node._AngeloAreaPromptTarget === "negative"
        ? "area_text_negative" : "area_text_positive";
    const w = findWidget(node, wname);
    if (w) setWidget(w, next);
}

function showSmartPhrasingPopup(node) {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed; inset:0; background:rgba(0,0,0,0.6); "
        + "display:flex; align-items:center; justify-content:center; z-index:10000; "
        + "font-family:Arial,sans-serif;";

    const modal = document.createElement("div");
    modal.style.cssText = "background:#2a2a2a; color:#ddd; border:1px solid #555; "
        + "border-radius:8px; padding:16px; width:340px; max-width:90vw; "
        + "display:flex; flex-direction:column; gap:8px;";

    const header = document.createElement("div");
    header.textContent = "Insert Smart Phrasing";
    header.style.cssText = "font-size:14px; font-weight:bold; color:#aaa;";
    modal.appendChild(header);

    const hint = document.createElement("div");
    hint.textContent = "Tick the constraints to add to the Area Prompt.";
    hint.style.cssText = "font-size:11px; color:#888; margin-bottom:4px;";
    modal.appendChild(hint);

    const checks = [];
    for (const phrase of _Angelo_SMART_PHRASES) {
        const row = document.createElement("label");
        row.style.cssText = "display:flex; align-items:center; gap:8px; font-size:13px; "
            + "cursor:pointer; padding:3px 2px;";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        const span = document.createElement("span");
        span.textContent = phrase;
        row.appendChild(cb);
        row.appendChild(span);
        modal.appendChild(row);
        checks.push({ cb, phrase });
    }

    const footer = document.createElement("div");
    footer.style.cssText = "display:flex; justify-content:flex-end; gap:8px; margin-top:8px;";

    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.cssText = "background:#444; color:#ddd; border:none; padding:6px 14px; "
        + "border-radius:4px; cursor:pointer;";

    const insertBtn = document.createElement("button");
    insertBtn.textContent = "Insert";
    insertBtn.style.cssText = "background:rgba(30,120,80,0.95); color:#fff; border:none; "
        + "padding:6px 14px; border-radius:4px; cursor:pointer;";

    footer.appendChild(cancelBtn);
    footer.appendChild(insertBtn);
    modal.appendChild(footer);
    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);

    const close = () => { if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop); };
    insertBtn.addEventListener("click", () => {
        const chosen = checks.filter((c) => c.cb.checked).map((c) => c.phrase);
        appendSmartPhrases(node, chosen);
        close();
    });
    cancelBtn.addEventListener("click", close);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });
    document.addEventListener("keydown", function onKey(e) {
        if (e.key === "Escape") { close(); document.removeEventListener("keydown", onKey); }
    });
}

// Reflect the current edit-target widget value into the textarea and
// update the Pos/Neg button styling. Called on creation, on toggle,
// and from the full toolbar sync (workflow load).
function syncAreaPromptBox(node) {
    const textarea = node._AngeloAreaPromptTextarea;
    const btn = node._AngeloAreaPromptPosNegBtn;
    if (!textarea || !btn) return;
    const onNeg = node._AngeloAreaPromptTarget === "negative";
    const wname = onNeg ? "area_text_negative" : "area_text_positive";
    const w = findWidget(node, wname);
    const val = w ? String(w.value ?? "") : "";
    if (textarea.value !== val) {
        textarea.value = val;
        // Re-measure height whenever text is reloaded from the widget.
        textarea.style.height = "auto";
        textarea.style.height = Math.max(48, textarea.scrollHeight) + "px";
    }
    btn.textContent = onNeg ? "Negative" : "Positive";
    btn.style.background = onNeg ? "rgba(140, 60, 60, 0.95)" : "rgba(30, 120, 80, 0.95)";
}

// Show the Area Prompt box only when Area Prompt is effectively ON —
// either the area_prompt widget is true, or Smart Inpaint mode forces
// it on. Hiding is display:none only; the text lives in the
// area_text_* widgets and is reloaded by syncAreaPromptBox, so toggling
// visibility never loses what was typed.
function syncAreaPromptVisibility(node) {
    const guided = isSmartGuidedInpaintMode(node);
    const anySmart = isSmartInpaintMode(node) || guided;

    // Smart Phrasing button: both smart modes (both use reference-image
    // conditioning, so the "keep X the same" constraints apply).
    if (node._AngeloSmartPhrasingBtn) {
        node._AngeloSmartPhrasingBtn.style.display = anySmart ? "block" : "none";
    }
    // Location dropdown + Generate button: Smart Guided Inpaint only.
    if (node._AngeloGuidedLocationSelect) {
        node._AngeloGuidedLocationSelect.style.display = guided ? "inline-flex" : "none";
    }
    if (node._AngeloGuidedRunBtn) {
        node._AngeloGuidedRunBtn.style.display = guided ? "block" : "none";
    }

    const wrap = node._AngeloAreaPromptWrap;
    if (!wrap) return;
    const w = findWidget(node, "area_prompt");
    const on = (w && !!w.value) || anySmart;
    const next = on ? "flex" : "none";
    if (wrap.style.display === next) return;  // no change → no reflow
    wrap.style.display = next;
    // The container's height just changed — nudge LiteGraph to recompute
    // the node's reserved space for the DOM widget so the box isn't clipped.
    if (node.graph && node.graph.setDirtyCanvas) node.graph.setDirtyCanvas(true, true);
}

function loadIntoCanvas(node, url) {
    if (!node._AngeloCanvas) {
        attachPreviewCanvas(node);
    }
    const canvas = node._AngeloCanvas;
    const placeholder = node._AngeloPlaceholder;

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
        const prev = node._AngeloImg;
        const sameDims = prev
            && prev.naturalWidth === img.naturalWidth
            && prev.naturalHeight === img.naturalHeight;
        node._AngeloImg = img;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0);
        if (placeholder) placeholder.style.display = "none";

        // Tell Python the actual image dimensions so it can derive the
        // correct pixel→latent scale per-axis (FLUX 2 is 16×, FLUX 1
        // / SDXL are 8×; we don't want to hardcode either).
        const wi = findWidget(node, "image_w");
        const hi = findWidget(node, "image_h");
        if (wi) setWidget(wi, img.naturalWidth);
        if (hi) setWidget(hi, img.naturalHeight);

        // If this is a refine of the SAME image (same dims) and the user
        // is zoomed in, keep their view — so clicking to refine a detail
        // doesn't pop them back to fit. A genuinely new image (different
        // dims) or the first load resets to fit.
        if (sameDims && _angeloIsZoomed(node)) {
            applyView(node);
        } else {
            resetView(node);
        }

        // Auto-resize the node height to suit the image aspect ratio so the
        // canvas is always a comfortable size to paint on.  We only do this
        // when the image dimensions change (new generation / first load) —
        // mid-refine updates to the same-size canvas keep whatever height the
        // user has set.  The target height is:
        //   LiteGraph node title (~32 px) + toolbar + clamped canvas height
        // where the canvas height mirrors the logic in getMinHeight().
        if (!sameDims) {
            const nodeW = node.size ? node.size[0] : 380;
            const contentW = Math.max(1, nodeW - 12);  // subtract side margins
            const ar = img.naturalHeight / img.naturalWidth;
            const idealCanvasH = Math.round(Math.min(contentW, _PREVIEW_MAX_W) * ar);
            const canvasH = Math.max(_PREVIEW_MIN_CANVAS_H, Math.min(_PREVIEW_MAX_H, idealCanvasH));
            // +32 for the LiteGraph node title bar; +8 extra breathing room.
            const idealNodeH = _TOOLBAR_H_APPROX + canvasH + 40;
            const newH = Math.max(480, Math.min(960, idealNodeH));
            if (!node.size) node.size = [nodeW, newH];
            else if (Math.abs(node.size[1] - newH) > 30) node.size[1] = newH;
        }

        // If we're in detect mode (candidates persist for batch editing),
        // re-overlay them on the freshly-refined preview so they stay put.
        if (node._AngeloDetections && node._AngeloDetections.length) {
            redrawCanvasWithOverlays(node);
        }

        // Force a node redraw so LiteGraph re-computes the canvas widget size.
        if (node.graph && node.graph.setDirtyCanvas) {
            node.graph.setDirtyCanvas(true, false);
        }
        dbg("image drawn", { w: img.naturalWidth, h: img.naturalHeight });
    };
    img.onerror = (e) => {
        dbg("image load error", e, url);
        // Show a visible notice so the user knows the preview failed to load,
        // rather than leaving the canvas blank or stale with no feedback.
        showAngeloNotice(node, "Preview failed to load. Try re-queuing the workflow.");
    };
    img.src = url;
}

/**
 * Redraw the canvas with the underlying image + any active overlays
 * (hover ring, paint stroke). Called on every mousemove + on demand
 * from the paint handlers. Cheap: ~1 image blit + a few arcs per frame.
 */
function redrawCanvasWithOverlays(node) {
    const canvas = node._AngeloCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (node._AngeloImg) {
        ctx.drawImage(node._AngeloImg, 0, 0);
    } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    }

    // Cursor changes by mode — only remaining visual indicator now
    // that the corner pills are gone.
    if (canvas) {
        if (isSmartGuidedInpaintMode(node)) {
            canvas.style.cursor = "default";  // no canvas interaction
        } else if (isSmartInpaintMode(node)) {
            canvas.style.cursor = "crosshair";
        } else {
            canvas.style.cursor = isPaintModeOn(node) ? "cell" : "crosshair";
        }
    }

    const rect = canvas.getBoundingClientRect();
    const scaleX = rect.width > 0 ? canvas.width / rect.width : 1;
    const scaleY = rect.height > 0 ? canvas.height / rect.height : 1;

    const radiusWidget = findWidget(node, "click_radius");
    const radiusPixel = radiusWidget ? radiusWidget.value : 96;

    // 1. Paint stroke (if actively painting) — draw the union of brush
    //    circles so far at 50% opacity (the user wanted it visibly
    //    blue without fully occluding the underlying image).
    if (node._AngeloPainting && node._AngeloStroke?.length) {
        ctx.save();
        ctx.fillStyle = "rgba(80, 180, 255, 0.5)";
        for (const [px, py] of node._AngeloStroke) {
            ctx.beginPath();
            ctx.arc(px, py, radiusPixel, 0, Math.PI * 2);
            ctx.fill();
        }
        ctx.restore();
    }

    // 1b. Smart Inpaint live rectangle while the user is dragging.
    if (node._AngeloDraggingRect) {
        const r = node._AngeloDraggingRect;
        const x = Math.min(r.x1, r.x2);
        const y = Math.min(r.y1, r.y2);
        const w = Math.abs(r.x2 - r.x1);
        const h = Math.abs(r.y2 - r.y1);
        ctx.save();
        ctx.fillStyle = "rgba(80, 180, 255, 0.35)";
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = "rgba(80, 180, 255, 1.0)";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.restore();
    }

    // 2. Hover ring at cursor position (skip during active paint or
    //    rect drag — those have their own active visuals; skip in
    //    Smart Inpaint mode entirely since the click_radius circle
    //    is irrelevant to a rectangle workflow).
    if (node._AngeloHover
        && !node._AngeloPainting
        && !node._AngeloDraggingRect
        && !isSmartInpaintMode(node)) {
        const px = node._AngeloHover.x * scaleX;
        const py = node._AngeloHover.y * scaleY;
        ctx.save();
        ctx.strokeStyle = "rgba(255, 200, 80, 0.7)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, radiusPixel, 0, Math.PI * 2);
        ctx.stroke();
        // Tiny cross-hair at the centre
        ctx.strokeStyle = "rgba(255, 200, 80, 0.9)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px - 6, py); ctx.lineTo(px + 6, py);
        ctx.moveTo(px, py - 6); ctx.lineTo(px, py + 6);
        ctx.stroke();
        ctx.restore();
    }

    // 3. Detection candidates (SAM 3 / YOLO) awaiting a click-to-confirm.
    drawDetections(node, ctx);
}

function flashClickOverlay(node, cx, cy) {
    // Draw a fading ring on the canvas at the click point.
    // We don't have the click_radius -> pixel scaling on hand here cheaply;
    // just draw a 24px ring as visual feedback that the click registered.
    const canvas = node._AngeloCanvas;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    // Scale the click point from display coords to canvas coords (canvas
    // is at natural image size; display is css-scaled).
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const px = cx * scaleX;
    const py = cy * scaleY;

    const radiusWidget = findWidget(node, "click_radius");
    const radiusPixel = radiusWidget ? radiusWidget.value : 96;

    let t0 = performance.now();
    const tick = (t) => {
        const elapsed = t - t0;
        const alpha = Math.max(0, 1 - elapsed / 1500);
        if (alpha <= 0) return;
        // Let redrawCanvasWithOverlays handle the base image + all active
        // overlays (hover ring, paint stroke, detections), then draw the
        // fading flash ring on top so both can coexist without flickering.
        redrawCanvasWithOverlays(node);
        ctx.save();
        ctx.strokeStyle = `rgba(255, 200, 80, ${alpha})`;
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.arc(px, py, radiusPixel, 0, Math.PI * 2);
        ctx.stroke();
        ctx.fillStyle = `rgba(255, 220, 120, ${alpha})`;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}


// ============================================================
// Refine trigger — update hidden widgets + queue
// ============================================================

function triggerRefine(node, pixelX, pixelY, displayCX, displayCY) {
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    const ws = findWidget(node, "click_seq");
    const wr = findWidget(node, "reset");
    const wsp = findWidget(node, "stroke_points");
    if (!wx || !wy || !ws) {
        dbg("ERROR: hidden widgets not found", { wx: !!wx, wy: !!wy, ws: !!ws });
        return;
    }

    setWidget(wx, pixelX);
    setWidget(wy, pixelY);
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    if (wr) setWidget(wr, false);
    // Clear stroke_points + seg_polygon so a leftover paint stroke or a
    // SAM-detected silhouette from earlier doesn't bleed into a single
    // click refine. (seg_polygon persists across Persistent Mask re-rolls
    // until a manual action like this resets it.)
    if (wsp) setWidget(wsp, "");
    const wseg = findWidget(node, "seg_polygon");
    if (wseg) setWidget(wseg, "");
    const wmaskpng = findWidget(node, "seg_mask_png");
    if (wmaskpng) setWidget(wmaskpng, "");

    dbg("queueing workflow (click)", { click_x: wx.value, click_y: wy.value, click_seq: ws.value });
    queuePrompt();
}

function triggerPaintRefine(node, strokePoints) {
    const wsp = findWidget(node, "stroke_points");
    const ws = findWidget(node, "click_seq");
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    const wr = findWidget(node, "reset");
    if (!wsp || !ws) {
        dbg("ERROR: stroke widgets not found", { wsp: !!wsp, ws: !!ws });
        return;
    }

    // Round to ints to keep the JSON tight. Sub-pixel precision is
    // wasted — Python rasterises into latent space anyway.
    const compact = strokePoints.map(([x, y]) => [Math.round(x), Math.round(y)]);
    setWidget(wsp, JSON.stringify(compact));
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    // Also set click_x/y to the first stroke point as a fallback target,
    // in case anything downstream reads them.
    if (wx) setWidget(wx, compact[0][0]);
    if (wy) setWidget(wy, compact[0][1]);
    if (wr) setWidget(wr, false);
    // A paint stroke replaces any SAM-detected silhouette.
    const wseg = findWidget(node, "seg_polygon");
    if (wseg) setWidget(wseg, "");
    const wmaskpng = findWidget(node, "seg_mask_png");
    if (wmaskpng) setWidget(wmaskpng, "");

    dbg("queueing workflow (paint)", { points: compact.length, click_seq: ws.value });
    queuePrompt();
}

function queuePrompt() {
    if (typeof app.queuePrompt === "function") {
        const ret = app.queuePrompt(0);
        if (ret && typeof ret.then === "function") {
            ret.catch(e => dbg("queuePrompt promise rejected", e));
        }
    } else {
        dbg("ERROR: app.queuePrompt is not a function");
    }
}

// =====================================================================
// Load Image — bring an external photo in as the base latent.
// Button → file picker → resolution popup → upload (/upload/image) →
// set hidden widgets → queue. run() VAE-encodes the upload into the
// base; Reset/Undo then return to it.
// =====================================================================

// Persistent in-app notice bar (top of the preview). For actionable
// messages that must be read — unlike _angeloToast which auto-hides.
function showAngeloNotice(node, message) {
    if (!node._AngeloNotice) return;
    node._AngeloNoticeText.textContent = message;
    node._AngeloNotice.style.display = "block";
}
function hideAngeloNotice(node) {
    if (node._AngeloNotice) node._AngeloNotice.style.display = "none";
}

function showAngeloLoading(node, message) {
    if (!node._AngeloLoading) return;
    if (message) node._AngeloLoadingText.textContent = message;
    node._AngeloLoading.style.display = "flex";
}
function hideAngeloLoading(node) {
    if (node._AngeloLoading) node._AngeloLoading.style.display = "none";
}

function _angeloToast(message) {
    // Deduplicate: don't stack identical toasts on top of each other.
    const existing = document.querySelectorAll("[data-angelo-toast]");
    for (const el of existing) {
        if (el.textContent === message) return;
    }
    const t = document.createElement("div");
    t.setAttribute("data-angelo-toast", "1");
    t.textContent = message;
    t.style.cssText = "position:fixed; top:20px; right:20px; background:#333; color:#fff; "
        + "padding:10px 16px; border-radius:6px; z-index:100000; font:13px Arial,sans-serif; "
        + "box-shadow:0 2px 8px rgba(0,0,0,0.5); opacity:0; transition:opacity 0.18s;";
    document.body.appendChild(t);
    requestAnimationFrame(() => { t.style.opacity = "1"; });
    setTimeout(() => { t.style.opacity = "0"; setTimeout(() => t.remove(), 220); }, 1600);
}

function triggerLoadImage(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.style.display = "none";
    input.addEventListener("change", () => {
        const file = input.files && input.files[0];
        if (file) showLoadImagePopup(node, file);
        input.remove();
    });
    document.body.appendChild(input);
    input.click();
}

// ---- Right-click image actions (#7) -------------------------------------
// Copy / Open, like ComfyUI's image nodes. Loading an image is also possible
// via drag-drop onto the node, the Load Image button, or clipboard paste
// (Ctrl+V / Cmd+V while the cursor hovers the Angelo canvas).

function _angeloOpenImageInTab(node) {
    let url = null;
    if (node._AngeloPreviewRef) url = makeViewUrl(node._AngeloPreviewRef);
    else if (node._AngeloImg && node._AngeloImg.src) url = node._AngeloImg.src;
    if (url) window.open(url, "_blank");
    else _angeloToast("No image to open yet");
}

async function _angeloCopyImageToClipboard(node) {
    const img = node._AngeloImg;
    if (!img) { _angeloToast("No image to copy yet"); return; }
    if (!navigator.clipboard || !window.ClipboardItem) {
        _angeloToast("Clipboard image copy not supported in this browser");
        return;
    }
    try {
        const c = document.createElement("canvas");
        c.width = img.naturalWidth || img.width;
        c.height = img.naturalHeight || img.height;
        c.getContext("2d").drawImage(img, 0, 0);
        const blob = await new Promise((res) => c.toBlob(res, "image/png"));
        await navigator.clipboard.write([new ClipboardItem({ "image/png": blob })]);
        _angeloToast("Image copied to clipboard");
    } catch (e) {
        dbg("[Angelo] copy image failed", e);
        _angeloToast("Copy failed — see console");
    }
}

function _angeloShowImageContextMenu(node, event) {
    const hasImg = !!node._AngeloImg;
    const items = [
        { content: "Open image in new tab", disabled: !hasImg,
          callback: () => _angeloOpenImageInTab(node) },
        { content: "Copy image", disabled: !hasImg,
          callback: () => _angeloCopyImageToClipboard(node) },
    ];
    const LG = window.LiteGraph;
    if (LG && LG.ContextMenu) {
        new LG.ContextMenu(items, { event, title: "Angelo" });
    }
}

async function _uploadLoadedImage(node, file, mode, mp) {
    const fd = new FormData();
    fd.append("image", file, file.name);
    fd.append("overwrite", "false");
    _angeloToast("Uploading image…");
    let data;
    try {
        const res = await api.fetchApi("/upload/image", { method: "POST", body: fd });
        if (!res.ok) { _angeloToast("Upload failed"); return; }
        data = await res.json();
    } catch (e) {
        dbg("[Angelo] upload failed", e);
        _angeloToast("Upload failed — see console");
        return;
    }
    const ref = JSON.stringify({
        name: data.name,
        subfolder: data.subfolder || "",
        type: data.type || "input",
    });
    const wImg = findWidget(node, "loaded_image");
    const wMode = findWidget(node, "loaded_resize_mode");
    const wMp = findWidget(node, "loaded_target_mp");
    const wSeq = findWidget(node, "loaded_image_seq");
    if (wImg) setWidget(wImg, ref);
    if (wMode) setWidget(wMode, mode);
    if (wMp) setWidget(wMp, mp);
    if (wSeq) setWidget(wSeq, ((wSeq.value || 0) + 1) & 0x7FFFFFFF);

    // Loading a photo means you want to EDIT it — flip to Edit Mode so
    // the queue refines/previews the loaded base instead of regenerating
    // it from noise (Sampler Mode at denoise 1.0). Fires the wrapped mode
    // callback (lock + grey sync); refresh the Mode dropdown to match.
    const wNodeMode = findWidget(node, "mode");
    if (wNodeMode && String(wNodeMode.value) !== "Edit Mode") {
        setWidget(wNodeMode, "Edit Mode");
        syncModeSelect(node);
    }

    syncLoadImageControls(node);   // reveal the Unload button
    _angeloToast("Loading as base…");
    queuePrompt();
}

// Clear the loaded image → the wired latent input takes over as base.
function unloadImage(node) {
    const wImg = findWidget(node, "loaded_image");
    if (wImg) setWidget(wImg, "");
    syncLoadImageControls(node);
    _angeloToast("Unloaded — using latent input");
    queuePrompt();
}

// Show the Unload button only while an image is loaded.
function syncLoadImageControls(node) {
    const btn = node._AngeloUnloadImageBtn;
    if (!btn) return;
    const w = findWidget(node, "loaded_image");
    const active = !!(w && String(w.value || "").trim());
    btn.style.display = active ? "" : "none";
}

// =====================================================================
// Detect (SAM 3 auto-segment): text → /angelo/detect → highlighted
// candidates on the canvas → click-to-confirm → mask per mode → refine.
// Refine uses the silhouette polygons; Smart Inpaint uses the bbox.
// =====================================================================

async function runDetect(node, conceptOverride) {
    // conceptOverride (from a quick-select preset) runs that term directly
    // WITHOUT touching the text box; otherwise use what the user typed.
    const text = (conceptOverride != null && String(conceptOverride).trim())
        ? String(conceptOverride).trim()
        : (node._AngeloDetectText?.value || "").trim();
    if (!text) { showAngeloNotice(node, "Type what to segment first (e.g. \"the face\")."); return; }
    const ref = node._AngeloPreviewRef;
    if (!ref || !ref.filename) { showAngeloNotice(node, "Generate or load an image first, then Detect."); return; }
    const confEl = node._AngeloDetectConf && node._AngeloDetectConf._AngeloInput;
    const conf = confEl ? Math.max(0.05, Math.min(0.95, parseFloat(confEl.value) || 0.3)) : 0.3;
    // Persist so it survives page reloads.
    try { localStorage.setItem("angelo_detect_conf", String(conf)); } catch (e) { /* noop */ }
    hideAngeloNotice(node);   // clear any prior error before a new attempt
    // In-app overlay (NOT a toast) while the request is in flight — the
    // first detect builds the SAM 3 model and can take several seconds.
    showAngeloLoading(node, "Loading SAM 3…");
    try {
        const res = await api.fetchApi("/angelo/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                method: "sam3_text",
                text,
                confidence_threshold: conf,
                filename: ref.filename,
                subfolder: ref.subfolder || "",
                type: ref.type || "temp",
            }),
        });
        const data = await res.json();
        if (!res.ok || data.error) {
            dbg("[Angelo] detect error", data);
            // Persistent in-app notice — esp. the "SAM 3 not installed →
            // run the installer" message, which needs reading + acting on.
            showAngeloNotice(node, data.error || `Detect failed (HTTP ${res.status}).`);
            return;
        }
        const dets = data.detections || [];
        node._AngeloDetections = dets;
        node._AngeloHoverDet = -1;
        node._AngeloEditedDets = new Set();   // which candidates have been edited
        node._AngeloMaskGrow = 0;             // each detect starts at no grow
        _syncMaskGrowReadout(node);
        syncDetectModeButton(node);
        if (!dets.length) {
            // Nothing highlights, so say so where it can be read + acted on.
            showAngeloNotice(node, "No matches — try a lower Conf or different words.");
        }
        redrawCanvasWithOverlays(node);
    } catch (e) {
        dbg("[Angelo] detect failed", e);
        showAngeloNotice(node, "Detect request failed — is the ComfyUI server reachable? See the console.");
    } finally {
        hideAngeloLoading(node);   // self-dismiss on every exit path
    }
}

function clearDetections(node) {
    if (!node._AngeloDetections) return;
    node._AngeloDetections = null;   // candidate objects (+ their _editMask) drop here
    node._AngeloHoverDet = -1;
    node._AngeloEditedDets = null;
    node._AngeloTouchup = null;
    node._AngeloBrushPreview = null;
    // Reset the highlight opacity to default so the next detect starts full.
    node._AngeloDetOpacity = 1.0;
    if (node._AngeloDetOpacitySlider) node._AngeloDetOpacitySlider.value = "1";
    node._AngeloMaskGrow = 0;
    _syncMaskGrowReadout(node);
    syncDetectModeButton(node);
    redrawCanvasWithOverlays(node);
}

// Show the floating detect-mode panel (Cancel + opacity slider) while
// candidates are active.
function syncDetectModeButton(node) {
    const panel = node._AngeloDetectPanel;
    if (!panel) return;
    panel.style.display = (node._AngeloDetections && node._AngeloDetections.length) ? "flex" : "none";
}

// Ray-casting point-in-polygon test. flat = [x0,y0,x1,y1,...] polygon
// coords in image-pixel space. Returns true if (px, py) is inside.
function _pointInPolygon(px, py, flat) {
    const n = flat.length >> 1;
    if (n < 3) return false;
    let inside = false;
    for (let i = 0, j = n - 1; i < n; j = i++) {
        const xi = flat[2 * i], yi = flat[2 * i + 1];
        const xj = flat[2 * j], yj = flat[2 * j + 1];
        if (((yi > py) !== (yj > py)) &&
            (px < (xj - xi) * (py - yi) / (yj - yi) + xi)) {
            inside = !inside;
        }
    }
    return inside;
}

// Topmost (tightest) detection whose silhouette contains the image-pixel point.
// Falls back to bbox containment when the detection has no polygons (e.g. a
// grown/edited raster mask where the original polygon may be stale). Selects
// the candidate with the smallest bbox area among those that contain the point
// so overlapping detections resolve to the tightest fit.
function _detAtPoint(node, px, py) {
    const dets = node._AngeloDetections || [];
    let best = null, bestArea = Infinity;
    for (const d of dets) {
        const b = _detBbox(node, d);
        if (!b) continue;
        // Quick bbox pre-filter — skip if the point is outside the bbox.
        if (px < b[0] || px > b[2] || py < b[1] || py > b[3]) continue;
        const area = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
        // If the detection has an edit-mask, bbox containment is the best we
        // can do (raster masks have no polygon representation).
        if (d._editMask) {
            if (area < bestArea) { bestArea = area; best = d; }
            continue;
        }
        // Polygon hit-test: accept if the point lands inside ANY of the
        // detection's polygons (detections can be multi-part, e.g. two eyes).
        const polys = _detPolys(node, d);
        if (polys.length === 0) {
            // No polygons at all — fall back to bbox.
            if (area < bestArea) { bestArea = area; best = d; }
        } else {
            for (const poly of polys) {
                if (poly && _pointInPolygon(px, py, poly)) {
                    if (area < bestArea) { bestArea = area; best = d; }
                    break;
                }
            }
        }
    }
    return best;
}

function confirmDetection(node, det) {
    if (isSmartGuidedInpaintMode(node)) return;
    const ws = findWidget(node, "click_seq");
    if (!ws) return;
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    const wr = findWidget(node, "reset");
    const wsp = findWidget(node, "stroke_points");
    const wrp = findWidget(node, "rect_points");
    const wseg = findWidget(node, "seg_polygon");
    const wmask = findWidget(node, "seg_mask_png");
    const smart = isSmartInpaintMode(node);
    // Apply the current Mask grow/shrink to the committed shape (the same
    // offset that's being drawn), so what you edit matches what you see.
    const b = _detBbox(node, det) || [0, 0, 0, 0];
    if (wx) setWidget(wx, Math.round((b[0] + b[2]) / 2));
    if (wy) setWidget(wy, Math.round((b[1] + b[3]) / 2));
    if (wr) setWidget(wr, false);
    if (wsp) setWidget(wsp, "");
    if (smart) {
        // Smart Inpaint: the (grown) bbox is the rectangle. (No touch-up
        // brush in Smart Inpaint — it's Refine-only.)
        if (wrp) setWidget(wrp, JSON.stringify([[
            Math.round(b[0]), Math.round(b[1]), Math.round(b[2]), Math.round(b[3]),
        ]]));
        if (wseg) setWidget(wseg, "");
        if (wmask) setWidget(wmask, "");
    } else if (det._editMask && wmask) {
        // Refine + brushed: send the raster edit-mask (handles brushed holes /
        // additions a polygon can't), and clear the polygon path.
        const png = det._editMask.toDataURL("image/png").split(",")[1] || "";
        setWidget(wmask, png);
        if (wseg) setWidget(wseg, "");
        if (wrp) setWidget(wrp, "");
    } else {
        // Refine: the (grown) silhouette polygons are the mask.
        if (wseg) setWidget(wseg, JSON.stringify(_detPolys(node, det)));
        if (wmask) setWidget(wmask, "");
        if (wrp) setWidget(wrp, "");
    }
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    // Keep the candidates up (batch editing): mark this one edited (drawn
    // green) and leave the rest clickable. Exit via Cancel / Esc / Space.
    const idx = node._AngeloDetections ? node._AngeloDetections.indexOf(det) : -1;
    if (idx >= 0) {
        node._AngeloEditedDets = node._AngeloEditedDets || new Set();
        node._AngeloEditedDets.add(idx);
    }
    redrawCanvasWithOverlays(node);   // immediate feedback: clicked one turns green
    queuePrompt();
}

// Offset a closed polygon outward (delta>0) or inward (delta<0) by ~|delta|
// pixels, perpendicular to its edges. Miter join with a spike clamp; the
// outward direction is fixed per-vertex via the centroid so winding order
// doesn't matter. flat = [x0,y0,x1,y1,...] in image-pixel coords.
function _offsetPolygon(flat, delta) {
    const n = flat.length >> 1;
    if (n < 3 || !delta) return flat.slice();
    const P = [];
    for (let i = 0; i < n; i++) P.push([flat[2 * i], flat[2 * i + 1]]);
    let cx = 0, cy = 0;
    for (const [x, y] of P) { cx += x; cy += y; }
    cx /= n; cy /= n;
    const out = new Array(n * 2);
    for (let i = 0; i < n; i++) {
        const prev = P[(i - 1 + n) % n], cur = P[i], next = P[(i + 1) % n];
        let e1x = cur[0] - prev[0], e1y = cur[1] - prev[1];
        let e2x = next[0] - cur[0], e2y = next[1] - cur[1];
        const l1 = Math.hypot(e1x, e1y) || 1, l2 = Math.hypot(e2x, e2y) || 1;
        e1x /= l1; e1y /= l1; e2x /= l2; e2y /= l2;
        // Unit edge normals (one consistent side); miter = their sum.
        const n1x = -e1y, n1y = e1x, n2x = -e2y, n2y = e2x;
        let mx = n1x + n2x, my = n1y + n2y;
        let ml = Math.hypot(mx, my);
        if (ml < 1e-6) { mx = n1x; my = n1y; ml = Math.hypot(mx, my) || 1; }
        mx /= ml; my /= ml;
        let cosA = Math.abs(mx * n1x + my * n1y);   // cos(half-angle)
        if (cosA < 0.25) cosA = 0.25;               // clamp miter spike at sharp corners
        const len = Math.abs(delta) / cosA;
        // Orient the miter outward (away from centroid), then push out/in.
        if (mx * (cur[0] - cx) + my * (cur[1] - cy) < 0) { mx = -mx; my = -my; }
        const s = delta >= 0 ? 1 : -1;
        out[2 * i] = cur[0] + mx * len * s;
        out[2 * i + 1] = cur[1] + my * len * s;
    }
    return out;
}

// The detection's polygons / bbox AS DISPLAYED + COMMITTED, with the
// current mask grow/shrink applied. grow==0 returns the originals.
function _detPolys(node, det) {
    const g = node._AngeloMaskGrow || 0;
    const polys = det.polygons || [];
    if (!g) return polys;
    return polys.map((p) => (p && p.length >= 6) ? _offsetPolygon(p, g) : p);
}
function _detBbox(node, det) {
    if (!det.bbox) return det.bbox;
    const g = node._AngeloMaskGrow || 0;
    let b = det.bbox;
    if (g) {
        const img = node._AngeloImg;
        const W = (img && img.naturalWidth) ? img.naturalWidth : 1e9;
        const H = (img && img.naturalHeight) ? img.naturalHeight : 1e9;
        b = [
            Math.max(0, b[0] - g), Math.max(0, b[1] - g),
            Math.min(W, b[2] + g), Math.min(H, b[3] + g),
        ];
    }
    // Include any brushed extent so the whole touched shape is hit-testable.
    const e = det._editBounds;
    if (e) {
        b = [Math.min(b[0], e[0]), Math.min(b[1], e[1]), Math.max(b[2], e[2]), Math.max(b[3], e[3])];
    }
    return b;
}

function _syncMaskGrowReadout(node) {
    const r = node._AngeloMaskGrowReadout;
    if (!r) return;
    const g = node._AngeloMaskGrow || 0;
    r.textContent = (g > 0 ? "+" : "") + g + "px";
}

// +/- buttons: nudge the grow value and re-draw every highlight together.
function adjustMaskGrow(node, delta) {
    if (!node._AngeloDetections || !node._AngeloDetections.length) return;
    const cur = node._AngeloMaskGrow || 0;
    node._AngeloMaskGrow = Math.max(-40, Math.min(200, cur + delta));
    _syncMaskGrowReadout(node);
    redrawCanvasWithOverlays(node);
}

// ===== Touch-up brush (Refine only) — Shift-drag adds to a detected mask,
//       Alt-drag subtracts (holes allowed). Once brushed, a detection holds
//       a raster edit-mask (`_editMask`, an offscreen canvas at image res,
//       white = masked) that is its source of truth for display + commit. =====

function _brushRadius(node) {
    const w = findWidget(node, "click_radius");
    return Math.max(2, (w && Number(w.value)) || 96);
}

// Lazily promote a detection to a raster edit-mask, seeded from its current
// (grown) silhouette so brushing starts from exactly what's on screen.
function _ensureEditMask(node, det) {
    if (det._editMask) return det._editMask;
    const img = node._AngeloImg;
    const W = (img && img.naturalWidth) || 512;
    const H = (img && img.naturalHeight) || 512;
    const c = document.createElement("canvas");
    c.width = W; c.height = H;
    const cx = c.getContext("2d");
    cx.fillStyle = "#fff";
    for (const poly of _detPolys(node, det)) {
        if (!poly || poly.length < 6) continue;
        cx.beginPath();
        cx.moveTo(poly[0], poly[1]);
        for (let k = 2; k < poly.length - 1; k += 2) cx.lineTo(poly[k], poly[k + 1]);
        cx.closePath();
        cx.fill();
    }
    det._editMask = c;
    return c;
}

function _brushStamp(det, px, py, radius, subtract) {
    const cx = det._editMask.getContext("2d");
    cx.save();
    cx.globalCompositeOperation = subtract ? "destination-out" : "source-over";
    cx.fillStyle = "#fff";
    cx.beginPath();
    cx.arc(px, py, Math.max(1, radius), 0, Math.PI * 2);
    cx.fill();
    cx.restore();
    // Grow the hit-box to cover added paint so the whole brushed shape stays
    // hoverable / clickable even where it extends past the original bbox.
    if (!subtract) {
        const nb = [px - radius, py - radius, px + radius, py + radius];
        const b = det._editBounds;
        det._editBounds = b
            ? [Math.min(b[0], nb[0]), Math.min(b[1], nb[1]), Math.max(b[2], nb[2]), Math.max(b[3], nb[3])]
            : nb;
    }
}

// Stamp circles along a segment so a fast drag leaves a continuous stroke.
function _brushLine(det, a, b, radius, subtract) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const dist = Math.hypot(dx, dy);
    const n = Math.max(1, Math.ceil(dist / Math.max(1, radius * 0.4)));
    for (let i = 1; i <= n; i++) {
        const t = i / n;
        _brushStamp(det, a[0] + dx * t, a[1] + dy * t, radius, subtract);
    }
}

// Which candidate a brush stroke acts on: the one under the cursor (smallest
// bbox), else the nearest by bbox-centre so a stroke starting just outside a
// mask still attaches to it.
function _pickTouchupTarget(node, px, py) {
    const dets = node._AngeloDetections || [];
    if (!dets.length) return null;
    const hit = _detAtPoint(node, px, py);
    if (hit) return hit;
    let best = null, bestD = Infinity;
    for (const d of dets) {
        const b = _detBbox(node, d);
        if (!b) continue;
        const dxc = (b[0] + b[2]) / 2 - px, dyc = (b[1] + b[3]) / 2 - py;
        const dist = Math.hypot(dxc, dyc);
        if (dist < bestD) { bestD = dist; best = d; }
    }
    return best;
}

// Per-node scratch canvas for tinting a raster edit-mask in a candidate's
// colour before compositing it onto the overlay. Using a per-node canvas
// (stored on node._AngeloScratch) avoids shared-state corruption if two
// nodes ever redraw in overlapping animation frames.
function _getScratch(node, w, h) {
    if (!node._AngeloScratch) node._AngeloScratch = document.createElement("canvas");
    const s = node._AngeloScratch;
    if (s.width !== w) s.width = w;
    if (s.height !== h) s.height = h;
    return s;
}
function _drawTintedMask(ctx, det, color, fillAlpha, node) {
    const m = det._editMask;
    const W = m.width, H = m.height;
    const s = _getScratch(node, W, H);
    const sx = s.getContext("2d");
    sx.clearRect(0, 0, W, H);
    sx.globalCompositeOperation = "source-over";
    sx.fillStyle = color;
    sx.fillRect(0, 0, W, H);
    sx.globalCompositeOperation = "destination-in";
    sx.drawImage(m, 0, 0);
    const prev = ctx.globalAlpha;
    ctx.globalAlpha = prev * fillAlpha;
    ctx.drawImage(s, 0, 0, W, H);
    ctx.globalAlpha = prev;
}

// Draw candidate outlines (called from redrawCanvasWithOverlays, image-px ctx).
function drawDetections(node, ctx) {
    const dets = node._AngeloDetections;
    if (!dets || !dets.length) return;
    const smart = isSmartInpaintMode(node);
    ctx.save();
    // Selection-highlight opacity (the floating slider) scales the whole
    // overlay so the user can dim it and inspect the edited region's edges.
    ctx.globalAlpha = (typeof node._AngeloDetOpacity === "number") ? node._AngeloDetOpacity : 1;
    const edited = node._AngeloEditedDets;
    dets.forEach((d, i) => {
        const hot = (i === node._AngeloHoverDet);
        const done = edited && edited.has(i);
        let tint;   // solid colour for the raster-mask fill
        if (hot) {
            ctx.lineWidth = 4;
            ctx.strokeStyle = "rgba(255, 220, 80, 1.0)";
            ctx.fillStyle = "rgba(255, 220, 80, 0.25)";
            tint = "rgb(255, 220, 80)";
        } else if (done) {
            // already edited this session — green so the user can track progress
            ctx.lineWidth = 2;
            ctx.strokeStyle = "rgba(90, 220, 120, 0.95)";
            ctx.fillStyle = "rgba(90, 220, 120, 0.18)";
            tint = "rgb(90, 220, 120)";
        } else {
            ctx.lineWidth = 2;
            ctx.strokeStyle = "rgba(80, 200, 255, 0.9)";
            ctx.fillStyle = "rgba(80, 200, 255, 0.15)";
            tint = "rgb(80, 200, 255)";
        }
        if (!smart && d._editMask) {
            // Brushed candidate: its raster edit-mask is the source of truth.
            // A raster has no outline to thicken on hover like the polygon
            // candidates, so instead brighten the fill + add a glow so the
            // hovered one still pops clearly.
            if (hot) {
                ctx.save();
                ctx.shadowColor = tint;
                ctx.shadowBlur = 16;
                _drawTintedMask(ctx, d, tint, 0.55, node);
                ctx.restore();
            } else {
                _drawTintedMask(ctx, d, tint, 0.30, node);
            }
        } else if (smart) {
            const b = _detBbox(node, d);
            if (b) {
                ctx.beginPath();
                ctx.rect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
                ctx.fill(); ctx.stroke();
            }
        } else {
            for (const poly of _detPolys(node, d)) {
                if (!poly || poly.length < 6) continue;
                ctx.beginPath();
                ctx.moveTo(poly[0], poly[1]);
                for (let k = 2; k < poly.length - 1; k += 2) ctx.lineTo(poly[k], poly[k + 1]);
                ctx.closePath();
                ctx.fill(); ctx.stroke();
            }
        }
    });

    // Touch-up brush preview — a circle at the cursor while Shift/Alt is held
    // (green = add, red = subtract).
    const bp = node._AngeloBrushPreview;
    if (bp) {
        ctx.globalAlpha = (typeof node._AngeloDetOpacity === "number") ? node._AngeloDetOpacity : 1;
        ctx.lineWidth = 2;
        ctx.strokeStyle = bp.subtract ? "rgba(255, 90, 90, 0.95)" : "rgba(90, 230, 130, 0.95)";
        ctx.fillStyle = bp.subtract ? "rgba(255, 90, 90, 0.12)" : "rgba(90, 230, 130, 0.12)";
        ctx.beginPath();
        ctx.arc(bp.x, bp.y, bp.r, 0, Math.PI * 2);
        ctx.fill(); ctx.stroke();
    }
    ctx.restore();
}

// Show the Detect row only in Edit Mode's masked sub-modes (Refine +
// Smart Inpaint); hide in Smart Guided (no mask) and Sampler Mode.
function syncDetectControls(node) {
    const row = node._AngeloDetectRow;
    if (!row) return;
    const modeW = findWidget(node, "mode");
    const inEdit = modeW && String(modeW.value) === "Edit Mode";
    const show = inEdit && !isSmartGuidedInpaintMode(node);
    // Must restore "flex" (not "") — an empty string reverts the row to a
    // <div>'s default display:block, which kills flex-wrap:nowrap and the
    // separator's align-self:stretch (dropdown drops to a new line, sep
    // vanishes).
    row.style.display = show ? "flex" : "none";
    if (!show) clearDetections(node);
}

function showLoadImagePopup(node, file) {
    const backdrop = document.createElement("div");
    backdrop.style.cssText = "position:fixed; inset:0; background:rgba(0,0,0,0.6); "
        + "display:flex; align-items:center; justify-content:center; z-index:10000; "
        + "font-family:Arial,sans-serif;";

    const modal = document.createElement("div");
    modal.style.cssText = "background:#2a2a2a; color:#ddd; border:1px solid #555; "
        + "border-radius:8px; padding:16px; width:360px; max-width:90vw; "
        + "display:flex; flex-direction:column; gap:10px;";

    const header = document.createElement("div");
    header.textContent = "Load Image — resolution";
    header.style.cssText = "font-size:14px; font-weight:bold; color:#aaa;";
    modal.appendChild(header);

    const hint = document.createElement("div");
    hint.textContent = `"${file.name}" — both options round dimensions to a multiple of 16.`;
    hint.style.cssText = "font-size:11px; color:#888;";
    modal.appendChild(hint);

    // Two mutually-exclusive choices via radio inputs.
    const mkRadio = (value, labelText, checked) => {
        const row = document.createElement("label");
        row.style.cssText = "display:flex; align-items:center; gap:8px; font-size:13px; cursor:pointer; padding:3px 2px;";
        const r = document.createElement("input");
        r.type = "radio";
        r.name = "angelo_loadres";
        r.value = value;
        r.checked = !!checked;
        const span = document.createElement("span");
        span.textContent = labelText;
        row.appendChild(r);
        row.appendChild(span);
        return { row, radio: r };
    };
    const keep = mkRadio("keep", "Keep current resolution", true);
    const resize = mkRadio("mp", "Resize to", false);
    modal.appendChild(keep.row);

    // Resize row: radio + MP input + "MP" label, inline.
    const mpInput = document.createElement("input");
    mpInput.type = "number";
    mpInput.min = "0.1"; mpInput.max = "8"; mpInput.step = "0.1"; mpInput.value = "1.5";
    mpInput.style.cssText = "width:60px; background:#1a1a1a; color:#eee; border:1px solid #555; border-radius:3px; padding:2px 6px; font-size:12px;";
    const mpLabel = document.createElement("span");
    mpLabel.textContent = "MP";
    mpLabel.style.cssText = "font-size:13px;";
    resize.row.appendChild(mpInput);
    resize.row.appendChild(mpLabel);
    modal.appendChild(resize.row);
    // Picking the MP field implies the resize choice.
    mpInput.addEventListener("focus", () => { resize.radio.checked = true; });

    const footer = document.createElement("div");
    footer.style.cssText = "display:flex; justify-content:flex-end; gap:8px; margin-top:8px;";
    const cancelBtn = document.createElement("button");
    cancelBtn.textContent = "Cancel";
    cancelBtn.style.cssText = "background:#444; color:#ddd; border:none; padding:6px 14px; border-radius:4px; cursor:pointer;";
    const loadBtn = document.createElement("button");
    loadBtn.textContent = "Load";
    loadBtn.style.cssText = "background:rgba(30,120,80,0.95); color:#fff; border:none; padding:6px 14px; border-radius:4px; cursor:pointer;";
    footer.appendChild(cancelBtn);
    footer.appendChild(loadBtn);
    modal.appendChild(footer);

    backdrop.appendChild(modal);
    document.body.appendChild(backdrop);

    const close = () => { if (backdrop.parentNode) backdrop.parentNode.removeChild(backdrop); };
    cancelBtn.addEventListener("click", close);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) close(); });
    loadBtn.addEventListener("click", () => {
        const mode = resize.radio.checked ? "mp" : "keep";
        const mp = Math.max(0.1, Math.min(8, parseFloat(mpInput.value) || 1.5));
        close();
        _uploadLoadedImage(node, file, mode, mp);
    });
}

function isPaintModeOn(node) {
    const w = findWidget(node, "paint_mode");
    if (!w) {
        dbg("isPaintModeOn: paint_mode widget NOT FOUND on node — widget list:",
            (node.widgets || []).map(x => x.name));
        return false;
    }
    const v = w.value;
    // Coerce defensively — different ComfyUI versions may store BOOLEAN
    // widgets as actual booleans, the strings "true"/"false", or 0/1.
    const on = (v === true || v === 1 || v === "true" || v === "True" || v === "1");
    return on;
}

function isSmartInpaintMode(node) {
    const w = findWidget(node, "inpainting_mode");
    if (!w) return false;
    return w.value === "Smart Inpaint";
}

function isSmartGuidedInpaintMode(node) {
    const w = findWidget(node, "inpainting_mode");
    if (!w) return false;
    return w.value === "Smart Guided Inpaint";
}

// Either edit-model mode (both inject reference_latents).
function isAnySmartMode(node) {
    return isSmartInpaintMode(node) || isSmartGuidedInpaintMode(node);
}

// Smart Guided Inpaint location labels — MUST match the Python
// _GUIDED_LOCATION_PREFIXES keys exactly (Python owns the label→prefix
// mapping; JS only stores the chosen label).
const _Angelo_GUIDED_LOCATIONS = [
    "(none)", "Whole image",
    "Top left", "Top middle", "Top right",
    "Middle left", "Center", "Middle right",
    "Bottom left", "Bottom middle", "Bottom right",
    "Left edge", "Right edge", "Top edge", "Bottom edge",
    "Top half", "Bottom half", "Left half", "Right half",
    "Top of the image", "Bottom of the image",
];

// Smart Guided Inpaint has no click/drag — this fires the backend's
// new_click gate so the whole-image guided edit runs. Sets a valid
// (image-centre) click point, bumps click_seq, clears any stale
// stroke/rect, and queues.
function triggerGuidedRefine(node) {
    const ws = findWidget(node, "click_seq");
    if (!ws) return;
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    const wr = findWidget(node, "reset");
    const wsp = findWidget(node, "stroke_points");
    const wrp = findWidget(node, "rect_points");
    const img = node._AngeloImg;
    const cx = img && img.naturalWidth ? Math.round(img.naturalWidth / 2) : 0;
    const cy = img && img.naturalHeight ? Math.round(img.naturalHeight / 2) : 0;
    if (wx) setWidget(wx, cx);
    if (wy) setWidget(wy, cy);
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    if (wr) setWidget(wr, false);
    if (wsp) setWidget(wsp, "");
    if (wrp) setWidget(wrp, "");
    dbg("queueing workflow (smart guided edit)", { click_seq: ws.value });
    queuePrompt();
}

function triggerRectRefine(node, rect) {
    const wrp = findWidget(node, "rect_points");
    const ws = findWidget(node, "click_seq");
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    const wr = findWidget(node, "reset");
    const wsp = findWidget(node, "stroke_points");
    if (!wrp || !ws) {
        dbg("ERROR: rect widgets not found", { wrp: !!wrp, ws: !!ws });
        return;
    }

    // Round to ints; the backend rasterises into latent space anyway.
    const [x1, y1, x2, y2] = rect.map(v => Math.round(v));
    setWidget(wrp, JSON.stringify([[x1, y1, x2, y2]]));
    setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    // Stash a fallback target at the rect centre (some downstream UI
    // reads click_x/y).
    if (wx) setWidget(wx, Math.round((x1 + x2) / 2));
    if (wy) setWidget(wy, Math.round((y1 + y2) / 2));
    if (wr) setWidget(wr, false);
    // Clear stroke_points + seg_polygon so a previous paint stroke or a
    // SAM-detected silhouette can't fall through.
    if (wsp) setWidget(wsp, "");
    const wseg = findWidget(node, "seg_polygon");
    if (wseg) setWidget(wseg, "");
    const wmaskpng = findWidget(node, "seg_mask_png");
    if (wmaskpng) setWidget(wmaskpng, "");

    dbg("queueing workflow (smart inpaint rect)", { x1, y1, x2, y2, click_seq: ws.value });
    queuePrompt();
}


// ============================================================
// Reset button (canvas-rendered on the title bar)
// ============================================================

// (Title-bar button rect/draw helpers removed when Reset/Undo moved to
// the DOM toggle bar. Earlier versions had resetButtonRect /
// undoButtonRect / drawTitleButton / hitRect / roundedRect here; see
// git history if we ever want canvas-rendered buttons again.)

function triggerUndo(node) {
    const wu = findWidget(node, "undo_seq");
    if (!wu) return;
    setWidget(wu, ((wu.value || 0) + 1) & 0x7FFFFFFF);
    dbg("queue undo", { undo_seq: wu.value });
    if (typeof app.queuePrompt === "function") app.queuePrompt(0);
}

// Redo: re-apply the edit Undo most recently removed. Pure restore (like
// Undo), so it just bumps redo_seq and re-queues — Python pops its redo
// stack back onto history. No seed change, no re-sample.
function triggerRedo(node) {
    const wr = findWidget(node, "redo_seq");
    if (!wr) return;
    setWidget(wr, ((wr.value || 0) + 1) & 0x7FFFFFFF);
    dbg("queue redo", { redo_seq: wr.value });
    if (typeof app.queuePrompt === "function") app.queuePrompt(0);
}

// Re-roll: redo the most recent edit with a fresh seed, same mask, same
// pre-edit base. Force a NEW random seed even if Seed Ctrl is "fixed" —
// a re-roll is by definition new dice — then bump reroll_seq so Python
// pops the last attempt and re-runs the (unchanged) mask widgets in its
// place. The mask widgets (click_x/y, stroke_points, rect_points,
// seg_polygon) are deliberately left untouched so the same region is
// reused.
function triggerReroll(node) {
    const wr = findWidget(node, "reroll_seq");
    if (!wr) return;
    const wseed = findWidget(node, "seed");
    if (wseed) {
        setWidget(wseed, Math.floor(Math.random() * Number.MAX_SAFE_INTEGER));
        syncSeedInput(node);
    }
    setWidget(wr, ((wr.value || 0) + 1) & 0x7FFFFFFF);
    dbg("queue reroll", { reroll_seq: wr.value, seed: wseed && wseed.value });
    queuePrompt();
}

function triggerReset(node) {
    const wr = findWidget(node, "reset");
    const ws = findWidget(node, "click_seq");
    const wx = findWidget(node, "click_x");
    const wy = findWidget(node, "click_y");
    if (!wr) return;
    // Use setWidget() for all mutations so the widget callback fires and
    // serialisation stays consistent — direct .value assignment would
    // bypass the callback in some ComfyUI versions.
    setWidget(wr, true);
    if (wx) setWidget(wx, -1);
    if (wy) setWidget(wy, -1);
    if (ws) setWidget(ws, ((ws.value || 0) + 1) & 0x7FFFFFFF);
    node._AngeloImg = null;
    if (node._AngeloCanvas) {
        const ctx = node._AngeloCanvas.getContext("2d");
        ctx.clearRect(0, 0, node._AngeloCanvas.width, node._AngeloCanvas.height);
    }
    if (node._AngeloPlaceholder) node._AngeloPlaceholder.style.display = "flex";
    app.graph.setDirtyCanvas(true, true);
    if (typeof app.queuePrompt === "function") app.queuePrompt(0);
    // Safety-net: if onExecuted never fires (e.g. server error), clear
    // `reset` after 5 s so a stuck widget doesn't wipe the cache on the
    // next unrelated Queue. onExecuted clears it immediately on success.
    setTimeout(() => {
        if (wr.value === true) {
            setWidget(wr, false);
            app.graph.setDirtyCanvas(true, true);
        }
    }, 5000);
}


// ============================================================
// Helpers
// ============================================================

function findWidget(node, name) {
    if (!node.widgets) return null;
    return node.widgets.find(w => w.name === name);
}

/**
 * Build a momentary action button (Reset, Undo). Click → onClick().
 * `kind` selects a colour theme so different actions are visually
 * distinct in the bar.
 */
function makeActionButton(label, onClick, kind = "neutral") {
    const themes = {
        reset:   { fg: "#ffe0d0", bg: "rgba(70, 50, 50, 0.95)",  border: "rgba(220, 140, 100, 0.9)" },
        undo:    { fg: "#dde7ff", bg: "rgba(50, 60, 70, 0.95)",  border: "rgba(120, 170, 220, 0.9)" },
        redo:    { fg: "#d2f3e2", bg: "rgba(48, 66, 60, 0.95)",  border: "rgba(110, 200, 160, 0.9)" },
        reroll:  { fg: "#ecdcff", bg: "rgba(58, 50, 72, 0.95)",  border: "rgba(170, 130, 220, 0.9)" },
        neutral: { fg: "#ccc",    bg: "#2a2a2a",                  border: "#555" },
    };
    const th = themes[kind] || themes.neutral;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    btn.style.cursor = "pointer";
    btn.style.padding = "3px 10px";
    btn.style.fontSize = "11px";
    btn.style.fontWeight = "bold";
    btn.style.border = `1px solid ${th.border}`;
    btn.style.borderRadius = "3px";
    btn.style.background = th.bg;
    btn.style.color = th.fg;
    btn.style.userSelect = "none";
    btn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        onClick();
    });
    return btn;
}

/** Thin vertical separator for the toggle bar. */
function makeSeparator() {
    const sep = document.createElement("div");
    sep.style.width = "1px";
    sep.style.alignSelf = "stretch";
    sep.style.background = "#444";
    sep.style.margin = "0 4px";
    return sep;
}

/** A horizontal row of controls inside the toolbar wrapper. */
function makeToolbarRow() {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.flexDirection = "row";
    row.style.flexWrap = "wrap";
    row.style.alignItems = "center";
    row.style.gap = "4px";
    row.style.padding = "4px 6px";
    return row;
}

/**
 * Build a click-to-toggle button. Returns the DOM element; the caller
 * is responsible for syncing its visual state to whatever underlying
 * widget value it's bound to (call syncPersistentMaskToggle / similar on
 * the relevant node after the widget value changes).
 */
function makeToggleButton(label, onToggle) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label + ": OFF";
    btn.dataset.state = "off";
    btn.style.cursor = "pointer";
    btn.style.padding = "3px 10px";
    btn.style.fontSize = "11px";
    btn.style.fontWeight = "bold";
    btn.style.border = "1px solid #555";
    btn.style.borderRadius = "3px";
    btn.style.background = "#2a2a2a";
    btn.style.color = "#bbb";
    btn.style.userSelect = "none";
    btn._AngeloLabel = label;
    btn.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        onToggle();
    });
    return btn;
}

// --- Sync helpers: keep DOM toolbar controls in lockstep with the
//     underlying widget values. Called once on node creation (so persisted
//     widget state reflects in the UI) and after every toggle click.

function _syncToggle(btn, widgetValue, onColor) {
    if (!btn) return;
    const on = !!widgetValue;
    btn.textContent = btn._AngeloLabel + (on ? ": ON" : ": OFF");
    btn.dataset.state = on ? "on" : "off";
    btn.style.background = on ? onColor.bg : "#2a2a2a";
    btn.style.color = on ? "#fff" : "#bbb";
    btn.style.borderColor = on ? onColor.border : "#555";
}

const _TOGGLE_ON_COLORS = {
    blue:   { bg: "rgba(20, 80, 140, 0.95)",  border: "rgba(120, 170, 220, 0.9)" },
    green:  { bg: "rgba(30, 120, 80, 0.95)",  border: "rgba(140, 220, 170, 0.9)" },
    purple: { bg: "rgba(95, 50, 130, 0.95)",  border: "rgba(180, 140, 220, 0.9)" },
    teal:   { bg: "rgba(30, 110, 130, 0.95)", border: "rgba(140, 200, 220, 0.9)" },
    amber:  { bg: "rgba(135, 90, 25, 0.95)",  border: "rgba(235, 185, 95, 0.9)" },
};

function syncPersistentMaskToggle(node) {
    // Backend forces persistent_mask OFF in Smart Guided Inpaint (no
    // mask to persist) — reflect that as OFF rather than the stale
    // widget value.
    const effective = isSmartGuidedInpaintMode(node)
        ? false
        : findWidget(node, "persistent_mask")?.value;
    _syncToggle(node._AngeloPersistentMaskToggle, effective, _TOGGLE_ON_COLORS.blue);
}

function syncAreaPromptToggle(node) {
    // Backend forces area_prompt=True in Smart Inpaint regardless of
    // widget state — reflect that in the displayed toggle so it reads
    // ON instead of misleadingly showing the underlying widget value.
    const effective = isSmartInpaintMode(node)
        ? true
        : findWidget(node, "area_prompt")?.value;
    _syncToggle(node._AngeloAreaPromptToggle, effective, _TOGGLE_ON_COLORS.purple);
}

function syncPaintModeToggle(node) {
    _syncToggle(node._AngeloPaintModeToggle, findWidget(node, "paint_mode")?.value, _TOGGLE_ON_COLORS.teal);
}

function syncRestoreToggle(node) {
    // Backend honours restore_mode in Refine only. Show OFF in Smart modes
    // so the toolbar reflects the effective behavior, not a stale widget.
    const effective = isAnySmartMode(node)
        ? false
        : findWidget(node, "restore_mode")?.value;
    _syncToggle(node._AngeloRestoreToggle, effective, _TOGGLE_ON_COLORS.amber);
}

function syncFineUpscaleToggle(node) {
    // Backend forces fine_upscaling ON in Smart Inpaint and OFF in
    // Smart Guided Inpaint regardless of widget state — reflect that in
    // the displayed toggle rather than the (stale) underlying value.
    let effective;
    if (isSmartInpaintMode(node)) effective = true;
    else if (isSmartGuidedInpaintMode(node)) effective = false;
    else effective = findWidget(node, "fine_upscaling")?.value;
    _syncToggle(node._AngeloFineUpscaleToggle, effective, _TOGGLE_ON_COLORS.green);

    const showFine = !!effective;
    const fineGroup = [
        [node._AngeloFineSep, ""],
        [node._AngeloMpInput, "inline-flex"],
        [node._AngeloMaxInput, "inline-flex"],
        [node._AngeloCtxPadInput, "inline-flex"],
        [node._AngeloMethodSelect, "inline-flex"],
    ];
    for (const [el, shown] of fineGroup) {
        if (el) el.style.display = showFine ? shown : "none";
    }
}

// Smart Inpaint mode hard-locks several params on the backend
// regardless of widget state:
//   denoise=1.0, fine_upscaling=ON, fine_context_pad=0, area_prompt=ON
// Dim the corresponding UI controls + paint_mode + click_radius so
// the user can see at a glance that they're not driving anything.
// Feather is NOT locked — a soft edge can help blend the insert, so
// it stays under user control.
const _Angelo_LOCK_SUFFIX = "\n\n[Locked in this Inpaint mode.]";
function _dimControls(node, ids, dim) {
    for (const id of ids) {
        const el = node[id];
        if (!el) continue;
        el.style.opacity = dim ? "0.35" : "";
        el.style.pointerEvents = dim ? "none" : "";
        if (dim) {
            el.title = (el.title?.split("\n\n[")[0] || el.title || "") + _Angelo_LOCK_SUFFIX;
        } else if (el.title && el.title.includes(_Angelo_LOCK_SUFFIX.trim())) {
            el.title = el.title.split(_Angelo_LOCK_SUFFIX)[0];
        }
    }
}

function syncSmartInpaintLockedWidgets(node) {
    const guided = isSmartGuidedInpaintMode(node);
    const anySmart = isSmartInpaintMode(node) || guided;

    // Common locks for BOTH smart modes — backend forces these or they
    // don't apply: denoise, fine_upscale toggle, paint_mode, click
    // radius, area_prompt toggle.
    _dimControls(node, [
        "_AngeloDenoiseInput",
        "_AngeloFineUpscaleToggle",
        "_AngeloPaintModeToggle",
        "_AngeloRestoreToggle",
        "_AngeloClickRadiusInput",
        "_AngeloAreaPromptToggle",
        "_AngeloCtxPadInput",
    ], anySmart);

    // Feather: live in Smart Inpaint (a soft edge can help blend the
    // insert), disabled in Smart Guided (whole-image edit, no mask edge).
    _dimControls(node, ["_AngeloFeatherInput"], guided);

    // Persistent Mask: meaningless in Smart Guided (no mask). Dimmed +
    // forced OFF there; left alone in Smart Inpaint (re-rolls the rect).
    _dimControls(node, ["_AngeloPersistentMaskToggle"], guided);

    // Fine Upscale + Area Prompt toggles' displayed state is forced by
    // the backend (ON for Smart Inpaint; Fine Upscale OFF for Smart
    // Guided; Area Prompt ON for both). Re-run their sync so the labels
    // reflect the forced state, and refresh the Area Prompt box + guided
    // controls visibility.
    syncFineUpscaleToggle(node);
    syncAreaPromptToggle(node);
    syncPersistentMaskToggle(node);
    syncRestoreToggle(node);
    syncAreaPromptVisibility(node);
    // Detect row hides in Smart Guided (no mask), shows in Refine/Smart Inpaint.
    syncDetectControls(node);
}

function _syncNumberInput(wrap, widgetValue) {
    if (!wrap || !wrap._AngeloInput || widgetValue == null) return;
    if (wrap._AngeloInput.value !== String(widgetValue)) {
        wrap._AngeloInput.value = String(widgetValue);
    }
}

function syncClickRadiusInput(node) {
    _syncNumberInput(node._AngeloClickRadiusInput, findWidget(node, "click_radius")?.value);
}
function syncFeatherInput(node) {
    _syncNumberInput(node._AngeloFeatherInput, findWidget(node, "feather_radius")?.value);
}
function syncDenoiseInput(node) {
    _syncNumberInput(node._AngeloDenoiseInput, findWidget(node, "denoise")?.value);
}
function syncSeedInput(node) {
    _syncNumberInput(node._AngeloSeedInput, findWidget(node, "seed")?.value);
}
function syncSeedCtrlSelect(node) {
    const wrap = node._AngeloSeedCtrlSelect;
    const w = findWidget(node, "seed_control");
    if (!wrap || !w || !wrap._AngeloSelect) return;
    if (wrap._AngeloSelect.value !== String(w.value)) {
        wrap._AngeloSelect.value = String(w.value);
    }
}

/**
 * Mode-state sync: when mode == "Sampler Mode" grey out the toolbar
 * + canvas (controls inert); when "Edit Mode" un-grey. Also
 * auto-forces sampler_seed_control to "fixed" when switching INTO
 * Edit Mode, so subsequent Queue presses don't regenerate the
 * cached base.
 */
function syncModeState(node) {
    const modeW = findWidget(node, "mode");
    if (!modeW) return;
    const inSampler = String(modeW.value) === "Sampler Mode";

    // Row 3 (mode + shared gen config) is ALWAYS active — both modes use
    // steps/cfg/sampler/scheduler, and the Mode dropdown must always work.
    //
    // Refinement control rows (1+2) grey out in Sampler Mode — they don't
    // apply while generating a base.
    const refineRows = node._AngeloRefineRowsWrap;
    if (refineRows) {
        refineRows.style.opacity = inSampler ? "0.4" : "1";
        refineRows.style.pointerEvents = inSampler ? "none" : "auto";
    }
    // Sampler-seed row (4) greys in Edit Mode — that seed group only
    // drives the base generation.
    const samplerSeedRow = node._AngeloSamplerSeedRow;
    if (samplerSeedRow) {
        samplerSeedRow.style.opacity = inSampler ? "1" : "0.4";
        samplerSeedRow.style.pointerEvents = inSampler ? "auto" : "none";
    }
    // Detect row shows only in Edit Mode's masked sub-modes.
    syncDetectControls(node);
    // Cursor: default in Sampler Mode (clicks do nothing); crosshair
    // in Smart Inpaint (rectangle drag); cell when paint mode is on
    // for Refine; crosshair otherwise.
    if (node._AngeloCanvas) {
        if (inSampler || isSmartGuidedInpaintMode(node)) {
            // Sampler Mode: clicks do nothing. Smart Guided: no canvas
            // interaction (location dropdown + Generate button drive it).
            node._AngeloCanvas.style.cursor = "default";
        } else if (isSmartInpaintMode(node)) {
            node._AngeloCanvas.style.cursor = "crosshair";
        } else {
            node._AngeloCanvas.style.cursor = isPaintModeOn(node) ? "cell" : "crosshair";
        }
    }
}

function syncModeSwitchToFixed(node, prevMode) {
    // Called from the mode widget's callback. If switching INTO Refinement
    // Mode, force sampler_seed_control to "fixed" AND restore sampler_seed
    // to the seed Python actually used for the cached base (which after
    // after-gen control may have drifted from the current widget value).
    const modeW = findWidget(node, "mode");
    if (!modeW) return;
    const nowInRefine = String(modeW.value) === "Edit Mode";
    if (nowInRefine && prevMode === "Sampler Mode") {
        lockSeedToAtRun(node, "sampler_seed", "sampler_seed_control");
        dbg("syncMode: forced sampler_seed_control → fixed + restored sampler_seed");
    }
}

/**
 * Apply after-gen control (fixed / increment / decrement / randomize) to
 * a seed widget. ComfyUI's built-in seed widgets auto-attach this dropdown;
 * ours are explicit ENUM widgets so we do the modification ourselves.
 * Runs AFTER the response is processed so seed_at_run capture happens
 * first (lock-on-fixed restores from the pre-modification value).
 *
 * NOTE: JavaScript's float64 can only represent integers exactly up to
 * Number.MAX_SAFE_INTEGER (2^53 − 1 ≈ 9×10^15). The Python widget max is
 * 0xFFFFFFFFFFFFFFFF (uint64), which is beyond float64 precision. We cap
 * all arithmetic at Number.MAX_SAFE_INTEGER so increment/decrement/randomize
 * never produce a value that can't be represented exactly.
 */
function applyAfterGenControl(node, seedWidgetName, controlWidgetName) {
    const seedW = findWidget(node, seedWidgetName);
    const ctrlW = findWidget(node, controlWidgetName);
    if (!seedW || !ctrlW) return;
    const ctrl = String(ctrlW.value);
    const maxSeed = Number.MAX_SAFE_INTEGER;  // 2^53−1; safe float64 integer range
    let cur = Number(seedW.value || 0);
    let next = cur;
    switch (ctrl) {
        case "fixed":     return;  // no change
        case "increment": next = cur >= maxSeed ? 0 : cur + 1; break;
        case "decrement": next = cur > 0 ? cur - 1 : maxSeed; break;
        case "randomize":
            // 53 bits is the safe integer range; that's plenty of seed
            // entropy for any sampler.
            next = Math.floor(Math.random() * maxSeed);
            break;
        default: return;
    }
    setWidget(seedW, next);
    // Mirror to the matching toolbar input.
    if (seedWidgetName === "seed") {
        syncSeedInput(node);
    } else if (seedWidgetName === "sampler_seed") {
        syncSamplerSeedInput(node);
    }
    dbg("after-gen", seedWidgetName, ctrl, cur, "→", next);
}

/**
 * "Lock seed to the value Python actually used at the last run." Used when
 * a control widget flips to "fixed" — restores the seed widget to what was
 * sent to Python (before ComfyUI's after-gen modified it), then forces the
 * control to "fixed". The "_AngeloSamplerSeedAtRun" / "_AngeloRefineSeedAtRun"
 * fields are captured from the Angelo_*_seed_at_run keys in the ui message
 * each onExecuted.
 */
function lockSeedToAtRun(node, seedWidgetName, controlWidgetName) {
    const seedW = findWidget(node, seedWidgetName);
    const ctrlW = findWidget(node, controlWidgetName);
    if (!seedW || !ctrlW) return;
    const stored = seedWidgetName === "sampler_seed"
        ? node._AngeloSamplerSeedAtRun
        : node._AngeloRefineSeedAtRun;
    // Always set control to fixed.
    if (ctrlW.value !== "fixed") setWidget(ctrlW, "fixed");
    // Restore seed value if we have a known-used value to fall back to.
    if (stored != null) {
        if (Number(seedW.value) !== Number(stored)) {
            setWidget(seedW, Number(stored));
        }
    }
    // Mirror to toolbar.
    if (seedWidgetName === "seed") {
        syncSeedInput(node);
        syncSeedCtrlSelect(node);
    } else if (seedWidgetName === "sampler_seed") {
        syncSamplerSeedInput(node);
        syncSamplerSeedCtrlSelect(node);
    }
}
function syncMpInput(node) {
    _syncNumberInput(node._AngeloMpInput, findWidget(node, "min_megapixels")?.value);
}
function syncMaxInput(node) {
    _syncNumberInput(node._AngeloMaxInput, findWidget(node, "max_upscale")?.value);
}

function syncCtxPadInput(node) {
    _syncNumberInput(node._AngeloCtxPadInput, findWidget(node, "fine_context_pad")?.value);
}

function syncMethodSelect(node) {
    const wrap = node._AngeloMethodSelect;
    const w = findWidget(node, "resize_method");
    if (!wrap || !w || !wrap._AngeloSelect) return;
    if (wrap._AngeloSelect.value !== String(w.value)) {
        wrap._AngeloSelect.value = String(w.value);
    }
}

// --- Sync helpers for the row 3/4 sampler + generation controls ---
function _syncDropdownWrap(wrap, widgetValue) {
    if (!wrap || !wrap._AngeloSelect || widgetValue == null) return;
    if (wrap._AngeloSelect.value !== String(widgetValue)) {
        wrap._AngeloSelect.value = String(widgetValue);
    }
}
function syncModeSelect(node) {
    _syncDropdownWrap(node._AngeloModeSelect, findWidget(node, "mode")?.value);
}
function syncStepsInput(node) {
    // steps widget removed — no-op kept so any external callers don't throw.
}
function syncSamplerSeedInput(node) {
    _syncNumberInput(node._AngeloSamplerSeedInput, findWidget(node, "sampler_seed")?.value);
}
function syncSamplerSeedCtrlSelect(node) {
    _syncDropdownWrap(node._AngeloSamplerSeedCtrlSelect, findWidget(node, "sampler_seed_control")?.value);
}
function syncSamplerDenoiseInput(node) {
    // sampler_denoise widget removed — sampler-mode denoise is now encoded in the
    // connected SIGMAS (BasicScheduler denoise param). No-op kept for compatibility
    // with any external code that might call this.
}
function syncGuidedLocationSelect(node) {
    _syncDropdownWrap(node._AngeloGuidedLocationSelect, findWidget(node, "guided_location")?.value);
}

function syncInpaintModeSelect(node) {
    const wrap = node._AngeloInpaintModeSelect;
    const w = findWidget(node, "inpainting_mode");
    if (!wrap || !w || !wrap._AngeloSelect) return;
    if (wrap._AngeloSelect.value !== String(w.value)) {
        wrap._AngeloSelect.value = String(w.value);
    }
    // Keep the locked-widget UI in sync — the mode value may have
    // changed via a workflow load, undo, or any path other than the
    // dropdown's own click handler.
    syncSmartInpaintLockedWidgets(node);
}

// Reflect ALL persisted widget state into the DOM toolbar controls.
// Call this both on node creation AND on configure (workflow load) —
// onNodeCreated runs BEFORE ComfyUI restores serialized widget values,
// so a sync there alone leaves toggles like Paint Mode showing the
// default instead of the saved value. onConfigure fires after the
// restore, so a second pass there fixes the mismatch.
function syncAllToolbarControls(node) {
    syncPersistentMaskToggle(node);
    syncAreaPromptToggle(node);
    syncPaintModeToggle(node);
    syncRestoreToggle(node);
    syncFineUpscaleToggle(node);
    syncClickRadiusInput(node);
    syncFeatherInput(node);
    syncDenoiseInput(node);
    syncSeedInput(node);
    syncSeedCtrlSelect(node);
    syncMpInput(node);
    syncMaxInput(node);
    syncCtxPadInput(node);
    syncMethodSelect(node);
    syncInpaintModeSelect(node);
    // Row 3/4 sampler + generation controls.
    syncModeSelect(node);
    syncStepsInput(node);
    syncSamplerSeedInput(node);
    syncSamplerSeedCtrlSelect(node);
    syncSamplerDenoiseInput(node);
    syncGuidedLocationSelect(node);
    syncLoadImageControls(node);
    syncDetectControls(node);
    syncSmartInpaintLockedWidgets(node);
    syncAreaPromptBox(node);
    syncAreaPromptVisibility(node);
    syncModeState(node);
}

function makeDropdown(label, options, onChange) {
    const wrap = document.createElement("div");
    wrap.style.display = "inline-flex";
    wrap.style.alignItems = "center";
    wrap.style.gap = "4px";
    wrap.style.fontSize = "11px";
    wrap.style.color = "#bbb";
    wrap.style.padding = "0 4px";

    const lbl = document.createElement("span");
    lbl.textContent = label;
    wrap.appendChild(lbl);

    const sel = document.createElement("select");
    sel.style.fontSize = "11px";
    sel.style.padding = "2px 4px";
    sel.style.border = "1px solid #555";
    sel.style.borderRadius = "3px";
    sel.style.background = "#1a1a1a";
    sel.style.color = "#ddd";
    for (const opt of options) {
        const o = document.createElement("option");
        o.value = opt;
        o.textContent = opt;
        sel.appendChild(o);
    }
    sel.addEventListener("change", () => onChange(sel.value));
    sel.addEventListener("mousedown", (e) => e.stopPropagation());
    sel.addEventListener("pointerdown", (e) => e.stopPropagation());
    wrap.appendChild(sel);
    wrap._AngeloSelect = sel;
    return wrap;
}

/**
 * Small inline numeric input with a label. width is the input width in
 * pixels (label adds ~20px). Calls onChange(numericValue) when committed.
 */
function makeNumberInput(label, opts, onChange) {
    const wrap = document.createElement("div");
    wrap.style.display = "inline-flex";
    wrap.style.alignItems = "center";
    wrap.style.gap = "4px";
    wrap.style.fontSize = "11px";
    wrap.style.color = "#bbb";
    wrap.style.padding = "0 4px";

    const lbl = document.createElement("span");
    lbl.textContent = label;
    wrap.appendChild(lbl);

    const input = document.createElement("input");
    input.type = "number";
    input.min = String(opts.min ?? 0);
    input.max = String(opts.max ?? 100);
    input.step = String(opts.step ?? 1);
    input.style.width = (opts.width || 60) + "px";
    input.style.padding = "2px 4px";
    input.style.fontSize = "11px";
    input.style.border = "1px solid #555";
    input.style.borderRadius = "3px";
    input.style.background = "#1a1a1a";
    input.style.color = "#ddd";
    function _commitNumberInput() {
        let v = parseFloat(input.value);
        if (!isFinite(v)) v = opts.min ?? 0;
        const lo = opts.min ?? -Infinity, hi = opts.max ?? Infinity;
        v = Math.max(lo, Math.min(hi, v));
        input.value = String(v);
        onChange(v);
    }
    // Fire on blur (user tabs away) OR on Enter key so changes aren't lost
    // when the user types a value and presses Enter without leaving the field.
    input.addEventListener("change", _commitNumberInput);
    input.addEventListener("keydown", (e) => {
        if (e.key === "Enter") { e.preventDefault(); _commitNumberInput(); }
    });
    // Stop click on the input from selecting / dragging the node behind it.
    input.addEventListener("mousedown", (e) => e.stopPropagation());
    input.addEventListener("pointerdown", (e) => e.stopPropagation());
    wrap.appendChild(input);
    wrap._AngeloInput = input;
    return wrap;
}

function setWidget(widget, value) {
    if (!widget) return;
    widget.value = value;
    // Some ComfyUI versions sync the value-for-serialization through the
    // widget's callback rather than reading widget.value directly. Fire
    // it with the owning node (widget.node may be unset on hidden widgets
    // in some builds — fall back to null rather than crashing).
    try {
        if (typeof widget.callback === "function") {
            const ownerNode = widget.node || null;
            widget.callback(value, app.canvas, ownerNode);
        }
    } catch (e) {
        dbg("widget callback threw", widget.name, e);
    }
}

function hideMechanicalWidgets(node) {
    // JS-driven widgets that don't need to clutter the user-visible UI.
    // Skipped while Angelo_DEBUG is on so we can watch them update.
    if (Angelo_DEBUG) return;

    const hideNames = [
        // JS-driven plumbing (never user-visible)
        "click_x", "click_y", "click_seq",
        "image_w", "image_h", "undo_seq",
        "stroke_points", "rect_points",
        // Area Prompt text (driven by the DOM textarea below the canvas)
        "area_text_positive", "area_text_negative",
        // Smart Guided Inpaint location (driven by the DOM dropdown)
        "guided_location",
        // Load Image — driven by the Load Image button + popup
        "loaded_image", "loaded_image_seq", "loaded_resize_mode", "loaded_target_mp",
        // Detect (SAM 3 / YOLO) — driven by the Detect row + click-confirm
        "seg_polygon",
        // Detect touch-up brush — base64 raster mask
        "seg_mask_png",
        // Re-roll button — bumps to re-run the last edit with a new seed
        "reroll_seq",
        // Redo button — bumps to restore an edit that Undo removed
        "redo_seq",
        // Restore brush — driven by the Restore toggle on the toolbar
        "restore_mode",
        // Toolbar-driven (visible via the bar above the canvas)
        "persistent_mask", "area_prompt", "paint_mode", "fine_upscaling",
        "click_radius", "feather_radius", "denoise",
        "seed", "seed_control",
        "min_megapixels", "max_upscale", "resize_method", "fine_context_pad",
        "inpainting_mode",
        // Sampler / generation config — now in toolbar rows 3 & 4
        "mode", "sampler_seed", "sampler_seed_control",
        // Deprecated control — always-on, kept declared for serialization
        "auto_decode",
        // Driven by the Reset button on the toolbar — no need to see the widget too
        "reset",
    ];
    if (!node.widgets) return;
    for (const w of node.widgets) {
        // Hide explicit-name matches PLUS any ComfyUI-auto-added
        // "control_after_generate" dropdowns that may have been
        // attached to seed widgets (legacy auto-attach for widgets
        // literally named "seed"). The Python widgets opt out via
        // `"control_after_generate": False`, but some ComfyUI versions
        // ignore that opt-out, so this pattern-match catches the
        // orphaned dropdowns either way.
        const isAutoControl = typeof w.name === "string"
            && /control_after_generate/i.test(w.name);
        if (!hideNames.includes(w.name) && !isAutoControl) continue;
        // The "type = hidden" trick alone isn't enough — some renderers
        // still reserve the widget's natural height. Belt + suspenders:
        // use ComfyUI's own "converted-widget" type (which the frontend
        // explicitly skips), set hidden=true (newer LiteGraph respects
        // it), zero-out computeSize (-4 cancels the 4px inter-widget
        // gap), and null the draw function so nothing paints even if a
        // renderer iterates over hidden widgets.
        w.type = "converted-widget";
        w.hidden = true;
        w.computeSize = () => [0, -4];
        w.draw = () => {};
    }
}

function makeViewUrl(ref) {
    const params = new URLSearchParams();
    params.set("filename", ref.filename);
    if (ref.subfolder) params.set("subfolder", ref.subfolder);
    if (ref.type) params.set("type", ref.type);
    return `/view?${params.toString()}`;
}
