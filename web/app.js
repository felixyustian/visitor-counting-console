/* Crowd console front-end: crowd management panel + one MJPEG panel per camera.
   State arrives over /ws (JSON snapshot ~4 Hz); controls are small POSTs. */
(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const grid = $("#grid");
  const toastEl = $("#toast");
  const tpl = $("#cam-tpl");
  let state = null;
  let history = null;         // last history summary (sent every ~5 s)
  let toastTimer = 0;
  let setupShown = false;     // the day set-up screen was shown / dismissed this session

  const panels = new Map();   // cid -> panel element
  const drafts = new Map();   // cid -> {points: [[nx,ny],...], side: +1|-1}

  const fmtTime = (t) => new Date(t * 1000).toLocaleTimeString("en-GB", { hour12: false });
  const fmtUptime = (s) => `${String(Math.floor(s / 3600)).padStart(2, "0")}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  const GENDER = { M: "male", F: "female", "?": "?" };
  const AGE = { A: "adult", C: "child", "?": "?" };
  const LEVEL_TEXT = { normal: "NORMAL", busy: "BUSY", crowded: "APPROACHING CAPACITY", over: "OVER CAPACITY" };

  function toast(msg, ms = 2400) {
    toastEl.textContent = msg;
    toastEl.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastEl.classList.remove("show"), ms);
  }
  function esc(s) { return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  async function api(method, url, body) {
    const r = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (_) {}
      throw new Error(msg);
    }
    return r.json();
  }
  const tryApi = (...a) => api(...a).catch((e) => { toast(`error: ${e.message}`); throw e; });

  // ------------------------------------------------------------ modal
  const modal = $("#modal"), modalForm = $("#modal-form"), modalFields = $("#modal-fields"), modalMsg = $("#modal-msg");
  let modalSubmit = null;
  function openModal(title, fieldsHtml, onSubmit, okLabel = "apply") {
    $("#modal-title").textContent = title;
    modalFields.innerHTML = fieldsHtml;
    $("#modal-ok").textContent = okLabel;
    modalMsg.textContent = "";
    modalSubmit = onSubmit;
    modal.classList.remove("hidden");
    const first = $("input", modalFields);
    if (first) { first.focus(); first.select && first.select(); }
  }
  function closeModal() { modal.classList.add("hidden"); modalSubmit = null; }
  $("#modal-cancel").addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
  modalForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!modalSubmit) return;
    const data = Object.fromEntries(new FormData(modalForm).entries());
    try { await modalSubmit(data); closeModal(); }
    catch (err) { modalMsg.textContent = err.message; }
  });

  // ------------------------------------------------------------ day set-up screen
  const setup = $("#setup"), setupForm = $("#setup-form"), setupMsg = $("#setup-msg");
  function showSetup(site) {
    const f = setupForm;
    f.name.value = site.name || "";
    f.date.value = site.configured_today ? site.date : new Date().toISOString().slice(0, 10);
    f.open.value = site.open; f.close.value = site.close;
    f.capacity.value = site.capacity; f.warn_pct.value = site.warn_pct; f.baseline.value = site.baseline;
    f.apply_baseline.checked = !site.configured_today;
    f.current.value = state ? state.occupancy.current : "";   // head-count correction lives here now
    setupMsg.textContent = "";
    $("#setup-cancel").style.display = site.configured ? "" : "none";
    setup.classList.remove("hidden");
    setupShown = true;
  }
  $("#setup-cancel").addEventListener("click", () => setup.classList.add("hidden"));
  setupForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const d = Object.fromEntries(new FormData(setupForm).entries());
    d.apply_baseline = setupForm.apply_baseline.checked;
    // "people inside now" is a separate endpoint; only send it when it was changed
    const correction = d.current === "" ? null : Number(d.current);
    delete d.current;
    try {
      await api("POST", "/api/site", d);
      if (correction !== null && state && correction !== state.occupancy.current && !d.apply_baseline) {
        await api("POST", "/api/occupancy", { current: correction, note: "corrected from site settings" });
      }
      setup.classList.add("hidden");
      toast(d.apply_baseline ? "day started - count set to the starting occupancy" : "site settings saved");
    } catch (err) { setupMsg.textContent = err.message; }
  });
  $("#btn-site").addEventListener("click", (e) => { e.preventDefault(); if (state) showSetup(state.site); });
  $("#btn-history").addEventListener("click", (e) => {
    e.preventDefault();
    $("#history-head").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  // ------------------------------------------------------------ camera panels
  function createPanel(cam) {
    const panel = tpl.content.firstElementChild.cloneNode(true);
    panel.dataset.cid = cam.cid;
    const cid = cam.cid;
    const img = $(".feed", panel);
    img.src = `/stream/${cid}?t=${Date.now()}`;
    img.addEventListener("error", () => setTimeout(() => { if (panels.has(cid)) img.src = `/stream/${cid}?t=${Date.now()}`; }, 2000));

    // ---- trigger line calibration (click two points, or drag the handles of the existing line) ----
    $(".panel-body", panel).addEventListener("click", (ev) => {
      if (!panel.classList.contains("calibrating") || ev.target.closest(".tools, .source-form, .handle")) return;
      const box = contentBox(panel);
      if (!box) return;
      const nx = (ev.clientX - box.left) / box.width, ny = (ev.clientY - box.top) / box.height;
      if (nx < 0 || nx > 1 || ny < 0 || ny > 1) return;
      const d = drafts.get(cid);
      if (!d || d.dragged) { if (d) d.dragged = false; return; }
      if (d.points.length >= 2) d.points = [];
      d.points.push([clamp01(nx), clamp01(ny)]);
      if (d.mode === "region" && d.points.length === 2) saveRegion(panel, cid, d);
      renderDraft(panel);
    });
    // The tools row sits ON the picture, so the bottom strip of the frame cannot be
    // clicked - and a click that lands there fires whatever button is under it. Press
    // and drag instead: only the press has to start on the picture, so the box can be
    // pulled right out to any edge. Clicking two corners still works.
    $(".panel-body", panel).addEventListener("pointerdown", (ev) => {
      if (!panel.classList.contains("calibrating") || ev.target.closest(".tools, .source-form, .handle")) return;
      const d = drafts.get(cid);
      if (!d || d.mode !== "region") return;
      const box = contentBox(panel);
      if (!box) return;
      const at = (e) => [clamp01((e.clientX - box.left) / box.width),
                         clamp01((e.clientY - box.top) / box.height)];
      const before = d.points.slice(), start = at(ev);
      ev.preventDefault();
      d.points = [start];
      renderDraft(panel);
      const move = (e) => { d.points = [start, at(e)]; renderDraft(panel); };
      const up = (e) => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
        const end = at(e);
        const w = Math.abs(end[0] - start[0]), h = Math.abs(end[1] - start[1]);
        if (w < 0.02 && h < 0.02) {       // a tap, not a drag - let the click flow have it
          d.points = before;
          renderDraft(panel);
          return;
        }
        d.dragged = true;                 // swallow the click that follows this pointerup
        if (w < MIN_REGION || h < MIN_REGION) {
          d.points = [];
          renderDraft(panel);
          toast(`that area is too small - drag a box at least ${Math.round(MIN_REGION * 100)} % of the frame across`);
          return;
        }
        d.points = [start, end];
        saveRegion(panel, cid, d);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    });
    $(".calibrate", panel).addEventListener("click", async () => {
      const cam = camOf(cid);
      // start from the saved line so its end points can simply be dragged
      drafts.set(cid, { mode: "line", points: cam && cam.line ? [cam.line.p1.slice(), cam.line.p2.slice()] : [], side: cam && cam.line ? cam.line.entry_side : 1, fromSaved: !!(cam && cam.line) });
      panel.classList.add("calibrating");
      await tryApi("POST", `/api/cameras/${cid}/editing`, { on: true });
      renderDraft(panel);
    });
    $(".done", panel).addEventListener("click", async () => {
      drafts.delete(cid);
      panel.classList.remove("calibrating");
      renderDraft(panel);
      await tryApi("POST", `/api/cameras/${cid}/editing`, { on: false });
    });
    $(".clear", panel).addEventListener("click", () => { const d = drafts.get(cid); if (d) { d.points = []; d.fromSaved = false; renderDraft(panel); } });
    $(".region", panel).addEventListener("click", () => {
      const d = drafts.get(cid);
      if (!d) return;
      d.mode = d.mode === "region" ? "line" : "region";
      d.points = []; d.fromSaved = false;
      if (d.mode === "region") {
        const c = camOf(cid);
        d.role = (c && c.region_role) || d.role || "count_inside";
        $(".region-role", panel).value = d.role;
      }
      renderDraft(panel);
    });
    $(".region-role", panel).addEventListener("change", (ev) => {
      const d = drafts.get(cid);
      if (!d) return;
      d.role = ev.target.value;
      renderDraft(panel);                    // recolour and re-caption before the corners go down
    });
    $(".del-region", panel).addEventListener("click", async () => {
      await tryApi("DELETE", `/api/cameras/${cid}/region`);
      const d = drafts.get(cid); if (d) { d.mode = "line"; d.points = []; }
      renderDraft(panel);
      toast("area cleared - the whole frame counts and classifies again");
    });
    $(".flip", panel).addEventListener("click", async () => {
      const d = drafts.get(cid);
      if (!d) return;
      d.side = -d.side;
      renderDraft(panel);
      if (d.points.length < 2) { const r = await tryApi("POST", `/api/cameras/${cid}/flip`); d.side = r.entry_side; toast("entry side flipped and saved"); }
    });
    $(".save", panel).addEventListener("click", async () => {
      const d = drafts.get(cid);
      if (!d || d.points.length !== 2) return;
      const r = await tryApi("POST", `/api/cameras/${cid}/line`, { p1: d.points[0], p2: d.points[1], entry_side: d.side });
      d.fromSaved = true;
      renderDraft(panel);
      toast(r.warning ? `saved, but ${r.warning}` : "trigger line saved to config/lines.json",
            r.warning ? 9000 : undefined);
    });
    $(".del", panel).addEventListener("click", async () => {
      await tryApi("DELETE", `/api/cameras/${cid}/line`);
      const d = drafts.get(cid); if (d) { d.points = []; d.fromSaved = false; }
      renderDraft(panel);
      toast("trigger line deleted");
    });

    // ---- recording ----
    $(".swap", panel).addEventListener("click", () => swapFeed(cid));
    $(".record", panel).addEventListener("click", () => {
      const cam = camOf(cid);
      openModal(`Record event - ${cam ? cam.name : "camera"}`,
        `<label>Note <input name="note" type="text" placeholder="what happened (optional)"></label>
         <div class="hint">Saves the last 10 s and the next 20 s of this camera as a clip plus a snapshot to output/events/, and logs the event in history.csv.</div>`,
        async (d) => { await api("POST", `/api/cameras/${cid}/record`, { note: d.note || "" }); toast("recording 30 s clip…"); }, "record");
    });

    // ---- source editor ----
    const form = $(".source-form", panel), src = $(".sf-src", panel), name = $(".sf-name", panel), area = $(".sf-area", panel), msg = $(".sf-msg", panel);
    const open = () => {
      const cam = camOf(cid);
      // never echo a masked URL back into the field: the operator re-enters it in full
      src.value = cam && cam.kind !== "none" && !cam.src.includes("***") ? cam.src : "";
      src.placeholder = cam && cam.kind !== "none" ? `current: ${cam.src}` : "rtsp://user:pass@host:554/stream  or  path/to/video.mp4";
      name.value = (cam && cam.label) || "";
      area.value = cam && cam.area_capacity ? cam.area_capacity : "";
      msg.textContent = ""; msg.className = "sf-msg";
      $(".remove", panel).style.display = cam && cam.kind !== "none" ? "" : "none";
      $(".remove-slot", panel).style.display = state && state.cameras.length > 1 ? "" : "none";
      panel.classList.add("editing");
      src.focus();
    };
    const close = () => panel.classList.remove("editing");
    $(".source", panel).addEventListener("click", open);
    $(".add-camera", panel).addEventListener("click", open);
    $(".cancel", panel).addEventListener("click", close);
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const v = src.value.trim();
      if (!v) { msg.textContent = "enter an RTSP URL or a video file path"; msg.className = "sf-msg err"; return; }
      msg.textContent = "connecting…"; msg.className = "sf-msg";
      try {
        await api("POST", `/api/cameras/${cid}/source`, { src: v, name: name.value.trim(), area_capacity: Number(area.value) || 0 });
        close();
        toast(v.startsWith("rtsp") || v.startsWith("http") ? "camera set - connecting to the stream" : "video source set");
      } catch (err) { msg.textContent = err.message; msg.className = "sf-msg err"; }
    });
    $(".remove", panel).addEventListener("click", async () => { await tryApi("DELETE", `/api/cameras/${cid}/source`); close(); toast("camera removed"); });
    $(".remove-slot", panel).addEventListener("click", async () => { await tryApi("DELETE", `/api/cameras/${cid}`); close(); toast("camera panel removed"); });

    $(".expand", panel).addEventListener("click", () => toggleExpand(panel));
    return panel;
  }

  function camOf(cid) { return state ? state.cameras.find((c) => c.cid === cid) : null; }
  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  function syncPanels(cams) {
    const wanted = cams.map((c) => c.cid);
    for (const [cid, el] of panels) if (!wanted.includes(cid)) { el.remove(); panels.delete(cid); drafts.delete(cid); }
    let prev = $("#console");
    for (const cam of cams) {
      let el = panels.get(cam.cid);
      if (!el) { el = createPanel(cam); panels.set(cam.cid, el); }
      if (prev.nextElementSibling !== el) prev.after(el);
      prev = el;
    }
    const tiles = cams.length + 1;
    const cols = tiles <= 4 ? 2 : tiles <= 6 ? 3 : 4;
    grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    // Once the grid is 3 or 4 wide the console would get a third or a quarter of
    // the width - about 414px at six cameras - and its three metric columns
    // collapse. Let it span two cells so it keeps a usable width while the
    // cameras share the rest.
    grid.classList.toggle("wide-console", cols >= 3);
    // four or more cameras: the location cards pair up rather than stacking, or
    // the column grows taller than the console tile and clips the other two
    grid.classList.toggle("many-cams", cams.length >= 4);
    // At five the grid turns 3x3 = 9 cells for 6 tiles, so a 2x2 console uses the
    // slack instead of leaving a hole, and gets the height the alert list needs.
    grid.classList.toggle("tall-console", cams.length >= 5);
    $("#btn-add-camera").disabled = state && cams.length >= state.max_cameras;
  }

  function toggleExpand(panel) {
    const on = !panel.classList.contains("expanded");
    $$(".panel").forEach((p) => p.classList.remove("expanded"));
    grid.classList.toggle("has-expanded", on);
    if (on) panel.classList.add("expanded");
    $$(".expand").forEach((b) => { b.textContent = "⤢"; });
    if (on) $(".expand", panel).textContent = "⤡";
    panels.forEach((p) => renderDraft(p));
  }
  $(".expand", $("#console")).addEventListener("click", () => toggleExpand($("#console")));
  $("#btn-add-camera").addEventListener("click", async () => { await tryApi("POST", "/api/cameras"); toast("camera panel added"); });

  /* Rectangle (page coords) of the visible image content inside a panel body (object-fit: contain). */
  function contentBox(panel) {
    const img = $(".feed", panel);
    const cam = camOf(Number(panel.dataset.cid));
    const r = img.getBoundingClientRect();
    let nw = img.naturalWidth, nh = img.naturalHeight;
    if (!(nw && nh) && cam && cam.w && cam.h) { nw = cam.w; nh = cam.h; }
    if (!(nw && nh) || !r.width || !r.height) return null;
    const scale = Math.min(r.width / nw, r.height / nh);
    const w = nw * scale, h = nh * scale;
    return { left: r.left + (r.width - w) / 2, top: r.top + (r.height - h) / 2, width: w, height: h };
  }

  // counter.py bounds a crossing to the drawn segment plus this fraction of its
  // length beyond each end (LineCounter.segment_tolerance). Drawing it is the
  // difference between "the line counts further than it looks" and seeing why.
  const SEGMENT_TOLERANCE = 0.08;

  // What a drawn area is for. Only count_inside constrains the trigger line:
  // the other two leave tracking alone, so the line may sit outside the box.
  const ROLES = {
    count_inside:    { colour: "#ffb020", title: "COUNTING AREA",
                       help: "only people inside it are counted - keep the line inside" },
    ignore_inside:   { colour: "#eb5050", title: "IGNORED AREA",
                       help: "people inside it are skipped (a window, a mirror, a poster)" },
    classify_inside: { colour: "#3cb4eb", title: "CLASSIFY HERE",
                       help: "everyone is counted; only those inside are sexed and aged" },
  };
  const roleOf = (d) => ROLES[d && d.role] || ROLES.count_inside;
  const MIN_REGION = 0.05;              // matches the server's "region is too small to be useful"

  function saveRegion(panel, cid, d) {
    const [[ax, ay], [bx, by]] = d.points;
    return tryApi("POST", `/api/cameras/${cid}/region`,
                  { x1: ax, y1: ay, x2: bx, y2: by, role: d.role || "count_inside" })
      .then((r) => { toast(r.warning ? `area saved, but ${r.warning}`
                                     : `${roleOf(d).title.toLowerCase()} saved`,
                           r.warning ? 9000 : undefined);
                     d.mode = "line"; d.points = []; renderDraft(panel); })
      .catch(() => { d.points = []; renderDraft(panel); });
  }

  function renderDraft(panel) {
    const cid = Number(panel.dataset.cid);
    const svg = $(".draft", panel);
    const d = drafts.get(cid);
    const box = contentBox(panel);
    const body = $(".panel-body", panel).getBoundingClientRect();
    if (!d || !box) { svg.innerHTML = ""; return; }
    Object.assign(svg.style, { left: `${box.left - body.left}px`, top: `${box.top - body.top}px`, width: `${box.width}px`, height: `${box.height}px`, inset: "auto" });
    svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
    const P = d.points.map(([x, y]) => [x * box.width, y * box.height]);
    const col = d.mode === "region" ? roleOf(d).colour
                                    : (d.fromSaved ? "#3cdc3c" : "#00c8ff");
    let caption = [`click ${2 - P.length} point${P.length === 1 ? "" : "s"} on the picture`];
    if (d.mode === "region") {
      caption = [`${roleOf(d).title} - click 2 opposite corners`, roleOf(d).help];
    }
    // the panel is usually wider than the picture, so the image sits letterboxed
    // inside it; mark where it actually ends while a line is being placed
    let s = `<rect x="0.75" y="0.75" width="${box.width - 1.5}" height="${box.height - 1.5}" `
          + `fill="none" stroke="rgba(255,255,255,.38)" stroke-width="1.5" stroke-dasharray="7 6"/>`;
    if (d.mode === "region") {
      const rc = col;
      if (P.length === 1) {
        s += `<circle cx="${P[0][0]}" cy="${P[0][1]}" r="6" fill="${rc}"/>`;
      } else if (P.length === 2) {
        const rx = Math.min(P[0][0], P[1][0]), ry = Math.min(P[0][1], P[1][1]);
        const rw = Math.abs(P[1][0] - P[0][0]), rh = Math.abs(P[1][1] - P[0][1]);
        if (d.role === "ignore_inside") {
          s += `<rect x="${rx}" y="${ry}" width="${rw}" height="${rh}" fill="rgba(0,0,0,.45)"/>`;
        } else if ((d.role || "count_inside") === "count_inside") {
          s += `<path d="M0,0 H${box.width} V${box.height} H0 Z M${rx},${ry} h${rw} v${rh} h${-rw} Z" `
             + `fill="rgba(0,0,0,.45)" fill-rule="evenodd"/>`;
        }
        s += `<rect x="${rx}" y="${ry}" width="${rw}" height="${rh}" fill="none" stroke="${rc}" `
           + `stroke-width="2.5" stroke-dasharray="8 6"/>`;
      }
    } else if (P.length === 2) {
      const [[x1, y1], [x2, y2]] = P;
      // a crossing still counts this far past each end, so show it rather than
      // leave it as a surprise in the numbers
      const ux = (x2 - x1), uy = (y2 - y1), ulen = Math.hypot(ux, uy) || 1;
      const ex = (ux / ulen) * ulen * SEGMENT_TOLERANCE, ey = (uy / ulen) * ulen * SEGMENT_TOLERANCE;
      s += `<line x1="${x1 - ex}" y1="${y1 - ey}" x2="${x2 + ex}" y2="${y2 + ey}" `
         + `stroke="${col}" stroke-width="2" stroke-opacity=".45" stroke-dasharray="5 5"/>`;
      for (const [hx, hy] of [[x1 - ex, y1 - ey], [x2 + ex, y2 + ey]]) {
        s += `<line x1="${hx - uy / ulen * 7}" y1="${hy + ux / ulen * 7}" `
           + `x2="${hx + uy / ulen * 7}" y2="${hy - ux / ulen * 7}" `
           + `stroke="${col}" stroke-width="2" stroke-opacity=".45"/>`;
      }
      s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#000" stroke-width="6" stroke-linecap="round"/>`;
      s += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="3.5" stroke-linecap="round"/>`;
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, dx = x2 - x1, dy = y2 - y1;
      const n = Math.hypot(dx, dy) || 1, nx = -dy / n, ny = dx / n;   // +1 side of the counter's cross product
      const L = Math.max(26, Math.min(box.width, box.height) * 0.09), sg = d.side >= 0 ? 1 : -1;
      const tx = mx + sg * nx * L, ty = my + sg * ny * L;
      s += `<line x1="${mx}" y1="${my}" x2="${tx}" y2="${ty}" stroke="${col}" stroke-width="2.5"/>`;
      s += `<circle cx="${tx}" cy="${ty}" r="5" fill="${col}"/>`;
      s += `<text x="${tx + 9}" y="${ty + 5}" fill="${col}" font-size="15" font-weight="700" font-family="Ubuntu, sans-serif">IN</text>`;
      // The caption used to sit at the top-left of the line's bounding box, which
      // is the line's own start point whenever it runs down-right - so it landed
      // on top of the line. Pin it to a corner instead, on a plate so it stays
      // readable over any frame, and it can never collide with the line.
      caption = [d.fromSaved ? "drag the end points, then save" : "NEW LINE - press save",
                 `dashed = still counts (${Math.round(SEGMENT_TOLERANCE * 100)} % past each end)`];
    }
    P.forEach(([x, y], i) => {
      s += `<circle class="handle" data-i="${i}" cx="${x}" cy="${y}" r="11" fill="rgba(0,0,0,.35)" stroke="${col}" stroke-width="2.5"/><circle cx="${x}" cy="${y}" r="3" fill="${col}"/>`;
    });
    // caption plate, last so nothing is drawn over it
    {
      const pad = 9, lh = 17, fs = [13, 11];
      const wch = Math.max(...caption.map((c, i) => c.length * fs[i] * 0.53));
      const bw = Math.min(box.width - 20, wch + pad * 2), bh = pad * 2 + lh * caption.length - 4;
      s += `<rect x="10" y="10" rx="4" width="${bw}" height="${bh}" fill="rgba(8,10,14,.62)" stroke="${col}" stroke-opacity=".45"/>`;
      caption.forEach((c, i) => {
        s += `<text x="${10 + pad}" y="${10 + pad + 12 + i * lh}" fill="${col}" `
           + `fill-opacity="${i ? .72 : 1}" font-size="${fs[i]}" font-family="Ubuntu, sans-serif">${c}</text>`;
      });
    }
    svg.innerHTML = s;
    $$(".handle", svg).forEach((h) => h.addEventListener("pointerdown", (ev) => startDrag(ev, panel, Number(h.dataset.i))));
    $(".save", panel).disabled = d.mode === "region" || P.length !== 2;
    const rolePick = $(".region-role", panel);
    rolePick.classList.toggle("hidden", d.mode !== "region");
    rolePick.dataset.role = d.role || "count_inside";
    // These sit on the drawing surface and none of them belongs to the area flow.
    // Leaving them live means a corner clicked near the bottom of the picture
    // deletes the trigger line or swaps the feed instead of placing the corner.
    for (const sel of [".flip", ".clear", ".del", ".swap", ".record"]) {
      const el = $(sel, panel);
      if (el) el.disabled = d.mode === "region";
    }
    $(".hint", panel).textContent = d.mode === "region"
      ? `${roleOf(d).title.toLowerCase()} - click ${2 - P.length} corner${P.length === 1 ? "" : "s"}`
      : (P.length === 2 ? (d.fromSaved ? "line loaded" : "line ready") : `click ${2 - P.length} point${P.length === 1 ? "" : "s"}`);
  }

  function startDrag(ev, panel, i) {
    ev.preventDefault(); ev.stopPropagation();
    const cid = Number(panel.dataset.cid);
    const d = drafts.get(cid);
    if (!d) return;
    const move = (e) => {
      const box = contentBox(panel);
      if (!box) return;
      d.points[i] = [clamp01((e.clientX - box.left) / box.width), clamp01((e.clientY - box.top) / box.height)];
      d.dragged = true; d.fromSaved = false;
      renderDraft(panel);
    };
    const up = () => { window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); setTimeout(() => { if (d) d.dragged = false; }, 50); };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }
  window.addEventListener("resize", () => panels.forEach((p) => renderDraft(p)));

  // ------------------------------------------------------------ console
  const el = {
    siteName: $("#site-name"), session: $("#session"), mode: $("#mode"),
    banner: $("#alarm-banner"), abTitle: $("#ab-title"), abDetail: $("#ab-detail"),
    occLevel: $("#occ-level"), occMeta: $("#occ-meta"),
    oMale: $("#o-male"), oFemale: $("#o-female"), oAdult: $("#o-adult"), oChild: $("#o-child"),
    obarMale: $("#obar-male"), obarFemale: $("#obar-female"), obarAdult: $("#obar-adult"), obarChild: $("#obar-child"),
    btnHistory: $("#btn-history"),
    alerts: $("#alerts"), alertsSub: $("#alerts-sub"),
    visLabel: $("#vis-label"),
    events: $("#events tbody"), eventsSub: $("#events-sub"),
    cams: $("#cams tbody"), camsSub: $("#cams-sub"),
    histLast: $("#hist-last"), histToday: $("#hist-today tbody"), histDays: $("#hist-days tbody"), histEvents: $("#hist-events tbody"),
    pause: $("#btn-pause"), foot: $("#foot-info"),
    vTotal: $("#v-total"), vMale: $("#v-male"), vFemale: $("#v-female"), vAdult: $("#v-adult"), vChild: $("#v-child"),
    vbarMale: $("#vbar-male"), vbarFemale: $("#vbar-female"), vbarAdult: $("#vbar-adult"), vbarChild: $("#vbar-child"), visSub: $("#vis-sub"),
    locRow: $("#loc-row"), locSub: $("#loc-sub"), occLabel: $("#occ-label"), occSub: $("#occ-sub"), oTotal: $("#o-total"),
    assumedNote: $("#assumed-note"),
  };

  // which camera the metrics row describes: null = the whole site
  let selectedCam = null;


  $("#ack-all").addEventListener("click", async (e) => { e.preventDefault(); await tryApi("POST", "/api/alerts/ack", {}); });
  $("#ab-ack").addEventListener("click", async (e) => { e.preventDefault(); await tryApi("POST", "/api/alerts/ack", { id: state.occupancy.level === "over" ? "capacity_over" : "capacity_crowded" }); });
  el.alerts.addEventListener("click", async (e) => {
    const a = e.target.closest("a[data-ack]");
    if (!a) return;
    e.preventDefault();
    await tryApi("POST", "/api/alerts/ack", { id: a.dataset.ack });
  });
  el.pause.addEventListener("click", async (e) => { e.preventDefault(); await tryApi("POST", "/api/pause", { paused: !state.paused }); });
  $("#btn-reset").addEventListener("click", (e) => {
    e.preventDefault();
    openModal("Reset counters", `<div class="hint">Restart the count from the starting occupancy (${state.site.baseline}) and clear the per-camera counts. Logged in history.csv.</div>`,
      async () => { await api("POST", "/api/reset"); toast("counters reset"); }, "reset");
  });

  function dropStaleSelection(st) {
    // a focused camera that has been removed must not keep the panel filtered
    if (selectedCam !== null && !st.cameras.some((c) => c.kind !== "none" && c.cid === selectedCam)) {
      selectedCam = null;
    }
  }
  $("#loc-row").addEventListener("click", (e) => {
    const card = e.target.closest(".loc-card");
    if (!card || !state) return;
    const cid = Number(card.dataset.cid);
    selectedCam = selectedCam === cid ? null : cid;      // click again to go back to the whole site
    render(state);
  });

  // ---------------------------------------------------------- feed swapping
  const swapping = new Set();                 // cids with a swap in flight
  async function swapFeed(cid, quiet = false) {
    const cam = camOf(cid);
    if (!cam || !cam.alt_src) {
      if (!quiet) toast("no paired feed for this camera — set one in config/sources.json", 4200);
      return false;
    }
    if (swapping.has(cid)) return false;      // ignore repeats while the POST is out
    swapping.add(cid);
    const btn = panels.get(cid) && $(".swap", panels.get(cid));
    if (btn) btn.disabled = true;
    try {
      const r = await tryApi("POST", `/api/cameras/${cid}/swap`);
      if (!quiet) toast(`${r.camera.name} now showing ${r.camera.kind === "live" ? "the live feed" : "the video file"}`);
      return true;
    } catch (e) {
      return false;
    } finally {
      swapping.delete(cid);
      if (btn) btn.disabled = false;
    }
  }

  // ---------------------------------------------------------- keyboard
  // v        swap the panel under the pointer between its file and its live feed
  // shift+V  swap every panel that has a paired feed
  let ptr = { x: -1, y: -1 };
  document.addEventListener("mousemove", (e) => { ptr = { x: e.clientX, y: e.clientY }; }, { passive: true });

  function typingTarget(el) {
    if (!el) return false;
    const tag = el.tagName;
    return el.isContentEditable || tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
  }
  function overlayOpen() {
    return !$("#setup").classList.contains("hidden") || !$("#modal").classList.contains("hidden");
  }
  function panelUnderPointer() {
    if (ptr.x < 0) return null;
    const el = document.elementFromPoint(ptr.x, ptr.y);
    const p = el && el.closest(".panel.cam");
    return p && p.dataset.cid ? Number(p.dataset.cid) : null;
  }

  document.addEventListener("keydown", async (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;         // leave browser shortcuts alone
    if (typingTarget(e.target) || overlayOpen()) return;    // never while typing or in a dialog
    if (e.key !== "v" && e.key !== "V") return;
    e.preventDefault();
    if (e.shiftKey) {
      const live = (state ? state.cameras : []).filter((c) => c.alt_src);
      if (!live.length) { toast("no camera has a paired feed to swap to", 4200); return; }
      await Promise.all(live.map((c) => swapFeed(c.cid, true)));
      toast(`swapped ${live.length} camera${live.length === 1 ? "" : "s"}`);
      return;
    }
    const cid = panelUnderPointer();
    if (cid === null) { toast("point at a camera panel, then press v  (shift+V swaps all)", 4200); return; }
    await swapFeed(cid);
  });

  function render(st) {
    state = st;
    if (st.history) history = st.history;
    if (!setupShown && st.site && !st.site.configured_today) showSetup(st.site);

    el.siteName.textContent = st.site && st.site.name ? `${st.site.name} - Crowd Console` : "Crowd Console";
    el.session.textContent = st.session;
    el.mode.textContent = { live: "LIVE CCTV", video: "VIDEO PLAYBACK", empty: "NO CAMERAS" }[st.mode] || st.mode;
    el.mode.className = `mode-badge ${st.mode}`;

    syncPanels(st.cameras);
    const o = st.occupancy;
    const siteOver = o.level === "over";

    // camera panels
    for (const cam of st.cameras) {
      const panel = panels.get(cam.cid);
      const dot = $(".dot", panel), fps = $(".fps", panel), hud = $(".hud", panel), title = $(".title", panel), ribbon = $(".ribbon", panel), areaPill = $(".pill.area", panel);
      panel.classList.toggle("recording", !!cam.recording);
      if (cam.kind === "none") {
        panel.classList.add("empty");
        panel.classList.remove("alarm", "tipped", "area-alarm");
        title.textContent = `camera_${String(cam.idx + 1).padStart(2, "0")}`;
        title.title = "";
        dot.className = "dot"; fps.textContent = "no camera"; hud.innerHTML = "";
        ribbon.classList.add("hidden"); areaPill.classList.add("hidden");
        $(".calibrate", panel).disabled = true;
        $(".swap", panel).classList.add("hidden");
        continue;
      }
      panel.classList.remove("empty");
      title.textContent = `camera_${String(cam.idx + 1).padStart(2, "0")} · ${cam.name}`;
      title.title = cam.src;
      const idle = cam.ok && cam.fps < 0.5;
      dot.className = `dot ${cam.ok ? (idle ? "idle" : "ok") : "bad"}`;
      fps.textContent = cam.ok ? (idle ? "idle" : `${cam.fps.toFixed(1)} fps`) : cam.status.toLowerCase();
      const c = cam.counts;
      const net = c.entries - c.exits;
      // Who is in shot right now, split M/F and A/C. Nothing to do with the line:
      // with a counting area drawn this is the population of the box, so it reads
      // correctly from the first frame instead of starting at zero.
      const pr = cam.present || {n: 0};
      const hereLabel = cam.region_role === "count_inside" ? "in area now" : "in view now";
      const hereLine =
        `<div class="hud-line hud-here"><span class="hud-k">${hereLabel}</span><b class="hud-mid">${pr.n || 0}</b>` +
        `<span class="hud-sub">M <b class="male">${pr.male || 0}</b> F <b class="female">${pr.female || 0}</b>` +
        ` · A <b class="adult">${pr.adult || 0}</b> C <b class="child">${pr.child || 0}</b>` +
        `${pr.gender_unknown || pr.age_unknown ? ` · <span class="unk">${Math.max(pr.gender_unknown || 0, pr.age_unknown || 0)} ?</span>` : ""}` +
        `</span></div>`;
      hud.innerHTML = (cam.has_line
        ? `<div class="hud-line"><span class="hud-k">net visitors</span><b class="hud-big">${net > 0 ? "+" : ""}${net}</b>` +
          `<span class="hud-sub">M <b class="male">${c.n_male}</b> F <b class="female">${c.n_female}</b>` +
          ` · A <b class="adult">${c.n_adult}</b> C <b class="child">${c.n_child}</b></span></div>` +
          `<div class="hud-line hud-flow"><span class="in">${c.entries} entered</span> · <span class="out">${c.exits} left</span>` +
          `${cam.area_capacity ? ` · <span class="${cam.area_over ? "warn" : ""}">area ${Math.max(0, net)}/${cam.area_capacity}</span>` : ""}</div>`
        : `<span class="warn">no trigger line - press calibrate</span>`) + hereLine;
      $(".calibrate", panel).disabled = !(cam.w && cam.h);
      const hasRegion = !!cam.region;
      $(".del-region", panel).classList.toggle("hidden", !hasRegion);
      $(".region", panel).textContent = hasRegion ? "redraw area" : "draw area";
      $(".region", panel).title = hasRegion
        ? `area set: ${(ROLES[cam.region_role] || ROLES.count_inside).help}`
        : "limit where this camera counts or classifies people";
      const swap = $(".swap", panel);
      swap.classList.toggle("hidden", !cam.alt_src);
      if (cam.alt_src) {
        swap.textContent = cam.alt_kind === "live" ? "▸ live" : "▸ video";
        swap.title = `switch this panel to ${cam.alt_src}   (or hover and press v)`;
      }
      areaPill.classList.toggle("hidden", !cam.area_capacity);
      areaPill.textContent = cam.area_capacity ? `area ${Math.max(0, c.visitors)} / ${cam.area_capacity}` : "";
      areaPill.classList.toggle("over", !!cam.area_over);
      // alarm decoration
      panel.classList.toggle("alarm", siteOver || cam.area_over);
      panel.classList.toggle("tipped", !!cam.tipped && siteOver);
      panel.classList.toggle("area-alarm", !!cam.area_over);
      // the alarm text is drawn into the frame by the server (so clips carry it); the HTML ribbon only covers a feed without frames
      if (cam.area_over && !cam.ok) { ribbon.textContent = `AREA OVER CAPACITY  ${Math.max(0, c.visitors)} / ${cam.area_capacity}`; ribbon.classList.remove("hidden"); }
      else if (siteOver && !cam.ok) { ribbon.textContent = `SITE OVER CAPACITY  ${o.current} / ${o.capacity}${cam.tipped ? "  -  TRIGGERED HERE" : ""}`; ribbon.classList.remove("hidden"); }
      else ribbon.classList.add("hidden");
      const d = drafts.get(cam.cid);
      if (d && d.points.length < 2 && cam.line) d.side = cam.line.entry_side;
    }

    // alarm banner
    if (siteOver || o.level === "crowded") {
      el.banner.classList.remove("hidden");
      el.banner.classList.toggle("warn", !siteOver);
      el.abTitle.textContent = siteOver ? "OVER CAPACITY" : "APPROACHING CAPACITY";
      const where = o.over && o.over.cam_name ? ` · triggered at ${o.over.cam_name} ${fmtTime(o.over.since)}` : "";
      el.abDetail.textContent = `${o.current} inside · limit ${o.capacity} (${o.pct} %)${where}`;
    } else el.banner.classList.add("hidden");

    // occupancy / net flow - the metrics row describes the whole site or one camera
    dropStaleSelection(st);
    const cam = selectedCam === null ? null : st.cameras.find((c) => c.cid === selectedCam);
    const m = cam ? cam.counts : st.totals;                 // the counts the metrics row reads
    const net = cam ? m.entries - m.exits : o.current;      // one door's net flow can be negative
    const baseline = cam ? 0 : st.totals.initial;

    el.occLabel.textContent = cam ? "NET FLOW" : "CURRENT OCCUPANCY";
    el.occSub.textContent = cam
      ? `through ${cam.name} — entries minus exits at this door`
      : "current number of visitors in the zone/area";
    // the headline figure and the percentage were removed from this column; the
    // level badge moved up beside the heading and is the only status left here
    el.occLevel.style.display = cam ? "none" : "";
    el.occLevel.textContent = LEVEL_TEXT[o.level];
    el.occLevel.className = `occ-level ${o.level}`;
    el.occMeta.textContent = cam
      ? `${cam.status}${cam.area_capacity ? ` · area capacity ${cam.area_capacity}` : ""} · ${m.entries} in · ${m.exits} out`
      : `capacity ${o.capacity} · warning at ${o.warn_pct} % · ${st.site.open}–${st.site.close}${st.site.is_open ? "" : " (closed)"}`
        + ` · ${m.entries} in · ${m.exits} out`;
    const raw = baseline + m.entries - m.exits;             // before the site count is clamped at zero
    // the occupancy figure, which is also what the two tiles below sum to
    el.oTotal.textContent = m.in_male + m.in_female;
    // OCCUPANCY: who is inside right now. in_* are the occupancy figure shared out
    // by the measured mix, so the parts always add up to the number above them.
    el.oMale.textContent = m.in_male; el.oFemale.textContent = m.in_female;
    el.oAdult.textContent = m.in_adult; el.oChild.textContent = m.in_child;
    const og = m.in_male + m.in_female, oa = m.in_adult + m.in_child;
    el.obarMale.style.width = `${og ? (m.in_male / og) * 100 : 50}%`; el.obarFemale.style.width = `${og ? (m.in_female / og) * 100 : 50}%`;
    el.obarAdult.style.width = `${oa ? (m.in_adult / oa) * 100 : 50}%`; el.obarChild.style.width = `${oa ? (m.in_child / oa) * 100 : 50}%`;

    // VISITORS TODAY: cumulative entries, never decreasing. Someone who came in
    // and left again still visited today, so this
    // is entries and its v_* split - not the net figure, which is OCCUPANCY.
    const v = m;
    el.visLabel.textContent = "VISITORS TODAY";
    el.visSub.textContent = cam ? `entered at ${cam.name}` : "total number of visitors for the day";
    el.vTotal.textContent = v.entries;
    el.vMale.textContent = v.v_male; el.vFemale.textContent = v.v_female;
    el.vAdult.textContent = v.v_adult; el.vChild.textContent = v.v_child;
    const vg = v.v_male + v.v_female, va = v.v_adult + v.v_child;
    el.vbarMale.style.width = `${vg ? (v.v_male / vg) * 100 : 50}%`; el.vbarFemale.style.width = `${vg ? (v.v_female / vg) * 100 : 50}%`;
    el.vbarAdult.style.width = `${va ? (v.v_adult / va) * 100 : 50}%`; el.vbarChild.style.width = `${va ? (v.v_child / va) * 100 : 50}%`;

    // alerts
    const active = st.alerts.filter((a) => !a.acked).length;
    el.alertsSub.textContent = `${active} active`;
    el.alerts.innerHTML = st.alerts.length
      ? st.alerts.map((a) => `<li class="${a.acked ? "acked" : ""}"><div class="t ${a.level}">${esc(a.title)}</div><div class="d">${esc(a.detail)} · ${fmtTime(a.since)}${a.acked ? "" : ` <a href="#" class="link" data-ack="${esc(a.id)}">ack</a>`}</div></li>`).join("")
      : `<li class="empty">no alerts</li>`;

    // the assumed opening count describes the OCCUPANCY number; no gender/age figure carries it any more
    const assumed = cam ? 0 : Math.min(baseline, Math.max(0, net));
    if (!cam && raw < 0) {
      // more people were seen leaving than entering: they were inside before
      // counting started, which is what the opening baseline is for
      el.assumedNote.classList.remove("hidden");
      el.assumedNote.classList.add("warn");
      el.assumedNote.textContent =
        `${m.exits} left but only ${m.entries} entered, so at least ${-raw} ${-raw === 1 ? "person was" : "people were"} already inside before counting started`;
    } else if (assumed > 0) {
      const share = net > 0 ? assumed / net : 0;
      el.assumedNote.classList.remove("hidden");
      el.assumedNote.classList.toggle("warn", share > 0.5);
      el.assumedNote.textContent = share >= 1
        ? `all ${assumed} were assumed at opening — none of them was detected entering`
        : `${assumed} of ${net} inside were assumed at opening (${Math.round(share * 100)} %) — not detected`;
    } else el.assumedNote.classList.add("hidden");

    // cameras table
    const live = st.cameras.filter((c) => c.kind !== "none");
    el.camsSub.textContent = `${live.filter((c) => c.ok).length} of ${live.length} online · ${st.cameras.length} panel${st.cameras.length === 1 ? "" : "s"}`;
    el.cams.innerHTML = live.map((c) => {
      const k = c.counts;
      const cls = c.ok ? "ok" : "bad";
      return `<tr class="${c.area_over || (siteOver && c.tipped) ? "alarm-row" : ""}"><td>${esc(c.name)}${c.tipped && siteOver ? " ⚠" : ""}</td><td class="${cls}">${esc(c.status)}</td><td class="mono">${c.fps.toFixed(1)}</td><td class="num-cell">${k.entries} / ${k.exits}</td><td class="num-cell">${k.entries - k.exits}</td><td>${c.area_capacity ? `${c.area_capacity}${c.area_over ? " OVER" : ""}` : "–"}</td><td><span class="M">${k.male}</span> / <span class="F">${k.female}</span></td><td><span class="A">${k.adult}</span> / <span class="C">${k.child}</span></td><td class="${c.has_line ? "ok" : "warn"}">${c.has_line ? "set" : "missing"}</td></tr>`;
    }).join("") || `<tr class="empty"><td colspan="9">no cameras - press + camera or add one on a panel</td></tr>`;

    // by location - one card per camera, so the site total can be read back to its entrances
    const ico = (n, c) => `<svg class="ico ${c}"><use href="#i-${n}"/></svg>`;
    el.locSub.textContent = live.length ? `${live.length} camera${live.length === 1 ? "" : "s"}` : "no cameras yet";
    // the cards shrink as cameras are added, so up to --max-cameras of them still
    // fit the column without scrolling
    el.locRow.dataset.n = String(live.length);
    el.locRow.innerHTML = live.map((c) => {
      const k = c.counts, net = k.entries - k.exits;
      const alarm = c.area_over || (siteOver && c.tipped);
      const head = `<div class="loc-name" title="${esc(c.src)}">${esc(c.name)}</div>`;
      // A dropped stream keeps its last counts, which then sit there looking live.
      // Show nothing but the state; the reason stays in the tooltip for diagnosis.
      if (!c.ok) {
        return `<div class="loc-card offline${selectedCam === c.cid ? " on" : ""}" data-cid="${c.cid}">
          ${head}
          <div class="loc-offline" title="${esc(c.status)}">OFFLINE</div>
        </div>`;
      }
      return `<div class="loc-card${alarm ? " alarm" : ""}${selectedCam === c.cid ? " on" : ""}" data-cid="${c.cid}">
        ${head}
        <div class="loc-total"><span class="k">net</span><b class="total">${net > 0 ? "+" : ""}${net}</b></div>
        <div class="loc-split">
          <span>${ico("male", "male")}<b class="male">${k.n_male}</b><span class="sep">/</span><b class="female">${k.n_female}</b>${ico("female", "female")}</span>
          <span>${ico("adult", "adult")}<b class="adult">${k.n_adult}</b><span class="sep">/</span><b class="child">${k.n_child}</b>${ico("child", "child")}</span>
          ${c.area_capacity ? `<span class="loc-area${c.area_over ? " warn" : ""}">area ${Math.max(0, net)}/${c.area_capacity}</span>` : ""}
        </div>
      </div>`;
    }).join("") || `<div class="loc-empty">add a camera to see its numbers here</div>`;

    // events
    el.eventsSub.textContent = st.events.length ? `latest ${st.events.length} crossings` : "no crossings yet";
    el.events.innerHTML = st.events.length
      ? st.events.slice(0, 10).map((e) => `<tr><td class="mono">${fmtTime(e.t)}</td><td>${esc(e.cam_name)}</td><td class="${e.dir > 0 ? "entry" : "exit"}">${e.dir > 0 ? "ENTRY" : "EXIT"}</td><td class="${e.gender === "?" ? "q" : e.gender}">${GENDER[e.gender] || "?"}</td><td class="${e.age === "?" ? "q" : e.age}">${AGE[e.age] || "?"}</td><td class="mono q">#${e.track}</td></tr>`).join("")
      : `<tr class="empty"><td colspan="6">waiting for the first crossing…</td></tr>`;

    renderHistory(history, st.recordings);

    // footer
    el.pause.textContent = st.paused ? "resume" : "pause";
    el.foot.textContent = `${st.pipeline_fps} pipeline fps · ${st.device} · up ${fmtUptime(st.uptime)}${st.setup_mode ? " · CALIBRATING" : ""}`;
  }

  function renderHistory(h, recordings) {
    if (!h) return;
    const lc = h.last_closing;
    el.histLast.innerHTML = lc
      ? `<div><span class="k">last closing</span><span class="v small">${esc(lc.date)} ${esc(lc.timestamp.slice(11, 16))}</span></div>
         <div><span class="k">total visitors</span><span class="v total">${esc(lc.visitors || lc.entries)}</span></div>
         <div><span class="k">inside at close</span><span class="v">${esc(lc.occupancy)}</span></div>
         <div><span class="k">entries / exits</span><span class="v"><span class="in">${esc(lc.entries)}</span> / <span class="out">${esc(lc.exits)}</span></span></div>
         <div><span class="k">male / female</span><span class="v"><span class="M">${esc(lc.male)}</span> / <span class="F">${esc(lc.female)}</span></span></div>
         <div><span class="k">adult / child</span><span class="v"><span class="A">${esc(lc.adult)}</span> / <span class="C">${esc(lc.child)}</span></span></div>`
      : `<span class="empty">no closing totals recorded yet - the first closing row is written at ${state ? state.site.close : "closing time"}</span>`;
    const row = (r) => `<td class="num-cell">${esc(r.occupancy)}</td><td class="num-cell total">${esc(r.visitors || r.entries)}</td><td><span class="in">${esc(r.entries)}</span> / <span class="out">${esc(r.exits)}</span></td><td><span class="M">${esc(r.male)}</span> / <span class="F">${esc(r.female)}</span></td><td><span class="A">${esc(r.adult)}</span> / <span class="C">${esc(r.child)}</span></td>`;
    el.histToday.innerHTML = h.today.length
      ? h.today.slice().reverse().map((r) => `<tr><td class="mono">${esc(r.timestamp.slice(11, 16))}</td><td>${esc(r.kind)}${r.note ? ` <span class="q" title="${esc(r.note)}">·</span>` : ""}</td>${row(r)}</tr>`).join("")
      : `<tr class="empty"><td colspan="7">no rows yet today (hourly rows are written on the hour while open)</td></tr>`;
    el.histDays.innerHTML = h.days.length
      ? h.days.slice().reverse().map((r) => `<tr><td class="mono">${esc(r.date)}</td>${row(r)}</tr>`).join("")
      : `<tr class="empty"><td colspan="6">no previous days</td></tr>`;
    const recs = (recordings || []).filter((r) => !r.done).map((r) => `<tr><td class="mono">${fmtTime(r.started)}</td><td class="wrap">${esc(r.cam_name)} - ${esc(r.kind)}${r.note ? ": " + esc(r.note) : ""}</td><td></td><td class="bad">recording…</td></tr>`);
    const evs = h.events.slice().reverse().map((r) => `<tr><td class="mono">${esc(r.timestamp.slice(0, 16).replace("T", " "))}</td><td class="wrap">${esc(r.note)}</td><td class="num-cell">${esc(r.occupancy)}</td><td>${r.clip ? `<a class="link" href="/events/${encodeURIComponent(r.clip)}" target="_blank">${r.clip.endsWith(".mp4") ? "clip" : "snapshot"}</a>` : "–"}</td></tr>`);
    el.histEvents.innerHTML = recs.concat(evs).join("") || `<tr class="empty"><td colspan="4">no recorded events - use ● record on a camera, or they are saved automatically on an over-capacity alarm</td></tr>`;
  }

  // ------------------------------------------------------------ transport
  function connect() {
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
    ws.onmessage = (m) => render(JSON.parse(m.data));
    ws.onclose = () => { toast("connection lost - reconnecting"); setTimeout(connect, 1500); };
    ws.onerror = () => ws.close();
  }
  fetch("/api/state").then((r) => r.json()).then(render).catch(() => {});
  connect();
})();
