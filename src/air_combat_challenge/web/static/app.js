const form = document.querySelector("#run-form");
const startButton = form.querySelector(".launch");
const stopButton = document.querySelector("#stop-run");
const map = document.querySelector("#live-globe");
const unitHistory = new Map();
let websocket;
let replayTelemetry = null;
let liveMap = null;
const liveLayers = new Map();
let liveGlobe = null;
const liveGlobeLayers = new Map();
const liveGlobeMissiles = [];
let globeMissileAnimationFrame = null;
let globeViewer = null;
let globeTimeline = null;
let telemetryChartState = null;
let latestCompletedSummary = null;
const liveMissiles = [];
let missileAnimationFrame = null;
const destroyedPlatforms = new Set();
const visuallyDestroyedPlatforms = new Set();
const lastKnownUnits = new Map();

function formData() {
  const data = Object.fromEntries(new FormData(form));
  const numbers = ["blue_count", "red_count", "blue_start_latitude", "blue_start_longitude", "red_start_latitude", "red_start_longitude", "blue_formation_spacing_km", "red_formation_spacing_km", "altitude_m", "speed_mps", "max_steps", "time_scale", "afsim_port"];
  numbers.forEach((key) => { data[key] = Number(data[key]); });
  data.seed = data.seed === "" ? null : Number(data.seed);
  data.global_view = form.elements.global_view.checked;
  data.auto_start_afsim = form.elements.auto_start_afsim.checked;
  return data;
}

async function pickLocalFile(button) {
  button.disabled = true;
  try {
    const response = await fetch("/api/pick-file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: button.dataset.picker }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "无法打开文件选择框");
    if (payload.path) document.querySelector(`#${button.dataset.target}`).value = payload.path;
  } catch (error) {
    document.querySelector("#run-state").textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

function setConnection(label, kind = "online") {
  document.querySelector("#connection-label").textContent = label;
  const dot = document.querySelector("#connection-dot");
  dot.className = kind;
}

function speed(unit) {
  const velocity = unit.velocity;
  return Math.hypot(velocity.north_mps, velocity.east_mps, velocity.up_mps);
}

function weaponDisplayName(name) {
  return { aam_medium: "中距弹", aam_short: "近距弹", agm_short: "空地弹" }[name] || name;
}

function unitsFromObservations(observations) {
  const units = [];
  for (const side of ["blue", "red"]) {
    for (const unit of observations?.[side]?.own_units || []) units.push(unit);
  }
  return units;
}

function pointBounds(units) {
  const longitudes = units.map((unit) => unit.position.longitude);
  const latitudes = units.map((unit) => unit.position.latitude);
  const padLon = Math.max((Math.max(...longitudes) - Math.min(...longitudes)) * .12, .08);
  const padLat = Math.max((Math.max(...latitudes) - Math.min(...latitudes)) * .3, .08);
  return { minLon: Math.min(...longitudes) - padLon, maxLon: Math.max(...longitudes) + padLon, minLat: Math.min(...latitudes) - padLat, maxLat: Math.max(...latitudes) + padLat };
}

function percent(position, bounds) {
  return {
    x: ((position.longitude - bounds.minLon) / (bounds.maxLon - bounds.minLon)) * 100,
    y: 100 - ((position.latitude - bounds.minLat) / (bounds.maxLat - bounds.minLat)) * 100,
  };
}

function drawTrail(history, side, bounds) {
  if (history.length < 2) return;
  for (let index = 1; index < history.length; index += 1) {
    const from = percent(history[index - 1], bounds);
    const to = percent(history[index], bounds);
    const length = Math.hypot(to.x - from.x, to.y - from.y);
    const angle = Math.atan2(to.y - from.y, to.x - from.x) * 180 / Math.PI;
    const trail = document.createElement("i");
    trail.className = `trail ${side}`;
    trail.style.left = `${from.x}%`;
    trail.style.top = `${from.y}%`;
    trail.style.width = `${length}%`;
    trail.style.transform = `rotate(${angle}deg)`;
    map.append(trail);
  }
}

function renderMap(units) {
  if (renderCesiumMap(units)) return;
  if (renderLeafletMap(units)) return;
  map.querySelectorAll(".plane, .trail").forEach((node) => node.remove());
  document.querySelector("#map-empty").style.display = units.length ? "none" : "grid";
  if (!units.length) return;
  const bounds = pointBounds(units);
  for (const unit of units) {
    const history = unitHistory.get(unit.platform_id) || [];
    history.push(unit.position);
    if (history.length > 5000) history.shift();
    unitHistory.set(unit.platform_id, history);
    drawTrail(history, unit.side, bounds);
  }
  for (const unit of units) {
    const place = percent(unit.position, bounds);
    const plane = document.createElement("div");
    plane.className = `plane ${unit.side}`;
    plane.style.left = `${place.x}%`;
    plane.style.top = `${place.y}%`;
    plane.style.transform = `rotate(${unit.attitude.heading_deg - 45}deg)`;
    const label = document.createElement("span");
    label.className = "plane-label";
    label.textContent = unit.platform_id.replace("_fighter_", "-");
    plane.append(label);
    map.append(plane);
  }
}

function renderCesiumMap(units) {
  if (!window.Cesium) return false;
  const Cesium = window.Cesium;
  const empty = document.querySelector("#map-empty");
  if (!liveGlobe) {
    liveGlobe = new Cesium.Viewer("live-globe", {
      animation: false, timeline: false, baseLayerPicker: false, geocoder: false,
      homeButton: false, sceneModePicker: true, navigationHelpButton: false,
      infoBox: false, selectionIndicator: false, shouldAnimate: false,
    });
    liveGlobe.scene.globe.baseColor = Cesium.Color.fromCssColorString("#182a38");
    liveGlobe.scene.backgroundColor = Cesium.Color.fromCssColorString("#0d1b27");
    liveGlobe.scene.screenSpaceCameraController.minimumZoomDistance = 400;
    liveGlobe.camera.setView({ destination: Cesium.Cartesian3.fromDegrees(0, 0, 12000000) });
    Cesium.ArcGisMapServerImageryProvider.fromUrl("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer").then((provider) => {
      liveGlobe.imageryLayers.removeAll();
      liveGlobe.imageryLayers.addImageryProvider(provider);
    }).catch(() => {});
  }
  empty.style.display = units.length ? "none" : "grid";
  const active = new Set();
  units.forEach((unit) => {
    active.add(unit.platform_id);
    const history = unitHistory.get(unit.platform_id) || [];
    history.push(unit.position);
    if (history.length > 5000) history.shift();
    unitHistory.set(unit.platform_id, history);
    const color = unit.side === "blue" ? Cesium.Color.fromCssColorString("#38a0ff") : Cesium.Color.fromCssColorString("#ff6148");
    const position = Cesium.Cartesian3.fromDegrees(unit.position.longitude, unit.position.latitude, unit.position.altitude_m);
    const trail = history.map((point) => Cesium.Cartesian3.fromDegrees(point.longitude, point.latitude, point.altitude_m));
    let layer = liveGlobeLayers.get(unit.platform_id);
    if (!layer) {
      layer = {
        aircraft: liveGlobe.entities.add({
          position,
          point: { pixelSize: 10, color, outlineColor: Cesium.Color.WHITE, outlineWidth: 2 },
          label: {
            text: unit.platform_id.replace("_fighter_", "-"), font: "12px sans-serif",
            fillColor: Cesium.Color.WHITE, outlineColor: Cesium.Color.BLACK, outlineWidth: 2,
            style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(10, -10),
          },
        }),
        trail: liveGlobe.entities.add({ polyline: { positions: trail, width: 3, material: color.withAlpha(.72) } }),
      };
      liveGlobeLayers.set(unit.platform_id, layer);
    } else {
      layer.aircraft.position = position;
      layer.aircraft.show = true;
      layer.trail.polyline.positions = trail;
      layer.trail.show = true;
    }
  });
  liveGlobeLayers.forEach((layer, platformId) => {
    if (!active.has(platformId)) {
      layer.aircraft.show = false;
      if (!destroyedPlatforms.has(platformId)) {
        liveGlobe.entities.remove(layer.trail);
        liveGlobeLayers.delete(platformId);
      }
    }
  });
  if (units.length && !liveGlobe._airCombatFitted) {
    liveGlobe._airCombatFitted = true;
    requestAnimationFrame(() => liveGlobe.zoomTo([...liveGlobeLayers.values()].flatMap((layer) => [layer.aircraft, layer.trail])));
  }
  return true;
}

function renderLeafletMap(units) {
  if (!window.L) return false;
  document.querySelector("#map-empty").style.display = units.length ? "none" : "grid";
  if (!liveMap) {
    liveMap = window.L.map(map, { zoomControl: true, attributionControl: true }).setView([0, 0], 3);
    window.L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 19,
      attribution: "Tiles © Esri",
    }).addTo(liveMap);
  }
  const active = new Set();
  units.forEach((unit) => {
    active.add(unit.platform_id);
    const history = unitHistory.get(unit.platform_id) || [];
    history.push(unit.position);
    if (history.length > 5000) history.shift();
    unitHistory.set(unit.platform_id, history);
    const coordinates = history.map((point) => [point.latitude, point.longitude]);
    let layer = liveLayers.get(unit.platform_id);
    if (!layer) {
      const icon = aircraftIcon(unit);
      layer = { marker: window.L.marker(coordinates.at(-1), { icon }).addTo(liveMap), trail: window.L.polyline(coordinates, { color: unit.side === "blue" ? "#38a0ff" : "#ff6148", weight: 3 }).addTo(liveMap) };
      liveLayers.set(unit.platform_id, layer);
    } else {
      if (layer.destroyed) {
        layer.marker.addTo(liveMap);
        layer.destroyed = false;
      }
      layer.marker.setLatLng(coordinates.at(-1));
      layer.marker.setIcon(aircraftIcon(unit));
      layer.trail.setLatLngs(coordinates);
    }
  });
  liveLayers.forEach((layer, platformId) => {
    if (!active.has(platformId)) {
      layer.marker.remove();
      if (destroyedPlatforms.has(platformId)) { layer.destroyed = true; } else { layer.trail.remove(); liveLayers.delete(platformId); }
    }
  });
  if (units.length && !liveMap._airCombatFitted) {
    liveMap.fitBounds(units.map((unit) => [unit.position.latitude, unit.position.longitude]), { padding: [45, 45], maxZoom: 8 });
    liveMap._airCombatFitted = true;
  }
  return true;
}

function aircraftIcon(unit) {
  return window.L.divIcon({
    className: "",
    html: `<div class="live-aircraft ${unit.side}"></div>`,
    iconSize: [16, 16], iconAnchor: [8, 8],
  });
}

function renderTelemetry(units) {
  const body = document.querySelector("#telemetry-body");
  if (!units.length) return;
  body.innerHTML = units.map((unit) => {
    const weaponCount = (name) => unit.weapons.find((weapon) => weapon.name === name)?.count ?? "--";
    const position = unit.position;
    return `<tr><td><span class="platform-tag ${unit.side}"><i></i>${unit.platform_id}</span></td><td>${position.latitude.toFixed(3)}°, ${position.longitude.toFixed(3)}°</td><td>${Math.round(position.altitude_m).toLocaleString()} m</td><td>${Math.round(speed(unit))} m/s</td><td>${Math.round((unit.attitude.heading_deg + 360) % 360)}°</td><td>${weaponCount("aam_medium")}</td><td>${weaponCount("aam_short")}</td><td>${weaponCount("agm_short")}</td></tr>`;
  }).join("");
  document.querySelector("#blue-alive").textContent = units.filter((unit) => unit.side === "blue").length;
  document.querySelector("#red-alive").textContent = units.filter((unit) => unit.side === "red").length;
}

function renderEvents(events = []) {
  if (!events.length) return;
  const event = events[events.length - 1];
  document.querySelector("#event-title").textContent = event.event_type;
  document.querySelector("#event-detail").textContent = [event.shooter || event.platform, event.target, event.weapon].filter(Boolean).join(" → ") || `T+${Math.round(event.sim_time)}s`;
}

function renderStep(step) {
  const units = unitsFromObservations(step.observations);
  units.forEach((unit) => lastKnownUnits.set(unit.platform_id, unit));
  step.events.filter((event) => event.event_type === "WeaponMissed" && event.target).forEach((event) => {
    liveMissiles.forEach((missile) => { if (missile.targetId === event.target) missile.missed = true; });
    if (visuallyDestroyedPlatforms.delete(event.target)) destroyedPlatforms.delete(event.target);
  });
  step.events.filter((event) => event.event_type === "WeaponHit" && event.target).forEach((event) => {
    visuallyDestroyedPlatforms.delete(event.target);
    destroyedPlatforms.add(event.target);
  });
  const activeUnits = units.filter((unit) => !destroyedPlatforms.has(unit.platform_id));
  renderMap(activeUnits);
  renderLiveMissileEvents(step.events, activeUnits);
  renderHitEffects(step.events);
  renderTelemetry(units);
  renderEvents(step.events);
  document.querySelector("#sim-time").textContent = `T+${Math.round(step.sim_time)}s`;
  document.querySelector("#step-index").textContent = `STEP ${step.step_index}`;
  document.querySelector("#formation-readout").textContent = `${units.length} 架飞机 / 实时同步`;
}

function renderLiveMissileEvents(events, units) {
  if (liveGlobe) {
    renderCesiumMissileEvents(events, units);
    return;
  }
  if (!liveMap) return;
  const byPlatform = new Map(units.map((unit) => [unit.platform_id, unit]));
  events.filter((event) => event.event_type === "WeaponFired").forEach((event) => {
    const shooter = byPlatform.get(event.shooter);
    const target = byPlatform.get(event.target);
    if (!shooter || !target) return;
    const color = shooter.side === "blue" ? "#55adff" : "#ff7058";
    const start = [shooter.position.latitude, shooter.position.longitude];
    const end = [target.position.latitude, target.position.longitude];
    const icon = window.L.divIcon({ className: "", html: '<div class="live-missile"></div>', iconSize: [10, 10], iconAnchor: [5, 5] });
    liveMissiles.push({
      startedAt: performance.now(), duration: 2600, start, end,
      targetId: event.target,
      marker: window.L.marker(start, { icon, interactive: false }).addTo(liveMap),
      trail: window.L.polyline([start, start], { color, weight: 2, opacity: .9, dashArray: "5 5" }).addTo(liveMap),
    });
  });
  if (liveMissiles.length && !missileAnimationFrame) missileAnimationFrame = requestAnimationFrame(animateLiveMissiles);
}

function renderCesiumMissileEvents(events, units) {
  const Cesium = window.Cesium;
  const byPlatform = new Map(units.map((unit) => [unit.platform_id, unit]));
  events.filter((event) => event.event_type === "WeaponFired").forEach((event) => {
    const shooter = byPlatform.get(event.shooter);
    const target = byPlatform.get(event.target);
    if (!shooter || !target) return;
    const start = shooter.position;
    const end = target.position;
    const startPoint = Cesium.Cartesian3.fromDegrees(start.longitude, start.latitude, start.altitude_m);
    const color = shooter.side === "blue" ? "#55adff" : "#ff7058";
    const entity = liveGlobe.entities.add({
      position: startPoint,
      point: { pixelSize: 7, color: Cesium.Color.YELLOW, outlineColor: Cesium.Color.ORANGERED, outlineWidth: 2 },
      polyline: { positions: [startPoint, startPoint], width: 2, material: Cesium.Color.fromCssColorString(color).withAlpha(.9) },
    });
    liveGlobeMissiles.push({ startedAt: performance.now(), duration: 2600, start, end, targetId: event.target, entity });
  });
  if (liveGlobeMissiles.length && !globeMissileAnimationFrame) globeMissileAnimationFrame = requestAnimationFrame(animateCesiumMissiles);
}

function animateCesiumMissiles(now) {
  const Cesium = window.Cesium;
  for (let index = liveGlobeMissiles.length - 1; index >= 0; index -= 1) {
    const missile = liveGlobeMissiles[index];
    const progress = Math.min(1, (now - missile.startedAt) / missile.duration);
    const position = {
      latitude: missile.start.latitude + (missile.end.latitude - missile.start.latitude) * progress,
      longitude: missile.start.longitude + (missile.end.longitude - missile.start.longitude) * progress,
      altitude_m: missile.start.altitude_m + (missile.end.altitude_m - missile.start.altitude_m) * progress,
    };
    const current = Cesium.Cartesian3.fromDegrees(position.longitude, position.latitude, position.altitude_m);
    missile.entity.position = current;
    missile.entity.polyline.positions = [Cesium.Cartesian3.fromDegrees(missile.start.longitude, missile.start.latitude, missile.start.altitude_m), current];
    if (progress >= 1) {
      if (!missile.missed) destroyCesiumPlatform(missile.targetId, missile.end);
      liveGlobe.entities.remove(missile.entity);
      liveGlobeMissiles.splice(index, 1);
    }
  }
  globeMissileAnimationFrame = liveGlobeMissiles.length ? requestAnimationFrame(animateCesiumMissiles) : null;
}

function destroyCesiumPlatform(platformId, position) {
  if (!platformId || destroyedPlatforms.has(platformId)) return;
  visuallyDestroyedPlatforms.add(platformId);
  destroyedPlatforms.add(platformId);
  const layer = liveGlobeLayers.get(platformId);
  if (layer) layer.aircraft.show = false;
  const Cesium = window.Cesium;
  const effect = liveGlobe.entities.add({
    position: Cesium.Cartesian3.fromDegrees(position.longitude, position.latitude, position.altitude_m),
    point: { pixelSize: 18, color: Cesium.Color.YELLOW.withAlpha(.9), outlineColor: Cesium.Color.ORANGERED, outlineWidth: 3 },
  });
  setTimeout(() => liveGlobe?.entities.remove(effect), 1500);
}

function animateLiveMissiles(now) {
  for (let index = liveMissiles.length - 1; index >= 0; index -= 1) {
    const missile = liveMissiles[index];
    const progress = Math.min(1, (now - missile.startedAt) / missile.duration);
    const position = [missile.start[0] + (missile.end[0] - missile.start[0]) * progress, missile.start[1] + (missile.end[1] - missile.start[1]) * progress];
    missile.marker.setLatLng(position); missile.trail.setLatLngs([missile.start, position]);
    if (progress >= 1) {
      if (!missile.missed) visuallyDestroyPlatform(missile.targetId, missile.end);
      missile.marker.remove(); missile.trail.remove(); liveMissiles.splice(index, 1);
    }
  }
  missileAnimationFrame = liveMissiles.length ? requestAnimationFrame(animateLiveMissiles) : null;
}

function visuallyDestroyPlatform(platformId, impactPosition) {
  if (!platformId || destroyedPlatforms.has(platformId)) return;
  visuallyDestroyedPlatforms.add(platformId);
  destroyedPlatforms.add(platformId);
  const layer = liveLayers.get(platformId);
  if (layer) {
    layer.marker.remove();
    layer.destroyed = true;
  }
  const icon = window.L.divIcon({ className: "", html: '<div class="live-explosion">✦</div>', iconSize: [30, 30], iconAnchor: [15, 15] });
  const marker = window.L.marker(impactPosition, { icon, interactive: false }).addTo(liveMap);
  setTimeout(() => marker.remove(), 1500);
}

function renderHitEffects(events) {
  if (liveGlobe) {
    const Cesium = window.Cesium;
    events.filter((event) => event.event_type === "WeaponHit" && event.target).forEach((event) => {
      const target = lastKnownUnits.get(event.target);
      if (!target) return;
      const effect = liveGlobe.entities.add({
        position: Cesium.Cartesian3.fromDegrees(target.position.longitude, target.position.latitude, target.position.altitude_m),
        point: { pixelSize: 18, color: Cesium.Color.YELLOW.withAlpha(.9), outlineColor: Cesium.Color.ORANGERED, outlineWidth: 3 },
      });
      setTimeout(() => liveGlobe?.entities.remove(effect), 1500);
    });
    return;
  }
  if (!liveMap) return;
  events.filter((event) => event.event_type === "WeaponHit" && event.target).forEach((event) => {
    const target = lastKnownUnits.get(event.target);
    if (!target) return;
    const icon = window.L.divIcon({ className: "", html: '<div class="live-explosion">✹</div>', iconSize: [30, 30], iconAnchor: [15, 15] });
    const marker = window.L.marker([target.position.latitude, target.position.longitude], { icon, interactive: false }).addTo(liveMap);
    setTimeout(() => marker.remove(), 1500);
  });
}

function renderResult(summary) {
  const names = { blue: "蓝方胜利", red: "红方胜利", draw: "平局" };
  const reason = { blue_eliminated: "蓝方战机全部被击落", red_eliminated: "红方战机全部被击落", simultaneous_elimination: "双方同时失去全部战机", max_steps: "达到最大决策步数", max_sim_time: "达到最大仿真时间", agent_violation_limit: "Agent 违规达到上限", stopped_by_user: "已通过网页安全结束仿真" };
  document.querySelector("#result-title").textContent = names[summary.winner] || "任务结束";
  document.querySelector("#result-copy").textContent = reason[summary.reason] || summary.reason || "本局仿真结束。";
  document.querySelector("#result-steps").textContent = summary.executed_steps ?? "--";
  const violations = summary.violations || {};
  document.querySelector("#violation-count").textContent = Object.values(violations).reduce((total, value) => total + (value.total || 0), 0);
}

function renderAnalysisMarkdown(markdown) {
  const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;",
  })[character]);
  const inline = (value) => escapeHtml(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  const output = [];
  let paragraph = [];
  let listType = null;
  const flushParagraph = () => {
    if (paragraph.length) output.push(`<p>${inline(paragraph.join(" "))}</p>`);
    paragraph = [];
  };
  const closeList = () => {
    if (listType) output.push(`</${listType}>`);
    listType = null;
  };
  for (const rawLine of String(markdown || "").replace(/\r/g, "").split("\n")) {
    const line = rawLine.trim();
    const heading = line.match(/^(#{1,3})\s+(.+)$/);
    const bullet = line.match(/^[-*]\s+(.+)$/);
    const ordered = line.match(/^\d+\.\s+(.+)$/);
    if (!line) { flushParagraph(); closeList(); continue; }
    if (heading) {
      flushParagraph(); closeList();
      const level = heading[1].length;
      output.push(`<h${level}>${inline(heading[2])}</h${level}>`);
    } else if (bullet || ordered) {
      flushParagraph();
      const nextType = bullet ? "ul" : "ol";
      if (listType && listType !== nextType) closeList();
      if (!listType) { output.push(`<${nextType}>`); listType = nextType; }
      output.push(`<li>${inline((bullet || ordered)[1])}</li>`);
    } else {
      closeList();
      paragraph.push(line);
    }
  }
  flushParagraph(); closeList();
  return output.join("");
}

async function requestAnalysis(summary) {
  const panel = document.querySelector("#analysis-panel");
  const state = document.querySelector("#analysis-state");
  const content = document.querySelector("#analysis-content");
  const button = document.querySelector("#request-analysis");
  button.disabled = true;
  panel.hidden = false;
  state.hidden = false;
  state.textContent = "正在整理回放数据并请求大模型分析。";
  content.hidden = true;
  document.querySelector("#analysis-model").textContent = "";
  try {
    const response = await fetch(`/api/replays/${summary.episode_id}/analysis`, { method: "POST" });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "无法生成大模型复盘");
    content.innerHTML = renderAnalysisMarkdown(data.analysis);
    content.hidden = false;
    state.hidden = true;
    document.querySelector("#analysis-model").textContent = data.model;
  } catch (error) {
    state.textContent = `大模型复盘未生成：${error.message}`;
  } finally {
    button.disabled = !latestCompletedSummary;
  }
}

async function requestLatestAnalysis() {
  if (!latestCompletedSummary) {
    const response = await fetch("/api/status");
    const status = await response.json();
    latestCompletedSummary = status.last_summary || null;
  }
  if (!latestCompletedSummary) {
    document.querySelector("#analysis-state").hidden = false;
    document.querySelector("#analysis-state").textContent = "尚无已完成仿真，无法生成分析。";
    return;
  }
  requestAnalysis(latestCompletedSummary);
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { context, width: rect.width, height: rect.height };
}

function range(values) {
  const low = Math.min(...values);
  const high = Math.max(...values);
  return [low, high === low ? low + 1 : high];
}

function unitSpeed(sample) {
  const velocity = sample.velocity;
  return Math.hypot(velocity.north_mps, velocity.east_mps, velocity.up_mps);
}

function trajectoryColor(side) {
  return side === "blue" ? "#55adff" : "#ff7b62";
}

function projectPoint(point, yawDeg, pitchDeg, width, height) {
  const yaw = yawDeg * Math.PI / 180;
  const pitch = pitchDeg * Math.PI / 180;
  const x1 = point.x * Math.cos(yaw) - point.z * Math.sin(yaw);
  const z1 = point.x * Math.sin(yaw) + point.z * Math.cos(yaw);
  const y1 = point.y * Math.cos(pitch) - z1 * Math.sin(pitch);
  const perspective = 1 + (z1 + .5) * .22;
  const scale = Math.min(width, height) * .78 / perspective;
  return { x: width / 2 + x1 * scale, y: height / 2 - y1 * scale };
}

function draw3DTrajectories() {
  const canvas = document.querySelector("#trajectory-3d");
  const { context, width, height } = prepareCanvas(canvas);
  context.fillStyle = "#152721";
  context.fillRect(0, 0, width, height);
  if (!replayTelemetry?.trajectories?.length) return;

  const samples = replayTelemetry.trajectories.flatMap((trajectory) => trajectory.samples);
  const lonRange = range(samples.map((sample) => sample.position.longitude));
  const latRange = range(samples.map((sample) => sample.position.latitude));
  const altRange = range(samples.map((sample) => sample.position.altitude_m));
  const yaw = Number(document.querySelector("#trajectory-yaw").value);
  const pitch = Number(document.querySelector("#trajectory-pitch").value);
  const normalize = (sample) => ({
    x: (sample.position.longitude - lonRange[0]) / (lonRange[1] - lonRange[0]) - .5,
    y: (sample.position.latitude - latRange[0]) / (latRange[1] - latRange[0]) - .5,
    z: (sample.position.altitude_m - altRange[0]) / (altRange[1] - altRange[0]) - .5,
  });

  context.lineWidth = 1;
  context.strokeStyle = "rgba(210,235,215,.3)";
  [[-.55, -.55, -.55], [.55, -.55, -.55], [.55, .55, -.55], [-.55, .55, -.55], [-.55, -.55, -.55]].forEach((item, index, list) => {
    const projected = projectPoint({ x: item[0], y: item[1], z: item[2] }, yaw, pitch, width, height);
    if (index === 0) context.moveTo(projected.x, projected.y); else context.lineTo(projected.x, projected.y);
    if (index === list.length - 1) context.stroke();
  });

  replayTelemetry.trajectories.forEach((trajectory) => {
    const color = trajectoryColor(trajectory.side);
    context.beginPath();
    trajectory.samples.forEach((sample, index) => {
      const point = projectPoint(normalize(sample), yaw, pitch, width, height);
      if (index === 0) context.moveTo(point.x, point.y); else context.lineTo(point.x, point.y);
    });
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.stroke();
    const last = projectPoint(normalize(trajectory.samples.at(-1)), yaw, pitch, width, height);
    context.fillStyle = color;
    context.beginPath(); context.arc(last.x, last.y, 4, 0, Math.PI * 2); context.fill();
    context.fillStyle = "#edf5e8";
    context.font = "11px Consolas";
    context.fillText(trajectory.platform_id, last.x + 6, last.y - 5);
  });
  context.fillStyle = "#aebeb6";
  context.font = "10px Consolas";
  context.fillText("经度", 12, height - 14); context.fillText("纬度", 12, height - 2); context.fillText("高度", width - 38, 14);
}

function accelerationSeries(samples) {
  return samples.map((sample, index) => {
    if (index === 0) return 0;
    const previous = samples[index - 1];
    return (unitSpeed(sample) - unitSpeed(previous)) / Math.max(.001, sample.sim_time - previous.sim_time);
  });
}

function unwrapAngleSeries(values, period = 360) {
  if (!values.length) return [];
  const halfPeriod = period / 2;
  const unwrapped = [values[0]];
  for (let index = 1; index < values.length; index += 1) {
    let delta = values[index] - values[index - 1];
    if (delta > halfPeriod) delta -= period;
    if (delta < -halfPeriod) delta += period;
    unwrapped.push(unwrapped[index - 1] + delta);
  }
  return unwrapped;
}

function drawMetricChart(canvasId, label, values, color, samples, acceleration) {
  const canvas = document.querySelector(`#${canvasId}`);
  const { context, width, height } = prepareCanvas(canvas);
  context.fillStyle = "#fbfcf8"; context.fillRect(0, 0, width, height);
  const times = samples.map((sample) => sample.sim_time);
  const [timeMin, timeMax] = range(times); const [valueMin, valueMax] = range(values);
  const left = 46; const right = width - 14; const top = 24; const bottom = height - 24;
  const x = (time) => left + (time - timeMin) / (timeMax - timeMin) * (right - left);
  const y = (value) => bottom - (value - valueMin) / (valueMax - valueMin) * (bottom - top);
  context.strokeStyle = "#d8e0d7"; context.beginPath(); context.moveTo(left, top); context.lineTo(right, top); context.moveTo(left, bottom); context.lineTo(right, bottom); context.stroke();
  context.beginPath(); values.forEach((value, index) => { const pointX = x(times[index]); const pointY = y(value); if (index === 0) context.moveTo(pointX, pointY); else context.lineTo(pointX, pointY); }); context.strokeStyle = color; context.lineWidth = 2; context.stroke();
  const platformId = document.querySelector("#trajectory-unit").value;
  (replayTelemetry.fire_actions || []).filter((item) => item.platform_id === platformId).forEach((item) => { const fireX = x(item.sim_time); context.strokeStyle = "rgba(227,74,49,.75)"; context.lineWidth = 1; context.beginPath(); context.moveTo(fireX, top); context.lineTo(fireX, bottom); context.stroke(); context.fillStyle = "#ba3d2a"; context.font = "10px Consolas"; context.fillText("F", fireX + 2, top + 11); });
  context.fillStyle = "#34433c"; context.font = "10px Consolas"; context.fillText(`${label}  ${Math.round(valueMin)}–${Math.round(valueMax)}`, left, 13); context.fillText(`T+${Math.round(timeMin)}s`, left, height - 7); context.fillText(`T+${Math.round(timeMax)}s`, right - 40, height - 7);
  telemetryChartState.set(canvasId, { samples, acceleration, left, right, timeMin, timeMax });
}

function drawTelemetryChart() {
  const platformId = document.querySelector("#trajectory-unit").value;
  const trajectory = replayTelemetry?.trajectories?.find((item) => item.platform_id === platformId);
  if (!trajectory?.samples?.length) return;
  const samples = trajectory.samples;
  const altitude = samples.map((sample) => sample.position.altitude_m);
  const speedKmh = samples.map((sample) => unitSpeed(sample) * 3.6);
  const pitch = samples.map((sample) => sample.attitude.pitch_deg);
  const roll = unwrapAngleSeries(samples.map((sample) => sample.attitude.roll_deg));
  const heading = unwrapAngleSeries(samples.map((sample) => (sample.attitude.heading_deg + 360) % 360));
  const acceleration = accelerationSeries(samples);
  telemetryChartState = new Map();
  drawMetricChart("telemetry-altitude", "高度 (m)", altitude, "#7da91c", samples, acceleration);
  drawMetricChart("telemetry-speed", "速度 (km/h)", speedKmh, "#167cd0", samples, acceleration);
  drawMetricChart("telemetry-pitch", "俯仰角 (°)", pitch, "#c4890d", samples, acceleration);
  drawMetricChart("telemetry-roll", "滚转角 (°)", roll, "#a361d1", samples, acceleration);
  drawMetricChart("telemetry-heading", "航向 / 偏航 (°)", heading, "#108d83", samples, acceleration);
  drawMetricChart("telemetry-acceleration", "加速度 (m/s²)", acceleration, "#d6533d", samples, acceleration);
  renderTelemetryDetails(trajectory, acceleration);
}

function showTelemetryTooltip(event) {
  const canvas = event.currentTarget;
  const state = telemetryChartState?.get(canvas.id);
  if (!state) return;
  const rect = canvas.getBoundingClientRect();
  const pointerX = Math.max(state.left, Math.min(state.right, event.clientX - rect.left));
  const targetTime = state.timeMin + (pointerX - state.left) / (state.right - state.left) * (state.timeMax - state.timeMin);
  const index = state.samples.reduce((nearest, sample, sampleIndex) => Math.abs(sample.sim_time - targetTime) < Math.abs(state.samples[nearest].sim_time - targetTime) ? sampleIndex : nearest, 0);
  const sample = state.samples[index];
  const tooltip = document.querySelector("#chart-tooltip");
  tooltip.innerHTML = `<b>T+${Math.round(sample.sim_time)}s · STEP ${sample.step_index}</b><br>高度: ${Math.round(sample.position.altitude_m)} m<br>速度: ${Math.round(unitSpeed(sample) * 3.6)} km/h<br>航向 / 俯仰 / 滚转: ${Math.round((sample.attitude.heading_deg + 360) % 360)}° / ${Math.round(sample.attitude.pitch_deg)}° / ${Math.round(sample.attitude.roll_deg)}°<br>加速度: ${state.acceleration[index].toFixed(2)} m/s²`;
  tooltip.hidden = false;
  const articleRect = canvas.closest("article").getBoundingClientRect();
  const left = Math.min(articleRect.width - 193, Math.max(8, event.clientX - articleRect.left + 12));
  const top = Math.min(rect.bottom - articleRect.top - 82, Math.max(8, event.clientY - articleRect.top - 30));
  tooltip.style.left = `${left}px`; tooltip.style.top = `${top}px`;
}

function renderTelemetryDetails(trajectory, acceleration) {
  const sample = trajectory.samples.at(-1);
  const platformId = trajectory.platform_id;
  const firstLocks = [...new Map((replayTelemetry.locks || []).filter((item) => item.platform_id === platformId).map((item) => [item.target_id, item])).values()];
  const fires = (replayTelemetry.fire_actions || []).filter((item) => item.platform_id === platformId);
  const initialWeapons = new Map(trajectory.samples[0].weapons.map((weapon) => [weapon.name, weapon.count]));
  const weaponRemaining = sample.weapons.map((weapon) => `${weaponDisplayName(weapon.name)} ${weapon.count}/${initialWeapons.get(weapon.name) ?? weapon.count}`).join(" · ") || "无";
  const cards = [
    ["末端位置", `${sample.position.latitude.toFixed(4)}°, ${sample.position.longitude.toFixed(4)}°`], ["末端高度", `${Math.round(sample.position.altitude_m)} m`], ["三维速度", `${Math.round(unitSpeed(sample) * 3.6)} km/h`], ["北/东/天速度", `${Math.round(sample.velocity.north_mps)} / ${Math.round(sample.velocity.east_mps)} / ${Math.round(sample.velocity.up_mps)} m/s`],
    ["航向 / 俯仰 / 滚转", `${Math.round((sample.attitude.heading_deg + 360) % 360)}° / ${Math.round(sample.attitude.pitch_deg)}° / ${Math.round(sample.attitude.roll_deg)}°`], ["最大加速度", `${Math.max(...acceleration.map(Math.abs)).toFixed(1)} m/s²`], ["传感器", sample.sensor?.enabled ? "开启" : "关闭"], ["武器余量（剩余/初始）", weaponRemaining],
    ["开火指令时刻", fires.length ? fires.map((item) => `T+${Math.round(item.sim_time)}s`).join("，") : "无"], ["航迹获取", firstLocks.length ? firstLocks.map((item) => `${item.target_id} T+${Math.round(item.sim_time)}s`).join("，") : "未获取"],
  ];
  document.querySelector("#telemetry-details").innerHTML = cards.map(([label, value]) => `<div><small>${label}</small><b>${value}</b></div>`).join("");
}

function buildGlobeReplay() {
  if (!window.Cesium || !replayTelemetry?.trajectories?.length) {
    document.querySelector("#replay-status").textContent = "无法加载 Cesium 三维地球，请检查网络连接";
    return;
  }
  const Cesium = window.Cesium;
  if (!globeViewer) {
    globeViewer = new Cesium.Viewer("trajectory-globe", {
      animation: false, timeline: false, baseLayerPicker: false, geocoder: false,
      homeButton: false, sceneModePicker: false, navigationHelpButton: false,
      infoBox: false, selectionIndicator: false, shouldAnimate: false,
    });
    globeViewer.scene.screenSpaceCameraController.zoomFactor = 1.2;
    globeViewer.scene.screenSpaceCameraController.minimumZoomDistance = 250;
    Cesium.ArcGisMapServerImageryProvider.fromUrl("https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer").then((provider) => {
      globeViewer.imageryLayers.removeAll(); globeViewer.imageryLayers.addImageryProvider(provider);
    }).catch(() => { document.querySelector("#replay-status").textContent = "卫星影像加载失败，已保留默认地球底图"; });
  }
  globeViewer.entities.removeAll();
  const allSamples = replayTelemetry.trajectories.flatMap((trajectory) => trajectory.samples);
  const firstTime = Math.min(...allSamples.map((sample) => sample.sim_time));
  const lastTime = Math.max(...allSamples.map((sample) => sample.sim_time));
  const start = Cesium.JulianDate.fromDate(new Date(0));
  const stop = Cesium.JulianDate.addSeconds(start, Math.max(1, lastTime - firstTime), new Cesium.JulianDate());
  const entities = [];
  replayTelemetry.trajectories.forEach((trajectory) => {
    const color = trajectory.side === "blue" ? Cesium.Color.fromCssColorString("#38a0ff") : Cesium.Color.fromCssColorString("#ff6148");
    const property = new Cesium.SampledPositionProperty();
    const positions = trajectory.samples.map((sample) => {
      const time = Cesium.JulianDate.addSeconds(start, sample.sim_time - firstTime, new Cesium.JulianDate());
      const position = Cesium.Cartesian3.fromDegrees(sample.position.longitude, sample.position.latitude, sample.position.altitude_m);
      property.addSample(time, position);
      return position;
    });
    entities.push(globeViewer.entities.add({ polyline: { positions, width: 3, material: color.withAlpha(.72) } }));
    entities.push(globeViewer.entities.add({
      position: property, point: { pixelSize: 9, color, outlineColor: Cesium.Color.WHITE, outlineWidth: 1 },
      label: { text: trajectory.platform_id, font: "12px sans-serif", fillColor: Cesium.Color.WHITE, outlineColor: Cesium.Color.BLACK, outlineWidth: 2, style: Cesium.LabelStyle.FILL_AND_OUTLINE, pixelOffset: new Cesium.Cartesian2(10, -10) },
      path: { resolution: 1, material: color, width: 4, trailTime: lastTime - firstTime, leadTime: 0 },
    }));
  });
  const trajectoryByPlatform = new Map(replayTelemetry.trajectories.map((trajectory) => [trajectory.platform_id, trajectory]));
  const sampleAt = (trajectory, simTime) => trajectory?.samples.reduce((nearest, sample) => Math.abs(sample.sim_time - simTime) < Math.abs(nearest.sim_time - simTime) ? sample : nearest, trajectory.samples[0]);
  (replayTelemetry.events || []).filter((event) => event.event_type === "WeaponFired").forEach((event) => {
    const shooter = trajectoryByPlatform.get(event.shooter);
    const target = trajectoryByPlatform.get(event.target);
    if (!shooter || !target) return;
    const launchTime = Math.max(firstTime, event.sim_time);
    const impactTime = Math.min(lastTime, launchTime + 20);
    const launchSample = sampleAt(shooter, launchTime);
    const impactSample = sampleAt(target, impactTime);
    const missilePosition = new Cesium.SampledPositionProperty();
    const launchDate = Cesium.JulianDate.addSeconds(start, launchTime - firstTime, new Cesium.JulianDate());
    const impactDate = Cesium.JulianDate.addSeconds(start, impactTime - firstTime, new Cesium.JulianDate());
    missilePosition.addSample(launchDate, Cesium.Cartesian3.fromDegrees(launchSample.position.longitude, launchSample.position.latitude, launchSample.position.altitude_m));
    missilePosition.addSample(impactDate, Cesium.Cartesian3.fromDegrees(impactSample.position.longitude, impactSample.position.latitude, impactSample.position.altitude_m));
    entities.push(globeViewer.entities.add({
      availability: new Cesium.TimeIntervalCollection([new Cesium.TimeInterval({ start: launchDate, stop: impactDate })]),
      position: missilePosition,
      point: { pixelSize: 7, color: Cesium.Color.YELLOW, outlineColor: Cesium.Color.ORANGERED, outlineWidth: 2 },
      path: { resolution: 1, material: Cesium.Color.YELLOW.withAlpha(.8), width: 2, trailTime: 20, leadTime: 0 },
    }));
  });
  globeViewer.clock.startTime = start.clone(); globeViewer.clock.stopTime = stop.clone(); globeViewer.clock.currentTime = start.clone();
  globeViewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP; globeViewer.clock.multiplier = Number(document.querySelector("#replay-speed").value); globeViewer.clock.shouldAnimate = false;
  globeTimeline = { start, stop };
  globeViewer.resize();
  requestAnimationFrame(() => { globeViewer.resize(); globeViewer.zoomTo(entities); });
}

function redrawReplay() {
  drawTelemetryChart();
}

async function loadReplay(summary) {
  const status = document.querySelector("#replay-status");
  status.textContent = "正在读取本地 JSONL 回放";
  try {
    const response = await fetch(`/api/replays/${summary.episode_id}/telemetry`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "无法读取回放");
    replayTelemetry = data;
    const replaySpeed = document.querySelector("#replay-speed");
    replaySpeed.value = ["0.2", "0.5", "1", "2", "3"].includes(String(summary.time_scale)) ? String(summary.time_scale) : "1";
    const selector = document.querySelector("#trajectory-unit");
    selector.replaceChildren();
    data.trajectories.forEach((trajectory) => { const option = document.createElement("option"); option.value = trajectory.platform_id; option.textContent = trajectory.platform_id; selector.append(option); });
    selector.disabled = !data.trajectories.length;
    status.textContent = `${data.trajectories.length} 架战机，${data.trajectories.reduce((total, item) => total + item.samples.length, 0)} 个遥测样本`;
    buildGlobeReplay(); redrawReplay();
  } catch (error) { status.textContent = error.message; }
}

function setRunControls(running) {
  startButton.disabled = running;
  stopButton.disabled = !running;
  startButton.querySelector("span").textContent = running ? "仿真运行中" : "启动仿真";
}

async function startRun(event) {
  event.preventDefault();
  startButton.disabled = true;
  startButton.querySelector("span").textContent = "正在启动";
  document.querySelector("#run-state").textContent = "正在连接 AFSIM";
  unitHistory.clear();
  destroyedPlatforms.clear(); visuallyDestroyedPlatforms.clear(); lastKnownUnits.clear();
  liveLayers.forEach((layer) => { layer.marker.remove(); layer.trail.remove(); }); liveLayers.clear();
  liveGlobeLayers.forEach((layer) => { liveGlobe?.entities.remove(layer.aircraft); liveGlobe?.entities.remove(layer.trail); }); liveGlobeLayers.clear();
  liveGlobeMissiles.splice(0).forEach((missile) => liveGlobe?.entities.remove(missile.entity));
  liveMissiles.splice(0).forEach((missile) => { missile.marker.remove(); missile.trail.remove(); });
  if (liveMap) liveMap._airCombatFitted = false;
  if (liveGlobe) liveGlobe._airCombatFitted = false;
  try {
    const response = await fetch("/api/runs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(formData()) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "无法启动仿真");
    document.querySelector("#run-state").textContent = `任务 ${payload.run_id.slice(0, 8)} 已提交`;
    setRunControls(true);
    replayTelemetry = null;
    latestCompletedSummary = null;
    document.querySelector("#analysis-panel").hidden = false;
    document.querySelector("#analysis-state").hidden = false;
    document.querySelector("#analysis-state").textContent = "本局仿真进行中，完成后将自动生成大模型复盘。";
    document.querySelector("#analysis-content").hidden = true;
    document.querySelector("#analysis-model").textContent = "";
    document.querySelector("#request-analysis").disabled = true;
    document.querySelector("#replay-status").textContent = "本局运行中，结束后加载回放";
  } catch (error) {
    document.querySelector("#run-state").textContent = "启动失败";
    document.querySelector("#result-title").textContent = "无法启动";
    document.querySelector("#result-copy").textContent = error.message;
    setRunControls(false);
  }
}

async function stopRun() {
  stopButton.disabled = true;
  stopButton.textContent = "正在结束";
  try {
    const response = await fetch("/api/runs/stop", { method: "POST" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "无法结束仿真");
    document.querySelector("#run-state").textContent = payload.afsim_stopped ? "AFSIM 已停止，正在整理结果" : "正在安全结束当前决策步";
  } catch (error) {
    document.querySelector("#run-state").textContent = error.message;
    stopButton.disabled = false;
  } finally {
    stopButton.textContent = "结束仿真";
  }
}

async function checkAfsim() {
  try {
    const data = formData();
    document.querySelector("#run-state").textContent = "检查 AFSIM 状态";
    const response = await fetch(`/api/afsim-status?ip=${encodeURIComponent(data.afsim_ip)}&port=${data.afsim_port}`);
    const status = await response.json();
    if (status.ready) {
      document.querySelector("#run-state").textContent = `AFSIM 已就绪 · ${status.endpoint}`;
    } else if (data.auto_start_afsim) {
      document.querySelector("#run-state").textContent = "正在启动并加载所选 AFSIM 场景";
      const startResponse = await fetch("/api/afsim/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      const started = await startResponse.json();
      if (!startResponse.ok) throw new Error(started.detail || "无法启动 AFSIM");
      document.querySelector("#run-state").textContent = started.started ? `AFSIM 已启动并加载场景 · ${started.endpoint}` : `AFSIM 已就绪 · ${started.endpoint}`;
    } else {
      document.querySelector("#run-state").textContent = `AFSIM 不可用 · ${status.error || status.endpoint}`;
    }
  } catch (error) {
    document.querySelector("#run-state").textContent = error.message;
  }
}

function connectWebSocket() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  websocket = new WebSocket(`${protocol}://${location.host}/ws`);
  websocket.onopen = () => { setConnection("控制台已连接", "online"); websocket.send("ready"); };
  websocket.onclose = () => { setConnection("控制台断开，正在重连", "fail"); setTimeout(connectWebSocket, 1500); };
  websocket.onmessage = ({ data }) => {
    const message = JSON.parse(data);
    if (message.type === "episode_start") { document.querySelector("#run-state").textContent = "仿真运行中"; setRunControls(true); renderStep({ ...message.payload.observations.blue, observations: message.payload.observations, events: [] }); }
    if (message.type === "step") renderStep(message.payload);
    if (message.type === "complete") { latestCompletedSummary = message.payload; document.querySelector("#request-analysis").disabled = false; renderResult(message.payload); document.querySelector("#run-state").textContent = "任务完成"; setRunControls(false); loadReplay(message.payload); requestAnalysis(message.payload); }
    if (message.type === "error") { document.querySelector("#run-state").textContent = "仿真错误"; document.querySelector("#result-title").textContent = "运行中断"; document.querySelector("#result-copy").textContent = message.payload.message; setRunControls(false); }
    if (message.type === "status") { latestCompletedSummary = message.payload.last_summary || latestCompletedSummary; document.querySelector("#request-analysis").disabled = !latestCompletedSummary; setRunControls(Boolean(message.payload.running)); }
  };
}

form.addEventListener("submit", startRun);
document.querySelector("#check-afsim").addEventListener("click", checkAfsim);
document.querySelector("#request-analysis").addEventListener("click", requestLatestAnalysis);
document.querySelectorAll("[data-picker]").forEach((button) => button.addEventListener("click", () => pickLocalFile(button)));
stopButton.addEventListener("click", stopRun);
document.querySelector("#trajectory-unit").addEventListener("change", redrawReplay);
document.querySelectorAll("[data-telemetry-chart]").forEach((canvas) => {
  canvas.addEventListener("mousemove", showTelemetryTooltip);
  canvas.addEventListener("mouseleave", () => { document.querySelector("#chart-tooltip").hidden = true; });
});
document.querySelector("#replay-speed").addEventListener("change", (event) => { if (globeViewer) globeViewer.clock.multiplier = Number(event.target.value); });
document.querySelector("#replay-play").addEventListener("click", () => {
  if (!globeViewer) return;
  globeViewer.clock.shouldAnimate = !globeViewer.clock.shouldAnimate;
  document.querySelector("#replay-play").textContent = globeViewer.clock.shouldAnimate ? "暂停" : "播放";
});
document.querySelector("#replay-reset").addEventListener("click", () => {
  if (!globeViewer || !globeTimeline) return;
  globeViewer.clock.currentTime = globeTimeline.start.clone(); globeViewer.clock.shouldAnimate = false;
  document.querySelector("#replay-play").textContent = "播放";
});
window.addEventListener("resize", () => { if (liveGlobe) liveGlobe.resize(); if (globeViewer) globeViewer.resize(); redrawReplay(); });
renderCesiumMap([]);
connectWebSocket();
