const form = document.getElementById("predictForm");
const healthStatus = document.getElementById("healthStatus");
const resultCard = document.getElementById("resultCard");
const recentPredictions = document.getElementById("recentPredictions");

const makeSelect = document.getElementById("makeSelect");
const modelSelect = document.getElementById("modelSelect");
const yearSelect = document.getElementById("yearSelect");
const variantSelect = document.getElementById("variantSelect");
const registeredSelect = document.getElementById("registeredSelect");
const colorSelect = document.getElementById("colorSelect");

const transmissionInput = document.getElementById("transmissionInput");
const fuelTypeInput = document.getElementById("fuelTypeInput");
const assemblyInput = document.getElementById("assemblyInput");
const bodyTypeInput = document.getElementById("bodyTypeInput");
const engineCapacityInput = document.getElementById("engineCapacityInput");

function money(value) {
  if (value == null) return "N/A";
  return new Intl.NumberFormat("en-PK", { maximumFractionDigits: 0 }).format(value);
}

async function getJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed: ${response.status}`);
  }
  return response.json();
}

function setOptions(select, options, placeholder, disabled = false) {
  select.innerHTML = `<option value="">${placeholder}</option>` + options.map((option) => (
    `<option value="${option}">${option}</option>`
  )).join("");
  select.disabled = disabled;
}

function setSpecFields(spec = {}) {
  transmissionInput.value = spec.transmission || "";
  fuelTypeInput.value = spec.fuel_type || "";
  assemblyInput.value = spec.assembly || "";
  bodyTypeInput.value = spec.body_type || "";
  engineCapacityInput.value = spec.engine_capacity_cc || "";
}

function appendLockedSpecFields(payload) {
  if (transmissionInput.value) payload.transmission = transmissionInput.value;
  if (fuelTypeInput.value) payload.fuel_type = fuelTypeInput.value;
  if (assemblyInput.value) payload.assembly = assemblyInput.value;
  if (bodyTypeInput.value) payload.body_type = bodyTypeInput.value;
  if (engineCapacityInput.value) payload.engine_capacity_cc = Number(engineCapacityInput.value);
}

async function loadHealth() {
  const health = await getJson("/api/health");
  healthStatus.textContent = `Service is live. Catalog: ${health.catalog_exists ? "ready" : "missing"} • GitHub mirror: ${health.github_mirror_enabled ? "enabled" : "disabled"} • Prediction store: ${health.prediction_store_exists ? "ready" : "missing"}`;
}

async function loadRecentPredictions() {
  const rows = await getJson("/api/predictions/recent?limit=8");
  if (!rows.length) {
    recentPredictions.innerHTML = `<div class="empty">No predictions stored yet.</div>`;
    return;
  }

  recentPredictions.innerHTML = rows.map((row) => `
    <article class="recent-card">
      <div class="recent-head">
        <strong>#${row.id}</strong>
        <span>${new Date(row.created_at).toLocaleString()}</span>
      </div>
      <div class="recent-grid">
        <span>Mode: ${row.prediction_mode}</span>
        <span>Price: PKR ${money(row.predicted_price)}</span>
        <span>Range: PKR ${money(row.price_range_low)} to ${money(row.price_range_high)}</span>
        <span>GitHub: ${row.github_status || "not mirrored"}</span>
      </div>
    </article>
  `).join("");
}

async function loadCatalogOptions(params = {}) {
  const query = new URLSearchParams(params);
  return getJson(`/api/catalog/options?${query.toString()}`);
}

async function initializeCatalog() {
  const options = await loadCatalogOptions();
  setOptions(makeSelect, options.makes || [], "Select make");
  setOptions(modelSelect, [], "Select model", true);
  setOptions(yearSelect, [], "Select year", true);
  setOptions(variantSelect, [], "Select variant", true);
  setOptions(registeredSelect, [], "Select registration", true);
  setOptions(colorSelect, [], "Select color", true);
}

async function onMakeChange() {
  const options = await loadCatalogOptions({ make: makeSelect.value });
  setOptions(modelSelect, options.models || [], "Select model", !makeSelect.value);
  setOptions(yearSelect, [], "Select year", true);
  setOptions(variantSelect, [], "Select variant", true);
  setOptions(registeredSelect, [], "Select registration", true);
  setOptions(colorSelect, [], "Select color", true);
  setSpecFields();
}

async function onModelChange() {
  const options = await loadCatalogOptions({ make: makeSelect.value, model: modelSelect.value });
  setOptions(yearSelect, (options.years || []).map(String), "Select year", !modelSelect.value);
  setOptions(variantSelect, [], "Select variant", true);
  setOptions(registeredSelect, [], "Select registration", true);
  setOptions(colorSelect, [], "Select color", true);
  setSpecFields();
}

async function onYearChange() {
  const options = await loadCatalogOptions({
    make: makeSelect.value,
    model: modelSelect.value,
    year: yearSelect.value,
  });
  setOptions(variantSelect, options.variants || [], "Select variant", !yearSelect.value);
  setOptions(registeredSelect, [], "Select registration", true);
  setOptions(colorSelect, [], "Select color", true);
  setSpecFields();
}

async function onVariantChange() {
  if (!variantSelect.value) {
    setSpecFields();
    return;
  }
  const spec = await getJson(`/api/catalog/spec?make=${encodeURIComponent(makeSelect.value)}&model=${encodeURIComponent(modelSelect.value)}&year=${encodeURIComponent(yearSelect.value)}&variant=${encodeURIComponent(variantSelect.value)}`);
  setSpecFields(spec.spec || {});
  setOptions(registeredSelect, spec.available_registered_in || [], "Select registration", false);
  setOptions(colorSelect, spec.available_colors || [], "Select color", false);
}

function activeInspectionMode() {
  return document.querySelector("input[name='inspection_mode']:checked").value;
}

function updateInspectionMode() {
  const mode = activeInspectionMode();
  document.getElementById("scoreFields").classList.toggle("active", mode === "score");
  document.getElementById("sectionFields").classList.toggle("active", mode === "sections");
}

function serializeForm() {
  const formData = new FormData(form);
  const payload = {};
  for (const [key, value] of formData.entries()) {
    if (value === "") continue;
    payload[key] = value;
  }
  appendLockedSpecFields(payload);

  [
    "year",
    "mileage",
    "engine_capacity_cc",
    "inspection_score",
    "section_interior_pct",
    "section_engine_transmission_clutch_pct",
    "section_electrical_electronics_pct",
    "section_body_frame_accident_pct",
    "section_exterior_body_pct",
    "section_ac_heater_pct",
    "section_brakes_pct",
    "section_suspension_steering_pct",
    "section_tyres_pct"
  ].forEach((field) => {
    if (payload[field] != null) payload[field] = Number(payload[field]);
  });

  const mode = payload.inspection_mode;
  delete payload.inspection_mode;

  if (mode === "none") {
    delete payload.inspection_score;
    Object.keys(payload).filter((key) => key.startsWith("section_")).forEach((key) => delete payload[key]);
  } else if (mode === "score") {
    Object.keys(payload).filter((key) => key.startsWith("section_")).forEach((key) => delete payload[key]);
  } else if (mode === "sections") {
    delete payload.inspection_score;
  }

  return payload;
}

function renderResult(result) {
  resultCard.innerHTML = `
    <div class="result-hero">
      <div class="result-kicker">Predicted Price</div>
      <div class="result-price">PKR ${money(result.predicted_price)}</div>
      <div class="result-range">Range: PKR ${money(result.price_range_low)} to PKR ${money(result.price_range_high)}</div>
      <div class="pill-row">
        <span class="pill">Mode: ${result.prediction_mode}</span>
        <span class="pill">Confidence: ${Number(result.confidence_score).toFixed(2)}</span>
        <span class="pill">Prediction ID: ${result.prediction_id}</span>
      </div>
    </div>

    <div class="kv-list">
      <div class="kv-row"><span>Anchor Price</span><strong>PKR ${money(result.anchor_price)}</strong></div>
      <div class="kv-row"><span>Condition Adjusted Price</span><strong>PKR ${money(result.condition_adjusted_price)}</strong></div>
      <div class="kv-row"><span>Comparable Reference</span><strong>PKR ${money(result.comparable_reference_price)}</strong></div>
      <div class="kv-row"><span>Comparable Count</span><strong>${result.comparable_count}</strong></div>
      <div class="kv-row"><span>GitHub Mirror</span><strong>${result.logged_to_github ? "stored" : "not configured"}</strong></div>
    </div>
  `;
}

async function onSubmit(event) {
  event.preventDefault();
  const button = document.getElementById("predictButton");
  button.disabled = true;
  button.textContent = "Running...";

  try {
    const result = await getJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(serializeForm()),
    });
    renderResult(result);
    await loadRecentPredictions();
  } catch (error) {
    resultCard.innerHTML = `<div class="empty">${error.message}</div>`;
  } finally {
    button.disabled = false;
    button.textContent = "Run Prediction";
  }
}

function bindEvents() {
  makeSelect.addEventListener("change", onMakeChange);
  modelSelect.addEventListener("change", onModelChange);
  yearSelect.addEventListener("change", onYearChange);
  variantSelect.addEventListener("change", onVariantChange);
  document.querySelectorAll("input[name='inspection_mode']").forEach((node) => {
    node.addEventListener("change", updateInspectionMode);
  });
  form.addEventListener("submit", onSubmit);
}

bindEvents();
updateInspectionMode();
Promise.all([loadHealth(), initializeCatalog(), loadRecentPredictions()]).catch((error) => {
  healthStatus.textContent = `Startup failed: ${error.message}`;
});
