const form = document.getElementById("predictForm");
const healthStatus = document.getElementById("healthStatus");
const resultCard = document.getElementById("resultCard");
const recentPredictions = document.getElementById("recentPredictions");
const selectionSummary = document.getElementById("selectionSummary");
const specLockState = document.getElementById("specLockState");
const toast = document.getElementById("toast");

const makeSelect = document.getElementById("makeSelect");
const modelSelect = document.getElementById("modelSelect");
const yearSelect = document.getElementById("yearSelect");
const variantSelect = document.getElementById("variantSelect");
const registeredSelect = document.getElementById("registeredSelect");
const colorSelect = document.getElementById("colorSelect");
const mileageInput = document.getElementById("mileageInput");
const inspectionScoreInput = document.getElementById("inspectionScoreInput");

const transmissionInput = document.getElementById("transmissionInput");
const fuelTypeInput = document.getElementById("fuelTypeInput");
const assemblyInput = document.getElementById("assemblyInput");
const bodyTypeInput = document.getElementById("bodyTypeInput");
const engineCapacityInput = document.getElementById("engineCapacityInput");

let toastTimer = null;

function money(value) {
  if (value == null) return "N/A";
  return new Intl.NumberFormat("en-PK", { maximumFractionDigits: 0 }).format(value);
}

function safeText(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 3200);
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
  select.innerHTML = `<option value="">${placeholder}</option>${options.map((option) => (
    `<option value="${safeText(option)}">${safeText(option)}</option>`
  )).join("")}`;
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

function activeInspectionMode() {
  return document.querySelector("input[name='inspection_mode']:checked").value;
}

function updateStepState() {
  const step1Ready = Boolean(makeSelect.value && modelSelect.value && yearSelect.value && variantSelect.value);
  const step2Ready = step1Ready && Boolean(mileageInput.value && registeredSelect.value && colorSelect.value);

  document.querySelectorAll(".step-chip").forEach((chip) => chip.classList.remove("active"));
  if (!step1Ready) {
    document.querySelector('[data-step="1"]').classList.add("active");
  } else if (!step2Ready) {
    document.querySelector('[data-step="2"]').classList.add("active");
  } else {
    document.querySelector('[data-step="3"]').classList.add("active");
  }
}

function updateSelectionSummary() {
  const bits = [makeSelect.value, modelSelect.value, yearSelect.value, variantSelect.value].filter(Boolean);
  if (!bits.length) {
    selectionSummary.textContent = "Select a valid vehicle path to unlock factory specs and localized options.";
    specLockState.textContent = "Waiting for exact variant";
    updateStepState();
    return;
  }

  const detailBits = [];
  if (registeredSelect.value) detailBits.push(`registered in ${registeredSelect.value}`);
  if (colorSelect.value) detailBits.push(`${colorSelect.value.toLowerCase()} color`);
  if (mileageInput.value) detailBits.push(`${money(mileageInput.value)} km`);

  selectionSummary.textContent = `${bits.join(" / ")}${detailBits.length ? ` • ${detailBits.join(" • ")}` : ""}`;
  specLockState.textContent = [transmissionInput.value, fuelTypeInput.value, assemblyInput.value].filter(Boolean).length
    ? "Factory specs locked from catalog"
    : "Waiting for exact variant";
  updateStepState();
}

function updateInspectionMode() {
  const mode = activeInspectionMode();
  document.getElementById("scoreFields").classList.toggle("active", mode === "score");
  document.getElementById("sectionFields").classList.toggle("active", mode === "sections");
  updateStepState();
}

async function loadHealth() {
  const health = await getJson("/api/health");
  healthStatus.textContent = `Live service • catalog ${health.catalog_exists ? "ready" : "missing"} • GitHub mirror ${health.github_mirror_enabled ? "enabled" : "disabled"} • prediction store ${health.prediction_store_exists ? "ready" : "missing"}`;
}

async function loadRecentPredictions() {
  const rows = await getJson("/api/predictions/recent?limit=6");
  if (!rows.length) {
    recentPredictions.innerHTML = `<div class="empty-state">No predictions stored yet.</div>`;
    return;
  }

  recentPredictions.innerHTML = rows.map((row) => `
    <article class="recent-card">
      <div class="recent-head">
        <strong>#${safeText(row.id)}</strong>
        <span>${safeText(new Date(row.created_at).toLocaleString())}</span>
      </div>
      <div class="recent-grid">
        <span>Mode: ${safeText(row.prediction_mode)}</span>
        <span>Predicted price: PKR ${money(row.predicted_price)}</span>
        <span>Range: PKR ${money(row.price_range_low)} to PKR ${money(row.price_range_high)}</span>
        <span>Stored status: ${safeText(row.github_status || "not mirrored")}</span>
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
  updateSelectionSummary();
}

async function onMakeChange() {
  const options = await loadCatalogOptions({ make: makeSelect.value });
  setOptions(modelSelect, options.models || [], "Select model", !makeSelect.value);
  setOptions(yearSelect, [], "Select year", true);
  setOptions(variantSelect, [], "Select variant", true);
  setOptions(registeredSelect, [], "Select registration", true);
  setOptions(colorSelect, [], "Select color", true);
  setSpecFields();
  updateSelectionSummary();
}

async function onModelChange() {
  const options = await loadCatalogOptions({ make: makeSelect.value, model: modelSelect.value });
  setOptions(yearSelect, (options.years || []).map(String), "Select year", !modelSelect.value);
  setOptions(variantSelect, [], "Select variant", true);
  setOptions(registeredSelect, [], "Select registration", true);
  setOptions(colorSelect, [], "Select color", true);
  setSpecFields();
  updateSelectionSummary();
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
  updateSelectionSummary();
}

async function onVariantChange() {
  if (!variantSelect.value) {
    setSpecFields();
    updateSelectionSummary();
    return;
  }

  const spec = await getJson(`/api/catalog/spec?make=${encodeURIComponent(makeSelect.value)}&model=${encodeURIComponent(modelSelect.value)}&year=${encodeURIComponent(yearSelect.value)}&variant=${encodeURIComponent(variantSelect.value)}`);
  setSpecFields(spec.spec || {});
  setOptions(registeredSelect, spec.available_registered_in || [], "Select registration", false);
  setOptions(colorSelect, spec.available_colors || [], "Select color", false);
  updateSelectionSummary();
  showToast("Factory specification fields were fetched from the catalog and locked.");
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
    "section_tyres_pct",
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

function renderComparables(comparables = []) {
  if (!comparables.length) {
    return `<div class="empty-state">No high-quality comparable listings were used for this request.</div>`;
  }

  return `
    <div class="comparables-list">
      ${comparables.map((comp) => `
        <article class="comp-card">
          <h4>${safeText(comp.make)} ${safeText(comp.model)} ${safeText(comp.variant)}</h4>
          <div class="comp-meta">
            <strong>${safeText(comp.year)}</strong> • ${money(comp.mileage)} km • PKR ${money(comp.price)}
          </div>
          <div class="comp-meta">
            ${safeText(comp.registered_in)} • ${safeText(comp.assembly)} • distance score ${Number(comp.score || 0).toFixed(2)}
          </div>
          ${comp.listing_url ? `<a class="comp-link" href="${safeText(comp.listing_url)}" target="_blank" rel="noreferrer">Open listing</a>` : ""}
        </article>
      `).join("")}
    </div>
  `;
}

function renderResult(result) {
  const supportLabel = `${result.support_tier} support • ${result.exact_variant_rows} exact variant rows`;
  const supportCopy = result.support_tier === "thin"
    ? "Thin support: this result leans more on anchor and comparable statistics than dense exact-example learning."
    : result.support_tier === "moderate"
      ? "Moderate support: the system has useful exact-market evidence, but the interval should still be treated as an estimate."
      : "Strong support: the system has dense exact-market evidence for this selection.";
  const compQualityCopy = result.comp_quality_passed
    ? "Comparable quality passed. Market comps contributed to the final blend."
    : "Comparable quality did not pass the gate. The output leans on anchor and condition evidence only.";

  resultCard.innerHTML = `
    <div class="result-hero">
      <div class="result-kicker">Predicted Price</div>
      <div class="result-price">PKR ${money(result.predicted_price)}</div>
      <div class="result-range">Expected range: PKR ${money(result.price_range_low)} to PKR ${money(result.price_range_high)}</div>
      <div class="pill-row">
        <span class="pill">${safeText(result.prediction_mode)}</span>
        <span class="pill">Confidence index ${Number(result.confidence_index).toFixed(2)}</span>
        <span class="pill">${safeText(supportLabel)}</span>
        <span class="pill">${safeText(result.blend_mode)}</span>
      </div>
    </div>

    <div class="selection-summary" style="margin-top:16px;">
      ${safeText(supportCopy)} ${safeText(compQualityCopy)}
    </div>

    <div class="result-grid">
      <article class="result-stat">
        <span class="label">Anchor price</span>
        <strong>PKR ${money(result.anchor_price)}</strong>
      </article>
      <article class="result-stat">
        <span class="label">Condition-adjusted</span>
        <strong>PKR ${money(result.condition_adjusted_price)}</strong>
      </article>
      <article class="result-stat">
        <span class="label">Comparable reference</span>
        <strong>PKR ${money(result.comparable_reference_price)}</strong>
      </article>
      <article class="result-stat">
        <span class="label">Comparable count</span>
        <strong>${safeText(result.comparable_count)}</strong>
      </article>
      <article class="result-stat">
        <span class="label">Usable comparables</span>
        <strong>${safeText(result.usable_comp_count)}</strong>
      </article>
      <article class="result-stat">
        <span class="label">Prediction ID</span>
        <strong>#${safeText(result.prediction_id)}</strong>
      </article>
      <article class="result-stat">
        <span class="label">Storage</span>
        <strong>${result.logged_to_github ? "Stored to GitHub" : "Stored locally"}</strong>
      </article>
      <article class="result-stat">
        <span class="label">Inspection source</span>
        <strong>${safeText(result.inspection_source)}</strong>
      </article>
      <article class="result-stat">
        <span class="label">Inspection completeness</span>
        <strong>${Number(result.inspection_completeness || 0).toFixed(2)}</strong>
      </article>
    </div>

    <div class="comparables-block">
      <h3>Nearest comparable listings</h3>
      ${renderComparables(result.comparables)}
    </div>
  `;
}

function renderError(error) {
  resultCard.innerHTML = `
    <div class="empty-state">
      <div>
        <strong>Prediction failed.</strong><br>
        ${safeText(error.message)}
      </div>
    </div>
  `;
}

async function onSubmit(event) {
  event.preventDefault();
  const button = document.getElementById("predictButton");
  button.disabled = true;
  button.textContent = "Running prediction...";
  resultCard.innerHTML = `<div class="empty-state">Running model blend and comparable search...</div>`;

  try {
    const result = await getJson("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(serializeForm()),
    });
    renderResult(result);
    await loadRecentPredictions();
  } catch (error) {
    renderError(error);
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
  registeredSelect.addEventListener("change", updateSelectionSummary);
  colorSelect.addEventListener("change", updateSelectionSummary);
  mileageInput.addEventListener("input", updateSelectionSummary);
  inspectionScoreInput.addEventListener("input", updateStepState);
  document.querySelectorAll("input[name='inspection_mode']").forEach((node) => {
    node.addEventListener("change", updateInspectionMode);
  });
  form.addEventListener("submit", onSubmit);
}

bindEvents();
updateInspectionMode();
updateSelectionSummary();

Promise.all([loadHealth(), initializeCatalog(), loadRecentPredictions()]).catch((error) => {
  healthStatus.textContent = `Startup failed: ${error.message}`;
  renderError(error);
});
