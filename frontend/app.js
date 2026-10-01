// Numeric fields; the limits match the backend's validation rules.
const NUMERIC = [
  { name: "policy_deductable", min: 0, step: 1 },
  { name: "policy_annual_premium", min: 0, step: 0.01 },
  { name: "umbrella_limit", min: 0, step: 1 },
  { name: "capital-gains", min: 0, step: 1 },
  { name: "capital-loss", min: 0, step: 1 },          // enter as a positive amount
  { name: "incident_hour_of_the_day", min: 0, max: 23, step: 1 },
  { name: "number_of_vehicles_involved", min: 1, max: 4, step: 1 },
  { name: "bodily_injuries", min: 0, max: 2, step: 1 },
  { name: "witnesses", min: 0, max: 3, step: 1 },
  { name: "total_claim_amount", min: 0, step: 1 },
  { name: "auto_year", min: 1980, max: 2026, step: 1 },
];

const EXAMPLE_HIGH = {
  policy_state: "OH", policy_csl: "250/500", policy_deductable: 1000,
  policy_annual_premium: 1406.91, umbrella_limit: 0,
  insured_education_level: "MD", insured_hobbies: "sleeping",
  insured_relationship: "husband", "capital-gains": 53300, "capital-loss": 0,
  incident_type: "Single Vehicle Collision", collision_type: "Side Collision",
  incident_severity: "Major Damage", authorities_contacted: "Police",
  incident_state: "SC", incident_city: "Columbus",
  incident_hour_of_the_day: 5, number_of_vehicles_involved: 1,
  property_damage: "YES", bodily_injuries: 1, witnesses: 2,
  police_report_available: "YES", total_claim_amount: 71610,
  auto_make: "Saab", auto_year: 2004,
};
const EXAMPLE_LOW = {
  ...EXAMPLE_HIGH,
  incident_severity: "Minor Damage", total_claim_amount: 5500, "capital-gains": 0,
};

const form = document.getElementById("claim-form");
const submitBtn = document.getElementById("submit-btn");
const errorBox = document.getElementById("error");
let chart = null;

function pretty(name) { return name.replace(/[-_]/g, " "); }   // CSS capitalises it

function showError(msg) { errorBox.textContent = msg; errorBox.classList.remove("hidden"); }
function hideError() { errorBox.classList.add("hidden"); }

async function buildForm() {
  const res = await fetch(`${API_URL}/options`);
  const options = await res.json();
  const box = document.getElementById("fields");
  box.innerHTML = "";

  for (const [name, values] of Object.entries(options)) {
    const opts = values.map(v => `<option value="${v}">${v}</option>`).join("");
    box.insertAdjacentHTML("beforeend",
      `<div><label for="${name}">${pretty(name)}</label>` +
      `<select id="${name}" name="${name}">${opts}</select></div>`);
  }
  for (const f of NUMERIC) {
    const max = f.max !== undefined ? `max="${f.max}"` : "";
    box.insertAdjacentHTML("beforeend",
      `<div><label for="${f.name}">${pretty(f.name)}</label>` +
      `<input type="number" id="${f.name}" name="${f.name}" min="${f.min}" ${max} step="${f.step}" required></div>`);
  }
}

function fill(values) {
  for (const [name, v] of Object.entries(values)) {
    const el = form.elements[name];
    if (el) el.value = v;
  }
}

function collect() {
  const data = {};
  for (const el of form.elements) {
    if (!el.name) continue;
    data[el.name] = el.type === "number" ? Number(el.value) : el.value;
  }
  return data;
}

function errorText(body) {
  const d = body.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) return d.map(x => `${x.loc[x.loc.length - 1]}: ${x.msg}`).join("; ");
  return "Request failed";
}

function showResult(r) {
  document.getElementById("result").classList.remove("hidden");
  document.getElementById("score").textContent = (r.fraud_probability * 100).toFixed(1) + "%";

  const badge = document.getElementById("badge");
  badge.textContent = r.flagged ? "Flagged for human review" : "Not flagged";
  badge.className = "badge " + (r.flagged ? "flag" : "ok");

  document.getElementById("threshold-note").textContent =
    `Claims scoring ${(r.threshold * 100).toFixed(0)}% or higher are sent for review. ` +
    `A flag is a prompt for a person to look, not a finding of fraud.`;

  const rows = [
    ...r.top_features.map(f => ({ label: `${f.feature} = ${f.value}`, shap: f.shap })),
    { label: "all other features", shap: r.other_features_shap },
  ];

  if (chart) chart.destroy();
  chart = new Chart(document.getElementById("chart"), {
    type: "bar",
    data: {
      labels: rows.map(x => x.label),
      datasets: [{
        data: rows.map(x => x.shap),
        backgroundColor: rows.map(x => (x.shap >= 0 ? "#d64545" : "#3b82c4")),
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.parsed.x > 0 ? "+" : ""}${ctx.parsed.x} (log-odds)` } },
      },
      scales: { x: { title: { display: true, text: "pushes score down  <----  |  ---->  pushes score up" } } },
    },
  });
  document.getElementById("result").scrollIntoView({ behavior: "smooth" });
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  submitBtn.disabled = true;
  submitBtn.textContent = "Scoring...";
  try {
    const res = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collect()),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(errorText(body));
    showResult(body);
  } catch (err) {
    showError(err.message.includes("Failed to fetch")
      ? `Cannot reach the API at ${API_URL}. Is uvicorn running?`
      : err.message);
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Score claim";
  }
});

document.getElementById("fill-high").addEventListener("click", () => fill(EXAMPLE_HIGH));
document.getElementById("fill-low").addEventListener("click", () => fill(EXAMPLE_LOW));

buildForm()
  .then(() => fill(EXAMPLE_HIGH))
  .catch(() => {
    document.getElementById("fields").textContent = "";
    showError(`Cannot reach the API at ${API_URL}. Start it with uvicorn, then reload this page.`);
  });