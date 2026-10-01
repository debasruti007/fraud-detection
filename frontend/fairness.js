const GROUP_NAMES = {
  policy_state: "Policy state",
  insured_sex: "Sex",
  age_bracket: "Age bracket",
  incident_state: "Incident state",
};

const pct = (v) => (v === null || v === undefined ? "-" : (v * 100).toFixed(1) + "%");
const errorBox = document.getElementById("error");
const select = document.getElementById("group-select");
let report = null;
let chart = null;

function showError(msg) { errorBox.textContent = msg; errorBox.classList.remove("hidden"); }

function renderSummary(metrics) {
  const t = metrics.test;
  document.getElementById("summary").textContent =
    `Held-out test set (${metrics.test_rows} claims): ROC-AUC ${t.roc_auc.toFixed(3)}. ` +
    `At the ${(metrics.threshold * 100).toFixed(0)}% threshold the model flags ${pct(t.flagged)} of claims, ` +
    `${pct(t.precision)} of flagged claims are fraud, and it catches ${pct(t.recall)} of all fraud. ` +
    `The audit below covers all ${metrics.train_rows + metrics.test_rows} claims, ` +
    `each scored by a model that had not seen it.`;
}

function renderGroup(col) {
  const info = report.groups[col];
  const rows = info.rows;

  // Ratios and pass / fail message
  const box = document.getElementById("ratio-box");
  if (info.not_flagged_ratio === null) {
    box.innerHTML = `<div class="warn">Not enough large groups here to compute a ratio.</div>`;
  } else {
    const ok = info.passes_four_fifths;
    let html = `<p>Flag-rate ratio (lowest ÷ highest): <b>${info.flag_rate_ratio}</b> &nbsp;·&nbsp; ` +
      `Not-flagged ratio (80% rule): <b class="${ok ? "pass" : "fail"}">${info.not_flagged_ratio} ` +
      `${ok ? "- passes" : "- below 0.80"}</b></p>`;
    if (!ok) {
      html += `<div class="warn">This breakdown falls below the 0.80 rule of thumb. That is a reason to look
        closer, not proof of bias: check whether the group really has more fraud ("Actually fraud"), and
        whether honest claims in it are flagged more often (last column). Flagged claims go to a person, not
        to an automatic rejection.</div>`;
    }
    box.innerHTML = html;
  }

  // Bar chart: faded bars for groups that are too small to trust
  const shade = (hex, r) => (r.reliable ? hex : hex + "55");
  if (chart) chart.destroy();
  chart = new Chart(document.getElementById("chart"), {
    type: "bar",
    data: {
      labels: rows.map((r) => `${r.group} (n=${r.n})`),
      datasets: [
        { label: "Flagged", data: rows.map((r) => (r.flag_rate ?? 0) * 100),
          backgroundColor: rows.map((r) => shade("#14213d", r)) },
        { label: "Actually fraud", data: rows.map((r) => (r.fraud_rate ?? 0) * 100),
          backgroundColor: rows.map((r) => shade("#fca311", r)) },
        { label: "Honest claims wrongly flagged", data: rows.map((r) => (r.false_flag_rate ?? 0) * 100),
          backgroundColor: rows.map((r) => shade("#d64545", r)) },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: { callbacks: { label: (c) => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}%` } },
      },
      scales: {
        y: { min: 0, title: { display: true, text: "% of the group's claims" },
             ticks: { callback: (v) => v + "%" } },
      },
    },
  });

  // Table
  document.getElementById("table-body").innerHTML = rows.map((r) => `
    <tr class="${r.reliable ? "" : "small"}">
      <td>${r.group}${r.reliable ? "" : " (too small)"}</td>
      <td>${r.n}</td>
      <td>${pct(r.flag_rate)}</td>
      <td>${pct(r.fraud_rate)}</td>
      <td>${pct(r.catch_rate)}</td>
      <td>${pct(r.false_flag_rate)}</td>
    </tr>`).join("");
}

async function init() {
  const [fRes, mRes] = await Promise.all([
    fetch(`${API_URL}/fairness`),
    fetch(`${API_URL}/metrics`),
  ]);
  if (!fRes.ok || !mRes.ok) throw new Error("The API returned an error.");
  report = await fRes.json();
  renderSummary(await mRes.json());

  select.innerHTML = Object.keys(report.groups)
    .map((c) => `<option value="${c}">${GROUP_NAMES[c] || c}</option>`).join("");
  select.addEventListener("change", () => renderGroup(select.value));
  renderGroup(select.value);
}

init().catch(() => {
  document.getElementById("summary").textContent = "";
  showError(`Cannot reach the API at ${API_URL}. Is uvicorn running? Start it, then reload this page.`);
});