const stockSelect = document.getElementById("stockSelect");
const rangeSelect = document.getElementById("rangeSelect");
const intervalSelect = document.getElementById("intervalSelect");
const refreshSelect = document.getElementById("refreshSelect");
const refreshButton = document.getElementById("refreshButton");

let refreshHandle = null;

async function loadConfig() {
  const response = await fetch("/api/config");
  const config = await response.json();

  stockSelect.innerHTML = config.stocks
    .map((stock) => `<option value="${stock.symbol}">${stock.name} (${stock.symbol})</option>`)
    .join("");

  document.getElementById("marketStatus").textContent = config.marketStatus;
  stockSelect.value = "RELIANCE.NS";
}

function formatNumber(value) {
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 }).format(value);
}

function moveClass(value) {
  return value >= 0 ? "positive" : "negative";
}

function setMetric(id, text) {
  document.getElementById(id).textContent = text;
}

function renderSignal(signal) {
  const badge = document.getElementById("signalBadge");
  badge.textContent = signal.signal;
  badge.className = `signal-badge ${signal.signal.toLowerCase()}`;

  const reasonList = document.getElementById("reasonList");
  reasonList.innerHTML = signal.reasoning.map((item) => `<li>${item}</li>`).join("");
}

function renderTable(targetId, headers, rows) {
  const target = document.getElementById(targetId);
  if (!rows.length) {
    target.innerHTML = "<p>No data available right now.</p>";
    return;
  }

  const headHtml = headers.map((header) => `<th>${header}</th>`).join("");
  const bodyHtml = rows
    .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("");

  target.innerHTML = `<table><thead><tr>${headHtml}</tr></thead><tbody>${bodyHtml}</tbody></table>`;
}

function renderChart(payload) {
  const rows = payload.history;
  const timestamps = rows.map((row) => row.timestamp);

  const candleTrace = {
    x: timestamps,
    open: rows.map((row) => row.open),
    high: rows.map((row) => row.high),
    low: rows.map((row) => row.low),
    close: rows.map((row) => row.close),
    type: "candlestick",
    name: "Price",
    increasing: { line: { color: "#138a4b" } },
    decreasing: { line: { color: "#d64545" } },
    xaxis: "x",
    yaxis: "y",
  };

  const volumeTrace = {
    x: timestamps,
    y: rows.map((row) => row.volume),
    type: "bar",
    name: "Volume",
    marker: {
      color: rows.map((row) => (row.close >= row.open ? "#95c11f" : "#f59e0b")),
    },
    xaxis: "x2",
    yaxis: "y2",
  };

  const forecastTrace = {
    x: [rows[rows.length - 1].timestamp, payload.predictedPoint.timestamp],
    y: [rows[rows.length - 1].close, payload.predictedPoint.price],
    mode: "lines+markers",
    type: "scatter",
    name: "Forecast",
    line: { color: "#111827", dash: "dot", width: 2 },
    marker: { size: 10, color: "#111827" },
    xaxis: "x",
    yaxis: "y",
  };

  const layout = {
    height: 620,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(255,255,255,0.75)",
    margin: { l: 20, r: 20, t: 20, b: 20 },
    legend: { orientation: "h", x: 0, y: 1.07 },
    xaxis: {
      domain: [0, 1],
      anchor: "y",
      rangeslider: { visible: false },
      showgrid: false,
    },
    yaxis: {
      domain: [0.32, 1],
      title: "Price",
      gridcolor: "rgba(20,37,59,0.08)",
    },
    xaxis2: {
      domain: [0, 1],
      anchor: "y2",
      showgrid: false,
    },
    yaxis2: {
      domain: [0, 0.22],
      title: "Volume",
      gridcolor: "rgba(20,37,59,0.08)",
    },
  };

  Plotly.newPlot("priceChart", [candleTrace, forecastTrace, volumeTrace], layout, {
    responsive: true,
    displaylogo: false,
  });
}

async function loadOverview() {
  const symbols = Array.from(stockSelect.options)
    .slice(0, 5)
    .map((option) => `symbols=${encodeURIComponent(option.value)}`)
    .join("&");

  const response = await fetch(`/api/overview?range=${rangeSelect.value}&interval=${intervalSelect.value}&${symbols}`);
  const overview = await response.json();

  renderTable(
    "indexTable",
    ["Index", "Value", "Move %"],
    overview.indices.map((row) => [
      row.name,
      formatNumber(row.value),
      `<span class="${moveClass(row.movePct)}">${row.movePct.toFixed(2)}%</span>`,
    ]),
  );

  renderTable(
    "watchlistTable",
    ["Symbol", "Last", "Move %", "Forecast %", "Signal", "Confidence"],
    overview.watchlist.map((row) => [
      row.symbol,
      formatNumber(row.lastPrice),
      `<span class="${moveClass(row.liveChangePct)}">${row.liveChangePct.toFixed(2)}%</span>`,
      `<span class="${moveClass(row.predictedReturnPct)}">${row.predictedReturnPct.toFixed(2)}%</span>`,
      row.signal,
      `${row.confidence.toFixed(1)}%`,
    ]),
  );
}

async function loadDashboard() {
  refreshButton.disabled = true;
  refreshButton.textContent = "Loading...";

  try {
    const symbol = stockSelect.value || "RELIANCE.NS";
    const response = await fetch(`/api/analyze?symbol=${encodeURIComponent(symbol)}&range=${rangeSelect.value}&interval=${intervalSelect.value}`);
    const payload = await response.json();

    if (payload.error) {
      throw new Error(payload.error);
    }

    setMetric("lastPrice", formatNumber(payload.lastPrice));
    setMetric("liveChange", `${payload.liveChangePct >= 0 ? "+" : ""}${payload.liveChangePct.toFixed(2)}% live move`);
    setMetric("forecastPrice", formatNumber(payload.signal.predictedPrice));
    setMetric("forecastMove", `${payload.signal.predictedReturnPct >= 0 ? "+" : ""}${payload.signal.predictedReturnPct.toFixed(2)}% next move`);
    setMetric("recommendation", payload.signal.signal);
    setMetric("confidence", `${payload.signal.confidence.toFixed(1)}% confidence`);
    setMetric("accuracy", `${payload.signal.accuracyPct.toFixed(1)}%`);
    document.getElementById("marketStatus").textContent = payload.marketStatus;
    document.getElementById("lastUpdated").textContent = `Last update: ${new Date(payload.lastUpdated).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" })}`;

    renderSignal(payload.signal);
    renderChart(payload);
    await loadOverview();
  } catch (error) {
    document.getElementById("reasonList").innerHTML = `<li>${error.message}</li>`;
  } finally {
    refreshButton.disabled = false;
    refreshButton.textContent = "Refresh live data";
  }
}

function applyRefreshTimer() {
  if (refreshHandle) {
    clearInterval(refreshHandle);
    refreshHandle = null;
  }

  const seconds = Number(refreshSelect.value);
  if (seconds > 0) {
    refreshHandle = setInterval(loadDashboard, seconds * 1000);
  }
}

refreshButton.addEventListener("click", loadDashboard);
stockSelect.addEventListener("change", loadDashboard);
rangeSelect.addEventListener("change", loadDashboard);
intervalSelect.addEventListener("change", loadDashboard);
refreshSelect.addEventListener("change", applyRefreshTimer);
refreshSelect.addEventListener("change", loadDashboard);

async function init() {
  await loadConfig();
  applyRefreshTimer();
  await loadDashboard();
}

init();
