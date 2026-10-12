const $ = (id) => document.getElementById(id);
const DAY_ORDER = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
const MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
let summaryData = null;

async function getJSON(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Request failed (${response.status})`);
  return data;
}
function fillSelect(id, values, firstLabel = null) {
  const select = $(id);
  select.innerHTML = "";
  if (firstLabel !== null) {
    const opt = document.createElement("option"); opt.value = "All"; opt.textContent = firstLabel; select.appendChild(opt);
  }
  values.forEach(value => {
    const opt = document.createElement("option"); opt.value = value; opt.textContent = value; select.appendChild(opt);
  });
}
function showToast(message) {
  const toast = $("toast"); toast.textContent = message; toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 4200);
}
function baseLayout() {
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: {family: "DM Sans, sans-serif", color: "#aab7d0", size: 10},
    margin: {l: 45, r: 14, t: 8, b: 42},
    xaxis: {gridcolor: "#26334b", zerolinecolor: "#26334b", linecolor: "#26334b"},
    yaxis: {gridcolor: "#26334b", zerolinecolor: "#26334b", linecolor: "#26334b"},
    showlegend: false
  };
}
function renderCharts(data) {
  const cats = [...(data.top_categories || [])].reverse();
  Plotly.react("categoryChart", [{
    type: "bar", orientation: "h", x: cats.map(x => x.count), y: cats.map(x => x.category),
    marker: {color: cats.map((_,i) => `rgba(255,107,134,${0.35 + 0.6*(i+1)/Math.max(cats.length,1)})`), line: {width:0}},
    hovertemplate: "%{y}<br>%{x:,} reports<extra></extra>"
  }], {...baseLayout(), margin:{l:125,r:12,t:8,b:30}, xaxis:{...baseLayout().xaxis,tickformat:",d"}}, {responsive:true, displayModeBar:false});

  const days = [...(data.days || [])].sort((a,b)=>DAY_ORDER.indexOf(a.day)-DAY_ORDER.indexOf(b.day));
  Plotly.react("dayChart", [{
    type:"bar", x:days.map(x=>x.day.slice(0,3)), y:days.map(x=>x.count),
    marker:{color:"#a78bfa",line:{width:0}}, hovertemplate:"%{x}<br>%{y:,} reports<extra></extra>"
  }], {...baseLayout(), margin:{l:45,r:10,t:8,b:35}, yaxis:{...baseLayout().yaxis,tickformat:"~s"}}, {responsive:true,displayModeBar:false});

  const hours = [...(data.hours || [])].sort((a,b)=>a.hour-b.hour);
  Plotly.react("hourChart", [{
    type:"scatter", mode:"lines+markers", x:hours.map(x=>x.hour), y:hours.map(x=>x.count),
    line:{color:"#ff7c96",width:3}, marker:{color:"#ff9eb0",size:5},
    fill:"tozeroy", fillcolor:"rgba(255,107,134,.09)", hovertemplate:"Hour %{x}:00<br>%{y:,} reports<extra></extra>"
  }], {...baseLayout(), margin:{l:45,r:10,t:8,b:35}, xaxis:{...baseLayout().xaxis,dtick:3}, yaxis:{...baseLayout().yaxis,tickformat:"~s"}}, {responsive:true,displayModeBar:false});

  const months = [...(data.months || [])].sort((a,b)=>a.month-b.month);
  Plotly.react("monthChart", [{
    type:"bar", x:months.map(x=>MONTH_NAMES[x.month-1]), y:months.map(x=>x.count),
    marker:{color:"#65b8ff",line:{width:0}}, hovertemplate:"%{x}<br>%{y:,} reports<extra></extra>"
  }], {...baseLayout(), margin:{l:45,r:10,t:8,b:35}, yaxis:{...baseLayout().yaxis,tickformat:"~s"}}, {responsive:true,displayModeBar:false});
}
function renderMetrics(data) {
  $("totalMetric").textContent = Number(data.total || 0).toLocaleString();
  $("categoryMetric").textContent = summaryData.category_count;
  $("districtMetric").textContent = summaryData.district_count;
  const top = (data.top_categories || [])[0];
  $("topMetric").textContent = top ? top.category : "N/A";
  $("topCount").textContent = top ? `${Number(top.count).toLocaleString()} historical reports` : "No records match filters";
}
async function loadAnalytics() {
  const params = new URLSearchParams({
    district: $("districtFilter").value,
    category: $("categoryFilter").value,
    year_from: $("yearFrom").value,
    year_to: $("yearTo").value
  });
  try {
    const data = await getJSON(`/api/analytics?${params.toString()}`);
    renderMetrics(data); renderCharts(data);
  } catch (err) { showToast(err.message); }
}
async function refreshModelStatus() {
  const el = $("modelStatus");
  try {
    const status = await getJSON("/api/model-status");
    if (status.ready) {
      el.className = "model-status ready";
      el.textContent = "✓ " + status.message;
    } else if (status.training) {
      el.className = "model-status";
      el.innerHTML = '<span class="spinner"></span> Model is training in the backend. This can take a little while on first run.';
      setTimeout(refreshModelStatus, 4000);
    } else {
      el.className = "model-status";
      el.innerHTML = '<span class="spinner"></span> ' + (status.message || "Preparing model...");
      setTimeout(refreshModelStatus, 4000);
    }
  } catch (err) {
    el.className = "model-status error"; el.textContent = err.message;
  }
}
async function predict() {
  const button = $("predictBtn");
  const result = $("predictionResult");
  button.disabled = true; button.innerHTML = '<span class="spinner"></span> Working…';
  result.innerHTML = '<div class="result-empty"><div class="result-orb"><span class="spinner"></span></div><h3>Preparing your estimate</h3><p>The backend is checking the trained model. The first run may take a minute.</p></div>';
  try {
    let status = await getJSON("/api/model-status");
    if (!status.ready) {
      await getJSON("/api/train", {method:"POST"});
      for (let i=0; i<90; i++) {
        await new Promise(resolve => setTimeout(resolve, 2000));
        status = await getJSON("/api/model-status");
        if (status.ready || (!status.training && !status.ready)) break;
      }
    }
    if (!status.ready) throw new Error(status.message || "Model is not ready yet. Please try again.");
    const payload = {
      district: $("predDistrict").value,
      day: $("predDay").value,
      hour: Number($("predHour").value),
      month: Number($("predMonth").value)
    };
    const prediction = await getJSON("/api/predict", {
      method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)
    });
    result.innerHTML = "";
    const wrap = document.createElement("div"); wrap.className = "result-content";
    const title = document.createElement("div"); title.className = "result-title"; title.textContent = "TOP MODEL ESTIMATE";
    const category = document.createElement("div"); category.className = "predicted-category"; category.textContent = prediction.predicted_category;
    const note = document.createElement("div"); note.className = "confidence-note";
    note.textContent = `Model score: ${(prediction.predictions[0].confidence*100).toFixed(1)}% · Not a real-world probability`;
    wrap.append(title, category, note);
    const sub = document.createElement("div"); sub.className = "result-title"; sub.textContent = "Top 5 model outputs"; sub.style.marginTop = "20px"; wrap.appendChild(sub);
    prediction.predictions.forEach(item => {
      const row = document.createElement("div"); row.className = "prob-row";
      const label = document.createElement("div"); label.className = "prob-label";
      const name = document.createElement("span"); name.textContent = item.category;
      const value = document.createElement("span"); value.textContent = `${(item.confidence*100).toFixed(1)}%`;
      label.append(name,value);
      const track = document.createElement("div"); track.className = "prob-track";
      const fill = document.createElement("div"); fill.className = "prob-fill"; fill.style.width = `${Math.min(100,item.confidence*100)}%`;
      track.appendChild(fill); row.append(label,track); wrap.appendChild(row);
    });
    const disclaimer = document.createElement("p"); disclaimer.className = "confidence-note"; disclaimer.style.marginTop = "18px"; disclaimer.textContent = prediction.note; wrap.appendChild(disclaimer);
    result.appendChild(wrap);
    $("modelStatus").className = "model-status ready"; $("modelStatus").textContent = "✓ " + status.message;
  } catch (err) {
    result.innerHTML = `<div class="result-empty"><div class="result-orb">!</div><h3>Could not generate estimate</h3><p>${escapeHTML(err.message)}</p></div>`;
    showToast(err.message);
    refreshModelStatus();
  } finally {
    button.disabled = false; button.innerHTML = 'Generate estimate <span>→</span>';
  }
}
function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
}
async function init() {
  try {
    summaryData = await getJSON("/api/summary");
    $("categoryMetric").textContent = summaryData.category_count;
    $("districtMetric").textContent = summaryData.district_count;
    fillSelect("districtFilter", summaryData.districts, "All districts");
    fillSelect("categoryFilter", summaryData.categories, "All categories");
    fillSelect("predDistrict", summaryData.districts);
    fillSelect("yearFrom", summaryData.years);
    fillSelect("yearTo", summaryData.years);
    $("yearFrom").value = summaryData.years[0];
    $("yearTo").value = summaryData.years[summaryData.years.length-1];
    const hourOptions = Array.from({length:24},(_,h)=>({value:h,label:`${String(h).padStart(2,"0")}:00`}));
    $("predHour").innerHTML = hourOptions.map(o=>`<option value="${o.value}" ${o.value===12?"selected":""}>${o.label}</option>`).join("");
    await loadAnalytics();
    refreshModelStatus();
  } catch (err) {
    showToast(err.message);
    document.querySelector(".main").innerHTML = `<div class="panel"><h2>Backend connection error</h2><p>${escapeHTML(err.message)}</p><p>Make sure Flask is running and train.csv is beside app.py.</p></div>`;
  }
}
$("applyFilters").addEventListener("click", loadAnalytics);
$("resetFilters").addEventListener("click", () => {
  $("districtFilter").value = "All"; $("categoryFilter").value = "All";
  $("yearFrom").value = summaryData.years[0]; $("yearTo").value = summaryData.years[summaryData.years.length-1];
  loadAnalytics();
});
$("predictBtn").addEventListener("click", predict);
window.addEventListener("resize", () => {
  ["categoryChart","dayChart","hourChart","monthChart"].forEach(id => { const el=$(id); if(el && el.data) Plotly.Plots.resize(el); });
});
init();
