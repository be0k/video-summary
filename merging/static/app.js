const $ = (id) => document.getElementById(id);

const ui = {
  navButtons: [...document.querySelectorAll(".nav-button")],
  pages: [...document.querySelectorAll(".page")],
  pageEyebrow: $("pageEyebrow"),
  pageTitle: $("pageTitle"),
  file: $("videoFile"),
  sourceVideo: $("sourceVideo"),
  sourceEmpty: $("sourceEmpty"),
  trimmedVideo: $("trimmedVideo"),
  retrievalVideo: $("retrievalVideo"),
  summaryVideo: $("summaryVideo"),
  uploadBtn: $("uploadBtn"),
  runBtn: $("runBtn"),
  refreshBtn: $("refreshBtn"),
  useCurrent: $("useCurrent"),
  useEndCurrent: $("useEndCurrent"),
  startRange: $("startRange"),
  endRange: $("endRange"),
  startSeconds: $("startSeconds"),
  endSeconds: $("endSeconds"),
  startLabel: $("startLabel"),
  endLabel: $("endLabel"),
  lingbotFps: $("lingbotFps"),
  summaryRatio: $("summaryRatio"),
  durationLabel: $("durationLabel"),
  jobLine: $("jobLine"),
  statePill: $("statePill"),
  stageMetric: $("stageMetric"),
  fpsMetric: $("fpsMetric"),
  startMetric: $("startMetric"),
  logBox: $("logBox"),
  rawViewerState: $("rawViewerState"),
  postViewerState: $("postViewerState"),
  trimmedState: $("trimmedState"),
  summaryState: $("summaryState"),
  openRawBtn: $("openRawBtn"),
  openPostBtn: $("openPostBtn"),
  rawViewerFrame: $("rawViewerFrame"),
  postViewerFrame: $("postViewerFrame"),
  rawExternalLink: $("rawExternalLink"),
  postExternalLink: $("postExternalLink"),
  reloadJobsBtn: $("reloadJobsBtn"),
  runningJobLine: $("runningJobLine"),
  jobList: $("jobList"),
  loadAnalysisBtn: $("loadAnalysisBtn"),
  scoreCanvas: $("scoreCanvas"),
  segmentList: $("segmentList"),
  shotList: $("shotList"),
  retrievalState: $("retrievalState"),
  retrievalQuery: $("retrievalQuery"),
  indexRetrievalBtn: $("indexRetrievalBtn"),
  queryRetrievalBtn: $("queryRetrievalBtn"),
  retrievedMoments: $("retrievedMoments"),
  retrievalSaliencyCanvas: $("retrievalSaliencyCanvas"),
  highlightFrames: $("highlightFrames"),
};

const app = {
  job: null,
  pollTimer: null,
  jobs: [],
  rawViewerUrl: "",
  postViewerUrl: "",
  retrievalSaliency: [],
};

function secondsLabel(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number.toFixed(1)}s`;
}

function scoreLabel(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return number.toFixed(3);
}

function fileState(url) {
  return url ? "ready" : "-";
}

function setPage(name) {
  const pageName = name || "upload";
  ui.navButtons.forEach((button) => button.classList.toggle("active", button.dataset.page === pageName));
  ui.pages.forEach((page) => page.classList.toggle("active", page.id === `page-${pageName}`));
  const current = ui.pages.find((page) => page.id === `page-${pageName}`) || ui.pages[0];
  ui.pageTitle.textContent = current.dataset.title || "Drone Mapping";
  ui.pageEyebrow.textContent = current.dataset.eyebrow || pageName;
  if (location.hash !== `#${pageName}`) {
    history.replaceState(null, "", `#${pageName}`);
  }
}

function setStart(value) {
  const max = Number(ui.startRange.max || 0);
  const end = Number(ui.endSeconds.value || max || 0);
  const upper = max > 0 ? Math.max(0, Math.min(max, end - 0.1)) : Number.MAX_SAFE_INTEGER;
  const next = Math.min(Math.max(Number(value) || 0, 0), upper);
  ui.startRange.value = String(next);
  ui.startSeconds.value = next.toFixed(1);
  ui.startLabel.textContent = secondsLabel(next);
}

function setEnd(value) {
  const max = Number(ui.endRange.max || 0);
  const start = Number(ui.startSeconds.value || 0);
  const lower = Math.min(max || Number.MAX_SAFE_INTEGER, start + 0.1);
  const fallback = max > 0 ? max : Number(value || 0);
  const next = Math.min(Math.max(Number(value) || fallback, lower), max || Number.MAX_SAFE_INTEGER);
  ui.endRange.value = String(next);
  ui.endSeconds.value = next.toFixed(1);
  ui.endLabel.textContent = secondsLabel(next);
}

function setDuration(duration) {
  const value = Number(duration);
  if (!Number.isFinite(value) || value <= 0) {
    ui.durationLabel.textContent = "-";
    return;
  }
  ui.startRange.max = String(value);
  ui.endRange.max = String(value);
  ui.startSeconds.max = String(value);
  ui.endSeconds.max = String(value);
  ui.startRange.disabled = false;
  ui.endRange.disabled = false;
  ui.startSeconds.disabled = false;
  ui.endSeconds.disabled = false;
  ui.useCurrent.disabled = false;
  ui.useEndCurrent.disabled = false;
  ui.durationLabel.textContent = secondsLabel(value);
  if (!Number(ui.endSeconds.value) || Number(ui.endSeconds.value) > value) {
    ui.endSeconds.value = value.toFixed(1);
    ui.endRange.value = String(value);
  }
  setStart(Math.min(Number(ui.startSeconds.value || 0), value));
  setEnd(Math.min(Number(ui.endSeconds.value || value), value));
}

function resetClipControls() {
  ui.durationLabel.textContent = "-";
  for (const input of [ui.startRange, ui.endRange, ui.startSeconds, ui.endSeconds]) {
    input.value = "0";
    input.max = "0";
    input.disabled = true;
  }
  ui.startLabel.textContent = "0.0s";
  ui.endLabel.textContent = "0.0s";
  ui.useCurrent.disabled = true;
  ui.useEndCurrent.disabled = true;
}

function setVideo(video, url) {
  if (!video) return;
  if (!url) {
    video.pause();
    video.removeAttribute("src");
    video.dataset.url = "";
    video.load();
    return;
  }
  if (video.dataset.url === url) return;
  video.dataset.url = url;
  video.src = url;
  video.load();
}

function seekRetrievalVideo(second) {
  const time = Math.max(0, Number(second) || 0);
  if (!ui.retrievalVideo.src && app.job?.artifacts?.trimmed_video_url) {
    setVideo(ui.retrievalVideo, app.job.artifacts.trimmed_video_url);
  }
  const video = ui.retrievalVideo?.src ? ui.retrievalVideo : ui.trimmedVideo?.src ? ui.trimmedVideo : null;
  if (!video) return;
  setPage("retrieval");
  video.currentTime = time;
  video.play().catch(() => {});
}

function setSourceVisible(hasVideo) {
  ui.sourceEmpty.classList.toggle("hidden", Boolean(hasVideo));
}

function setStatePill(state) {
  const next = state || "idle";
  ui.statePill.textContent = next;
  ui.statePill.className = "status-pill";
  if (["running", "partial", "failed"].includes(next)) {
    ui.statePill.classList.add(next);
  }
}

function setBusy(isBusy) {
  ui.uploadBtn.disabled = isBusy;
  ui.runBtn.disabled = isBusy || !app.job;
  ui.refreshBtn.disabled = !app.job;
}

function setLink(anchor, url) {
  if (!url) {
    anchor.removeAttribute("href");
    anchor.classList.add("disabled");
    return;
  }
  anchor.href = url;
  anchor.classList.remove("disabled");
}

async function readJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function loadJobs() {
  const payload = await readJson(await fetch("/api/jobs"));
  app.jobs = payload.jobs || [];
  if (app.job && !app.jobs.some((job) => job.id === app.job.id)) {
    clearCurrentJob();
  }
  renderJobs();
}

function renderJobs() {
  const running = app.jobs.find((job) => job.is_running_local || job.state === "running" || job.state === "stopping");
  ui.runningJobLine.textContent = running ? `running: ${running.filename || running.id}` : "running: none";
  ui.jobList.innerHTML = "";
  for (const job of app.jobs) {
    const row = document.createElement("div");
    row.className = `job-row${app.job?.id === job.id ? " active" : ""}`;
    const select = document.createElement("button");
    select.type = "button";
    select.className = "job-select";
    select.innerHTML = `<strong>${escapeHtml(job.filename || job.id)}</strong><span>${job.state} · ${job.id}</span>`;
    select.addEventListener("click", () => selectJob(job));
    const del = document.createElement("button");
    del.type = "button";
    del.className = "job-delete";
    del.textContent = "×";
    del.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteJob(job);
    });
    row.append(select, del);
    ui.jobList.append(row);
  }
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function selectJob(job) {
  if (app.job?.id === job.id) return;
  if (app.job && app.job.state !== "complete") {
    if (app.job.state === "running" || app.job.is_running_local) {
      const stopCurrent = confirm("현재 실행 중인 데이터가 있습니다. 중단하고 바꿀까요? 취소하면 실행은 유지하고 화면만 바꿉니다.");
      if (stopCurrent) {
        await fetch(`/api/jobs/${app.job.id}/stop`, { method: "POST" });
      }
    } else if (!confirm("현재 선택한 데이터가 complete 상태가 아닙니다. 그래도 다른 데이터로 바꿀까요?")) {
      return;
    }
  }
  const next = await readJson(await fetch(`/api/jobs/${job.id}`));
  applyStatus(next);
  setPage("upload");
}

async function deleteJob(job) {
  const running = job.is_running_local || job.state === "running" || job.state === "stopping";
  const ok = running
    ? confirm("이 데이터는 실행 중입니다. 중단하고 삭제할까요?")
    : confirm(`${job.filename || job.id} 데이터를 삭제할까요?`);
  if (!ok) return;
  await readJson(await fetch(`/api/jobs/${job.id}${running ? "?force=1" : ""}`, { method: "DELETE" }));
  if (app.job?.id === job.id) {
    clearCurrentJob();
  }
  await loadJobs();
}

function clearCurrentJob() {
  app.job = null;
  localStorage.removeItem("droneMappingJobId");
  app.rawViewerUrl = "";
  app.postViewerUrl = "";
  stopPolling();
  setStatePill("idle");
  ui.jobLine.textContent = "no job";
  ui.stageMetric.textContent = "-";
  ui.fpsMetric.textContent = "-";
  ui.startMetric.textContent = "-";
  ui.logBox.textContent = "";
  setViewer("raw", "");
  setViewer("post", "");
  setVideo(ui.sourceVideo, "");
  setVideo(ui.trimmedVideo, "");
  setVideo(ui.retrievalVideo, "");
  setVideo(ui.summaryVideo, "");
  setSourceVisible(false);
  resetClipControls();
  ui.trimmedState.textContent = "-";
  ui.summaryState.textContent = "-";
  ui.segmentList.textContent = "";
  ui.shotList.textContent = "";
  ui.retrievedMoments.innerHTML = "";
  ui.highlightFrames.innerHTML = "";
  ui.retrievalState.textContent = "-";
  app.retrievalSaliency = [];
  drawRetrievalSaliency([]);
  drawScoreChart([]);
  setBusy(false);
  updateArtifactButtons({});
}

async function uploadVideo() {
  const file = ui.file.files[0];
  if (!file) {
    alert("비디오 파일을 선택해 주세요.");
    return;
  }

  setBusy(true);
  ui.logBox.textContent = "";
  const form = new FormData();
  form.append("video", file);

  try {
    const job = await readJson(await fetch("/api/upload", { method: "POST", body: form }));
    localStorage.setItem("droneMappingJobId", job.id);
    applyStatus(job);
    await loadJobs();
    setPage("upload");
  } catch (error) {
    alert(error.message);
  } finally {
    setBusy(false);
  }
}

async function runJob() {
  if (!app.job) return;
  app.rawViewerUrl = "";
  app.postViewerUrl = "";
  setViewer("raw", "");
  setViewer("post", "");
  setBusy(true);

  const payload = {
    start_seconds: Number(ui.startSeconds.value || 0),
    end_seconds: Number(ui.endSeconds.value || 0),
    lingbot_fps: Number(ui.lingbotFps.value || 5),
    summary_ratio: Number(ui.summaryRatio.value || 0.15),
  };

  try {
    const job = await readJson(
      await fetch(`/api/jobs/${app.job.id}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }),
    );
    applyStatus(job);
    await loadJobs();
    startPolling();
    setPage("lingbot");
  } catch (error) {
    alert(error.message);
    setBusy(false);
  }
}

async function pollOnce() {
  if (!app.job) return;
  const response = await fetch(`/api/jobs/${app.job.id}`);
  if (response.status === 404) {
    clearCurrentJob();
    await loadJobs();
    return;
  }
  const job = await readJson(response);
  applyStatus(job);
  if (["complete", "partial", "failed", "stopped"].includes(job.state) && job.retrieval_state !== "indexing") {
    stopPolling();
    setBusy(false);
    loadJobs().catch(() => {});
  }
}

function startPolling() {
  stopPolling();
  app.pollTimer = window.setInterval(() => {
    pollOnce().catch((error) => {
      ui.logBox.textContent += `\n${error.message}`;
      stopPolling();
      setBusy(false);
    });
  }, 2000);
}

function stopPolling() {
  if (app.pollTimer) {
    window.clearInterval(app.pollTimer);
    app.pollTimer = null;
  }
}

function setViewer(kind, url) {
  const frame = kind === "raw" ? ui.rawViewerFrame : ui.postViewerFrame;
  const state = kind === "raw" ? ui.rawViewerState : ui.postViewerState;
  const link = kind === "raw" ? ui.rawExternalLink : ui.postExternalLink;
  if (!url) {
    frame.removeAttribute("src");
    state.textContent = "waiting";
    setLink(link, "");
    return;
  }
  state.textContent = "loading";
  frame.onload = () => {
    state.textContent = "running";
  };
  frame.src = url;
  setLink(link, url);
}

async function launchViewer(kind) {
  if (!app.job) return;
  const button = kind === "raw" ? ui.openRawBtn : ui.openPostBtn;
  const state = kind === "raw" ? ui.rawViewerState : ui.postViewerState;
  button.disabled = true;
  state.textContent = "starting";
  try {
    const body = kind === "post" ? { restart: true } : {};
    const result = await readJson(
      await fetch(`/api/jobs/${app.job.id}/view/${kind}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }),
    );
    if (kind === "raw") app.rawViewerUrl = result.url;
    if (kind === "post") app.postViewerUrl = result.url;
    setViewer(kind, result.url);
  } catch (error) {
    state.textContent = error.message;
  } finally {
    updateArtifactButtons(app.job?.artifacts || {});
  }
}

function updateArtifactButtons(artifacts) {
  ui.openRawBtn.disabled = !app.job || !artifacts.raw_npz_url;
  ui.openPostBtn.disabled = !app.job || !artifacts.post_pcd_url;
}

async function loadSummaryAnalysis() {
  if (!app.job) return;
  try {
    const analysis = await readJson(await fetch(`/api/jobs/${app.job.id}/summary/analysis`));
    renderSummaryAnalysis(analysis);
  } catch (error) {
    ui.segmentList.textContent = error.message;
    ui.shotList.textContent = "";
    drawScoreChart([]);
  }
}

function renderSummaryAnalysis(analysis) {
  ui.segmentList.innerHTML = "";
  ui.shotList.innerHTML = "";
  if (analysis.fallback) {
    const chip = document.createElement("span");
    chip.className = "segment-chip";
    chip.textContent = "fallback summary";
    ui.segmentList.append(chip);
  }
  for (const segment of analysis.segments || []) {
    const chip = document.createElement("span");
    chip.className = "segment-chip";
    chip.textContent = `${secondsLabel(segment.start)} - ${secondsLabel(segment.end)}`;
    ui.segmentList.append(chip);
  }
  if (Number.isFinite(Number(analysis.score_min)) && Number.isFinite(Number(analysis.score_max))) {
    const chip = document.createElement("span");
    chip.className = "segment-chip neutral";
    chip.textContent = `score ${scoreLabel(analysis.score_min)} - ${scoreLabel(analysis.score_max)}`;
    ui.segmentList.append(chip);
  }
  if (!ui.segmentList.children.length) {
    ui.segmentList.textContent = "no selected segments";
  }
  drawScoreChart(analysis.scores || []);
  renderShotList(analysis.shots || []);
}

function renderShotList(shots) {
  const ranked = [...shots]
    .sort((a, b) => Number(b.mean_score || 0) - Number(a.mean_score || 0))
    .slice(0, 10);
  for (const shot of ranked) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = `shot-row${shot.selected ? " selected" : ""}`;
    row.innerHTML = `
      <span>${secondsLabel(shot.time_start)} - ${secondsLabel(shot.time_end)}</span>
      <strong>${scoreLabel(shot.mean_score)}</strong>
      <em>${shot.selected ? "selected" : `${shot.length || 0} steps`}</em>
    `;
    row.addEventListener("click", () => {
      if (ui.trimmedVideo.src) {
        ui.trimmedVideo.currentTime = Math.max(0, Number(shot.time_start || 0));
        ui.trimmedVideo.play().catch(() => {});
      }
    });
    ui.shotList.append(row);
  }
}

function drawScoreChart(scores) {
  const canvas = ui.scoreCanvas;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(canvas.clientWidth * ratio));
  canvas.height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
  const ctx = canvas.getContext("2d");
  ctx.scale(ratio, ratio);
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfe";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d7dde5";
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
  if (!scores.length) return;
  const rawScores = scores.map((row) => Number(row.score)).filter(Number.isFinite);
  if (!rawScores.length) return;
  const minScore = Math.min(...rawScores);
  const maxScore = Math.max(...rawScores);
  const scoreSpan = Math.max(maxScore - minScore, 1e-6);
  const pad = 22;
  const innerW = width - pad * 2;
  const innerH = height - pad * 2;
  scores.forEach((row, i) => {
    const x = pad + (i / Math.max(1, scores.length - 1)) * innerW;
    const barH = ((Number(row.score) - minScore) / scoreSpan) * innerH;
    ctx.fillStyle = row.selected ? "#2563eb" : "#9aa6b2";
    ctx.fillRect(x - 2, pad + innerH - barH, 4, barH);
  });
}

async function indexRetrieval() {
  if (!app.job) return;
  ui.retrievalState.textContent = "indexing";
  ui.indexRetrievalBtn.disabled = true;
  try {
    const job = await readJson(
      await fetch(`/api/jobs/${app.job.id}/retrieval/index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sample_fps: 1.0 }),
      }),
    );
    applyStatus(job);
    startPolling();
  } catch (error) {
    ui.retrievalState.textContent = error.message;
  }
}

async function queryRetrieval() {
  if (!app.job) return;
  const query = ui.retrievalQuery.value.trim();
  if (!query) return;
  ui.retrievalState.textContent = "searching";
  ui.queryRetrievalBtn.disabled = true;
  try {
    const result = await readJson(
      await fetch(`/api/jobs/${app.job.id}/retrieval/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: 5 }),
      }),
    );
    renderRetrieval(result);
    ui.retrievalState.textContent = "ready";
  } catch (error) {
    ui.retrievalState.textContent = error.message;
  } finally {
    ui.queryRetrievalBtn.disabled = false;
  }
}

function renderRetrieval(result) {
  ui.retrievedMoments.innerHTML = "";
  ui.highlightFrames.innerHTML = "";
  app.retrievalSaliency = result.saliency || [];
  drawRetrievalSaliency(app.retrievalSaliency, result.duration);

  const moments = result.moments || [];
  if (!moments.length) {
    ui.retrievedMoments.textContent = "No retrieved moments.";
  }
  for (const moment of moments) {
    const button = document.createElement("button");
    button.className = "moment-card";
    button.type = "button";
    const start = secondsLabel(moment.start);
    const end = secondsLabel(moment.end);
    button.innerHTML = `
      <strong>#${moment.rank} · ${start} - ${end}</strong>
      <span>score ${Number(moment.score).toFixed(3)}</span>
    `;
    button.addEventListener("click", () => seekRetrievalVideo(moment.start));
    ui.retrievedMoments.append(button);
  }

  const highlights = result.highlights || result.hits || [];
  if (!highlights.length) {
    ui.highlightFrames.textContent = "No highlighted frames.";
  }
  for (const hit of highlights) {
    const card = document.createElement("article");
    card.className = "retrieval-card";
    card.innerHTML = `
      <img src="${hit.thumbnail_url}" alt="">
      <div>
        <strong>#${hit.rank} · ${secondsLabel(hit.time)}</strong>
        <span>score ${Number(hit.score).toFixed(3)}</span>
      </div>
    `;
    card.addEventListener("click", () => seekRetrievalVideo(hit.time));
    ui.highlightFrames.append(card);
  }
}

function drawRetrievalSaliency(scores, duration = 0) {
  const canvas = ui.retrievalSaliencyCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfe";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d7dde5";
  ctx.strokeRect(0.5, 0.5, width - 1, height - 1);
  if (!scores.length) return;

  const values = scores.map((row) => Number(row.score)).filter(Number.isFinite);
  if (!values.length) return;
  const minScore = Math.min(...values);
  const maxScore = Math.max(...values);
  const span = Math.max(maxScore - minScore, 1e-6);
  const maxTime = duration || Math.max(...scores.map((row) => Number(row.end || row.time || 0)));
  const padX = 34;
  const padY = 22;
  const innerW = width - padX * 2;
  const innerH = height - padY * 2;

  ctx.fillStyle = "#64748b";
  ctx.font = "11px system-ui, sans-serif";
  ctx.fillText(maxScore.toFixed(2), 6, padY + 4);
  ctx.fillText(minScore.toFixed(2), 6, height - padY);
  ctx.fillText(`${Math.round(maxTime)}s`, width - padX - 18, height - 6);

  scores.forEach((row) => {
    const start = Number(row.start ?? row.time ?? 0);
    const end = Number(row.end ?? start + 1);
    const score = Number(row.score);
    if (!Number.isFinite(score)) return;
    const x0 = padX + (start / Math.max(maxTime, 1e-6)) * innerW;
    const x1 = padX + (end / Math.max(maxTime, 1e-6)) * innerW;
    const barH = ((score - minScore) / span) * innerH;
    ctx.fillStyle = "#2563eb";
    ctx.fillRect(x0, padY + innerH - barH, Math.max(3, x1 - x0 - 1), barH);
  });
}

function applyStatus(job) {
  app.job = job;
  localStorage.setItem("droneMappingJobId", job.id);
  setStatePill(job.state);
  ui.jobLine.textContent = `job ${job.id} · ${job.stage}`;
  ui.stageMetric.textContent = job.stage || "-";
  ui.fpsMetric.textContent = job.params?.lingbot_fps ? `${job.params.lingbot_fps} fps` : "-";
  ui.startMetric.textContent =
    job.params?.start_seconds !== undefined
      ? `${secondsLabel(job.params.start_seconds)} - ${secondsLabel(job.params.end_seconds || job.duration)}`
      : "-";
  ui.logBox.textContent = job.log_tail || "";
  ui.logBox.scrollTop = ui.logBox.scrollHeight;
  if (job.duration) setDuration(job.duration);
  if (job.params?.start_seconds !== undefined) {
    setStart(job.params.start_seconds);
  }
  if (job.duration) {
    setEnd(job.params?.end_seconds || job.duration);
  }

  const artifacts = job.artifacts || {};
  setVideo(ui.sourceVideo, artifacts.input_video_url);
  setVideo(ui.trimmedVideo, artifacts.trimmed_video_url);
  setVideo(ui.retrievalVideo, artifacts.trimmed_video_url);
  setVideo(ui.summaryVideo, artifacts.summary_video_url);
  setSourceVisible(Boolean(artifacts.input_video_url || ui.sourceVideo.src));

  ui.trimmedState.textContent = fileState(artifacts.trimmed_video_url);
  ui.summaryState.textContent = fileState(artifacts.summary_video_url);
  ui.loadAnalysisBtn.disabled = !artifacts.summary_features_url;
  ui.indexRetrievalBtn.disabled = !artifacts.trimmed_video_url || job.state === "running";
  ui.queryRetrievalBtn.disabled = !artifacts.retrieval_index_url;
  const retrievalState = job.retrieval_state || (artifacts.retrieval_index_url ? "ready" : "-");
  ui.retrievalState.textContent = artifacts.retrieval_index_url
    ? retrievalState
    : ["indexing", "failed", "stopped"].includes(retrievalState)
      ? retrievalState
      : "-";
  updateArtifactButtons(artifacts);
  setBusy(job.state === "running");
  renderJobs();
}

ui.navButtons.forEach((button) => {
  button.addEventListener("click", () => setPage(button.dataset.page));
});

ui.file.addEventListener("change", () => {
  const file = ui.file.files[0];
  if (!file) return;
  const localUrl = URL.createObjectURL(file);
  ui.sourceVideo.dataset.url = localUrl;
  ui.sourceVideo.src = localUrl;
  ui.sourceVideo.load();
  setSourceVisible(true);
  ui.uploadBtn.disabled = false;
});

ui.sourceVideo.addEventListener("loadedmetadata", () => {
  setDuration(ui.sourceVideo.duration);
});

ui.startRange.addEventListener("input", () => setStart(ui.startRange.value));
ui.endRange.addEventListener("input", () => setEnd(ui.endRange.value));
ui.startSeconds.addEventListener("change", () => {
  setStart(ui.startSeconds.value);
  if (Number.isFinite(ui.sourceVideo.duration)) {
    ui.sourceVideo.currentTime = Number(ui.startSeconds.value || 0);
  }
});
ui.endSeconds.addEventListener("change", () => {
  setEnd(ui.endSeconds.value);
  if (Number.isFinite(ui.sourceVideo.duration)) {
    ui.sourceVideo.currentTime = Number(ui.endSeconds.value || 0);
  }
});
ui.useCurrent.addEventListener("click", () => setStart(ui.sourceVideo.currentTime));
ui.useEndCurrent.addEventListener("click", () => setEnd(ui.sourceVideo.currentTime));
ui.uploadBtn.addEventListener("click", uploadVideo);
ui.runBtn.addEventListener("click", runJob);
ui.refreshBtn.addEventListener("click", () => pollOnce().catch((error) => alert(error.message)));
ui.openRawBtn.addEventListener("click", () => launchViewer("raw"));
ui.openPostBtn.addEventListener("click", () => launchViewer("post"));
ui.reloadJobsBtn.addEventListener("click", () => loadJobs().catch((error) => alert(error.message)));
ui.loadAnalysisBtn.addEventListener("click", loadSummaryAnalysis);
ui.indexRetrievalBtn.addEventListener("click", indexRetrieval);
ui.queryRetrievalBtn.addEventListener("click", queryRetrieval);
ui.retrievalQuery.addEventListener("keydown", (event) => {
  if (event.key === "Enter") queryRetrieval();
});
ui.retrievalSaliencyCanvas.addEventListener("click", (event) => {
  if (!app.retrievalSaliency.length) return;
  const rect = ui.retrievalSaliencyCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const ratio = Math.min(1, Math.max(0, x / Math.max(rect.width, 1)));
  const maxTime = Math.max(...app.retrievalSaliency.map((row) => Number(row.end || row.time || 0)), 1);
  const target = ratio * maxTime;
  const nearest = app.retrievalSaliency.reduce((best, row) => {
    const time = Number(row.time || row.start || 0);
    return Math.abs(time - target) < Math.abs(Number(best.time || best.start || 0) - target) ? row : best;
  }, app.retrievalSaliency[0]);
  seekRetrievalVideo(nearest.time || nearest.start || target);
});

window.addEventListener("hashchange", () => setPage(location.hash.replace("#", "") || "upload"));

setPage(location.hash.replace("#", "") || "upload");
setStatePill("idle");
setBusy(false);
setSourceVisible(false);
setLink(ui.rawExternalLink, "");
setLink(ui.postExternalLink, "");
loadJobs().catch(() => {});

const params = new URLSearchParams(location.search);
const initialJobId = params.get("job") || localStorage.getItem("droneMappingJobId");
if (initialJobId) {
  app.job = { id: initialJobId };
  pollOnce().catch(() => {
    localStorage.removeItem("droneMappingJobId");
  });
}
