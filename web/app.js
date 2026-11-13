"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = { session: null, bootstrap: null, history: [], historySource: "single", result: null, resultId: null, contextId: null, contextDetails: null, contextPending: false, latestJob: null, jobId: null, busy: false, tab: "overview", selectedRoute: null, selectedDepot: "", labels: false, resultSequence: 0, pollToken: 0, viewBox: [0, 0, 780, 380] };
  const colors = ["#4569a3", "#398478", "#bc8051", "#8274ab", "#587e9b", "#ae6576", "#7d8751", "#786f61", "#4d8290", "#9b7848"];
  const stageLabels = { chat: "理解问题", selection: "JEV 选择模型", retrieval: "加载模型知识", mapping: "模型整理参数", build: "构建数学模型", solve: "求解与独立校验", report: "生成结果报告", done: "任务完成" };
  const costLabels = { opening_cost: "仓库开放成本", delivery_cost: "独立配送成本", vehicle_cost: "车辆出车成本", travel_cost: "行驶成本", energy_cost: "电能成本" };
  const STORAGE_KEY = "orlm.chat.session.v1";
  const MODEL_STORAGE_KEY = "orlm.chat.llm-source.v1";
  const formatter = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 });
  let toastTimer;

  const text = (node, value) => { node.textContent = value == null ? "" : String(value); return node; };
  function el(tag, className, value) { const node = document.createElement(tag); if (className) node.className = className; if (value !== undefined) text(node, value); return node; }
  function svg(tag, attributes = {}, value) { const node = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const [key, val] of Object.entries(attributes)) node.setAttribute(key, String(val)); if (value !== undefined) text(node, value); return node; }
  function depotColor(id) {
    if (id == null || id === "") return "#8d9fb6";
    const key = String(id), numbered = /^d([1-9]\d*)$/i.exec(key);
    // Warehouse identity, not the order or subset of rows, determines its color.
    const index = numbered ? Number(numbered[1]) - 1 : [...key].reduce((hash, char) => (hash * 31 + char.codePointAt(0)) >>> 0, 0);
    return colors[index % colors.length];
  }
  function depotBadge(id) {
    const badge = el("span", "depot-badge"), swatch = el("i", "legend-swatch legend-square");
    badge.dataset.depot = id; badge.style.setProperty("--depot-color", depotColor(id));
    swatch.style.setProperty("--legend-color", depotColor(id)); swatch.setAttribute("aria-hidden", "true");
    badge.append(swatch, el("span", "", id)); return badge;
  }
  function resultRoutes() { return state.result?.plot?.routes || state.result?.summary?.business?.routes || []; }
  function number(value, suffix = "") { return value !== null && value !== undefined && Number.isFinite(Number(value)) ? formatter.format(Number(value)) + suffix : "—"; }
  function seconds(value) { if (!Number.isFinite(Number(value)) || value == null) return "—"; const rounded = Math.round(Number(value)); return Number(value) < 60 ? number(value, " 秒") : `${Math.floor(rounded / 60)} 分 ${rounded % 60} 秒`; }
  function date(value) { if (!value) return ""; const parsed = new Date(value); return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleString("zh-CN", { hour12: false, timeZone: "Asia/Shanghai" }); }
  function storageGet() { try { return localStorage.getItem(STORAGE_KEY); } catch { return null; } }
  function storageSet(id) { try { localStorage.setItem(STORAGE_KEY, id); } catch { /* Private browsing can disable storage. */ } }
  function updateModelSource() {
    const source = $("llmSource").value;
    const profile = state.bootstrap?.config?.llm_profiles?.find((item) => item.id === source);
    text($("providerLine"), `JEV · 模型知识包 · ${profile?.model || (source === "qwen_local" ? "Qwen3.5-9B-Q8_0" : "DeepSeek")} · 数学规划求解器`);
    text($("modelSourceHint"), source === "qwen_local" ? `先启动本地模型 · ${profile?.context_size ? Math.round(profile.context_size / 1024) + "K" : "16K"} · 聊天思考${profile?.thinking ? "开" : "关"} / 映射${profile?.mapping_thinking ? "开" : "关"}` : "云端调用");
    try { localStorage.setItem(MODEL_STORAGE_KEY, source); } catch { /* Optional preference. */ }
  }
  function toast(message) { text($("toast"), message); $("toast").hidden = false; clearTimeout(toastTimer); toastTimer = setTimeout(() => { $("toast").hidden = true; }, 7000); }
  async function api(path, options) {
    const response = await fetch(path, { cache: "no-store", ...options, headers: { "Content-Type": "application/json", ...(options?.headers || {}) } });
    let value; try { value = await response.json(); } catch { throw new Error("本地服务返回了无法读取的内容，请检查服务是否正常。"); }
    if (!response.ok) { const error = new Error(value.error || value.message || `请求失败（${response.status}）`); error.status = response.status; throw error; }
    return value;
  }
  const query = (path, id) => `${path}?${new URLSearchParams({ id })}`;
  function safeUrl(value) { try { const url = new URL(value, location.origin); return url.origin === location.origin && url.protocol === location.protocol ? url.href : null; } catch { return null; } }
  function setBusy(busy) {
    state.busy = busy;
    $("sendButton").disabled = busy;
    $("newSessionButton").disabled = busy;
    for (const id of ["solver", "timeLimit", "mipGap", "instanceSelect", "llmSource"]) $(id).disabled = busy;
    for (const button of document.querySelectorAll(".example-button")) button.disabled = busy;
    text($("composerHint"), busy ? "任务正在进行，完成后可继续对话" : "Enter 发送 · Shift + Enter 换行");
    updateContext();
  }
  function updateSettingsLabel() { text($("settingsSummary"), `${$("timeLimit").value || "300"} 秒 · gap ${$("mipGap").value || "0"}%`); }
  function updateContext() {
    $("contextReference").hidden = !state.contextId;
    const item = state.history.find((entry) => entry.id === state.contextId) || (state.contextDetails?.id === state.contextId ? state.contextDetails : null);
    const model = state.contextId === state.resultId ? state.result?.summary?.models?.[0] : null;
    const label = item ? `${item.model_id || "结果"} · ${item.instance_id || item.title || ""}` : model ? `${model.model_id} · ${model.instance_id || ""}` : state.contextId;
    const stamp = item?.created_at || (model ? state.result?.summary?.generated_at : null);
    const source = item?.source_label || (state.contextId && state.session?.messages?.some((message) => message.result_id === state.contextId) ? "本次对话结果" : "保存结果");
    text($("contextLabel"), state.contextId ? `下一轮引用：${label}${stamp ? ` · ${date(stamp)}` : ""} · ${source}` : "");
    $("contextLabel").title = "本会话新完成的有效结果自动作为下一轮上下文；也可手动选择或取消。浏览历史结果不改变引用。";
    $("clearContext").disabled = state.busy || state.contextPending;
    $("sendButton").disabled = state.busy || state.contextPending;
    $("newSessionButton").disabled = state.busy || state.contextPending;
    $("reuseResultButton").hidden = !state.resultId;
    $("reuseResultButton").disabled = state.busy || state.contextPending || state.resultId === state.contextId;
    text($("reuseResultButton"), state.resultId === state.contextId && state.contextId ? "已沿用此结果" : "沿用此结果");
    $("browseNotice").hidden = !state.resultId;
    if (state.resultId) text($("browseNotice"), state.resultId === state.contextId ? "此结果已设为下一轮引用，可在左侧取消。" : state.contextId ? `当前仅浏览，下一轮仍引用 ${label}。点击“沿用此结果”才会切换引用。` : "当前仅查看结果。需要继续规划或解释时，点击“沿用此结果”设置下一轮引用。");
  }

  async function setContext(id) {
    if (!state.session || state.busy || state.contextPending) return false;
    state.contextPending = true; updateContext();
    try {
      const response = await api("/api/session/context", { method: "POST", body: JSON.stringify({ session_id: state.session.id, result_id: id || null }) });
      const session = response.session || response;
      state.contextId = Object.prototype.hasOwnProperty.call(session, "reference_result_id") ? session.reference_result_id || null : id || null;
      state.session.reference_result_id = state.contextId;
      if (state.contextId === state.resultId) state.contextDetails = resultReference(state.result, state.resultId);
      else if (!state.contextId) state.contextDetails = null;
      toast(state.contextId ? "已沿用此结果，下一轮会使用它。" : "已取消引用，下一轮不沿用保存结果。");
      return true;
    } catch (error) { toast(`引用未更新：${error.message}`); return false; }
    finally { state.contextPending = false; updateContext(); }
  }
  function resultReference(result, id) { const model = result?.summary?.models?.[0] || {}; return { id, model_id: model.model_id || result?.mapping?.model_id, instance_id: model.instance_id || result?.mapping?.instance_id, created_at: result?.summary?.generated_at, source_label: state.history.find((entry) => entry.id === id)?.source_label || (state.resultOrigin === "current" ? "本次对话结果" : "保存结果") }; }

  function appendMessage(message) {
    const container = el("article", `message ${message.role === "user" ? "user" : "assistant"}`);
    const meta = el("div", "message-meta");
    meta.append(el("span", "message-avatar", message.role === "user" ? "你" : "O"), el("span", "", message.role === "user" ? "你" : "ORLM"));
    if (message.created_at) meta.append(el("span", "", new Date(message.created_at).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false })));
    if (message.llm?.model) meta.append(el("span", "message-model", message.llm.provider_label || message.llm.model));
    container.append(meta, el("div", "message-content", message.content));
    if (message.result_id) {
      const button = el("button", "message-result-link", "查看此条任务的结果 ↗");
      button.addEventListener("click", () => loadResult(message.result_id, false, message.action === "run" ? "current" : "conversation"));
      container.append(button);
    }
    $("conversation").append(container);
  }
  function renderSession(session) {
    state.session = session;
    state.contextId = session.reference_result_id || null;
    const conversation = $("conversation");
    const welcome = $("welcome");
    conversation.replaceChildren(welcome);
    welcome.hidden = Boolean(session.messages?.length);
    for (const message of session.messages || []) appendMessage(message);
    if (state.busy) createPending();
    updateContext();
    scrollChat();
  }
  function scrollChat() { requestAnimationFrame(() => { $("conversation").scrollTop = $("conversation").scrollHeight; }); }
  function createPending() {
    let pending = $("pendingJob");
    if (pending) return pending;
    pending = el("div", "pending-card"); pending.id = "pendingJob";
    const heading = el("div", "pending-title"); heading.append(el("span", "spinner"), el("span", "", "正在准备任务…"), el("time", "", ""));
    const hint = el("p", "pending-hint", "结果会在右侧自动更新，刷新页面也可恢复任务。");
    const list = el("div", "phase-list");
    pending.append(heading, hint, list); $("conversation").append(pending); scrollChat(); return pending;
  }
  function renderJob(job) {
    state.latestJob = job;
    renderRunCard(job, true);
    const pending = createPending();
    text(pending.querySelector(".pending-title span:nth-child(2)"), job.stage_label || stageLabels[job.stage] || "正在运行…");
    text(pending.querySelector("time"), seconds(job.elapsed_seconds));
    pending.querySelector(".phase-list").replaceChildren();
    text(pending.querySelector(".pending-hint"), "阶段耗时和当前仓库见右侧运行卡片。刷新可恢复任务，不会重复提交。");
    if (job.preliminary_reply) { let summary = pending.querySelector(".pending-summary"); if (!summary) { summary = el("p", "pending-summary"); pending.append(summary); } text(summary, `任务准备：${job.preliminary_reply}`); }
  }
  function renderRunCard(progress, live = false) {
    // Saved timings belong with model evidence; live progress stays above the result.
    const host = live ? $("resultEmpty").parentElement : $("evidenceView");
    if ($("runCard").parentElement !== host) {
      if (live) host.insertBefore($("runCard"), $("resultEmpty"));
      else host.prepend($("runCard"));
    }
    $("runCard").hidden = false;
    const running = progress.status === "running";
    // Expand a newly running task once; preserve the user's disclosure choice on polls.
    const presentation = running ? "running" : "saved";
    if ($("runCard").dataset.presentation !== presentation) $("runCard").open = running;
    $("runCard").dataset.presentation = presentation;
    const stage = progress.stage || progress.current_stage;
    text($("runOrigin"), live ? running ? "本次对话 · 正在运行" : "本次对话 · 运行记录" : "当前浏览结果 · 运行记录");
    text($("runStage"), running ? progress.stage_label || stageLabels[stage] || "正在运行" : progress.status === "error" ? "任务未完成" : "运行耗时明细");
    text($("runElapsedLabel"), running ? "已用时间" : "记录总用时");
    text($("runElapsed"), seconds(progress.elapsed_seconds));
    $("runCard").classList.toggle("running", running);
    const subproblems = Array.isArray(progress.subproblems) ? progress.subproblems : [];
    const current = subproblems.find((item) => item.status === "running");
    const depot = current?.depot_id || progress.depot_id;
    const index = current?.subproblem_index ?? progress.subproblem_index;
    const count = current?.subproblem_count ?? progress.subproblem_count;
    const completed = progress.completed_subproblem_count ?? subproblems.filter((item) => item.status === "done").length;
    const phases = Array.isArray(progress.phase_timings) ? progress.phase_timings : [];
    let hint = progress.status === "error" ? progress.error || "任务未完成，保留最后一次记录的阶段与用时。请查看左侧说明。" : running ? stage === "solve" ? "求解器正在计算并进行独立校验。" : stage === "mapping" ? "所选模型正在结合业务描述与知识整理任务参数。" : "显示后台实际阶段和耗时，任务完成后继续保留记录。" : phases.length ? "阶段耗时已保留，可比较 API、检索与求解所用时间。" : "此历史结果未记录完整阶段耗时；下方展示已保存的求解器用时。";
    if (running && depot) hint += ` 当前仓库 ${depot}${index != null && count != null ? `（第 ${index} / ${count} 个子问题）` : ""}；已完成 ${completed}${count != null ? ` / ${count}` : ""} 个。`;
    else if (running && stage === "solve" && count) hint += ` 已完成 ${completed} / ${count} 个子问题。`;
    text($("runHint"), hint);
    const phaseBox = $("runPhases"); phaseBox.replaceChildren();
    for (const phase of phases) {
      const chip = el("div", `run-phase ${phase.state || "done"}`); chip.append(el("span", "phase-dot"), el("span", "", phase.label || stageLabels[phase.stage] || phase.stage || "阶段"), el("strong", "", seconds(phase.seconds ?? phase.elapsed_seconds))); phaseBox.append(chip);
    }
    const subBox = $("runSubproblems"); subBox.replaceChildren();
    for (const item of subproblems) {
      const active = running && item.status === "running";
      const row = el("div", `run-subproblem${active ? " active" : ""}`);
      row.append(el("span", "", item.depot_id || item.model_id || "整体子问题"), el("span", "run-subproblem-status", active ? "正在求解" : item.status === "running" ? "最终状态未记录" : item.status_label || item.solve_status || "已完成"), el("time", "", seconds(active ? item.elapsed_seconds : item.runtime_seconds)));
      subBox.append(row);
    }
    if (!phases.length && !subproblems.length) subBox.append(el("p", "run-record-missing", running ? "后台尚未返回阶段计时记录。" : "此结果未保存阶段或子问题计时。"));
  }
  async function refreshSession() { if (!state.session?.id) return; const session = await api(query("/api/session", state.session.id)); renderSession(session); return session; }
  async function pollJob(jobId) {
    const token = ++state.pollToken; state.jobId = jobId; setBusy(true); createPending();
    let failures = 0;
    while (token === state.pollToken) {
      try {
        const job = await api(query("/api/job", jobId)); if (token !== state.pollToken) return;
        failures = 0; renderJob(job);
        if (job.status !== "running") {
          state.jobId = null; setBusy(false); $("pendingJob")?.remove();
          try { await refreshSession(); } catch { if (job.reply || job.error) appendMessage({ role: "assistant", content: job.reply || job.error, result_id: job.result_id }); }
          if (job.result_id) await loadResult(job.result_id, false, job.action === "run" ? "current" : job.result_id === state.resultId ? state.resultOrigin || "history" : "history");
          if (job.status === "error") toast(job.error || "任务未完成，可查看对话中的原因并修改描述。");
          try { const history = await api("/api/history"); state.history = history.history || []; } catch { /* The result stays available if listing fails. */ }
          scrollChat(); return;
        }
      } catch (error) {
        failures += 1;
        if (error.status === 404 || error.status === 410) {
          try {
            const session = await refreshSession();
            if (!session?.active_job_id) { state.pollToken += 1; state.jobId = null; setBusy(false); $("pendingJob")?.remove(); renderRunCard({ ...state.latestJob, status: "error", error: "后台服务已重新启动或任务已结束，最终状态暂不可恢复。请从历史查看已保存结果；系统不会重新提交。" }, true); toast("后台服务已重新启动或任务已结束。已保存结果仍可从历史记录查看，不会自动重复提交。"); return; }
            if (session.active_job_id !== jobId) { pollJob(session.active_job_id); return; }
          } catch { /* Keep querying until the local service is reachable again. */ }
        }
        const pending = createPending(); text(pending.querySelector(".pending-title span:nth-child(2)"), "正在恢复后台连接…");
        text(pending.querySelector(".pending-hint"), "任务状态暂时不可读取。页面只查询现有任务，不会重新提交或重复调用 API。");
        if (failures === 1) toast(error.message);
      }
      await new Promise((resolve) => setTimeout(resolve, Math.min(5000, 1200 + failures * 1000)));
    }
  }
  async function sendPrompt(event) {
    event.preventDefault(); if (state.busy || state.contextPending || !state.session) return;
    const prompt = $("prompt").value.trim(); if (!prompt) { $("prompt").focus(); return; }
    const timeLimit = Number($("timeLimit").value), mipGap = Number($("mipGap").value) / 100;
    if (!Number.isFinite(timeLimit) || timeLimit < 1 || timeLimit > 86400 || !Number.isFinite(mipGap) || mipGap < 0 || mipGap >= 1) { toast("请填写 1–86400 秒的时限，以及大于等于 0、低于 100% 的 gap。"); $("settings").open = true; return; }
    const requestId = globalThis.crypto?.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setBusy(true); $("welcome").hidden = true; appendMessage({ role: "user", content: prompt, created_at: new Date().toISOString() }); $("prompt").value = ""; createPending();
    try {
      const data = await api("/api/chat", { method: "POST", body: JSON.stringify({ session_id: state.session.id, prompt, request_id: requestId, selected_result_id: state.contextId, llm_source: $("llmSource").value, settings: { solver: $("solver").value, time_limit: timeLimit, mip_gap: mipGap } }) });
      if (data.session_id) { state.session.id = data.session_id; storageSet(data.session_id); }
      if (!data.job_id) throw new Error("服务未返回任务编号，请刷新检查对话记录。");
      pollJob(data.job_id);
    } catch (error) {
      try {
        const session = await refreshSession();
        if (session?.active_job_id) { pollJob(session.active_job_id); toast("已恢复刚才提交的任务。"); return; }
      } catch { /* No second POST is made when acknowledgement is lost. */ }
      setBusy(false); $("pendingJob")?.remove(); $("prompt").value = prompt;
      appendMessage({ role: "assistant", content: `消息提交未完成：${error.message}\n若刚才网络断开，请刷新页面确认后台状态。系统没有自动重复提交。` }); scrollChat(); toast(error.message);
    }
  }

  function makeMetric(label, value, unit, note) { const item = el("div", "metric"); const metricValue = el("div", "metric-value", value); if (unit) metricValue.append(el("span", "metric-unit", unit)); item.append(el("div", "metric-label", label), metricValue, el("div", "metric-note", note)); return item; }
  function resultQuality(result) {
    if (result.quality) return result.quality;
    const summary = result.summary || {}, models = summary.models || [];
    const total = summary.subproblem_count ?? models.length, verified = summary.verified_subproblem_count ?? 0;
    const complete = summary.complete_solution === true && total > 0 && verified === total;
    const gaps = models.map((item) => item.mip_gap);
    const known = total > 0 && gaps.length === total && gaps.every((gap) => gap != null && Number.isFinite(Number(gap)) && Number(gap) >= 0);
    return { solution_state: complete ? "verified_complete" : verified > 0 ? "verified_partial" : "none", verified_count: verified, total_count: total, actual_max_gap: known ? Math.max(...gaps.map(Number)) : null, requested_gap: null, gap_state: "unknown", subproblems: models.map((item) => ({ depot_id: item.depot_id, model_id: item.model_id, status: item.status, status_label: item.status_label, verified: item.verified === true, gap: item.mip_gap, runtime_seconds: item.runtime_seconds, objective: item.objective })) };
  }
  function renderQuality(quality) {
    const complete = quality.solution_state === "verified_complete", partial = quality.solution_state === "verified_partial";
    const root = $("qualitySummary"); root.replaceChildren();
    const feasibility = el("div", `quality-card ${complete ? "verified" : "attention"}`), optimality = el("div", `quality-card ${complete && quality.gap_state === "met" ? "verified" : "attention"}`);
    const feasibilityLabel = complete ? "完整方案已通过独立校验" : partial ? "部分子问题已通过独立校验" : "尚无完整可行方案";
    feasibility.append(el("span", "quality-label", "方案可行性"), el("strong", "", feasibilityLabel), el("p", "", quality.total_count ? `${quality.verified_count ?? 0} / ${quality.total_count} 个子问题通过校验${partial ? "，不能作为完整方案使用。" : "。"}` : "此任务尚无已验证的求解结果。"));
    const gapTitle = !complete ? "暂不能判断整体 gap 是否达标" : quality.gap_state === "met" ? "已达到目标 gap" : quality.gap_state === "unmet" ? "尚未达到目标 gap" : "gap 达标状态未知";
    optimality.append(el("span", "quality-label", "最优性差距"), el("strong", "", gapTitle), el("p", "", quality.message || (!complete ? "先获得所有子问题的有效可行解，再判断整体达标状态。" : quality.gap_state === "unmet" ? "已有可行方案；实际 gap 仍高于目标，可继续计算以改善最优性证明。" : quality.gap_state === "unknown" ? "部分 gap 或目标值未记录，不能把缺失值当作 0%。" : "达到设置的容差即可停止，正 gap 仍保留可能的改善空间。")));
    root.append(feasibility, optimality);
    const entries = Array.isArray(quality.subproblems) ? quality.subproblems : [];
    $("subproblemDetails").hidden = !entries.length;
    text($("subproblemSummary"), quality.total_count ? `${quality.verified_count ?? 0} / ${quality.total_count} 个通过` : "");
    const target = quality.requested_gap != null ? `目标 ${number(quality.requested_gap * 100, "%")}` : "目标未记录";
    const table = el("table"), head = el("tr"), body = el("tbody"), thead = el("thead");
    for (const label of ["仓库 / 子问题", "独立校验", `实际 gap（${target}）`, "停止原因 / 状态", "求解用时", "目标值"]) head.append(el("th", "", label));
    thead.append(head);
    for (const item of entries) {
      const row = el("tr"), gapCell = el("td", "", item.gap != null ? number(item.gap * 100, "%") : "未记录");
      if (item.verified === true && item.gap_state === "met") gapCell.append(el("span", "inline-note", "达标"));
      else if (item.gap_state === "unmet") gapCell.append(el("span", "inline-note attention", "未达标"));
      const depotCell = el("td", "route-label"); depotCell.append(item.depot_id ? depotBadge(item.depot_id) : el("span", "", item.model_id || "整体子问题"));
      row.append(depotCell, el("td", item.verified === true ? "verified-text" : "", item.verified === true ? "通过" : item.verified === false ? "未通过 / 无有效解" : "未记录"), gapCell, el("td", "", item.termination_label || item.status_label || item.status || "未记录"), el("td", "", seconds(item.runtime_seconds)), el("td", "", item.verified === true ? number(item.objective) : "—")); body.append(row);
    }
    table.append(thead, body); $("subproblemTable").replaceChildren(table);
  }
  async function loadResult(id, _useContext = false, origin = "history") {
    const sequence = ++state.resultSequence; $("resultLoading").hidden = false;
    try {
      const result = await api(query("/api/result", id)); if (sequence !== state.resultSequence) return;
      state.result = result; state.resultId = id; state.selectedDepot = ""; state.selectedRoute = null; state.viewBox = [0, 0, 780, 380];
      state.resultOrigin = origin; renderResult(result); text($("resultOrigin"), origin === "current" ? "本次对话结果" : origin === "conversation" ? "对话中的保存结果 · 仅浏览" : "历史结果 · 仅浏览"); $("resultOrigin").hidden = false; updateContext();
    } catch (error) { if (sequence === state.resultSequence) toast(`无法读取结果：${error.message}`); }
    finally { if (sequence === state.resultSequence) $("resultLoading").hidden = true; }
  }
  function renderResult(result) {
    const summary = result.summary || {}, models = summary.models || [], model = models[0] || {}, business = summary.business || {}, depots = business.depots || [], routes = result.plot?.routes || business.routes || [];
    const quality = resultQuality(result), complete = quality.solution_state === "verified_complete";
    $("resultEmpty").hidden = true; $("resultContent").hidden = false;
    text($("resultTitle"), model.model_name || ({ M1: "仓库选址与客户分配", M2: "固定仓库的客户分配", M3: "常规车辆路径规划", M4: "电动车辆路径规划" }[model.model_id || result.mapping?.model_id]) || "任务结果");
    text($("resultSubtitle"), `${model.model_id || result.mapping?.model_id || result.selection?.model_id || "模型未确定"} · ${model.instance_id || result.mapping?.instance_id || summary.scenario_id || ""}${summary.generated_at ? ` · ${date(summary.generated_at)}` : ""}`);
    const status = summary.status || result.pipeline?.status || "unknown";
    text($("resultStatus"), summary.status_label || result.pipeline?.status_label || ({ needs_input: "需要补充信息", unsupported: "存在未支持要求", no_match: "没有匹配模型", error: "任务未完成" }[status]) || status);
    $("resultStatus").className = `status-badge ${/error|infeasible|failed/.test(status) ? "error" : summary.complete_solution === false || /limit|partial|input|unsupported|no_match|unknown/.test(status) ? "warning" : ""}`;
    const customerIds = new Set((result.plot?.nodes || []).filter((node) => node.type === "customer").map((node) => node.id));
    if (!customerIds.size) for (const item of business.assignments || []) customerIds.add(item.customer_id);
    const opened = depots.filter((depot) => depot.opened === true);
    const metrics = $("metrics"); metrics.replaceChildren(
      makeMetric(complete ? "方案总成本" : quality.verified_count > 0 ? "已验证部分成本" : "方案成本", complete ? number(summary.total_objective) : quality.verified_count > 0 ? number(summary.partial_objective) : "—", "", complete ? "原数据成本单位" : quality.verified_count > 0 ? "部分结果，不能视为完整方案" : "尚无通过验证的完整方案"),
      makeMetric("实例客户", customerIds.size ? number(customerIds.size) : number(models.reduce((sum, item) => sum + (item.customer_count || 0), 0)), "个", `${opened.length} 个开放仓库`),
      makeMetric(routes.length ? "车辆路线" : "客户分配", routes.length ? number(routes.length) : number((business.assignments || []).length), routes.length ? "条" : "项", routes.length ? "每条路线一次出车" : "独立配送口径"),
      makeMetric("最大子问题 gap", quality.actual_max_gap != null ? number(quality.actual_max_gap * 100, "%") : "未知", "", `${quality.requested_gap != null ? `目标 ${number(quality.requested_gap * 100, "%")}` : "目标未记录"} · ${!complete ? "整体未确认" : quality.gap_state === "met" ? "已达标" : quality.gap_state === "unmet" ? "未达标" : "状态未知"}`)
    );
    const gapMetric = metrics.lastElementChild;
    const gapHelp = el("button", "help-tip", "?"); gapHelp.type = "button"; gapHelp.setAttribute("aria-label", "了解最大子问题 gap"); gapHelp.title = "gap 比较当前可行方案成本和求解器证明的成本下界。5% 表示最多可能再省当前成本的 5%，实际能省多少未知。这里取所有仓库中最大的 gap；缺失记录不会当作 0%。";
    gapHelp.addEventListener("click", () => toast(gapHelp.title)); gapMetric.querySelector(".metric-label").append(gapHelp);
    renderQuality(quality);
    if (state.busy && state.latestJob) renderRunCard(state.latestJob, true);
    else {
      const progress = result.progress || {};
      renderRunCard({ ...progress, status: progress.status || "done", subproblems: progress.subproblems?.length ? progress.subproblems : (quality.subproblems || []).map((item) => ({ ...item, solve_status: item.status_label || item.status, status: "done" })) }, false);
    }
    renderAlerts(summary, result);
    const filter = $("depotFilter"); filter.replaceChildren(el("option", "", "全部仓库")); filter.firstChild.value = "";
    const graphDepots = (result.plot?.nodes || []).filter((node) => node.type === "depot");
    for (const depot of graphDepots) { const option = el("option", "", `${depot.id}${depot.opened === false ? " · 未开放" : depot.opened === true ? "" : " · 候选"}`); option.value = depot.id; filter.append(option); }
    text($("plotTitle"), routes.length ? "车辆路线与客户分布" : "仓库与客户分配"); text($("plotDescription"), routes.length ? `${routes.length} 条路线 · 颜色对应所属仓库` : "连线表示客户归属，不代表串点车辆路线");
    text($("routeCount"), routes.length ? routes.length : (business.assignments || []).length || "");
    renderPlot(); renderDepotLoads(depots); renderDetailTable(business, routes); renderEvidence(result); updateReportLink(result); setTab(state.tab);
  }
  function renderAlerts(summary, result) {
    const box = $("resultAlerts"); box.replaceChildren();
    const messages = [];
    if (summary.complete_solution === false) messages.push(summary.verified_subproblem_count > 0 ? "当前仅有部分子模型结果，请补全或处理剩余子问题后再使用完整方案。" : "当前尚未获得通过独立验证的完整方案，请查看需要补充的信息或检查提示。");
    for (const entry of summary.errors || []) messages.push(typeof entry === "string" ? entry : entry.message || JSON.stringify(entry));
    for (const entry of summary.missing_fields || result.mapping?.missing_fields || []) messages.push(`需要补充：${typeof entry === "string" ? entry : entry.message || entry.field || JSON.stringify(entry)}`);
    for (const entry of summary.unsupported_requirements || result.mapping?.unsupported_requirements || []) messages.push(`当前不支持：${typeof entry === "string" ? entry : entry.message || JSON.stringify(entry)}`);
    if (summary.warning_messages?.length) messages.push(...summary.warning_messages);
    const unique = [...new Set(messages)];
    for (const message of unique.slice(0, 3)) box.append(el("div", "result-alert", message));
    if (unique.length > 3) box.append(el("div", "result-alert", `还有 ${unique.length - 3} 条备注，可在“模型与依据”中查看。`));
  }

  function selectRoute(id, { toggle = true, fromTable = false } = {}) {
    const route = resultRoutes().find((item) => item.route_id === id);
    state.selectedRoute = route && !(toggle && state.selectedRoute === id) ? id : null;
    state.fitRoute = Boolean(state.selectedRoute);
    if (!state.selectedRoute) state.viewBox = [0, 0, 780, 380];
    if (state.selectedRoute && state.selectedDepot && String(route.depot_id) !== state.selectedDepot) {
      state.selectedDepot = ""; $("depotFilter").value = "";
    }
    renderPlot(); highlightRouteRows();
    if (fromTable) { setTab("overview"); $("routePicker").scrollIntoView({ block: "start", behavior: "smooth" }); $("routeSelect").focus({ preventScroll: true }); }
  }
  function renderRouteSelection(routes, selected, nodes) {
    $("routePicker").hidden = !routes.length;
    const select = $("routeSelect"); select.replaceChildren(el("option", "", "选择一条路线…")); select.firstChild.value = "";
    for (const route of routes.filter((item) => !state.selectedDepot || String(item.depot_id) === state.selectedDepot)) {
      const option = el("option", "", `${route.route_id} · ${route.depot_id}${route.customer_count != null ? ` · ${route.customer_count} 个客户` : ""}`); option.value = route.route_id; select.append(option);
    }
    select.value = selected?.route_id || "";
    const card = $("routeDetail"); card.replaceChildren(); card.hidden = !selected; if (!selected) return;
    card.style.setProperty("--depot-color", depotColor(selected.depot_id));
    const heading = el("div", "route-detail-heading"), title = el("div", "route-detail-title"), clear = el("button", "route-clear", "取消选择");
    title.append(el("strong", "", selected.route_id), depotBadge(selected.depot_id)); clear.type = "button";
    clear.addEventListener("click", () => { selectRoute(null); select.focus({ preventScroll: true }); }); heading.append(title, clear);
    const nodeIds = (selected.nodes || []).map(String), customers = nodeIds.filter((id) => id !== String(selected.depot_id));
    const stats = el("dl", "route-stats");
    const metric = (label, value) => { const item = el("div"); item.append(el("dt", "", label), el("dd", "", value)); stats.append(item); };
    metric("服务客户", number(selected.customer_count ?? (nodeIds.length ? customers.length : null), " 个"));
    metric("载重 / 容量", `${number(selected.demand)} / ${number(selected.capacity)}`);
    const utilization = selected.utilization ?? (selected.demand != null && Number(selected.capacity) > 0 ? Number(selected.demand) / Number(selected.capacity) : null);
    metric("装载率", utilization != null ? number(utilization * 100, "%") : "—"); metric("路线成本", number(selected.total_cost));
    const sequence = el("ol", "route-sequence"); sequence.setAttribute("aria-label", "路线访问顺序");
    let order = 0;
    for (const [index, id] of nodeIds.entries()) {
      const isDepot = id === String(selected.depot_id), stop = el("li", `route-stop${isDepot ? " depot-stop" : ""}`);
      const label = isDepot ? index === 0 ? "出发" : index === nodeIds.length - 1 ? "返回" : "仓库" : String(++order);
      stop.append(el("span", "stop-order", label), el("span", "", id)); sequence.append(stop);
    }
    const notes = [];
    if (selected.distance_km != null && Number.isFinite(Number(selected.distance_km))) notes.push(`里程 ${number(selected.distance_km)} km`);
    notes.push("成本沿用原数据单位");
    const byId = new Map(nodes.map((node) => [String(node.id), node]));
    if (nodeIds.some((id) => !byId.has(id))) notes.push("部分节点缺少坐标，图中仅显示有记录的路段");
    if (nodeIds.some((id, index) => { const a = byId.get(id), b = byId.get(nodeIds[index + 1]); return a && b && Number(a.x) === Number(b.x) && Number(a.y) === Number(b.y); })) notes.push("有节点坐标重合，到访次序请以此处列表为准");
    card.append(heading, stats, el("p", "route-sequence-label", "访问顺序 · 图中数字为客户到访次序"));
    card.append(nodeIds.length ? sequence : el("p", "route-detail-note", selected.path || "此结果未记录节点顺序。"), el("p", "route-detail-note", notes.join(" · ")));
  }
  function routeGeometry(route, byId, position, selected) {
    const ids = (route.nodes || []).map(String), legs = ids.slice(1).map((to, index) => ({ from: ids[index], to }));
    const edges = new Set(legs.map(({ from, to }) => JSON.stringify([from, to]))), arrows = [], paths = [];
    for (const [index, { from, to }] of legs.entries()) {
      const start = byId.get(from), end = byId.get(to); if (!start || !end) continue;
      const [x1, y1] = position(start), [x2, y2] = position(end), dx = x2 - x1, dy = y2 - y1, length = Math.hypot(dx, dy);
      if (length < .01) continue;
      // Split reciprocal legs (including a one-customer round trip) so both directions remain visible.
      const bend = selected && edges.has(JSON.stringify([to, from])) ? Math.min(20, length * .3) : 0;
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, cx = mx - dy / length * bend, cy = my + dx / length * bend;
      paths.push(`M${x1} ${y1} ${bend ? `Q${cx} ${cy} ` : "L"}${x2} ${y2}`);
      if (selected && length >= 6) arrows.push({ x: (mx + cx) / 2, y: (my + cy) / 2, angle: Math.atan2(dy, dx) * 180 / Math.PI, scale: Math.min(1, length / 18), from, to, order: index + 1 });
    }
    return { path: paths.join(" "), arrows };
  }
  function visitMarkers(visitOrder, byId, position, depots) {
    const markers = new Map(), occupied = depots.map((node) => { const [x, y] = position(node); return { left: x - 13, right: x + 75, top: y - 13, bottom: y + 13 }; });
    const actualStops = [...visitOrder.keys()].filter((id) => byId.has(id)).map((id) => { const [x, y] = position(byId.get(id)); return { id, left: x - 5, right: x + 5, top: y - 5, bottom: y + 5 }; });
    const overlaps = (a, b) => a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top;
    for (const id of visitOrder.keys()) {
      const node = byId.get(id); if (!node) continue;
      const [x, y] = position(node), offsets = [[0, 0]];
      for (const radius of [24, 36, 50, 68]) for (const [dx, dy] of [[0, -1], [1, 0], [0, 1], [-1, 0], [.71, -.71], [-.71, -.71], [.71, .71], [-.71, .71]]) offsets.push([dx * radius, dy * radius]);
      const candidates = offsets.map(([dx, dy]) => {
        const box = { left: x + dx - 12, right: x + dx + (state.labels ? 14 + String(id).length * 5.5 : 12), top: y + dy - 12, bottom: y + dy + 12 };
        return { x: x + dx, y: y + dy, shifted: Boolean(dx || dy), box, score: [...occupied, ...actualStops.filter((stop) => stop.id !== id)].filter((other) => overlaps(box, other)).length * 1000 + Math.hypot(dx, dy) };
      });
      const best = candidates.reduce((a, b) => a.score <= b.score ? a : b); occupied.push(best.box); markers.set(id, best);
    }
    return markers;
  }
  function renderPlot() {
    const plot = $("networkPlot"); plot.replaceChildren(); plot.setAttribute("viewBox", state.viewBox.join(" "));
    const fitRoute = state.fitRoute; state.fitRoute = false;
    const nodes = (state.result?.plot?.nodes || []).filter((node) => Number.isFinite(Number(node.x)) && Number.isFinite(Number(node.y)) && node.x != null && node.y != null);
    const routes = resultRoutes(), selectedRoute = routes.find((route) => route.route_id === state.selectedRoute);
    renderRouteSelection(routes, selectedRoute, nodes); $("plotTooltip").hidden = true;
    $("plotEmpty").hidden = Boolean(nodes.length); const legend = $("plotLegend"); legend.replaceChildren(); if (!nodes.length) return;
    const depots = nodes.filter((node) => node.type === "depot");
    const assignments = state.result.plot?.assignments || state.result.summary?.business?.assignments || [];
    const assignmentMap = new Map(assignments.map((assignment) => [String(assignment.customer_id), assignment.depot_id]));
    const byId = new Map(nodes.map((node) => [String(node.id), node]));
    const visitOrder = new Map((selectedRoute?.nodes || []).map(String).filter((id) => id !== String(selectedRoute?.depot_id)).map((id, index) => [id, index + 1]));
    const minX = Math.min(...nodes.map((node) => Number(node.x))), maxX = Math.max(...nodes.map((node) => Number(node.x))), minY = Math.min(...nodes.map((node) => Number(node.y))), maxY = Math.max(...nodes.map((node) => Number(node.y)));
    const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1;
    const scale = Math.min(664 / rangeX, 292 / rangeY), xOffset = 58 + (664 - rangeX * scale) / 2, yOffset = 27 + (292 - rangeY * scale) / 2;
    const position = (node) => [xOffset + (Number(node.x) - minX) * scale, yOffset + (maxY - Number(node.y)) * scale];
    const markers = visitMarkers(visitOrder, byId, position, depots);
    if (fitRoute && selectedRoute) {
      const stops = (selectedRoute.nodes || []).map((id) => byId.get(String(id))).filter(Boolean).map(position);
      if (stops.length) {
        const boxes = [...markers.values()].map((marker) => marker.box);
        const left = Math.min(...stops.map(([x]) => x), ...boxes.map((box) => box.left)), right = Math.max(...stops.map(([x]) => x), ...boxes.map((box) => box.right)), top = Math.min(...stops.map(([, y]) => y), ...boxes.map((box) => box.top)), bottom = Math.max(...stops.map(([, y]) => y), ...boxes.map((box) => box.bottom));
        const width = Math.max(260, right - left + 90, (bottom - top + 60) * 780 / 380), height = width * 380 / 780;
        state.viewBox = [(left + right - width) / 2, (top + bottom - height) / 2, width, height]; plot.setAttribute("viewBox", state.viewBox.join(" "));
      }
    }
    const grid = svg("g", { "aria-hidden": "true" });
    for (let tick = 0; tick <= 4; tick++) {
      const x = 58 + tick * 166, y = 27 + tick * 73;
      grid.append(svg("line", { x1: x, y1: 27, x2: x, y2: 319, stroke: "#f0f3f7", "stroke-width": 1 }), svg("line", { x1: 58, y1: y, x2: 722, y2: y, stroke: "#f0f3f7", "stroke-width": 1 }));
      const dataX = minX + (x - xOffset) / scale, dataY = maxY - (y - yOffset) / scale;
      grid.append(svg("text", { x, y: 340, fill: "#62738a", "font-size": 12, "text-anchor": "middle" }, number(dataX)), svg("text", { x: 45, y: y + 3, fill: "#62738a", "font-size": 12, "text-anchor": "end" }, number(dataY)));
    }
    grid.append(svg("text", { x: 737, y: 340, fill: "#62738a", "font-size": 12 }, "X"), svg("text", { x: 45, y: 14, fill: "#62738a", "font-size": 12 }, "Y")); plot.append(grid);
    const connections = svg("g"); plot.append(connections);
    if (routes.length) {
      for (const route of [...routes.filter((item) => item !== selectedRoute), ...(selectedRoute ? [selectedRoute] : [])]) {
        const visible = !state.selectedDepot || String(route.depot_id) === state.selectedDepot;
        const selected = state.selectedRoute === route.route_id;
        const dimmed = !visible || (state.selectedRoute && !selected);
        const { path, arrows } = routeGeometry(route, byId, position, selected); if (!path) continue;
        const line = svg("path", { d: path, fill: "none", stroke: depotColor(route.depot_id), "stroke-width": selected ? 2.8 : 1.3, opacity: dimmed ? .08 : selected ? .95 : .45, "stroke-linejoin": "round", "stroke-linecap": "round", tabindex: visible ? 0 : -1, role: "button", "aria-pressed": selected, "aria-label": `路线 ${route.route_id}，${route.path || (route.nodes || []).join("、")}`, "data-route": route.route_id, "data-depot": route.depot_id });
        line.append(svg("title", {}, `${route.route_id}\n${route.path || ""}\n需求 ${number(route.demand)} / 容量 ${number(route.capacity)}`));
        const choose = () => selectRoute(route.route_id);
        line.addEventListener("click", choose); line.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); choose(); [...plot.querySelectorAll("path[data-route]")].find((item) => item.dataset.route === route.route_id)?.focus(); } }); connections.append(line);
        const hit = svg("path", { d: path, fill: "none", stroke: "transparent", "stroke-width": 10, "pointer-events": visible ? "stroke" : "none", cursor: "pointer", "aria-hidden": "true" }); hit.addEventListener("click", choose); connections.append(hit);
        for (const arrow of arrows) connections.append(svg("path", { d: "M6 0 L-4.5 -4.5 L-2.2 0 L-4.5 4.5 Z", transform: `translate(${arrow.x} ${arrow.y}) rotate(${arrow.angle}) scale(${arrow.scale})`, fill: depotColor(route.depot_id), stroke: "#fff", "stroke-width": 1, "stroke-linejoin": "round", "pointer-events": "none", "aria-hidden": "true", class: "route-direction", "data-from": arrow.from, "data-to": arrow.to, "data-order": arrow.order }));
      }
    } else {
      for (const item of assignments) { const customer = byId.get(String(item.customer_id)), depot = byId.get(String(item.depot_id)); if (!customer || !depot) continue; const [x1, y1] = position(depot), [x2, y2] = position(customer); connections.append(svg("line", { x1, y1, x2, y2, stroke: depotColor(item.depot_id), "stroke-width": 1, opacity: !state.selectedDepot || state.selectedDepot === String(item.depot_id) ? .22 : .04 })); }
    }
    const guides = svg("g", { "pointer-events": "none", "aria-hidden": "true" }), dots = svg("g"); plot.append(guides, dots);
    for (const node of [...nodes.filter((item) => item.type !== "depot" && !visitOrder.has(String(item.id))), ...nodes.filter((item) => item.type !== "depot" && visitOrder.has(String(item.id))), ...depots]) {
      const [x, y] = position(node), depotId = node.type === "depot" ? node.id : node.depot_id || assignmentMap.get(String(node.id));
      const relevant = (!state.selectedDepot || String(depotId) === state.selectedDepot) && (!selectedRoute || (selectedRoute.nodes || []).map(String).includes(String(node.id)));
      const depot = node.type === "depot", opened = node.opened === true, color = depotColor(depotId), order = visitOrder.get(String(node.id)), marker = markers.get(String(node.id));
      const group = svg("g", { opacity: relevant ? 1 : .18, tabindex: 0, role: "button", "aria-label": `${depot ? "仓库" : "客户"} ${node.id}${order ? `，第 ${order} 站` : ""}`, "data-node": node.id, "data-depot": depotId || "" });
      if (depot) {
        if (opened) group.append(svg("rect", { x: x - 11, y: y - 11, width: 22, height: 22, rx: 6, fill: color, opacity: .1 }));
        group.append(svg("rect", { x: x - 6, y: y - 6, width: 12, height: 12, rx: 3, fill: opened ? color : "#fff", stroke: opened ? "#fff" : color, "stroke-width": 1.5, "stroke-dasharray": node.opened == null ? "2 2" : "none" }));
      } else {
        if (marker?.shifted) { guides.append(svg("line", { x1: x, y1: y, x2: marker.x, y2: marker.y, stroke: color, "stroke-width": .8, opacity: .65, class: "route-number-guide" })); group.append(svg("circle", { cx: x, cy: y, r: 3.5, fill: color, stroke: "#fff", "stroke-width": 1 })); }
        group.append(svg("circle", { cx: marker?.x ?? x, cy: marker?.y ?? y, r: order ? 9.5 : 3.5, fill: color, stroke: "#fff", "stroke-width": order ? 1.5 : 1.1 }));
        if (order) group.append(svg("text", { x: marker.x, y: marker.y + .5, fill: "#fff", "font-size": order >= 10 ? 10 : 11, "font-weight": 600, "text-anchor": "middle", "dominant-baseline": "middle", class: "route-visit-number", "data-node": node.id }, order));
      }
      if (depot || state.labels) group.append(svg("text", { x: (marker?.x ?? x) + (depot || order ? 13 : 6), y: (marker?.y ?? y) + 3, fill: "#3c5573", "font-size": depot ? 14 : 12, "font-weight": depot || order ? 600 : 400, "paint-order": "stroke", stroke: "#fff", "stroke-width": 3 }, `${node.id}${depot && selectedRoute && relevant ? " · 起/终" : ""}`));
      const tooltipText = depot ? `${node.id} · ${opened ? "开放仓库" : node.opened === false ? "未开放仓库" : "候选仓库（状态待定）"}\n容量 ${number(node.capacity)}\n坐标 (${number(node.x)}, ${number(node.y)})` : `${node.id} · 客户\n需求 ${number(node.demand)}${depotId ? ` · 所属 ${depotId}` : ""}\n坐标 (${number(node.x)}, ${number(node.y)})`;
      group.append(svg("title", {}, tooltipText));
      group.addEventListener("pointerenter", (event) => showTooltip(event, tooltipText)); group.addEventListener("pointermove", (event) => showTooltip(event, tooltipText)); group.addEventListener("pointerleave", () => { $("plotTooltip").hidden = true; });
      const selectNode = () => { if (depot) { state.selectedDepot = state.selectedDepot === String(node.id) ? "" : String(node.id); $("depotFilter").value = state.selectedDepot; state.selectedRoute = null; state.viewBox = [0, 0, 780, 380]; renderPlot(); highlightRouteRows(); } else toast(tooltipText.replaceAll("\n", " · ")); };
      group.addEventListener("click", selectNode); group.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); selectNode(); } }); dots.append(group);
    }
    const legendItem = (label, color, square, ring) => { const item = el("span", "legend-item"); const swatch = el("i", `legend-swatch${square ? " legend-square" : ""}${ring ? " legend-ring" : ""}`); swatch.style.setProperty("--legend-color", color); item.append(swatch, el("span", "", label)); return item; };
    if (depots.some((depot) => depot.opened === true)) legend.append(legendItem("开放仓库", colors[0], true, false));
    legend.append(legendItem("客户", colors[0], false, false)); if (depots.some((depot) => depot.opened === false)) legend.append(legendItem("未开放仓库", "#fff", true, true));
    if (depots.some((depot) => depot.opened == null)) { const candidate = legendItem("候选仓库 · 状态待定", "#fff", true, true); candidate.firstChild.style.borderStyle = "dashed"; legend.append(candidate); }
    for (const depot of depots.filter((item) => item.opened === true)) legend.append(legendItem(depot.id, depotColor(depot.id), false, false));
    if (selectedRoute) legend.append(el("span", "route-legend-hint", `箭头表示行驶方向 · 数字表示到访次序${[...markers.values()].some((marker) => marker.shifted) ? " · 细线连接编号与实际节点" : ""}`));
  }
  function showTooltip(event, content) { const box = $("plotContainer").getBoundingClientRect(), tip = $("plotTooltip"); text(tip, content); tip.hidden = false; tip.style.left = `${Math.max(8, Math.min(event.clientX - box.left + 13, box.width - 215))}px`; tip.style.top = `${Math.max(6, Math.min(event.clientY - box.top + 10, box.height - 95))}px`; }
  function renderDepotLoads(depots) {
    const root = $("depotLoads"); root.replaceChildren(); text($("depotCount"), `${depots.filter((item) => item.opened === true).length} 个开放 / ${depots.length} 个记录仓库`);
    if (!depots.length) { root.append(el("p", "depot-empty", "此任务尚无仓库负载结果。")); return; }
    for (const depot of depots) {
      const row = el("div", `depot-load-row${depot.opened === false ? " closed" : ""}`); row.dataset.depot = depot.depot_id;
      const track = el("div", "load-track"), fill = el("div", "load-fill"); const utilization = depot.utilization ?? (depot.capacity ? depot.demand / depot.capacity : 0); fill.style.width = `${Math.max(0, Math.min(100, Number(utilization || 0) * 100))}%`; fill.style.background = depotColor(depot.depot_id); track.append(fill); row.append(depotBadge(depot.depot_id), track, el("span", "load-number", `${number(depot.demand)} / ${number(depot.capacity)}`)); row.title = `${depot.depot_id} · ${depot.opened === false ? "未开放" : depot.opened === true ? `利用率 ${number(Number(utilization || 0) * 100, "%")}` : "候选仓库（状态待定）"} · ${depot.customer_count || 0} 个客户`; root.append(row);
    }
  }
  function renderDetailTable(business, routes) {
    const root = $("detailTable"); root.replaceChildren(); const assignments = business.assignments || [];
    const isRoutes = routes.length > 0; text($("detailTitle"), isRoutes ? "车辆路线" : "客户分配"); text($("detailDescription"), isRoutes ? "点击路线，查看行驶方向、到访次序和简短详情" : "客户整单分配至仓库；这不是串点路线");
    if (!routes.length && !assignments.length) { root.append(el("p", "depot-empty", "暂无可展示的方案明细。")); root.style.padding = "20px"; return; } root.style.padding = "0";
    const table = el("table"), thead = el("thead"), head = el("tr"), body = el("tbody");
    const headers = isRoutes ? ["路线 / 仓库", "访问顺序", "载重 / 容量", "里程（km）", "成本"] : ["客户", "所属仓库", "需求", "独立配送成本"];
    for (const header of headers) head.append(el("th", "", header)); thead.append(head);
    for (const item of isRoutes ? routes : assignments) {
      const row = el("tr");
      if (isRoutes) {
        row.dataset.route = item.route_id; row.dataset.depot = item.depot_id; row.style.setProperty("--depot-color", depotColor(item.depot_id));
        const routeCell = el("td", "route-label"), routeButton = el("button", "route-table-button", item.route_id); routeButton.type = "button"; routeButton.setAttribute("aria-pressed", "false"); routeButton.setAttribute("aria-label", `查看路线 ${item.route_id}`);
        routeCell.append(routeButton, depotBadge(item.depot_id));
        row.append(routeCell, el("td", "path-cell", (item.nodes || []).length ? item.nodes.join(" → ") : item.path || "—"), el("td", "", `${number(item.demand)} / ${number(item.capacity)}`), el("td", "", number(item.distance_km)), el("td", "", number(item.total_cost)));
        const select = () => selectRoute(item.route_id, { fromTable: true });
        row.addEventListener("click", select); row.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); select(); } });
      } else { const depotCell = el("td"); depotCell.append(depotBadge(item.depot_id)); row.append(el("td", "route-label", item.customer_id), depotCell, el("td", "", number(item.demand)), el("td", "", number(item.delivery_cost))); }
      body.append(row);
    }
    table.append(thead, body); root.append(table); highlightRouteRows();
  }
  function highlightRouteRows() { for (const row of document.querySelectorAll("tr[data-route]")) { const selected = row.dataset.route === state.selectedRoute; row.classList.toggle("selected", selected); row.querySelector(".route-table-button")?.setAttribute("aria-pressed", String(selected)); row.style.opacity = state.selectedDepot && row.dataset.depot !== state.selectedDepot ? ".35" : "1"; } }
  function jsonEvidence(root, data, empty = "此结果没有保存这项记录。") { root.replaceChildren(data ? el("pre", "", JSON.stringify(data, null, 2)) : el("p", "", empty)); }
  function renderEvidence(result) {
    const summary = result.summary || {}, model = summary.models?.[0] || {}, retrieval = result.retrieval || {}, pipeline = result.pipeline || {}, overview = $("modelOverview"); overview.replaceChildren();
    const info = (label, value) => { const item = el("dl", "info-item"); item.append(el("dt", "", label), el("dd", "", value)); overview.append(item); };
    info("数学模型", `${model.model_id || result.selection?.model_id || result.mapping?.model_id || "—"} · ${model.model_name || "待确定"}`);
    info("求解器", [...new Set((summary.models || []).map((item) => item.solver).filter(Boolean))].join(" / ").toUpperCase() || pipeline.solver || "—");
    info("模型选择", result.selection?.provider_model || result.selection?.provider || state.bootstrap?.config?.selector || "—");
    if (result.llm) {
      info("模型来源", `${result.llm.provider_label || ""} · ${result.llm.model || "未记录"}`);
      const thinking = result.llm.chat_template_kwargs?.enable_thinking;
      if (typeof thinking === "boolean") {
        const chatThinking = result.llm_chat?.chat_template_kwargs?.enable_thinking ?? thinking;
        const mappingThinking = result.llm_mapping?.chat_template_kwargs?.enable_thinking ?? result.llm.mapping_enable_thinking ?? thinking;
        info("思考模式", `聊天${chatThinking ? "开启" : "关闭"} / 参数映射${mappingThinking ? "开启" : "关闭"}`);
      }
      for (const [label, trace] of [["聊天", result.llm_chat], ["参数映射", result.llm_mapping]]) {
        if (!trace) continue;
        const usage = trace.usage || {}, context = trace.context;
        info(`${label}用量`, `输入 ${usage.prompt_tokens ?? "—"} / 输出 ${usage.completion_tokens ?? "—"} token · ${trace.finish_reason || "未记录"}`);
        if (context) info(`${label}上下文`, `${context.prompt_tokens} 输入 + ${context.output_budget} 输出预算 / ${context.effective_context} token`);
      }
    }
    info("知识加载", retrieval.backend === "model_pack" ? `按已选模型完整加载 · ${(retrieval.context || []).length} 条规则` : `${retrieval.backend || pipeline.retrieval_backend || "—"}${retrieval.top_k ? ` · Top ${retrieval.top_k}` : ""}`);
    info("实例", model.instance_id || result.mapping?.instance_id || "—");
    if (result.binding_state) {
      info("绑定版本", result.binding_state.revision.slice(0, 12));
      const updates = result.binding_state.scenario?.parameter_values || {};
      if (Object.keys(updates).length) info("已确认参数修改", Object.entries(updates).map(([key, value]) => `${key} = ${number(value)}`).join(" · "));
      info("需求记录", `${result.mapping?.requirement_coverage?.length || 0} 条原始需求；记录与证据一致性已检查，语义仍需审阅`);
    }
    info("独立校验", summary.subproblem_count ? `${summary.verified_subproblem_count ?? 0} / ${summary.subproblem_count} 个子模型通过` : "尚无已验证求解结果");
    const costs = summary.costs || {};
    if (Object.keys(costs).length) info("成本组成", Object.entries(costs).map(([key, value]) => `${costLabels[key] || key} ${number(value)}`).join(" · "));
    const timings = $("resultTimings"); timings.replaceChildren();
    const phaseRecords = pipeline.phase_timings || pipeline.timings || [];
    for (const phase of Array.isArray(phaseRecords) ? phaseRecords : []) { if (typeof phase !== "object") continue; const chip = el("span", "timing-chip", phase.label || stageLabels[phase.stage] || phase.stage || "阶段"); chip.append(el("strong", "", seconds(phase.seconds))); timings.append(chip); }
    jsonEvidence($("selectionEvidence"), result.selection);
    const knowledge = retrieval.context || retrieval.records || [], root = $("ragEvidence"); root.replaceChildren(); text($("knowledgeCount"), `${knowledge.length} 条`);
    if (!knowledge.length) root.append(el("p", "", "此结果未保存检索知识。"));
    for (const item of knowledge) { const section = el("section", "knowledge-item"), title = el("div", "knowledge-title", item.title || item.id || "知识条目"); if (item.category) title.append(el("span", "knowledge-category", item.category)); section.append(title, el("div", "", item.text || ""), el("div", "knowledge-source", `${item.source || ""}${item.model_id ? ` · ${item.model_id}` : ""}${item.distance !== undefined ? ` · 距离 ${number(item.distance)}` : ""}`)); root.append(section); }
    jsonEvidence($("mappingEvidence"), result.binding_state ? {original_mapping: result.mapping, current_parameter_values: result.binding_state.scenario.parameter_values, last_changes: result.binding_state.last_changes || []} : result.mapping);
    const notes = $("notesEvidence"); notes.replaceChildren(); const list = el("ul", "notes-list");
    const allNotes = [...(summary.warning_messages || []), ...(summary.notes || []), ...(summary.unsupported_requirements || [])].map((item) => typeof item === "string" ? item : item.message || JSON.stringify(item));
    if (!allNotes.length) allNotes.push("此任务没有额外备注。坐标图展示数据的平面位置，不代表真实道路地图。");
    for (const note of [...new Set(allNotes)]) list.append(el("li", "", note)); notes.append(list);
    const files = $("fileList"); files.replaceChildren();
    for (const file of result.files || []) { const url = safeUrl(file.url); if (!url) continue; const link = el("a", "file-link", file.name); link.href = url; if (String(file.name).endsWith(".html")) { link.target = "_blank"; link.rel = "noopener"; } else link.setAttribute("download", ""); files.append(link); }
    text($("outputPath"), result.output_path ? `本地保存位置：${result.output_path}` : "");
  }
  function updateReportLink(result) { const file = (result.files || []).find((item) => item.name === "report.html") || (result.files || []).find((item) => String(item.name).endsWith(".html")); const url = file && safeUrl(file.url); $("reportLink").hidden = !url; if (url) $("reportLink").href = url; }
  function setTab(name) { state.tab = name; for (const button of document.querySelectorAll(".tab")) { const selected = button.dataset.tab === name; button.classList.toggle("active", selected); button.setAttribute("aria-selected", String(selected)); } for (const key of ["overview", "details", "evidence"]) $(`${key}View`).hidden = key !== name; }

  async function openHistory() {
    if (!$("historyDialog").open) $("historyDialog").showModal(); renderHistory();
    try { const data = await api("/api/history"); state.history = data.history || []; renderHistory(); } catch (error) { toast(error.message); }
  }
  function sourceKind(item) { return item.source_kind || (String(item.id || "").startsWith("experiments/") ? "batch_unknown" : "single"); }
  function resultDate(value) { const parsed = new Date(value); if (!value || Number.isNaN(parsed.getTime())) return ""; const parts = new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit" }).formatToParts(parsed); return ["year", "month", "day"].map((kind) => parts.find((item) => item.type === kind)?.value || "").join("-"); }
  function historyButton(item) {
    const row = el("div", "history-row");
    const button = el("button", `history-item${item.id === state.resultId ? " active" : ""}`), info = el("div");
    info.append(el("div", "history-title", item.title || item.instance_id || item.id), el("div", "history-meta", `${item.instance_id || ""} · ${item.status_label || item.status || ""}${item.llm?.model ? ` · ${item.llm.provider_label || item.llm.model}` : ""}${item.created_at ? ` · ${date(item.created_at)}` : " · 时间未记录"}`));
    button.append(el("span", "history-model", item.model_id || "—"), info, el("span", "history-arrow", "↗"));
    button.addEventListener("click", () => { $("historyDialog").close(); loadResult(item.id, false, "history"); });
    const remove = el("button", "button history-delete", "删除记录");
    remove.type = "button";
    remove.setAttribute("aria-label", `删除历史记录：${item.title || item.id}`);
    remove.disabled = state.busy;
    remove.addEventListener("click", () => deleteHistory(item, remove));
    row.append(button, remove); return row;
  }
  async function deleteHistory(item, button) {
    if (state.busy || !window.confirm(`删除历史记录“${item.title || item.id}”？\n记录将从历史列表移除，并取消对话中的引用。原始结果文件仍保留在磁盘上。`)) return;
    button.disabled = true;
    try {
      await api("/api/history/delete", { method: "POST", body: JSON.stringify({ result_id: item.id }) });
      state.history = state.history.filter((entry) => entry.id !== item.id);
      if (state.contextId === item.id) { state.contextId = null; state.contextDetails = null; }
      // Invalidate any result fetch that started before deletion.
      state.resultSequence += 1; $("resultLoading").hidden = true;
      if (state.resultId === item.id) {
        state.result = null; state.resultId = null; state.latestJob = null;
        $("resultContent").hidden = true; $("resultEmpty").hidden = false;
        $("resultOrigin").hidden = true; $("reportLink").hidden = true; $("runCard").hidden = true;
        text($("resultTitle"), "方案可视化"); text($("resultSubtitle"), "请选择历史结果或提交新任务");
        text($("resultStatus"), "等待结果"); $("resultStatus").className = "status-badge neutral";
      }
      renderHistory(); updateContext();
      try { await refreshSession(); } catch { toast("记录已删除；请刷新页面以同步对话引用。"); return; }
      toast("历史记录已删除，原始结果文件已保留。");
    } catch (error) { toast(`删除失败：${error.message}`); }
    finally { button.disabled = state.busy; }
  }
  function renderHistory() {
    const root = $("historyList"), search = $("historySearch").value.trim().toLowerCase(); root.replaceChildren();
    const hasUnknown = state.history.some((item) => sourceKind(item) === "batch_unknown");
    let unknownTab = $("historyUnknownSource");
    if (hasUnknown && !unknownTab) {
      unknownTab = el("button", "", "其他批量 "); unknownTab.id = "historyUnknownSource";
      unknownTab.dataset.source = "batch_unknown"; unknownTab.append(el("span"));
      unknownTab.addEventListener("click", () => { state.historySource = "batch_unknown"; renderHistory(); });
      $("historySourceTabs").append(unknownTab);
    }
    if (unknownTab) unknownTab.hidden = !hasUnknown;
    const from = $("historyDateFrom").value, to = $("historyDateTo").value, model = $("historyModel").value;
    const sourceEntries = state.history.filter((item) => sourceKind(item) === state.historySource);
    const entries = sourceEntries.filter((item) => {
      const stamp = resultDate(item.created_at);
      return (!model || item.model_id === model) && (!from || stamp && stamp >= from) && (!to || stamp && stamp <= to) && [item.title, item.model_id, item.instance_id, item.status_label, item.id, item.batch_label, item.case_id].join(" ").toLowerCase().includes(search);
    }).sort((a, b) => (Date.parse(b.created_at) || 0) - (Date.parse(a.created_at) || 0));
    for (const button of $("historySourceTabs").querySelectorAll("button[data-source]")) { const active = button.dataset.source === state.historySource; button.classList.toggle("active", active); button.setAttribute("aria-pressed", String(active)); text(button.querySelector("span"), state.history.filter((item) => sourceKind(item) === button.dataset.source).length); }
    const labels = { single: "单次运行", batch_fixed: "固定配置批量", batch_live: "JEV + DeepSeek 批量", batch_unknown: "来源未记录的批量" };
    text($("historyCounts"), `${labels[state.historySource]}：共 ${sourceEntries.length} 条，当前显示 ${entries.length} 条 · 全部保存结果 ${state.history.length} 条`);
    if (from && to && from > to) { root.append(el("p", "depot-empty", "起始日期晚于截止日期，请调整日期范围。")); return; }
    if (!entries.length) { root.append(el("p", "depot-empty", state.history.length ? "没有匹配的结果。" : "暂无保存结果。完成一次任务后会显示在这里。")); return; }
    if (state.historySource === "single") { for (const item of entries) root.append(historyButton(item)); return; }
    const batches = new Map();
    for (const item of entries) { const key = item.batch_id || item.batch_label || "saved_batch"; if (!batches.has(key)) batches.set(key, []); batches.get(key).push(item); }
    for (const [batchId, batchEntries] of batches) {
      const batch = el("details", "history-batch"), summary = el("summary"), title = el("div"), cases = new Map();
      for (const item of batchEntries) { const key = item.case_id || item.instance_id || item.id; if (!cases.has(key)) cases.set(key, []); cases.get(key).push(item); }
      title.append(el("strong", "", batchEntries[0].batch_label || batchId), el("span", "history-meta", `${labels[state.historySource]} · ${batchEntries[0].created_at ? date(batchEntries[0].created_at) : "时间未记录"} · ${cases.size} 个数据组 / ${batchEntries.length} 条结果`));
      summary.append(title, el("span", "chevron", "⌄")); batch.append(summary);
      const caseList = el("div", "history-case-list");
      for (const [caseId, caseEntries] of cases) {
        const group = el("details", "history-case"), groupSummary = el("summary"), info = el("div");
        info.append(el("strong", "", caseEntries[0].instance_id || caseId), el("span", "history-meta", `${caseEntries.map((entry) => entry.model_id).filter(Boolean).sort().join(" → ")} · ${caseEntries.length} 条结果`));
        groupSummary.append(info, el("span", "chevron", "⌄")); group.append(groupSummary);
        const items = el("div", "history-case-items"); for (const item of caseEntries.sort((a, b) => String(a.model_id).localeCompare(String(b.model_id)))) items.append(historyButton(item)); group.append(items); caseList.append(group);
      }
      batch.append(caseList); root.append(batch);
    }
  }
  async function newSession() { if (state.busy || state.contextPending) return; $("newSessionButton").disabled = true; try { const session = await api("/api/session", { method: "POST", body: "{}" }); state.contextId = null; state.contextDetails = null; state.latestJob = null; renderSession(session); storageSet(session.id); updateContext(); if (state.result) { state.resultOrigin = "history"; text($("resultOrigin"), "历史结果 · 仅浏览"); renderResult(state.result); } else $("runCard").hidden = true; $("prompt").value = ""; $("prompt").focus(); toast("已开始新对话，不继承引用。历史求解结果仍然保留。"); } catch (error) { toast(error.message); } finally { $("newSessionButton").disabled = state.busy; } }
  async function examplePrompt(key) {
    if (state.busy) return;
    const preferred = state.bootstrap?.instances?.find((item) => item.instance_id === "G20-5-1a__Q70") || state.bootstrap?.instances?.[0];
    const instance = $("instanceSelect").value || preferred?.instance_id || "20-5-1a__Q70";
    const prompts = { location: `使用 ${instance} 数据，选择开放仓库并把所有客户分配至仓库，在仓库容量约束下使开放成本与配送成本之和最小。请帮我求解。`, route: "沿用当前结果的开放仓库和每个客户的所属仓库，规划常规车辆配送路线。每个客户恰好访问一次，车辆载重不能超过容量，每条路线从所属仓库出发并返回。请继续求解。", explain: "请解释当前结果：用了什么模型，成本由哪些部分组成，仓库负载如何，方案是否通过独立校验，gap 代表什么？不要重新求解。" };
    if (key !== "location" && state.contextId !== state.resultId) {
      if (!state.resultId) { toast("请先查看一个结果，再点击“沿用此结果”或继续规划快捷项。"); openHistory(); return; }
      if (!await setContext(state.resultId)) return;
    }
    if (key !== "location" && !state.contextId) { toast("请先查看并沿用一个结果，再继续规划或解释。"); openHistory(); return; }
    text($("prompt"), ""); $("prompt").value = prompts[key]; $("prompt").focus();
  }
  async function initialize() {
    setBusy(true);
    try {
      const bootstrap = await api("/api/bootstrap"); state.bootstrap = bootstrap; state.history = bootstrap.history || [];
      const config = bootstrap.config || {};
      $("llmSource").replaceChildren();
      for (const profile of config.llm_profiles || [{ id: "deepseek", label: "DeepSeek API" }, { id: "qwen_local", label: "本地 Qwen3.5 9B" }]) {
        const option = el("option", "", profile.label); option.value = profile.id; option.disabled = profile.available === false; $("llmSource").append(option);
      }
      let preferredSource = config.llm_source || "deepseek";
      try { preferredSource = localStorage.getItem(MODEL_STORAGE_KEY) || preferredSource; } catch { /* Optional preference. */ }
      if ([...$("llmSource").options].some((option) => option.value === preferredSource && !option.disabled)) $("llmSource").value = preferredSource;
      updateModelSource();
      $("timeLimit").value = config.time_limit ?? 300; $("mipGap").value = (config.mip_gap ?? .08) * 100; $("solver").value = "copt"; updateSettingsLabel();
      for (const item of bootstrap.instances || []) { const option = el("option", "", `${item.instance_id} · ${item.customers} 客户 / ${item.depots} 仓库`); option.value = item.instance_id; $("instanceSelect").append(option); }
      const stored = storageGet(); let session;
      if (stored) { try { session = await api(query("/api/session", stored)); } catch { /* An expired local session starts a new conversation. */ } }
      if (!session) session = await api("/api/session", { method: "POST", body: "{}" });
      storageSet(session.id); state.contextId = session.reference_result_id || null; setBusy(false); renderSession(session);
      $("connectionDot").classList.add("connected"); text($("connectionLabel"), "本地服务已连接");
      const initialResult = session.current_result_id || state.history.find((item) => sourceKind(item) === "single")?.id || state.history[0]?.id;
      const originatedHere = session.messages?.some((message) => message.result_id === initialResult && message.action === "run");
      if (initialResult) await loadResult(initialResult, false, originatedHere ? "current" : session.current_result_id ? "conversation" : "history");
      if (session.active_job_id) pollJob(session.active_job_id);
    } catch (error) {
      setBusy(false); $("sendButton").disabled = true; text($("connectionLabel"), "本地服务连接失败"); text($("composerHint"), "请刷新页面重试，或重新打开启动快捷方式"); toast(error.message);
    }
  }

  $("chatForm").addEventListener("submit", sendPrompt);
  $("llmSource").addEventListener("change", updateModelSource);
  $("prompt").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); if (!state.busy) $("chatForm").requestSubmit(); } });
  for (const id of ["timeLimit", "mipGap"]) $(id).addEventListener("input", updateSettingsLabel);
  $("instanceSelect").addEventListener("change", () => { if (!$("instanceSelect").value) return; const value = $("instanceSelect").value; $("prompt").value = $("prompt").value.trim() ? `${$("prompt").value.trim()}\n数据实例：${value}。` : `使用 ${value} 数据，`; $("prompt").focus(); });
  for (const button of document.querySelectorAll(".example-button")) button.addEventListener("click", () => examplePrompt(button.dataset.example));
  for (const button of document.querySelectorAll(".tab")) button.addEventListener("click", () => setTab(button.dataset.tab));
  $("historyButton").addEventListener("click", openHistory); $("emptyHistoryButton").addEventListener("click", openHistory); $("closeHistory").addEventListener("click", () => $("historyDialog").close()); $("historySearch").addEventListener("input", renderHistory);
  for (const button of $("historySourceTabs").querySelectorAll("button[data-source]")) button.addEventListener("click", () => { state.historySource = button.dataset.source; renderHistory(); });
  for (const id of ["historyModel", "historyDateFrom", "historyDateTo"]) $(id).addEventListener("change", renderHistory);
  $("historyDialog").addEventListener("click", (event) => { const bounds = $("historyDialog").getBoundingClientRect(); if (event.target === $("historyDialog") && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) $("historyDialog").close(); });
  $("newSessionButton").addEventListener("click", newSession); $("clearContext").addEventListener("click", () => setContext(null)); $("reuseResultButton").addEventListener("click", () => setContext(state.resultId));
  $("depotFilter").addEventListener("change", () => { state.selectedDepot = $("depotFilter").value; state.selectedRoute = null; state.viewBox = [0, 0, 780, 380]; renderPlot(); highlightRouteRows(); });
  $("routeSelect").addEventListener("change", () => selectRoute($("routeSelect").value, { toggle: false }));
  $("labelsToggle").addEventListener("click", () => { state.labels = !state.labels; state.fitRoute = Boolean(state.selectedRoute); $("labelsToggle").setAttribute("aria-pressed", String(state.labels)); renderPlot(); });
  $("plotReset").addEventListener("click", () => { state.selectedRoute = null; state.selectedDepot = ""; $("depotFilter").value = ""; state.viewBox = [0, 0, 780, 380]; renderPlot(); highlightRouteRows(); });
  $("networkPlot").addEventListener("wheel", (event) => {
    if (!state.result?.plot?.nodes?.length) return; event.preventDefault(); const box = $("networkPlot").getBoundingClientRect();
    const [x, y, width, height] = state.viewBox, factor = event.deltaY > 0 ? 1.12 : .89, nextWidth = Math.min(1170, Math.max(156, width * factor)), nextHeight = nextWidth * 380 / 780, rx = Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)), ry = Math.max(0, Math.min(1, (event.clientY - box.top) / box.height));
    state.viewBox = [x + (width - nextWidth) * rx, y + (height - nextHeight) * ry, nextWidth, nextHeight]; $("networkPlot").setAttribute("viewBox", state.viewBox.join(" ")); $("plotTooltip").hidden = true;
  }, { passive: false });
  initialize();
})();
