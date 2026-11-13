"""Local conversation workspace. No frontend dependencies or public hosting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, urlencode, urlsplit
import uuid

from orllm.llm_profiles import load_profile, profile_options, model_identity

ROOT = Path(__file__).resolve().parent
APP_KIND = "orllm-chat-workspace-v1"
UI_VERSION = "20261101-verified-continuity-v3"
SOLVE_LABELS = {"optimal": "达到最优性容差", "feasible_limit": "保留可行方案", "infeasible": "模型不可行",
                "unbounded": "模型无界", "no_solution": "未找到已验证方案", "error": "执行错误",
                "not_solved": "尚未求解", "invalid_solution": "解未通过独立校验"}
LABELS = {"chat": "理解问题", "selection": "JEV 选择模型", "retrieval": "加载模型知识",
          "mapping": "模型整理任务", "build": "校验与建模", "solve": "求解与复核",
          "report": "整理结果", "done": "完成"}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, ValueError):
        return default


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat()


def project_id(root):
    return sha256(str(Path(root).resolve()).encode("utf-8")).hexdigest()[:16]


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("求解设置必须是对象。")
    solver = settings.get("solver", "copt")
    seconds = settings.get("time_limit", 300)
    gap = settings.get("mip_gap", 0.08)
    if solver not in {"copt", "none"}:
        raise ValueError("本工作台使用 COPT 求解，请刷新页面后重新提交。")
    for value in (seconds, gap):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("时间和 gap 必须是有限数值。")
    if not 0 < seconds <= 86400 or not 0 <= gap < 1:
        raise ValueError("时间需在 0～86400 秒内，gap 需在 0～100% 范围内。")
    return {"solver": solver, "time_limit": seconds, "mip_gap": gap}


def finite_number(value, minimum=None, maximum=None):
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if valid and (minimum is None or value >= minimum) and (maximum is None or value < maximum):
        return value
    return None


def result_quality(summary, build_report, mapping=None):
    """Separate verified feasibility from the recorded optimality requirement."""
    rows = summary.get("models", [])
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    declared = summary.get("subproblem_count")
    if isinstance(declared, bool) or not isinstance(declared, int) or declared < 0:
        declared = None
    count = max(declared or 0, len(rows))
    target = finite_number(build_report.get("requested_mip_gap"), 0, 1)
    limit = finite_number(build_report.get("time_limit_seconds"), 0)
    limit = limit if limit and limit > 0 else None
    parts = []
    for row in rows:
        check = row.get("objective_check")
        primal = row.get("primal_check")
        verified = (row.get("has_verified_solution") is True and isinstance(primal, dict) and primal.get("valid") is True
                    and finite_number(row.get("objective")) is not None
                    and (check is None or isinstance(check, dict) and check.get("valid") is True))
        gap = finite_number(row.get("mip_gap"), 0) if verified else None
        runtime = finite_number(row.get("runtime_seconds"), 0)
        status = row.get("status", "unknown")
        if isinstance(row.get("artifacts"), dict) and row["artifacts"].get("solution_reused"):
            termination = "复用已验证方案"
        elif status == "optimal":
            termination = "达到最优性容差" if gap is None or gap > 1e-8 else "最优（gap 约为 0）"
        elif status == "feasible_limit":
            termination = ("达到时间上限，保留可行方案" if limit is not None and runtime is not None
                           and runtime >= limit - max(0.1, limit * 0.001) else "保留可行方案，求解因限制或提前停止")
        else:
            termination = row.get("status_label") or {"not_solved": "尚未求解", "no_solution": "没有已验证的可行解"}.get(status, status)
        parts.append({"depot_id": row.get("depot_id") or "全部仓库", "model_id": row.get("model_id"),
                      "status": status, "status_label": row.get("status_label") or status, "verified": verified,
                      "gap": gap, "runtime_seconds": runtime,
                      "objective": row.get("objective") if verified else None, "termination_label": termination,
                      "gap_state": "unknown" if gap is None or target is None else "met" if gap <= target + 1e-10 else "unmet"})
    verified_count = sum(part["verified"] for part in parts)
    complete = (count > 0 and declared in (None, len(rows)) and verified_count == count
                and summary.get("complete_solution") is True)
    state = "verified_complete" if complete else "verified_partial" if verified_count else "none"
    actual = max(part["gap"] for part in parts) if complete and all(part["gap"] is not None for part in parts) else None
    gap_state = "unknown" if actual is None or target is None else "met" if actual <= target + 1e-10 else "unmet"
    message = {"verified_complete": "完整方案已通过独立校验。", "verified_partial": "仅部分方案已通过独立校验，不能当作完整方案。",
               "none": "尚无通过独立校验的可行方案。"}[state]
    if gap_state == "unmet":
        message += f"实际最大 gap 为 {actual:.2%}，尚未达到本次目标 {target:.2%}。"
    elif gap_state == "met":
        message += f"实际最大 gap 为 {actual:.2%}，已达到本次目标 {target:.2%}。"
    elif complete:
        message += "部分 gap 或目标 gap 未记录，无法判断是否达标。"
    return {"solution_state": state, "verified_count": verified_count, "total_count": count,
            "actual_max_gap": actual, "requested_gap": target, "gap_state": gap_state,
            "time_limit_seconds": limit, "subproblems": parts, "message": message}


def result_progress(summary, pipeline, web_job):
    phases = web_job.get("phase_timings") or pipeline.get("phase_timings") or []
    phases = [dict(phase) for phase in phases if isinstance(phase, dict)] if isinstance(phases, list) else []
    problems = web_job.get("subproblems") or pipeline.get("subproblems") or []
    if not problems:
        problems = [{"depot_id": row.get("depot_id") or "全部仓库", "subproblem_index": i + 1,
                     "subproblem_count": len(summary.get("models", [])),
                     "status": "not_started" if row.get("status") == "not_solved" else "done" if row.get("status") else "unknown",
                     "solve_status": row.get("status"), "runtime_seconds": row.get("runtime_seconds"),
                     "status_label": row.get("status_label") or SOLVE_LABELS.get(row.get("status"), "未记录"),
                     "mip_gap": row.get("mip_gap"), "verified": row.get("has_verified_solution") is True}
                    for i, row in enumerate(summary.get("models", [])) if isinstance(row, dict)]
    return {"phase_timings": phases, "phases_recorded": bool(phases),
            "elapsed_seconds": web_job.get("elapsed_seconds", pipeline.get("elapsed_seconds")),
            "solver_runtime_seconds": summary.get("runtime_seconds"),
            "current_stage": web_job.get("stage", pipeline.get("stage", "done")),
            "status": web_job.get("status", pipeline.get("status", summary.get("status"))),
            "subproblems": problems,
            "completed_subproblem_count": sum(p.get("status") == "done" for p in problems)}


def history_source(result_id, outputs):
    parts = Path(result_id).parts
    metadata = {"source_kind": "single", "source_label": "单次运行", "batch_id": None,
                "batch_label": None, "case_id": None}
    if not parts or parts[0] != "experiments" or len(parts) < 4:
        return metadata
    batch_id = parts[1]
    manifest = read_json(Path(outputs) / "experiments" / batch_id / "batch_manifest.json", {})
    if not isinstance(manifest, dict):
        manifest = {}
    settings = manifest.get("settings", {}) if isinstance(manifest, dict) else {}
    kind = settings.get("pipeline") if isinstance(settings, dict) else None
    if kind == "live":
        source_kind, label = "batch_live", "JEV＋DeepSeek 批量"
    elif kind == "fixed" or kind is None and type(manifest.get("api_calls")) is int and manifest["api_calls"] == 0:
        # The original fixed batch predates settings.pipeline, but records zero API calls.
        source_kind, label = "batch_fixed", "固定配置批量"
    else:
        source_kind, label = "batch_unknown", "来源未记录的批量"
    stamp = re.search(r"(\d{8})_(\d{6})", batch_id)
    date_label = batch_id
    if stamp:
        try:
            date_label = datetime.strptime("".join(stamp.groups()), "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M")
        except ValueError:
            pass
    return {"source_kind": source_kind, "source_label": label, "batch_id": batch_id,
            "batch_label": f"{date_label} · {label}", "case_id": parts[2]}


class Workspace:
    def __init__(self, root=ROOT):
        self.root = Path(root).resolve()
        self.outputs = self.root / "outputs"
        self.sessions_path = self.outputs / "chat/sessions"
        self.deleted_path = self.outputs / "chat/deleted_results.json"
        self.lock = threading.RLock()
        self.sessions = {}
        self.jobs = {}
        self.active_job_id = None
        self._history_cache = (0, [])
        self.identity = project_id(self.root)

    def result_path(self, result_id):
        if not isinstance(result_id, str) or not result_id or len(result_id) > 500:
            raise ValueError("结果编号无效。")
        relative = Path(result_id)
        path = (self.outputs / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.outputs.resolve()) or path == self.outputs.resolve():
            raise ValueError("结果路径不在输出目录内。")
        if path.relative_to(self.outputs.resolve()).as_posix() in self.deleted_results():
            raise ValueError("该历史记录已删除。")
        if not (path / "summary.json").is_file():
            raise ValueError("该结果尚未生成或不存在。")
        return path

    def deleted_results(self):
        if not self.deleted_path.exists():
            return {}
        records = read_json(self.deleted_path)
        if not isinstance(records, dict):
            raise ValueError("历史删除记录无法读取，请检查 deleted_results.json。")
        return records

    def delete_result(self, result_id):
        with self.lock:
            if self.active_job_id:
                raise RuntimeError("请等待当前任务完成后再删除历史记录。")
            self.result_path(result_id)
            if result_id not in {row["id"] for row in self.history()}:
                raise ValueError("只能删除历史列表中的记录。")
            records = self.deleted_results()
            records[result_id] = {"deleted_at": now()}
            save_json(self.deleted_path, records)
            self._history_cache = (0, [])
            # Session reads clear references using the persisted deletion index,
            # including sessions loaded after a server restart.
            return {"deleted_id": result_id, "files_preserved": True}

    def history(self):
        with self.lock:
            if time.monotonic() - self._history_cache[0] < 3:
                deleted = self.deleted_results()
                return [row for row in self._history_cache[1] if row["id"] not in deleted]
        paths = list((self.outputs / "runs").glob("*/summary.json"))
        paths += list((self.outputs / "experiments").glob("*/*/M[1-4]/summary.json"))
        rows = []
        deleted = self.deleted_results()
        for path in paths:
            if not path.resolve().is_relative_to(self.outputs.resolve()):
                continue
            summary = read_json(path, {})
            if not isinstance(summary, dict) or not summary.get("schema_version"):
                continue
            models = summary.get("models", [])
            model = models[0] if models else {}
            selection = read_json(path.parent / "selection.json", {})
            model_id = model.get("model_id") or selection.get("model_id")
            instance = model.get("instance_id")
            result_id = path.parent.relative_to(self.outputs).as_posix()
            if result_id in deleted:
                continue
            run_metadata = read_json(path.parent / "web_job.json", {})
            if not run_metadata.get("llm"):
                run_metadata = read_json(path.parent / "pipeline_report.json", {})
            rows.append({"id": result_id, "llm": run_metadata.get("llm"),
                         "title": f"{model_id or '任务'} · {instance or summary.get('scenario_id') or path.parent.name}",
                         "model_id": model_id, "instance_id": instance, "status": summary.get("status"),
                         "status_label": summary.get("status_label"), "created_at": summary.get("generated_at", ""),
                         **history_source(result_id, self.outputs), "_mtime": path.stat().st_mtime_ns})
        rows.sort(key=lambda r: r.pop("_mtime"), reverse=True)
        with self.lock:
            self._history_cache = (time.monotonic(), rows)
        return rows

    def result(self, result_id):
        path = self.result_path(result_id)
        summary = read_json(path / "summary.json", {})
        selection = read_json(path / "selection.json", {})
        retrieval = read_json(path / "retrieval.json", {})
        mapping = read_json(path / "mapping.json", {})
        models = summary.get("models", [])
        instance_id = models[0].get("instance_id") if models else mapping.get("instance_id")
        base = {}
        if instance_id:
            from orllm.data import load_base
            try:
                base = load_base(self.root, instance_id)
            except (ValueError, OSError):
                pass
        state = read_json(path / 'binding_state.json', None)
        if state:
            for field, value in state['scenario'].get('parameter_values', {}).items():
                parts = field.split('.')
                if len(parts) == 2 and parts[0] in {'q', 'U'}:
                    base.get(parts[0], {})[parts[1]] = value
        business = summary.get("business", {})
        assignments = list(business.get("assignments", []))
        # Show only decoded, verified decisions; an unsolved mapping is not a solution.
        if summary.get("verified_subproblem_count", 0) and not assignments:
            for route in business.get("routes", []):
                assignments.extend({"customer_id": c, "depot_id": route["depot_id"]}
                                   for c in route.get("nodes", [])[1:-1])
        belong = {r["customer_id"]: r["depot_id"] for r in assignments}
        depots = {d["depot_id"]: d for d in business.get("depots", [])}
        fixed_opened = mapping.get("opened_depots") if mapping.get("status") == "ready" and mapping.get("model_id") in {"M2", "M3", "M4"} else None
        nodes = []
        for node_id, xy in base.get("coordinates", {}).items():
            is_depot = node_id in base.get("D", [])
            nodes.append({"id": node_id, "type": "depot" if is_depot else "customer", **xy,
                          "demand": base.get("q", {}).get(node_id),
                          "capacity": base.get("U", {}).get(node_id),
                          "opened": (depots[node_id].get("opened") if node_id in depots else
                                     node_id in fixed_opened if isinstance(fixed_opened, list) else None) if is_depot else None,
                          "depot_id": belong.get(node_id)})
        files = []
        for file in sorted(path.iterdir()):
            if self.allowed_artifact(path, file.name):
                files.append({"name": file.name, "url": "/artifact?" + urlencode({"id": result_id, "name": file.name})})
        pipeline = read_json(path / "pipeline_report.json", {})
        web_job = read_json(path / "web_job.json", {})
        build_report = read_json(path / "build_report.json", {})
        if isinstance(web_job.get("phase_timings"), list):
            pipeline = {**pipeline, "phase_timings": web_job["phase_timings"],
                        "chat_elapsed_seconds": web_job.get("elapsed_seconds")}
        update=None
        if state:
            from orllm.workflow import update_contract
            update=update_contract(self.root,state)
        return {"id": result_id, "summary": summary, "selection": selection, "retrieval": retrieval,
                "binding_state":state,"update_contract":update,
                "llm": web_job.get("llm") or pipeline.get("llm"),
                "llm_chat": web_job.get("llm_chat"), "llm_mapping": pipeline.get("llm_mapping"),
                "mapping": mapping, "pipeline": pipeline,
                "quality": result_quality(summary, build_report, mapping),
                "progress": result_progress(summary, pipeline, web_job),
                "source": history_source(result_id, self.outputs),
                "plot": {"nodes": nodes, "assignments": assignments, "routes": business.get("routes", [])},
                "files": files, "output_path": str(path)}

    @staticmethod
    def allowed_artifact(path, name):
        if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
            return False
        file = (path / name).resolve()
        allowed = name in {"report.html", "summary.md", "summary.json", "selection.json", "retrieval.json",
                            "mapping.json", "build_report.json", "pipeline_report.json", "problem.txt",
                            "llm_mapping.json", "llm_chat.json", "planning_context.json", "binding_state.json", "mapping_contract.json", "mapping_review.json", "incremental_update.json"}
        allowed |= name.endswith((".csv", ".lp", ".instance.json", ".model.json", ".solution.json", ".solver.log"))
        return bool(allowed and file.is_relative_to(path.resolve()) and file.is_file())

    def bootstrap(self):
        config = read_json(self.root / "config/llm.local.json", {})
        return {"config": {"llm_model": config.get("model", "DeepSeek"), "selector": "JEV",
                            "llm_source": "deepseek", "llm_profiles": profile_options(self.root),
                           "rag_backend": "model_pack", "time_limit": 300, "mip_gap": 0.08, "solver": "copt"},
                "instances": read_json(self.root / "data/processed/manifest.json", {}).get("instances", []),
                "history": self.history()}

    def session(self, session_id):
        if not isinstance(session_id, str) or not re.fullmatch(r"[a-f0-9]{32}", session_id):
            raise ValueError("会话编号无效。")
        with self.lock:
            if session_id not in self.sessions:
                data = read_json(self.sessions_path / (session_id + ".json"))
                if not isinstance(data, dict):
                    raise ValueError("会话不存在，请新建对话。")
                self.sessions[session_id] = data
            session = self.sessions[session_id]
            deleted = self.deleted_results()
            changed = False
            for field in ("current_result_id", "reference_result_id"):
                if session.get(field) in deleted:
                    session[field] = None
                    changed = True
            for message in session.get("messages", []):
                if message.get("result_id") in deleted:
                    message.pop("result_id", None)
                    changed = True
            if changed:
                self.save_session(session)
            session.setdefault("reference_result_id", None)
            if session.get("active_job_id") and session["active_job_id"] not in self.jobs:
                # Do not replay paid requests when reconnecting after a restart.
                session["active_job_id"] = None
                session["messages"].append({"role": "assistant", "content": "网页服务已重新启动。上次任务的已保存结果可从历史记录查看；不会自动重复提交。", "created_at": now()})
                self.save_session(session)
            return session

    def new_session(self):
        with self.lock:
            data = {"id": uuid.uuid4().hex, "messages": [], "current_result_id": None, "reference_result_id": None,
                    "active_job_id": None, "requests": {}}
            self.sessions[data["id"]] = data
            self.save_session(data)
            return data

    def save_session(self, session):
        save_json(self.sessions_path / (session["id"] + ".json"), session)

    def public_session(self, session):
        return {k: session.get(k) for k in ("id", "messages", "current_result_id", "reference_result_id", "active_job_id")}

    def set_reference(self, session_id, result_id):
        if result_id is not None:
            self.result_path(result_id)
        with self.lock:
            session = self.session(session_id)
            if session.get("active_job_id"):
                raise RuntimeError("任务运行期间请先等待完成，再更换引用结果。")
            session["reference_result_id"] = result_id
            self.save_session(session)
            return self.public_session(session)

    def submit(self, body):
        prompt = body.get("prompt")
        request_id = body.get("request_id")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000:
            raise ValueError("请输入 1～12000 字的业务描述或问题。")
        if not isinstance(request_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{8,100}", request_id):
            raise ValueError("提交编号无效，请刷新后重试。")
        settings = validate_settings(body.get("settings", {}))
        source = body.get("llm_source", "deepseek")
        llm_config = load_profile(self.root, source)
        with self.lock:
            session = self.session(body.get("session_id"))
            selected_id = body.get("selected_result_id") if "selected_result_id" in body else session.get("reference_result_id")
            if selected_id is not None:
                self.result_path(selected_id)
            fingerprint = sha256(json.dumps([prompt, settings, selected_id, source], sort_keys=True).encode()).hexdigest()
            existing = session["requests"].get(request_id)
            if existing:
                if existing["fingerprint"] != fingerprint:
                    raise ValueError("相同提交编号不能用于不同问题。")
                return {"job_id": existing["job_id"], "session_id": session["id"]}
            if self.active_job_id:
                raise RuntimeError("已有任务正在运行，请等待完成后再发送。")
            job_id = uuid.uuid4().hex
            stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
            # Chat-only messages have no run folders. Create an output folder only for a solve request.
            job = {"id": job_id, "status": "running", "stage": "chat", "stage_label": LABELS["chat"],
                   "created_at": now(), "started_monotonic": time.monotonic(), "phase_started": time.monotonic(),
                   "phase_timings": [], "session_id": session["id"], "reply": None, "result_id": None,
                   "error": None, "settings": settings, "run_stamp": stamp, "reference_result_id": selected_id,
                   "subproblems": [], "completed_subproblem_count": 0,
                   "llm": model_identity(llm_config), "_llm_config": llm_config, "llm_chat": {}}
            self.jobs[job_id] = job
            self.active_job_id = job_id
            session["requests"][request_id] = {"job_id": job_id, "fingerprint": fingerprint}
            context_messages = list(session["messages"][-16:])
            session["messages"].append({"role": "user", "content": prompt.strip(), "created_at": now(), "job_id": job_id})
            session["active_job_id"] = job_id
            self.save_session(session)
            reference = selected_id
            threading.Thread(target=self.execute, args=(job_id, prompt.strip(), context_messages, reference), daemon=True).start()
            return {"job_id": job_id, "session_id": session["id"]}

    def phase(self, job_id, stage, **details):
        with self.lock:
            job = self.jobs[job_id]
            tick = time.monotonic()
            if stage != job["stage"]:
                job["phase_timings"].append({"stage": job["stage"], "label": LABELS.get(job["stage"], job["stage"]),
                                              "seconds": tick - job["phase_started"], "state": "done"})
                job.update(stage=stage, stage_label=LABELS.get(stage, stage), phase_started=tick)
            if details.get("subproblem_count"):
                index, total = details.get("subproblem_index"), details.get("subproblem_count")
                if isinstance(index, int) and not isinstance(index, bool) and isinstance(total, int) and 0 < index <= total:
                    event_name = details.get("event", "subproblem_started")
                    problems = job.setdefault("subproblems", [])
                    part = next((p for p in problems if p["subproblem_index"] == index), None)
                    if part is None:
                        part = {"depot_id": details.get("depot_id") or "全部仓库", "subproblem_index": index,
                                "subproblem_count": total, "status": "running", "started_at": details.get("started_at") or now(),
                                "_started_tick": tick}
                        problems.append(part)
                    if event_name == "subproblem_finished" and part["status"] != "done":
                        part.update(status="done", finished_at=details.get("finished_at") or now(), elapsed_seconds=tick - part["_started_tick"],
                                    solve_status=details.get("solve_status"), runtime_seconds=details.get("runtime_seconds"),
                                    mip_gap=details.get("mip_gap"), verified=details.get("verified") is True)
                        part["status_label"] = (SOLVE_LABELS.get(part["solve_status"], part["solve_status"] or "已结束")
                                                if part["verified"] or part["solve_status"] not in {"optimal", "feasible_limit"}
                                                else "解未通过独立校验")
                        if part["verified"] and finite_number(details.get("objective")) is not None:
                            part["objective"] = details["objective"]
                    job.update(depot_id=part["depot_id"], subproblem_index=index, subproblem_count=total,
                               completed_subproblem_count=sum(p["status"] == "done" for p in problems))
                    job["stage_label"] = f"求解与复核 · {part['depot_id']}（{index}/{total}，已完成 {job['completed_subproblem_count']}）"
            if job.get("result_id"):
                save_json(self.outputs / job["result_id"] / "web_job.json", self.job_snapshot(job_id))

    def job_snapshot(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                raise ValueError("任务不存在；服务可能已重新启动，请查看历史结果。")
            public = {k: v for k, v in job.items() if not k.startswith("_") and k not in {"started_monotonic", "phase_started", "run_stamp", "subproblems"}}
            public["llm_chat"] = {k: v for k, v in job.get("llm_chat", {}).items() if not k.startswith("_")}
            public["phase_timings"] = list(job["phase_timings"])
            public["subproblems"] = [{k: v for k, v in part.items() if not k.startswith("_")} for part in job.get("subproblems", [])]
            for part, source in zip(public["subproblems"], job.get("subproblems", [])):
                if part.get("status") == "running":
                    part["elapsed_seconds"] = time.monotonic() - source["_started_tick"]
            if job["status"] == "running":
                tick = time.monotonic()
                public["elapsed_seconds"] = tick - job["started_monotonic"]
                public["phase_timings"].append({"stage": job["stage"], "label": job["stage_label"],
                                                  "seconds": tick - job["phase_started"], "state": "running"})
            return public

    def execute(self, job_id, prompt, messages, reference_id):
        job = self.jobs[job_id]
        result_data = None
        try:
            from orllm.web_chat import plan_reply, build_planning_context, prepare_description, reference_constraints
            if reference_id:
                result_data = self.result(reference_id)
            planning_context = build_planning_context(self.root, result_data)
            job['planning_context'] = planning_context
            plan = plan_reply(self.root, prompt, messages, planning_context,
                              config=job["_llm_config"], trace=job["llm_chat"])
            with self.lock:
                job["action"] = plan["action"]
            if plan["action"] == "answer":
                self.finish(job_id, plan["reply"], result_id=reference_id)
                return
            try:
                if plan['action']=='update':
                    if not result_data or not result_data.get('binding_state'):raise ValueError('当前引用结果没有可更新的绑定快照，请先运行新流程。')
                    description=prompt;fixed=None
                else:
                    reference_state = planning_context['reference_state']
                    if plan['reuse_decisions'] != 'none' and reference_state['status'] != 'verified':
                        raise ValueError(reference_state['reason'] or '引用数据未通过本轮核验。')
                    if plan['reuse_decisions'] != 'none':
                        fresh_result = self.result(reference_id)
                        fresh_reference = build_planning_context(self.root, fresh_result)['reference_state']
                        if fresh_reference != reference_state:
                            raise ValueError('对话判断期间前序结果或数据已变化，请刷新引用后重新提交。')
                        result_data = fresh_result
                    if result_data and (result_data.get('binding_state') or {}).get('scenario',{}).get('parameter_values') and plan['reuse_decisions']!='none':
                        raise ValueError('当前结果含参数修改，跨模型继承需显式定义数据接口；请继续修改当前模型，或从原始实例新建任务。')
                    description = prepare_description(self.root, plan, result_data)
                    fixed = reference_constraints(self.root, plan, result_data)
            except ValueError as exc:
                self.finish(job_id, str(exc), result_id=reference_id)
                return
            result_id = "runs/" + job["run_stamp"]
            output = self.outputs / result_id
            output.mkdir(parents=True, exist_ok=False)
            (output / "problem.txt").write_text(description + "\n", encoding="utf-8")
            save_json(output / "chat_plan.json", {**plan, "original_prompt": prompt, "reference_result_id": reference_id})
            save_json(output / 'planning_context.json', planning_context)
            save_json(output / "llm_config.json", job["_llm_config"])
            save_json(output / "llm_chat.json", job["llm_chat"])
            with self.lock:
                job["result_id"] = result_id
                job["preliminary_reply"] = plan["reply"]
            self.phase(job_id, "mapping" if plan['action']=='update' else "selection")
            request_path = output / "web_request.json"
            save_json(request_path, {"description_file": str(output / "problem.txt"), "output": str(output),
                                     "llm_config": str(output / "llm_config.json"), "llm": job["llm"],
                                     "mapping_input": {"user_description":prompt,
                                         "user_history":[m['content'] for m in messages if m.get('role')=='user'][-4:],
                                         "confirmed_reference":fixed},
                                     "reference_constraints": fixed, **job["settings"]})
            if plan['action']=='update':
                request=read_json(request_path)
                request['parameter_update']={'binding_state':result_data['binding_state'],'changes':plan['changes'],
                    'expected_revision':plan['expected_revision'],'request_id':job_id,'description':prompt}
                save_json(request_path,request)
            command = [sys.executable, "-X", "utf8", str(self.root / "web_worker.py"), "--request", str(request_path)]
            flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            with (output / "web_console.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace", shell=False, creationflags=flags)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith("ORLLM_EVENT "):
                        event = json.loads(line[len("ORLLM_EVENT "):])
                        self.phase(job_id, event.pop("stage"), **event)
                code = process.wait()
            pipeline = read_json(output / "pipeline_report.json", {})
            if not (output / "summary.json").is_file():
                raise RuntimeError("运行没有生成结果摘要，请查看本次目录中的 web_console.log。")
            result_data = self.result(result_id)
            if code == 1 or pipeline.get("status") == "error":
                error = pipeline.get("error") or "建模或接口执行出错，请查看报告。"
                self.finish(job_id, "本次运行未完成。" + error, result_id=result_id, error=error)
            else:
                self.finish(job_id, result_reply(result_data), result_id=result_id)
        except Exception as exc:
            # Our API clients already redact HTTP bodies and credentials; do not display tracebacks.
            message = str(exc) if isinstance(exc, (ValueError, RuntimeError)) else "本地服务执行异常，请查看 outputs/diagnostics/web_server.log。"
            self.finish(job_id, "本次请求没有完成：" + message, result_id=job.get("result_id"), error=message)

    def finish(self, job_id, reply, result_id=None, error=None):
        with self.lock:
            self.phase(job_id, "done")
            job = self.jobs[job_id]
            if error and job.get("llm_chat", {}).get("status") != "validated":
                job["llm_chat"].update(status="error", error=error)
            job.update(status="error" if error else "done", reply=reply, result_id=result_id, error=error,
                       elapsed_seconds=time.monotonic() - job["started_monotonic"])
            for part in job.get("subproblems", []):
                if part["status"] == "running":
                    part.update(status="error", solve_status="error", verified=False, finished_at=now(),
                                status_label="任务中断，未完成", elapsed_seconds=time.monotonic() - part["_started_tick"])
            session = self.sessions[job["session_id"]]
            session["messages"].append({"role": "assistant", "content": reply, "created_at": now(),
                                         "job_id": job_id, "result_id": result_id, "action": job.get("action"),
                                         "llm": job.get("llm"),
                                         "llm_chat": {k: v for k, v in job.get("llm_chat", {}).items() if not k.startswith("_")},
                                         "context_audit": {'reference_state': job.get('planning_context', {}).get('reference_state'),
                                             'knowledge_sha256': job.get('planning_context', {}).get('workflow_knowledge', {}).get('sha256')}})
            if result_id and (self.outputs / result_id / "summary.json").is_file():
                session["current_result_id"] = result_id
                # Newly completed work becomes this conversation's next context.
                # Browsing and chat-only replies never alter the selected reference.
                if not error and job.get('action') in {'run', 'update'} and job.get('run_stamp') in result_id:
                    completed = read_json(self.outputs / result_id / 'summary.json', {})
                    if completed.get('complete_solution') is True and completed.get('cost_breakdown_complete') is True:
                        session['reference_result_id'] = result_id
                        session['reference_origin'] = 'latest_completed_run'
            session["active_job_id"] = None
            self.active_job_id = None
            self.save_session(session)
            if result_id and (self.outputs / result_id).is_dir() and result_id.startswith("runs/"):
                # Never add metadata to a selected historical result for a chat-only answer.
                if (self.outputs / result_id / "web_request.json").exists() and job.get("run_stamp") in result_id:
                    save_json(self.outputs / result_id / "web_job.json", self.job_snapshot(job_id))
            self._history_cache = (0, [])


def result_reply(data):
    summary = data["summary"]
    models = summary.get("models", [])
    selection = data.get("selection", {})
    model_id = models[0].get("model_id") if models else selection.get("model_id")
    name = models[0].get("model_name") if models else None
    status = summary.get("status_label", summary.get("status", "未知"))
    lines = [f"本次采用 {model_id or '待确认模型'}" + (f"（{name}）" if name else "") + f"。状态：{status}。"]
    if summary.get("complete_solution") and summary.get("total_objective") is not None:
        lines.append(f"通过独立校验的目标值为 {summary['total_objective']:,.4f}（沿用数据成本单位）。")
        business = summary.get("business", {})
        opened = [d["depot_id"] for d in business.get("depots", []) if d.get("opened")]
        if opened:
            lines.append("开放仓库：" + "、".join(opened) + "。")
        routes = business.get("routes", [])
        if routes:
            lines.append(f"共 {len(routes)} 条配送路线，右侧可按仓库或路线查看。")
        quality = data.get("quality") or result_quality(summary, {})
        if quality.get("actual_max_gap") is not None:
            lines.append(f"最大子问题 gap 为 {quality['actual_max_gap']:.2%}。")
        if quality.get("gap_state") == "unmet":
            lines.append(f"尚未达到目标 gap {quality['requested_gap']:.2%}；当前方案仍然是通过校验的可行方案。")
    elif summary.get("partial_objective") is not None:
        lines.append("仅部分子问题得到已验证方案，右侧展示这些部分；不能当作完整总方案。")
    for key, title in (("missing_fields", "需要补充"), ("unsupported_requirements", "当前模型未支持"), ("errors", "检查信息")):
        items = summary.get(key, [])
        if items:
            text = "；".join(str(x.get("message", x)) if isinstance(x, dict) else str(x) for x in items)
            lines.append(title + "：" + text)
    if selection.get("status") == "needs_review":
        lines.append("JEV 选型置信度不足，流程已停止，需要补充更明确的业务要求。")
    lines.append("结果、模型说明及检索依据已放在右侧，完整文件可从报告入口打开。")
    return "\n\n".join(lines)


class Handler(BaseHTTPRequestHandler):
    server_version = "ORLLM/1.0"

    def log_message(self, format, *args):
        # No prompt, credentials, or query strings in server access logs.
        pass

    @property
    def workspace(self):
        return self.server.workspace

    def respond(self, payload, code=200):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        port = self.server.server_port
        return self.headers.get("Host") in {f"127.0.0.1:{port}", f"localhost:{port}"}

    def do_GET(self):
        if not self.valid_host():
            self.respond({"error": "仅支持本机访问。"}, 403)
            return
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        item_id = query.get("id", [None])[0]
        try:
            if url.path == "/api/health":
                self.respond({"kind": APP_KIND, "project_id": self.workspace.identity, "version": UI_VERSION,
                              "busy": bool(self.workspace.active_job_id)})
            elif url.path == "/api/bootstrap":
                self.respond(self.workspace.bootstrap())
            elif url.path == "/api/history":
                self.respond({"history": self.workspace.history()})
            elif url.path == "/api/session":
                self.respond(self.workspace.public_session(self.workspace.session(item_id)))
            elif url.path == "/api/job":
                if item_id not in self.workspace.jobs:
                    self.respond({"error": "任务不存在；服务可能已重新启动，请查看历史结果。"}, 404)
                else:
                    self.respond(self.workspace.job_snapshot(item_id))
            elif url.path == "/api/result":
                self.respond(self.workspace.result(item_id))
            elif url.path == "/artifact":
                path = self.workspace.result_path(item_id)
                name = query.get("name", [""])[0]
                if not self.workspace.allowed_artifact(path, name):
                    raise ValueError("该文件不可从网页打开。")
                self.send_file(path / name, artifact=True)
            elif url.path in {"/", "/index.html", "/app.js", "/style.css"}:
                self.send_file(ROOT / "web" / ("index.html" if url.path == "/" else url.path[1:]))
            elif url.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.respond({"error": "页面不存在。"}, 404)
        except (ValueError, KeyError, OSError) as exc:
            self.respond({"error": str(exc)}, 400)

    def send_file(self, path, artifact=False):
        data = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", mime + ("; charset=utf-8" if mime.startswith("text/") or mime in {"application/javascript", "application/json"} else ""))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-cache")
        if artifact:
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; img-src data:; sandbox")
        else:
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        port = self.server.server_port
        origin = self.headers.get("Origin")
        if not self.valid_host() or (origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}):
            self.respond({"error": "请求来源不受支持。"}, 403)
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self.respond({"error": "请使用 JSON 提交。"}, 415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 100000:
                raise ValueError("请求大小超出范围。")
            body = json.loads(self.rfile.read(length), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            if not isinstance(body, dict):
                raise ValueError("请求必须是 JSON 对象。")
            if self.path == "/api/session":
                self.respond(self.workspace.public_session(self.workspace.new_session()), 201)
            elif self.path == "/api/session/context":
                if "result_id" not in body:
                    raise ValueError("请明确选择要引用的结果，或传 null 取消引用。")
                self.respond(self.workspace.set_reference(body.get("session_id"), body["result_id"]))
            elif self.path == "/api/chat":
                self.respond(self.workspace.submit(body), 202)
            elif self.path == "/api/history/delete":
                self.respond(self.workspace.delete_result(body.get("result_id")))
            else:
                self.respond({"error": "接口不存在。"}, 404)
        except RuntimeError as exc:
            self.respond({"error": str(exc)}, 409)
        except (ValueError, OSError) as exc:
            self.respond({"error": str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description="ORLLM 本地聊天工作台")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    import main as pipeline
    pipeline.load_local_environment()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    server.workspace = Workspace()
    print(f"ORLLM workspace: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
