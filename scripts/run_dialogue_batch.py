"""Resumable 41-instance evaluation through the ordinary chat API only."""
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import html
import json
import math
import msvcrt
import os
from pathlib import Path
import re
import time
import traceback
import urllib.error
import urllib.request
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'outputs/location_m1_m4_20261101/m1_m3'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def source_check(iid):
    base = read_json(ROOT/'data/processed/instances'/f'{iid}.json')
    sheet = read_json(ROOT/'data/processed/source_sheets'/f'{base["source_sheet"]}.json')['cells']
    value = lambda coordinate: sheet[coordinate].get('cached_value', sheet[coordinate].get('value'))
    for field in ['q', 'U', 'F']:
        for key, number in base[field].items():
            assert value(base['source_cells'][field+'.'+key]) == number
    for node, xy in base['coordinates'].items():
        assert [value(c) for c in base['source_cells']['coordinates.'+node]] == [xy['x'], xy['y']]
    for field in ['h', 'route_cost_type']:
        assert value(base['source_cells'][field]) == base[field]
    assert value(base['source_cells']['Q']) == base['original_vehicle_capacity']
    return base


def travel(base, a, b):
    p, q = base['coordinates'][a], base['coordinates'][b]
    dist = math.hypot(p['x']-q['x'], p['y']-q['y'])
    return math.floor(100*dist) if base['route_cost_type'] == 0 else dist


def verify_complete(case, stage):
    job = stage['job']
    if job.get('action') != 'run' or job['status'] != 'done':
        return {'passed': False, 'reason': job.get('error') or job.get('reply') or 'Did not execute run'}
    folder = ROOT/'outputs'/job['result_id']
    summary, selection, mapping, review, retrieval = [read_json(folder/n) for n in
        ['summary.json','selection.json','mapping.json','mapping_review.json','retrieval.json']]
    # Expected outputs are evaluated only after execution, never sent as hints.
    expected = 'M1' if stage['name'] == 'location' else 'M3'
    assert selection['provider'] == 'typesafe' and selection['model_id'] == expected
    assert mapping['model_id'] == expected and mapping['instance_id'] == case['instance_id']
    assert summary['complete_solution'] and all(r['has_verified_solution'] and r['solver'] == 'copt' for r in summary['models'])
    assert review['status'] == 'ready'
    assert retrieval['backend'] == 'model_pack' and len(retrieval['context']) == 8
    assert all(r['model_id'] == expected for r in retrieval['context'])
    assert 'M1' not in stage['request']['prompt'] and 'M3' not in stage['request']['prompt']
    assert 'selected_result_id' not in stage['request']
    base = source_check(case['instance_id'])
    if expected == 'M1':
        rows = summary['business']['assignments']
        assignments = {r['customer_id']: r['depot_id'] for r in rows}
        assert len(rows) == len(base['C']) and set(assignments) == set(base['C'])
        opened = {r['depot_id'] for r in summary['business']['depots'] if r['opened']}
        assert opened <= set(base['D']) and set(assignments.values()) <= opened
        loads = {d: sum(base['q'][customer] for customer in assignments if assignments[customer] == d) for d in opened}
        assert all(loads[d] <= base['U'][d] for d in opened)
        assert all(base['q'][customer] <= base['Q'] for customer in assignments)
        cost = sum(base['F'][d] for d in opened) + sum(base['h']+travel(base,d,c)+travel(base,c,d) for c,d in assignments.items())
        route_rows = []
    else:
        original = read_json(ROOT/'outputs'/case['stages'][0]['job']['result_id']/'summary.json')
        assignments = {r['customer_id']:r['depot_id'] for r in original['business']['assignments']}
        opened = {r['depot_id'] for r in original['business']['depots'] if r['opened']}
        assert {r['customer_id']:r['depot_id'] for r in mapping['assignment']} == assignments
        assert set(mapping['opened_depots']) == opened
        assert job['reference_result_id'] == case['stages'][0]['job']['result_id']
        proof = read_json(folder/'planning_context.json')['reference_state']
        assert proof['status'] == 'verified' and proof['assignment_count'] == len(base['C'])
        seen, cost, route_rows = [], 0, []
        for r in summary['business']['routes']:
            nodes, depot = r['nodes'],r['depot_id']
            assert depot in opened and nodes[0] == nodes[-1] == depot
            customers = nodes[1:-1]
            assert customers and all(assignments[c] == depot for c in customers)
            demand = sum(base['q'][c] for c in customers)
            assert demand <= base['Q'] and demand == r['demand']
            route_cost = base['h'] + sum(travel(base,a,b) for a,b in zip(nodes,nodes[1:]))
            assert math.isclose(route_cost,r['total_cost'],rel_tol=1e-9,abs_tol=1e-5)
            cost += route_cost;seen += customers
            route_rows.append({'path':r['path'],'load':demand,'capacity':base['Q'],'cost':route_cost})
        assert len(seen) == len(base['C']) and set(seen) == set(base['C'])
    assert math.isclose(cost,summary['total_objective'],rel_tol=1e-9,abs_tol=1e-5)
    session = api('/api/session?id='+case['session_id'])
    assert session['reference_result_id'] == job['result_id']
    return {'passed':True,'selected_model':selection['model_id'],'confidence':selection['confidence'],
            'mapping_attempts':len(review['attempts']),'objective':summary['total_objective'],
            'independent_recomputed_cost':cost,'opened_depots':sorted(opened),
            'all_optimal':all(r['status']=='optimal' and r['mip_gap']==0 for r in summary['models']),
            'max_gap':max(r['mip_gap'] for r in summary['models']),
            'route_count':len(route_rows),'routes':route_rows,'elapsed_seconds':job['elapsed_seconds'],
            'mapping_usage':read_json(folder/'llm_mapping.json')['usage'],
            'automatic_context_correct':True,'customer_count':len(base['C'])}


check = SimpleNamespace(ROOT=ROOT, read=read_json, source_check=source_check, verify=verify_complete,
    ROUTING='沿用刚才确定的开放仓库和客户归属，继续优化配送路线，满足车辆载重限制，最小化出车固定成本与行驶成本。请直接求解。')

STATE = OUT / 'results.json'
SETTINGS = {'solver': 'copt', 'time_limit': 180, 'mip_gap': 0.1}
LOCK = None


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def api(path, body=None):
    # Retrying submission is idempotent: the same session/request ID is retained.
    for attempt in range(5):
        try:
            request = urllib.request.Request('http://127.0.0.1:8765' + path,
                data=None if body is None else json.dumps(body, ensure_ascii=False).encode(),
                headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.load(response)
        except urllib.error.HTTPError:
            raise
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(5)


check.http = api


def replace_with_retry(source, target):
    for retry in range(60):
        try:
            source.replace(target)
            return
        except PermissionError:
            if retry == 59:
                raise
            time.sleep(0.1)


def save(state):
    state['updated_at'] = stamp()
    temp = STATE.with_suffix('.writing')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    replace_with_retry(temp, STATE)


def fingerprints():
    paths = [ROOT / n for n in ['web_app.py', 'web_worker.py', 'main.py', 'config/qwen.local.json']]
    for directory in ['orllm', 'knowledge', 'models', 'schemas']:
        paths += [p for p in (ROOT / directory).rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.jsonl', '.md'}]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def final_attempt(stage):
    return (stage.get('attempts') or [{}])[-1]


def verify(case, stage, attempt):
    job = attempt['job']
    result = {'passed': False}
    if job.get('action') == 'run' and job.get('result_id'):
        folder = ROOT / 'outputs' / job['result_id']
        for filename, target in [('selection.json', 'selection'), ('mapping_review.json', 'mapping_review'), ('summary.json', 'solution_summary')]:
            p = folder / filename
            if p.exists():
                data = check.read(p)
                if target == 'mapping_review':
                    result['mapping_attempts'] = len(data.get('attempts', []))
                    result['mapping_errors'] = [a.get('error') for a in data.get('attempts', []) if a.get('error')]
                elif target == 'selection':
                    result['selected_model'] = data.get('model_id')
                else:
                    result['complete_solution'] = data.get('complete_solution')
                    result['subproblems'] = [{k: r.get(k) for k in ['depot_id', 'status', 'mip_gap', 'has_verified_solution']} for r in data.get('models', [])]
    if job.get('action') != 'run' or job['status'] != 'done':
        result.update(reason=job.get('error') or job.get('reply') or '没有执行求解', failure_stage=job.get('stage'))
        return result
    try:
        proxy_case = dict(case, stages=[{'job': final_attempt(case['stages'][0])['job']}])
        proxy_stage = dict(attempt, name=stage['name'])
        result.update(check.verify(proxy_case, proxy_stage))
        build = check.read(ROOT / 'outputs' / job['result_id'] / 'build_report.json')
        assert build['time_limit_seconds'] == 180 and build['requested_mip_gap'] == 0.1, '求解设置不匹配'
        result['gap_target_met'] = all(isinstance(r['mip_gap'], (int, float)) and math.isfinite(r['mip_gap']) and r['mip_gap'] <= 0.10000001 for r in result['subproblems'])
        result['failure_stage'] = None
    except Exception as exc:
        lines = traceback.extract_tb(exc.__traceback__)
        last = lines[-1]
        result.update(passed=False, reason=f'{type(exc).__name__}: {exc}; {last.name}:{last.lineno}: {last.line}', failure_stage='independent_validation')
    return result


def statistics(state):
    cases = state['cases']
    stages = [s for c in cases for s in c['stages']]
    terminal = [s for s in stages if s.get('done')]
    good = [s for s in terminal if final_attempt(s).get('verification', {}).get('passed')]
    return {'instances_total': len(cases), 'instances_finished': sum(all(s.get('done') for s in c['stages']) for c in cases),
        'chains_passed': sum(all(final_attempt(s).get('verification', {}).get('passed') for s in c['stages']) for c in cases),
        'stages_finished': len(terminal), 'stages_passed': len(good),
        'gap_target_met': sum(final_attempt(s)['verification'].get('gap_target_met', False) for s in good),
        'mapping_repaired_stages': sum(final_attempt(s).get('verification', {}).get('mapping_attempts', 0) > 1 for s in terminal),
        'transport_retried_stages': sum(len(s.get('attempts', [])) > 1 for s in stages),
        'first_submission_passed': sum(bool(s.get('attempts') and s['attempts'][0].get('verification', {}).get('passed')) for s in stages)}


def render(state):
    stats = statistics(state)
    state['statistics'] = stats
    rows = []
    details = []
    esc = html.escape
    def cell(value):
        if value is None: return '—'
        if isinstance(value, float): return f'{value:,.2f}'
        return esc(str(value))
    for c in state['cases']:
        for st in c['stages']:
            a = final_attempt(st); v = a.get('verification', {}); j = a.get('job', {})
            status = '等待'
            if st.get('skipped'): status = '前序失败，未提交'
            elif v.get('passed'): status = '通过 / gap达标' if v.get('gap_target_met') else '可行 / gap未达标'
            elif st.get('done'): status = '未通过'
            elif a: status = '运行中：' + j.get('stage', '提交')
            rid = j.get('result_id'); link = ''
            if rid:
                folder = ROOT / 'outputs' / rid
                file = 'report.html' if (folder/'report.html').exists() else 'web_job.json'
                link = f'<a href="../../../location/outputs/{rid}/{file}">记录</a>'
            gap = v.get('max_gap')
            fields = [c['instance_id'],c['customers'],'选址分配' if st['name']=='location' else '续问路线',status,v.get('selected_model'),v.get('objective'),None if gap is None else f'{gap:.2%}',v.get('route_count'),v.get('mapping_attempts'),j.get('elapsed_seconds')]
            rows.append('<tr>'+''.join('<td>'+cell(x)+'</td>' for x in fields)+'<td>'+link+'</td></tr>')
            if v.get('reason') or v.get('mapping_errors') or len(st.get('attempts', [])) > 1:
                details.append('<li><b>'+esc(c['instance_id']+' / '+st['name'])+'</b>：'+esc(v.get('reason') or '最终通过')+'；映射修复：'+esc('；'.join(v.get('mapping_errors', [])))+'；提交次数：'+str(len(st.get('attempts', [])))+'</li>')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>41 个实例完整对话测试汇总</title><style>body{font:15px/1.7 system-ui,"Microsoft YaHei";background:#f2f5f3;color:#243c31;margin:0}main{max-width:1450px;margin:24px auto;padding:28px;background:white;border-radius:12px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{padding:9px;border:1px solid #d4e0d9;text-align:left}th{background:#edf5ef}a{color:#08776b}.stats{font-size:20px}li{margin:8px 0}</style><main><h1>全部 41 个实例：选址分配 → 续问路线</h1>'''
    page += '<p>状态：'+esc(state['status'])+'；更新时间（北京时间）：'+esc(state['updated_at'])+'</p>'
    page += f'<p class="stats">完成 {stats["instances_finished"]}/41 个实例；完整通过 {stats["chains_passed"]}/41；阶段通过 {stats["stages_passed"]}/82；通过且 gap 达标 {stats["gap_target_met"]}/82。</p>'
    page += '<p>24 张原始表，17 张含不同容量版本，共 41 个实例全部重新测试。每个实例独立会话；两轮业务提示不含模型编号，不手动指定上轮结果。Qwen3.5 9B 整理和映射，JEV 自主选型，完整模型知识包，COPT 每个子问题 180 秒、目标 gap 10%。多仓库任务总耗时可能超过 3 分钟。</p>'
    page += '<p>完整通过表示原始数据、选型、映射、上下文继承、仓库/车辆容量、客户唯一覆盖与成本复算均通过。gap 达标另行统计；固定仓库与归属下的路线结果不是选址与路径联合全局最优。本轮每阶段只提交一次，失败原因保留；业务/映射失败不人工纠正，不强制选型。批次执行期间不改系统代码、知识库、原始数据。</p>'
    page += '<p>首个提交通过阶段数：'+str(stats['first_submission_passed'])+'；发生内部映射修复的阶段：'+str(stats['mapping_repaired_stages'])+'；发生连接补测的阶段：'+str(stats['transport_retried_stages'])+'。</p>'
    page += '<div class="scroll"><table><tr>'+''.join('<th>'+x+'</th>' for x in ['实例','客户数','阶段','状态','实际选型','目标值','最大子问题gap','路线数','映射次数','阶段秒数','证据'])+'</tr>'+''.join(rows)+'</table></div>'
    page += '<h2>问题与补测记录</h2><ul>'+(''.join(details) or '<li>暂无</li>')+'</ul><p><a href="results.json">完整请求、状态与独立核验记录</a></p></main></html>'
    temp = OUT / 'index.writing'
    temp.write_text(page, encoding='utf-8'); replace_with_retry(temp, OUT/'index.html')


def initialize():
    health = api('/api/health'); assert not health['busy'], '工作台正在运行其他任务'
    with urllib.request.urlopen('http://127.0.0.1:8080/v1/models', timeout=15) as response:
        model = json.load(response)
    assert model['data'][0]['id'] == 'Qwen3.5-9B-Q8_0', '本地模型身份不符'
    manifest = check.read(ROOT / 'data/processed/manifest.json')
    workbook = ROOT / 'data/raw' / manifest['workbook']
    assert hashlib.sha256(workbook.read_bytes()).hexdigest() == manifest['workbook_sha256']
    state = {'status':'running','started_at':stamp(),'updated_at':stamp(),'settings':SETTINGS,'service':health,
        'sheet_count':manifest['sheet_count'],'instance_count':manifest['instance_count'],
        'workbook':str(workbook),'workbook_sha256':manifest['workbook_sha256'],'fingerprints':fingerprints(),'cases':[]}
    for entry in manifest['instances']:
        base = check.source_check(entry['instance_id'])
        state['cases'].append({'instance_id':entry['instance_id'],'source_sheet':base['source_sheet'],'customers':len(base['C']),
            'capacity':base['Q'],'stages':[{'name':'location'},{'name':'routing'}]})
    assert len(state['cases']) == 41
    save(state); render(state)
    return state


def main():
    global LOCK
    (OUT / 'tmp').mkdir(parents=True, exist_ok=True)
    LOCK = (OUT / 'tmp/runner.lock').open('a+b')
    LOCK.seek(0); LOCK.write(b'1'); LOCK.flush(); LOCK.seek(0)
    msvcrt.locking(LOCK.fileno(), msvcrt.LK_NBLCK, 1)
    state = check.read(STATE) if STATE.exists() else initialize()
    if state['status'] == 'completed': return
    assert fingerprints() == state['fingerprints'], '系统文件已变化，不能混用批次'
    state.update(status='running', runner_pid=os.getpid())
    state.pop('runner_error', None)
    save(state)
    try:
        for index, case in enumerate(state['cases'], 1):
            if all(st.get('done') for st in case['stages']): continue
            if not case.get('session_id'):
                assert not api('/api/health')['busy']
                case['session_id'] = api('/api/session', {})['id']; save(state)
            for stage in case['stages']:
                if stage.get('done'): continue
                if stage['name'] == 'routing' and not final_attempt(case['stages'][0]).get('verification', {}).get('passed'):
                    stage.update(done=True, skipped=True, reason='前序未取得完整已核验的选址分配，未伪造上游决策。')
                    save(state); render(state); continue
                while True:
                    attempts = stage.setdefault('attempts', [])
                    if attempts and attempts[-1].get('verification'):
                        stage['done'] = True; save(state); render(state); break
                    if not attempts or attempts[-1].get('verification'):
                        prompt = check.ROUTING if stage['name'] == 'routing' else f"使用数据实例 {case['instance_id']}（原表 {case['source_sheet']}，选用车辆容量 {case['capacity']} 的版本），从全部候选仓库中选择开放仓库，并为全部客户确定唯一归属，满足仓库容量限制。最小化仓库启用成本与整单独立直送成本。请直接求解。"
                        assert not re.search(r'\bM[1-4]\b', prompt)
                        attempts.append({'name':stage['name'],'request':{'session_id':case['session_id'],
                            'request_id':f'{state.get("request_prefix", "all41_clean_20261101")}_{index}_{stage["name"]}_{len(attempts)+1}',
                            'prompt':prompt,'llm_source':'qwen_local','settings':SETTINGS},'started_at':stamp()})
                        save(state)
                    attempt = attempts[-1]
                    if not attempt.get('submission'):
                        assert not api('/api/health')['busy'], '工作台被其他任务占用'
                        attempt['submission'] = api('/api/chat', attempt['request']); save(state)
                        print(stamp(), index, case['instance_id'], stage['name'], 'submitted', len(attempts), flush=True)
                    last = None
                    while True:
                        job = api('/api/job?id=' + attempt['submission']['job_id'])
                        attempt['job'] = job; save(state)
                        if job['stage'] != last:
                            print(stamp(), index, stage['name'], job['stage'], round(job.get('elapsed_seconds',0),1), flush=True)
                            render(state); last = job['stage']
                        if job['status'] != 'running': break
                        time.sleep(5)
                    attempt['verification'] = verify(case, stage, attempt)
                    attempt['finished_at'] = stamp(); save(state)
                    print(stamp(), index, stage['name'], 'result', json.dumps({k:v for k,v in attempt['verification'].items() if k not in ['routes','subproblems','mapping_usage']},ensure_ascii=False), flush=True)
                    stage['done'] = True; save(state); render(state); break
            print(stamp(), 'CASE_FINISHED', index, json.dumps(statistics(state),ensure_ascii=False), flush=True)
        assert fingerprints() == state['fingerprints']
        assert hashlib.sha256(Path(state['workbook']).read_bytes()).hexdigest() == state['workbook_sha256']
        state.update(status='completed', finished_at=stamp(), system_and_data_unchanged=True)
        render(state); save(state)
        print('ALL_41_FINISHED', json.dumps(statistics(state),ensure_ascii=False), flush=True)
    except Exception as exc:
        state.update(status='runner_error', runner_error=f'{type(exc).__name__}: {exc}')
        save(state); render(state); raise


if __name__ == '__main__':
    main()
