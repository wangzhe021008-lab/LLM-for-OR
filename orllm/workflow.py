"""Versioned contracts, grounded mapping checks and allowlisted parameter updates.

The expert mathematical programs remain fixed. Coverage records are auditable
claims, not a proof that a language model understood every business requirement.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

from .data import load_base


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def model_version(root, model_id):
    root=Path(root)
    files=[root/'models/model_catalog.json', root/f'models/{model_id}.py',
           root/f'schemas/{model_id}.schema.json',root/'orllm/linear_model.py',root/'orllm/validation.py']
    return fingerprint({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


def requirements(description):
    rows=[]
    for match in re.finditer(r'[^。！？；;\n]+[。！？；;]?',description):
        text=match.group().strip()
        if text:rows.append({'id':f'R{len(rows)+1:03}','text':text,'start':match.start(),'end':match.end()})
    if len(rows)>80:raise ValueError('需求分句超过 80 条，请按独立任务拆分；没有截断业务要求。')
    return rows


def contract(root, description, model_id, instance_ids, schema, context=(), input_bundle=None):
    root=Path(root);schema=deepcopy(schema)
    bundle=deepcopy(input_bundle) if input_bundle is not None else {'user_description':description,'user_history':[],'confirmed_reference':None}
    description=bundle['user_description']
    if not isinstance(description,str) or not description.strip():raise ValueError('用户原文不能为空。')
    history=bundle.get('user_history',[])
    if not isinstance(history,list) or any(not isinstance(t,str) for t in history):raise ValueError('用户历史必须为原文列表。')
    reference=bundle.get('confirmed_reference')
    lookup=description+'\n'+'\n'.join(history)+('\n'+json.dumps(reference,ensure_ascii=False) if reference else '')
    mentioned=[i for i in instance_ids if re.search(r'(?<![A-Za-z0-9_-])'+re.escape(i)+r'(?![A-Za-z0-9_-])',lookup)]
    candidates=mentioned or list(instance_ids)
    bases={i:load_base(root,i) for i in candidates}
    depots=sorted({d for b in bases.values() for d in b['D']})
    customers=sorted({c for b in bases.values() for c in b['C']})
    schema['properties']['instance_id']['enum']=[None,*candidates]
    schema['properties']['opened_depots']['items']={'type':'string','enum':depots}
    schema['properties']['assignment']['items']['properties']['customer_id']['enum']=customers
    schema['properties']['assignment']['items']['properties']['depot_id']['enum']=depots
    clauses=requirements(description)
    catalog=json.loads((root/'models/model_catalog.json').read_text(encoding='utf-8'))
    index,model=next((i,m) for i,m in enumerate(catalog['models']) if m['model_id']==model_id)
    input_fields=['instance_id']
    if model_id!='M1':input_fields.append('opened_depots')
    if model_id in {'M3','M4'}:input_fields.append('assignment')
    if model_id=='M4':input_fields+=['distance_scale_km','distance_scale_source']
    targets=input_fields+['model_scope']+[c['id'] for c in model['constraints']]
    schema['properties']['requirement_coverage']={'type':'array','minItems':len(clauses),'maxItems':len(clauses),'items':{'type':'object','properties':{
        'requirement_id':{'type':'string','enum':[r['id'] for r in clauses]},
        'status':{'type':'string','enum':['mapped','model_capability','needs_clarification','unsupported']},
        'targets':{'type':'array','items':{'type':'string','enum':targets}},
        'explanation':{'type':'string','minLength':1}},
        'required':['requirement_id','status','targets','explanation'],'additionalProperties':False}}
    schema['required'].append('requirement_coverage')
    direct_sources={f'models/model_catalog.json#/models/{index}':json.dumps(model,ensure_ascii=False),
                    'models/model_catalog.json#/shared_conventions':json.dumps(catalog['shared_conventions'],ensure_ascii=False),
                    f'schemas/{model_id}.schema.json':(root/f'schemas/{model_id}.schema.json').read_text(encoding='utf-8')}
    # These are the actual payload keys supplied by llm._messages, not invented documents.
    direct_sources.update(fixed_model=direct_sources[f'models/model_catalog.json#/models/{index}'],
                          shared_conventions=direct_sources['models/model_catalog.json#/shared_conventions'],
                          complete_input_schema=direct_sources[f'schemas/{model_id}.schema.json'])
    business_sources={'user_description':description}
    business_sources.update({f'user_history:{i+1}':text for i,text in enumerate(history) if text.strip()})
    if reference:business_sources['confirmed_reference']=json.dumps(reference,ensure_ascii=False,sort_keys=True)
    allowed_sources={*business_sources,*direct_sources}
    allowed_sources.update(name for item in context for name in [item.get('source'),item.get('id')] if name)
    schema['properties']['evidence']['items']['properties']['source']['enum']=sorted(allowed_sources)
    spec={'schema':schema,'requirements':clauses,'model_id':model_id,'model_version':'1.0',
            'model_fingerprint':model_version(root,model_id),'data_versions':{i:fingerprint(b) for i,b in bases.items()},
            'entities':{i:{'depots':b['D'],'customers':b['C']} for i,b in bases.items()},
            'evidence_sources':direct_sources,'business_sources':business_sources,'business_input':bundle}
    program={'model_id':model_id,'model_version':'1.0',
             'scenario_id':'mapped_'+model_id.lower()+'_'+fingerprint(bundle)[:12]}
    for field in ['opened_depots','assignment','distance_scale_km','distance_scale_source']:
        if field not in input_fields:program[field]=None
    if reference:
        reuse=reference.get('reuse_decisions')
        if (reuse=='opened_depots' and model_id!='M2') or (reuse=='assignment' and model_id not in {'M3','M4'}):
            raise ValueError('选中模型与已确认的前序决策用途不一致。')
        if reuse not in {'opened_depots','assignment'}:raise ValueError('前序决策引用类型无效。')
        program.update(instance_id=reference['instance_id'],opened_depots=deepcopy(reference['opened_depots']))
        if reuse=='assignment':program['assignment']=deepcopy(reference['assignment'])
    spans={}
    for source,text in business_sources.items():
        prefix='U' if source=='user_description' else 'REF' if source=='confirmed_reference' else 'H'+source.split(':')[1]+'_'
        for i,row in enumerate(requirements(text),1):
            spans[f'{prefix}{i:03}']={'source':source,'text':row['text']}
    generation=deepcopy(schema)
    for field in program:
        generation['properties'].pop(field,None)
        if field in generation['required']:generation['required'].remove(field)
    editable=[f for f in input_fields if f not in program]
    generation['properties']['evidence']={'type':'array','items':{'type':'object',
        'properties':{'field':{'type':'string','enum':editable or ['unused']},
                      'span_id':{'type':'string','enum':list(spans)}},
        'required':['field','span_id'],'additionalProperties':False}}
    if not editable:generation['properties']['evidence']['maxItems']=0
    spec.update(generation_schema=generation,program_bindings=program,source_spans=spans,
                system_provenance={'model_id':{'value':model_id,'source':'selected_model'},
                                   'model_version':{'value':'1.0','source':'models/model_catalog.json'},
                                   'data_binding':'Python loads the selected prepared instance; knowledge is not user-supplied data.'})
    return spec


def expand_mapping(raw, spec):
    """Validate only editable outputs; materialize fixed fields and verbatim evidence."""
    from jsonschema import Draft202012Validator
    errors=list(Draft202012Validator(spec['generation_schema']).iter_errors(raw))
    if errors:raise ValueError('映射输出接口：'+errors[0].json_path+': '+errors[0].message)
    mapping=deepcopy(raw)
    mapping['evidence']=[{'field':e['field'],'source':spec['source_spans'][e['span_id']]['source'],
                          'excerpt':spec['source_spans'][e['span_id']]['text']} for e in raw['evidence']]
    mapping.update(deepcopy(spec['program_bindings']))
    reference=spec['business_sources'].get('confirmed_reference')
    if reference:
        mapping['evidence'] += [{'field':f,'source':'confirmed_reference','excerpt':reference}
                                for f in ['instance_id','opened_depots','assignment']
                                if spec['program_bindings'].get(f) is not None]
    return mapping


def validate_contract(mapping, spec, description, context):
    from jsonschema import Draft202012Validator
    errors=list(Draft202012Validator(spec['schema']).iter_errors(mapping))
    if errors:raise ValueError('动态接口校验：'+errors[0].json_path+': '+errors[0].message)
    coverage=mapping['requirement_coverage']
    ids=[r['requirement_id'] for r in coverage]
    if len(ids)!=len(set(ids)) or set(ids)!={r['id'] for r in spec['requirements']}:
        raise ValueError('需求覆盖：每个需求 ID 必须恰好说明一次，不得遗漏或重复。')
    if mapping['status']=='ready':
        if any(r['status'] in {'unsupported','needs_clarification'} for r in coverage):
            raise ValueError('需求覆盖：存在未覆盖或待澄清要求，不能返回 ready。')
        for row in coverage:
            if not row['targets']:raise ValueError('需求覆盖：已覆盖要求必须指向字段或模型能力。')
            for target in row['targets']:
                if target in mapping and mapping[target] is None:raise ValueError('需求覆盖：不能用空字段声称已经映射 '+target)
    sources={**spec.get('evidence_sources',{}),**spec.get('business_sources',{'user_description':description})}
    for item in context:
        for name in [item.get('source'),item.get('id')]:
            if name:sources[name]=sources.get(name,'')+'\n'+item.get('text','')
    norm=lambda s:re.sub(r'\s+','',s)
    for index,evidence in enumerate(mapping['evidence']):
        if evidence['field'] in {'model_id','model_version','scenario_id'}:
            raise ValueError(f'证据校验：evidence[{index}] 是系统字段，不允许由模型伪装为用户事实；其来源由程序登记。')
        if evidence['source'] not in sources or not norm(evidence['excerpt']) or norm(evidence['excerpt']) not in norm(sources[evidence['source']]):
            raise ValueError(f"证据校验：evidence[{index}] 的 {evidence['field']} 引用未出现在来源 {evidence['source']} 中；必须逐字摘录该来源，不可引用其他字段或改写。")
    if mapping['status']=='ready':
        expected={'instance_id'}
        expected|={f for f in ['opened_depots','assignment','distance_scale_km','distance_scale_source'] if mapping.get(f) is not None}
        present={e['field'] for e in mapping['evidence'] if e['source'] in spec.get('business_sources',{'user_description':description})}
        if not expected<=present:raise ValueError('证据校验：缺少业务配置的原文依据 '+', '.join(sorted(expected-present)))
        entity=spec['entities'][mapping['instance_id']]
        if not set(mapping.get('opened_depots') or [])<=set(entity['depots']):raise ValueError('实体校验：仓库不属于所选实例。')
        if mapping.get('assignment') is not None:
            if {r['customer_id'] for r in mapping['assignment']}!=set(entity['customers']):raise ValueError('实体校验：归属必须覆盖所选实例的全部客户。')
            if any(r['depot_id'] not in mapping['opened_depots'] for r in mapping['assignment']):raise ValueError('实体校验：客户不能归属关闭的仓库。')


def mapping_scenario(mapping, description):
    keys=['scenario_id','instance_id','model_id','model_version','opened_depots','distance_scale_km','distance_scale_source']
    return {**{k:deepcopy(mapping[k]) for k in keys},'assignment':None if mapping['assignment'] is None else
            {r['customer_id']:r['depot_id'] for r in mapping['assignment']},'description':description}


def new_state(root, mapping, description, spec):
    scenario=mapping_scenario(mapping,description)
    scenario['data_version']=spec['data_versions'][mapping['instance_id']]
    scenario['model_fingerprint']=spec['model_fingerprint']
    scenario['parameter_values']={}
    state={'schema_version':1,'model_fingerprint':spec['model_fingerprint'],'scenario':scenario,
           'system_provenance':deepcopy(spec.get('system_provenance',{})),
           'mapping':deepcopy(mapping),'applied_requests':{},'revision':None}
    state['revision']=fingerprint({k:v for k,v in state.items() if k!='revision'})
    return state


def parameter_catalog(root, state):
    scenario=state['scenario'];base=load_base(root,scenario['instance_id']);mid=scenario['model_id']
    result={}
    for field in ['U','q']+(['F'] if mid=='M1' else []):
        for entity,value in base[field].items():result[field+'.'+entity]={'value':value,'unit':'原数据单位','min_exclusive':field!='F'}
    for field in ['Q','h']:result[field]={'value':base[field],'unit':'原数据单位','min_exclusive':field=='Q'}
    if mid=='M4':
        for field,value in base['ev'].items():
            if field in {'B','p','r0','r1'}:result['ev.'+field]={'value':value,'unit':{'B':'kWh','p':'原货币/kWh','r0':'kWh/km','r1':'kWh/km'}[field],'min_exclusive':field=='B'}
    for field,value in scenario.get('parameter_values',{}).items():
        if field not in result:raise ValueError('参数白名单已变化，请新建任务。')
        result[field]['value']=value
    return result


def update_contract(root,state):
    return {'expected_revision':state['revision'],'parameters':parameter_catalog(root,state),
            'operations':['set','increase','decrease'],'model_id':state['scenario']['model_id'],
            'rule':'只修改列出的数值，单位沿用该字段。结构、实体集合、矩阵和未提及字段保持不变。'}


def apply_changes(root, state, changes, expected_revision, request_id, description):
    if state['revision']!=fingerprint({k:v for k,v in state.items() if k!='revision'}):raise ValueError('绑定快照校验失败。')
    if not request_id or not isinstance(request_id,str):raise ValueError('更新需要请求 ID。')
    request_hash=fingerprint({'changes':changes,'expected_revision':expected_revision,'description':description})
    if request_id in state['applied_requests']:
        if state['applied_requests'][request_id]!=request_hash:raise ValueError('同一请求 ID 不能修改内容。')
        return deepcopy(state)
    if state['revision']!=expected_revision:raise ValueError('会话绑定版本已变化，请刷新当前结果后重新提交。')
    scenario=state['scenario'];base=load_base(root,scenario['instance_id'])
    if fingerprint(base)!=scenario['data_version']:raise ValueError('原始数据版本已变化，请重新确认数据。')
    if model_version(root,scenario['model_id'])!=state['model_fingerprint']:raise ValueError('模型程序版本已变化，请重新选型。')
    allowed=parameter_catalog(root,state);updated=deepcopy(state);seen=set()
    if not isinstance(changes,list) or not 1<=len(changes)<=100:raise ValueError('每次提供 1 至 100 项参数修改。')
    for change in changes:
        if not isinstance(change,dict) or set(change)!={'field','operation','value'}:raise ValueError('修改格式必须为 field/operation/value。')
        field,operation,value=change['field'],change['operation'],change['value']
        if field not in allowed or field in seen:raise ValueError('不允许的字段或重复修改：'+str(field))
        if operation not in {'set','increase','decrease'} or isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):raise ValueError('修改操作或数值不合法。')
        if operation!='set' and value<0:raise ValueError('增减量必须非负；请使用正确的 increase/decrease。')
        current=allowed[field]['value'];result=value if operation=='set' else current+value if operation=='increase' else current-value
        if result<0 or (allowed[field]['min_exclusive'] and result==0):raise ValueError('修改后字段超出允许范围：'+field)
        seen.add(field);updated['scenario']['parameter_values'][field]=result
    updated['scenario']['description']=description
    updated['applied_requests'][request_id]=request_hash
    updated['parent_revision']=state['revision']
    updated['last_changes']=deepcopy(changes)
    updated['revision']=fingerprint({k:v for k,v in updated.items() if k!='revision'})
    from .binding import bind_scenario
    bound=bind_scenario(base,updated['scenario'])
    if bound['status']!='ready':raise ValueError('修改后的业务参数未通过校验：'+json.dumps(bound.get('errors') or bound['missing_fields'],ensure_ascii=False))
    return updated


def map_with_contract(root, description, selection, context, config, instance_ids, output, trace=None, input_bundle=None):
    from .llm import mapping_schema, extract_mapping
    from .binding import bind_scenario
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if any(item.get('model_id')!=selection['model_id'] or item.get('model_version')!='1.0' for item in context):
        raise ValueError('知识上下文包含其他模型或版本，不能进入映射。')
    spec=contract(root,description,selection['model_id'],instance_ids,mapping_schema(root,selection['model_id'],instance_ids),context=context,input_bundle=input_bundle)
    record={'version':3,'model_id':selection['model_id'],'model_version':'1.0','model_fingerprint':spec['model_fingerprint'],
            'requirements':spec['requirements'],'data_versions':spec['data_versions'],'attempts':[],
            'semantic_guarantee':'Coverage/evidence consistency checked; not a proof of semantic equivalence.'}
    (output/'mapping_contract.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    feedback=None;previous=None
    for attempt in range(2):
        mapping=None;attempt_trace={}
        try:
            mapping=extract_mapping(root,description,selection['model_id'],context,config,instance_ids,
                                    trace=attempt_trace,contract=spec,feedback=feedback)
            validate_contract(mapping,spec,description,context)
            if feedback and feedback['scope']=='evidence_only':
                for field in ['instance_id','model_id','model_version','opened_depots','assignment','distance_scale_km','distance_scale_source']:
                    if mapping[field]!=previous[field]:raise ValueError('只修证据时不允许改变参数字段：'+field)
            if mapping['status']=='ready':
                state=new_state(root,mapping,description,spec)
                bound=bind_scenario(load_base(root,mapping['instance_id']),state['scenario'])
                if bound['status']=='invalid':raise ValueError('业务校验：'+json.dumps(bound['errors'],ensure_ascii=False))
                if bound['status']=='needs_clarification':
                    mapping['status']='needs_clarification';mapping['missing_fields']=bound['missing_fields']
                else:(output/'binding_state.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
            record['attempts'].append({'attempt':attempt+1,'status':'accepted','mapping':mapping})
            record['status']=mapping['status']
            return mapping
        except ValueError as exc:
            record['attempts'].append({'attempt':attempt+1,'status':'rejected','error':str(exc),'mapping':mapping})
            record['status']='mapping_rejected'
            if attempt:raise
            previous=mapping
            feedback={'error':str(exc),'previous_mapping':mapping,'scope':'evidence_only' if mapping and str(exc).startswith(('证据校验','需求覆盖')) else 'mapping_only',
                      'rule':'只修指出的错误；模型、来源和原始需求固定。缺信息应澄清，不得编造。'}
        finally:
            safe_trace={k:v for k,v in attempt_trace.items() if not k.startswith('_')}
            if record['attempts'] and record['attempts'][-1]['attempt']==attempt+1:
                record['attempts'][-1]['llm_trace']=safe_trace
            if trace is not None:
                trace.update(safe_trace)
                calls=[r.get('llm_trace',{}) for r in record['attempts']]
                trace['mapping_attempts']=len(calls)
                trace['usage']={k:sum(c.get('usage',{}).get(k,0) for c in calls)
                                for k in {k for c in calls for k in c.get('usage',{})}}
                trace['elapsed_seconds']=sum(c.get('elapsed_seconds',0) for c in calls)
            (output/'mapping_review.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
