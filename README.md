# ORLM Location 选址与路径优化工作台

系统通过 JEV 选择固定模型，通过模型知识包和语言模型映射业务参数，再由 Python 校验数据、导出 LP、调用 COPT 求解并复核结果。M1 用于选址与客户分配，M2 在开放仓库已知时分配客户，M3 规划车辆路线，M4 加入电动车电量约束。

## 安装和本地验证

在解压后的 `location` 目录打开 PowerShell。项目在 Windows 和 Python 3.14 上验证过。以下命令安装基础依赖，用随包算例构建 M1，无需云端账户。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py list
.\.venv\Scripts\python.exe main.py build --scenario tests/fixtures/m1_assignment.scenario.json --output outputs/check
```

`outputs/check` 包含参数、LP 模型和构建报告。原始工作簿、41 个处理后的算例和模型知识包均已包含。要从原表重新生成算例，运行 `.\.venv\Scripts\python.exe main.py prepare`。

## 网页、云端选型和求解

首次配置时执行以下命令。打开已有安装时请保留自己已经填写的本地配置。

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt
Copy-Item .env.example .env
Copy-Item config/deepseek.example.json config/llm.local.json
Copy-Item config/jev.example.json config/jev.local.json
Copy-Item config/qwen.example.json config/qwen.local.json
Copy-Item config/rag.example.json config/rag.local.json
```

在本机的 `.env` 填写 `TYPESAFE_API_KEY`（JEV 选型）和 `DEEPSEEK_API_KEY`（DeepSeek 聊天和映射）。运行 `.\run_web.bat` 启动聊天工作台。云端端点和模型名可在对应的本地配置中修改。切换本地 Qwen 时，请先启动与配置匹配的 llama.cpp 服务；自动选型仍使用 JEV。完整求解使用 COPT，使用者需自行配置适用的求解授权。

要复现本次备份时的推理配置，可在本地 Qwen 配置中设置 `context_size` 为 65536、`max_tokens` 为 32768，并确认模型服务支持这一上下文长度。DeepSeek 配置中的 `json_schema_supported` 为 `false`。

默认检索使用随包模型知识。需要可选的 Chroma 向量检索时，先下载嵌入模型，再生成本地索引并把 `config/rag.local.json` 的 `backend` 改为 `chroma`、`embedding_model` 改为 `data/embedding_models/bge-small-zh-v1.5`。

```powershell
.\.venv\Scripts\python.exe scripts/download_embedding.py
.\.venv\Scripts\python.exe main.py index-rag --embedding-model data/embedding_models/bge-small-zh-v1.5
```

## 系统文件

`main.py` 提供命令行接口，`web/`、`web_app.py`、`web_worker.py` 和 `start_web.py` 提供本地网页。固定模型、字段约束和映射知识分别位于 `models/`、`schemas/`、`knowledge/`。`orllm/` 包含数据绑定、模型执行和校验逻辑，`tests/` 包含测试及示例输入，`scripts/` 提供诊断与批处理工具。批处理脚本中的实验配置可按本地任务调整。

算例作者、版本、许可及派生数据说明见 [数据来源与许可](DATA_NOTICE.md)。发布仓库时请保留该文件。本地密钥、服务配置、求解器许可证、模型缓存和生成的会话与结果仅保存在使用者自己的环境中。
