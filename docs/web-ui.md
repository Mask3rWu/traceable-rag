# 研究工作台

前端位于项目根目录 `web/`，保持 Node 构建链与 Python 包 `src/` 分离。所有 HTTP
接口统一位于 `src/api/`，前端不直接读取 `processed/` 或调用研究模块。

## 开发运行

安装后端和前端依赖：

```powershell
conda run -n dba-py311 python -m pip install -r requirements-api.txt
cd web
npm install
```

分别启动 API 和 Vite：

```powershell
conda run -n dba-py311 python scripts/run_api.py --reload
cd web
npm run dev
```

开发地址为 `http://127.0.0.1:5173`。Vite 将 `/api` 代理到
`http://127.0.0.1:8000`。

## 生产运行

```powershell
cd web
npm run build
cd ..
conda run -n dba-py311 python scripts/run_api.py
```

FastAPI 检测到 `web/dist` 后会同时托管构建产物，统一地址为
`http://127.0.0.1:8000`。

## API

```text
POST /api/runs
GET  /api/runs
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/events
POST /api/runs/{run_id}/cancel
GET  /api/runs/{run_id}/evidence/{evidence_id}/visuals/{block_id}
```

事件接口使用 SSE，只输出执行阶段、路由、工具摘要和终止状态，不传模型隐藏推理。
任务运行在进程内线程池中，最终结果持久化到 `processed/research/agent-runs/`。
进程重启后历史结果仍可读取，但正在运行的任务不会恢复。当前 provider 请求无法硬中断；
取消运行中的任务会返回 `cancel_requested`，排队中的任务可以直接取消。
