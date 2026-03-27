# Auto-Summary 工作流

## 1. 目标
将论文 PDF 自动转换为组内模板化摘要（`.md`），并自动抽取框架图，产出到固定目录。

---

## 2. 输入与输出

### 输入
- CLI：`待处理pdf/` 目录下的一个或多个 `.pdf`
- WebUI：用户上传单个 `.pdf`（可重复提交多个任务）

### 输出
- 摘要：`paper/[方向]-[会议]-[年份]-[标题].md`
- 图片：`image/YYYYMMDDNN.png`
- 原 PDF：移动到 `已处理pdf/`

---

## 3. 核心流程

![framework](./image/image.png)

### 阶段 A：任务初始化
1. 创建/检查工作目录：`待处理pdf/`、`已处理pdf/`、`paper/`、`image/`
2. 收集待处理 PDF 列表（CLI）或保存上传文件（WebUI）

### 阶段 B：文本抽取
1. 使用 `pypdf` 提取前 N 页文本（默认 16 页）
2. 限制最大字符数（默认 36000），用于后续 LLM 调用

### 阶段 C：框架图定位与截图
1. 使用 `pdftotext -bbox-layout` 提取版面文本坐标
2. 检测候选图注（Figure/Fig + framework/architecture/pipeline 等关键词）
3. 生成候选裁剪框并做视觉评分（边缘、线段、矩形块、文本行惩罚）
4. 可选 VLM 重排候选（`use_vlm_rerank`）
5. 裁剪并输出最终图片到 `image/`

### 阶段 D：摘要生成
1. LLM 结构化抽取（JSON）
2. LLM 文案润色（JSON）
3. 本地质量约束（去重、压缩、要点数量约束）
4. 低质量字段定向修复（仅重写缺陷字段）

### 阶段 E：落盘与归档
1. 组装 Markdown（模板化章节）
2. 写入 `paper/`
3. 将原始 PDF 移动到 `已处理pdf/`

---

## 4. WebUI 与 CLI 的关系

- CLI 入口：`generate_summary.py` -> `autosummary.cli` -> `autosummary.pipeline`
- WebUI 入口：`webui.py`
  - `/start_job` 创建后台线程任务
  - `/job_status/<job_id>` 返回进度（用于前端进度条）
  - `/result/<token>` 在线渲染摘要
  - `/download/<token>` 下载 `.md`

两者复用同一套 `pipeline`，保证结果一致。

---

## 5. 模块职责

- `autosummary/pipeline.py`：流程编排（主工作流）
- `autosummary/figure_extractor.py`：框架图候选检测、评分、裁剪
- `autosummary/llm_client.py`：LLM 请求与结构化 JSON 解析
- `autosummary/summary_writer.py`：Markdown 模板拼装
- `autosummary/text_utils.py`：文本处理、质量约束、文件名与路径工具
- `webui.py`：Web 接口、任务状态管理、页面路由

---

## 6. 关键可调参数（常用）

- `SUMMARY_MAX_PAGES`：文本抽取页数
- `SUMMARY_MAX_CHARS`：文本抽取字符上限
- `SUMMARY_SCAN_PAGES`：框架图扫描页数
- `SUMMARY_USE_VLM_RERANK`：是否启用 VLM 重排
- `SUMMARY_VLM_TOP_K`：VLM 重排候选数
- `SUMMARY_TIMEOUT` / `SUMMARY_RETRIES`：LLM 请求超时与重试策略

