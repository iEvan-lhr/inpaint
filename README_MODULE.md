# 在另一个 Go 服务中引入：图片 → OCR → HTML

模块路径：`github.com/iEvan-lhr/inpaint`。包名分别是 `textinpaint` 和 `reconstruct`。

业务服务直接调用 Go 函数即可，不需要启动本仓库的 HTTP 服务或 CLI 子进程。业务服务自己选择 OCR / 多模态模型；本库负责文字定位、裁图、选择性去字、颜色与笔画边界估算、HTML 组装。

```text
业务 Go 服务
  ├─ reconstruct.Analyze(ctx, engine, image, options)
  │    └─ Task：保存原图快照、稳定区域 ID 和 Go 坐标
  ├─ 调用你自己的 OCR / 多模态模型
  │    └─ 输入原图或裁图，返回 region_id + status + text/runs
  └─ task.Reconstruct(ctx, recognition, options)
       └─ HTML + 去字背景 + 遮罩 + 完整 JSON
```

## 1. 安装模块

在**你的业务项目**目录运行：

```sh
# 没有 go.mod 的新项目先执行：go mod init example.com/my-service
go get github.com/iEvan-lhr/inpaint@main
```

Go 会把当时的 main 解析为具体的伪版本并记录到 `go.mod/go.sum`。提交这两个文件即可固定业务项目的版本。需要特定修复时可以用 `go get github.com/iEvan-lhr/inpaint@提交SHA`。

```go
import (
    textinpaint "github.com/iEvan-lhr/inpaint"
    "github.com/iEvan-lhr/inpaint/reconstruct"
)
```

这是 cgo 库，**`go get` 只下载源码，不自动下载检测模型、OpenCV 或 ONNX Runtime**。要求 Go 1.24+、C/C++ 编译器及以下原生依赖。

## 2. 准备原生依赖

### Windows amd64

需要 MinGW-w64 的 gcc/g++，不能用 MSVC 编译的 OpenCV 库与 MinGW 链接。当前 Windows 默认链接 OpenCV 4.12.0 的 `core/imgproc/photo` 三个库，以及 ONNX Runtime 1.22.0。

准备一个独立、可写的依赖目录；它与业务项目、Go 模块缓存分别保存。例如：

```powershell
git clone https://github.com/iEvan-lhr/inpaint.git C:\deps\inpaint
& C:\deps\inpaint\scripts\setup-windows.ps1 -CompilerBin 'C:\Program Files\mingw64\bin'
```

首次运行 setup 还需要 CMake 和 `mingw32-make`。它会编译 OpenCV，下载 ONNX Runtime 和检测模型；已有完整 Windows 包时，可以直接把完整包目录当作 `DependencyRoot`，无需重新编译。

回到**业务项目**，点加载环境脚本：

```powershell
. C:\deps\inpaint\scripts\use-library.ps1 `
  -DependencyRoot 'C:\deps\inpaint' `
  -CompilerBin 'C:\Program Files\mingw64\bin'

go build -o .\bin\my-service.exe .\cmd\my-service
```

脚本只设置当前 PowerShell 会话：`CGO_ENABLED`、`CC/CXX`、OpenCV 的 `CGO_CXXFLAGS/CGO_LDFLAGS`、DLL 搜索路径、`INPAINT_MODEL` 和 `INPAINT_RUNTIME`。它不会修改系统 PATH、切换当前目录或向 Go 模块缓存写入 native 文件。

不要在 `GOMODCACHE` 中执行 `setup-windows.ps1`。`go get` 下载的模块不含 native 文件；通过上述绝对路径环境参数让模块缓存中的源码链接到外部依赖即可。**正常集成不需要 `replace`。**

部署业务程序时，把 `C:\deps\inpaint\native\windows-amd64\bin\*.dll` 复制到业务 EXE 同目录，并部署 `models\ch_PP-OCRv4_det_infer.onnx`。运行时给 `Config.ModelPath`、`Config.RuntimePath` 传入部署后的路径；构建机的环境变量不会自动跟随 EXE 到服务器。

### Linux / macOS

准备 OpenCV 4（包含 core/imgproc/photo）、`pkg-config`、支持 cgo 的 C/C++ 编译器、ONNX Runtime 1.22 CPU 动态库及检测模型。

```sh
# Linux 示例：apt-get install libopencv-dev pkg-config g++
# macOS 示例：brew install opencv pkg-config
export CGO_ENABLED=1
export INPAINT_MODEL=/opt/inpaint/models/ch_PP-OCRv4_det_infer.onnx
export INPAINT_RUNTIME=/opt/inpaint/lib/libonnxruntime.so  # macOS 改为 .dylib
pkg-config --modversion opencv4
go build ./...
```

运行时动态链接器须能找到 OpenCV 及 ONNX Runtime 的依赖库。本次实测平台为 Windows amd64；Linux/macOS 提供构建配置，未在此次环境实测。cgo 跨平台构建需要对应的目标工具链和 native 库，不能只设置 GOOS 就获得可运行程序。

## 3. 在业务服务里调用

下面是完整的业务封装。`Recognizer` 由你的项目实现，可以调用任意模型服务、现有 OCR 或人工校对流程。

```go
package documentservice

import (
    "context"
    "errors"
    "image"

    textinpaint "github.com/iEvan-lhr/inpaint"
    "github.com/iEvan-lhr/inpaint/reconstruct"
)

type Recognizer func(
    ctx context.Context,
    analysis reconstruct.Analysis,
    originalPNG []byte,
) (reconstruct.Recognition, error)

type Service struct {
    Engine *textinpaint.Engine
    OCR    Recognizer
}

// 服务启动时调用一次。Engine 应由业务服务复用。
func New(modelPath, runtimePath string, ocr Recognizer) (*Service, error) {
    if ocr == nil {
        return nil, errors.New("OCR recognizer is required")
    }
    engine, err := textinpaint.New(textinpaint.Config{
        ModelPath: modelPath,
        RuntimePath: runtimePath,
    })
    if err != nil {
        return nil, err
    }
    return &Service{Engine: engine, OCR: ocr}, nil
}

// 服务停止接收请求并等待任务结束后关闭引擎。
func (s *Service) Close() error { return s.Engine.Close() }

func (s *Service) ToHTML(ctx context.Context, img image.Image) (*reconstruct.Result, error) {
    task, err := reconstruct.Analyze(ctx, s.Engine, img, reconstruct.Options{})
    if err != nil {
        return nil, err
    }

    // 定位已完成，此时还没有擦除文字。
    // 模型接收原图或 CropPNG 的裁图，不能只接收坐标 JSON。
    recognition, err := s.OCR(ctx, task.Analysis(), task.OriginalPNG())
    if err != nil {
        return nil, err
    }

    // 复用当前任务的坐标和原图，不再次检测。
    return task.Reconstruct(ctx, recognition, reconstruct.RenderOptions{
        Title: "重建文档",
    })
}
```

取得结果后，`result.HTML` 可以直接作为 `text/html; charset=utf-8` 返回，或调用 `result.WriteDir("result")` 保存以下文件：

```text
index.html            自包含 HTML，内嵌图片，可离线打开
background.png        Go/OpenCV 去字背景
original.png          分析时的原图快照
mask.png              实际使用的擦除遮罩
reconstruction.json   文本、原始 Go 坐标、笔画边界、颜色、保留原因
```

`result.JSON` 不内嵌图片。`result.Background`、`result.Mask` 是 Go 图像对象，`result.Page` 可用于合并多页文档。

## 4. OCR / 多模态 JSON 协议

`task.Analysis()` 示例（尺寸和坐标仅为示意）：

```json
{
  "analysis_id": "本次任务的唯一标识",
  "width": 1200,
  "height": 1600,
  "source_sha256": "规范化原图PNG的SHA256",
  "regions": [
    {
      "id": "r000001",
      "polygon": [{"x": 80, "y": 90}, {"x": 300, "y": 90}, {"x": 300, "y": 125}, {"x": 80, "y": 125}],
      "score": 0.95,
      "text_height": 26
    }
  ]
}
```

识别结果必须原样带回 `analysis_id`，并用 `region_id` 关联文字。下面的响应假设当前任务包含三个区域：

```json
{
  "analysis_id": "原样返回本次任务标识",
  "regions": [
    {"region_id": "r000001", "status": "text", "text": "工作经历", "bold": true, "role": "heading"},
    {"region_id": "r000002", "status": "graphic"},
    {"region_id": "r000003", "status": "unreadable"}
  ],
  "reading_order": ["r000001"]
}
```

规则：

- `text` 且有非空内容：按 Go 坐标去字并放回 HTML。
- `graphic`、`unreadable`、空字符串、遗漏区域：保留原图，不擦除，也不生成该区域的 HTML 文本。
- 未知 ID、重复 ID、错误 `analysis_id`、非法状态或颜色：返回错误，避免不同任务的结果混用。
- 回传协议没有坐标字段；模型不能重定位或修改擦除范围。HTTP 解码适配器建议使用 `json.Decoder.DisallowUnknownFields()`。
- 默认从原图测量颜色。需要人工或模型覆盖时可以传 `"color":"#1F6F8B"`，只接受六位十六进制颜色。
- `role` 可选值：`text`、`heading`、`paragraph`、`list-item`、`caption`。它作为语义元数据保存，不自动重排文档。
- `reading_order` 控制复制正文和 DOM 中的顺序，不改变页面坐标。未列出的成功文本按检测顺序追加；可以包含保留区域的 ID，它们不会生成文本。

混合粗体可用 `runs` 替代 `text`，两者不能同时设置：

```json
{
  "region_id": "r000004",
  "status": "text",
  "runs": [
    {"text": "工程技术：", "bold": true},
    {"text": "Go、Python、SQL"}
  ]
}
```

对于同一框中左右相邻、颜色明显不同的文字块，可额外设置 `split_colors:true`。Go 按颜色拆分并依次对应 `runs`；若不能可靠分开，保留原图并返回 `no_reliable_ink`。本次实现的字号是依据 Go 笔画边界和浏览器字体度量拟合，识别模型无需给出字号。

## 5. 裁图、异步调用、小字与多页

按区域向 OCR 发送裁图：

```go
analysis := task.Analysis()
for _, region := range analysis.Regions {
    cropPNG, err := task.CropPNG(region.ID, 4) // 四周补 4 个原图像素
    if err != nil { return err }
    // 把 region.ID 和 cropPNG 交给你的 OCR；必要时另附原图提供上下文。
    _ = cropPNG
}
```

`Task` 持有规范化原图与检测结果，不连接外部服务、不开启后台线程。异步服务应自行把 `Task` 保存在任务存储中并控制容量、过期时间和访问权限；模型响应回来后取出**同一个 Task**。本版本不提供跨进程 Task 序列化，进程重启后需要重新分析并重新关联识别结果。`Analysis` JSON 本身不能恢复 Task。

默认沿用 Engine 的小字阈值（通常为 12 像素），并保留小字和设备界面保护。完整文档恢复需要页脚时，可仅对本次任务关闭过滤：

```go
task, err := reconstruct.Analyze(ctx, engine, img, reconstruct.Options{
    Processing: &textinpaint.ProcessingOptions{MinTextHeight: 0},
})
```

`Options.Mask` 为 nil 时采用默认图形保护和遮罩参数；需要自定义时从 `textinpaint.DefaultMaskOptions()` 开始修改。`InpaintRadius=0` 使用 3。检测阶段的界面保护由 Engine 的检测/Mask 配置决定；重建阶段的遮罩保护由 `Options.Mask` 决定。

PDF 先由业务服务渲染为图片，每页分别 Analyze、识别、Reconstruct，再合并：

```go
html, err := reconstruct.RenderHTML(reconstruct.Document{
    Title: "多页文档",
    Pages: []reconstruct.Page{firstResult.Page, secondResult.Page},
})
```

HTML 包含原图/重建/去字背景切换、文字框、JSON 下载、复制和打印。原图和背景都内嵌在 HTML 中，文件大小随页数增长。

## 6. 可运行的 HTTP OCR 适配示例

仓库里的 `examples/reconstruct/main.go` 是实际调用方示例。它只用 Go 函数处理图像，HTTP 仅用于调用**你提供的 OCR 适配服务**。

```powershell
# 先按第 2 节设置编译环境和 INPAINT_MODEL / INPAINT_RUNTIME。
go run github.com/iEvan-lhr/inpaint/examples/reconstruct `
  -input .\input.png `
  -ocr-url http://127.0.0.1:9000/recognize `
  -min-text-height 0 `
  -out .\result-html
```

该地址是你需要实现的服务，并非本库自带接口。它接收：

```json
{"analysis": {"analysis_id":"...","regions":[]}, "image_png_base64":"PNG的base64"}
```

返回第 4 节的 `Recognition` JSON。模型厂商原生的 Chat/Responses/OCR 协议通常不同，业务服务应负责提示词、结果解析和协议转换。示例读取可选的 `OCR_API_KEY` 并作为 Bearer token 发送到你指定的地址；本库不保存 API Key。

## 7. 生命周期与边界

- 服务启动时创建 Engine，复用它；同一个 Engine 的推理串行。业务方应限制请求并发和在途 Task 数量。
- Analyze 完成后 Task 不再需要 Engine；Reconstruct 不持有模型锁，也不重新识别。
- Task 复制输入图和检测结果，`Analysis()`、`OriginalPNG()` 返回副本；Task 可以被多次或并发重建，调用方不要并发改写自己的 Recognition 参数。
- `context` 在阶段间检查，不能打断正在执行的单次 OpenCV/ONNX 原生调用。
- 当前 HTML 优先恢复水平单行区域。明显旋转/竖排、单框内多行文字、无法可靠测量的区域保留为图片，并在 `preserved` 记录原因。
- 保留区域与待替换区域的包围框相交时，优先保留两者，避免背景文字与 HTML 重叠；因此极密排版可能需要上游确认框拆分。
- 复杂背景、混合颜色、艺术字和字体替代会影响视觉一致性。输出保留原页布局，窄屏按比例缩放。
- 普通的 `engine.Remove` 仍是直接去字接口；需要模型确认后再去字时，使用本文件的 Analyze → Reconstruct 流程。

## 8. 常见接入错误

| 错误 | 处理 |
| --- | --- |
| `opencv2/...: No such file` | 在业务项目终端点加载 `use-library.ps1`；确认 CGO_CXXFLAGS 指向独立依赖目录 |
| `cannot find -lopencv_*4120` | 检查 CGO_LDFLAGS 与 MinGW 版 OpenCV 4.12 导入库 |
| 程序启动缺 DLL | 部署整个 native/bin 的运行库至业务 EXE 同目录 |
| 模型或 ONNX Runtime 找不到 | 用服务器实际路径配置 ModelPath/RuntimePath，推荐绝对路径 |
| `analysis_id does not match` | 回传时找错了 Task，或重新分析生成了另一个 ID |
| 输出区域少于检测区域 | 查看 `result.Page.Preserved`，检查未识别、图案、过小文字及重叠保护 |
| 想在本地联调库改动 | 可临时 `go mod edit -replace=github.com/iEvan-lhr/inpaint=C:/path/inpaint`；发布业务代码前移除本机路径 |
