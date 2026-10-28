# inpaint / textinpaint

独立 Go 库：自动检测图片中的文字，生成遮罩，再用 OpenCV Telea 修复背景。整个运行过程在本地 CPU 上完成，不调用云服务，也不启动 Python 子进程。

定位使用 **PP-OCRv4 mobile ONNX**，Go 通过 `onnxruntime_go` 调用 ONNX Runtime。图像操作通过库内的小型 cgo/C++ 桥接层调用 OpenCV 的 `core`、`imgproc`、`photo` 三个模块。没有写死样例图的文字坐标。

## 作为内网 HTTP 服务使用

已包含 `textinpaint-server`，模型在启动时加载一次，其他语言的程序通过 HTTP 上传图片即可调用。

如果使用附带模型和 Windows 依赖的完整包，在目录中执行：

```powershell
.\bin\textinpaint-server.exe -listen 0.0.0.0:8080
```

然后浏览器打开 **http://127.0.0.1:8080**，选择图片、点击「开始去字」。其他内网电脑访问 `http://服务电脑的内网IP:8080`。

```powershell
# 健康检查
Invoke-RestMethod http://127.0.0.1:8080/healthz

# 上传图片，取得去字后的 PNG；把 input.png 换成自己的图片路径。
curl.exe --fail-with-body -F "image=@input.png" http://127.0.0.1:8080/v1/remove -o no_text.png
```

从 GitHub 获取的仓库只包含源码与依赖下载脚本，需要先准备依赖并编译：

```powershell
git clone https://github.com/iEvan-lhr/inpaint.git
cd inpaint
.\scripts\setup-windows.ps1 -CompilerBin 'C:\Program Files\mingw64\bin'
.\scripts\build.ps1 -CompilerBin 'C:\Program Files\mingw64\bin'
.\scripts\start-server.ps1
```

完整接口、ZIP 输出、鉴权、Go/Python 调用、内网部署说明见 **[HTTP 服务文档](HTTP.md)**。调用方不需要安装 OpenCV 或模型。

## 作为库使用

模块路径为 `github.com/iEvan-lhr/inpaint`，Go 包名为 `textinpaint`。准备好本地 native 依赖后，在调用项目中添加本地替换（路径换成实际目录）：

```sh
go mod edit -require=github.com/iEvan-lhr/inpaint@v0.0.0
go mod edit -replace=github.com/iEvan-lhr/inpaint=/absolute/path/to/textinpaint
go mod tidy
```

直接导入 Go 库的程序需要 cgo 和 native 依赖；通过 HTTP 调用的程序无需这些依赖。

```go
package main

import (
    "context"
    "image"
    _ "image/jpeg"
    "image/png"
    "log"
    "os"

    textinpaint "github.com/iEvan-lhr/inpaint"
)

func main() {
    engine, err := textinpaint.New(textinpaint.Config{
        ModelPath:   "models/ch_PP-OCRv4_det_infer.onnx",
        RuntimePath: "native/windows-amd64/bin/onnxruntime.dll",
    })
    if err != nil { log.Fatal(err) }
    defer engine.Close()

    f, err := os.Open("input.png")
    if err != nil { log.Fatal(err) }
    img, _, err := image.Decode(f)
    f.Close()
    if err != nil { log.Fatal(err) }

    result, err := engine.Remove(context.Background(), img)
    if err != nil { log.Fatal(err) }
    out, err := os.Create("no_text.png")
    if err != nil { log.Fatal(err) }
    if err = png.Encode(out, result.Image); err != nil { out.Close(); log.Fatal(err) }
    if err = out.Close(); err != nil { log.Fatal(err) }
    log.Printf("检测到 %d 个区域", len(result.Regions))
}
```

### 分步调用与人工补选

```go
regions, err := engine.Detect(ctx, img)
// 检查 err；可以在这里增删 regions，或追加人工选择的四边形。
options := textinpaint.DefaultMaskOptions()
mask, err := textinpaint.BuildMask(img, regions, options)
// 检查 err；也可以自行修改 mask，白色表示擦除，黑色表示保留。
cleaned, err := textinpaint.Inpaint(img, mask, 3)
```

主要 API：

| API | 用途 |
| --- | --- |
| `New(Config)` / `Close()` | 创建、释放模型会话 |
| `Engine.Detect(ctx, image)` | 只返回文字四边形和分数 |
| `BuildMask(image, regions, options)` | 使用自动或手工坐标生成遮罩 |
| `Inpaint(image, mask, radius)` | 使用任意同尺寸灰度遮罩修复 |
| `Engine.Remove(ctx, image)` | 完成检测、遮罩、修复，返回图片和坐标 |
| `DrawRegions(image, regions)` | 输出文字框预览 |

所有坐标相对于输入图片左上角，输出图片从 `(0,0)` 开始。`Region.Polygon` 的四个点按边界顺序排列；手工区域必须为凸四边形。检测只定位，不识别文字内容。`Result` 可以直接序列化为 JSON，图片字段会自动跳过。

## Windows 快速运行

需要 **Go 1.24+、启用 cgo、MinGW-w64 的 gcc/g++**。本次交付附带 Windows amd64 的 OpenCV 4.12.0、ONNX Runtime 1.22.0 和模型文件。

在库目录执行：

```powershell
. .\scripts\env.ps1 -CompilerBin 'C:\Program Files\mingw64\bin'
go test ./...
.\scripts\build.ps1 -CompilerBin 'C:\Program Files\mingw64\bin'
.\bin\textinpaint.exe -input input.png -out result `
  -model models\ch_PP-OCRv4_det_infer.onnx `
  -runtime native\windows-amd64\bin\onnxruntime.dll
```

`env.ps1` 只修改当前 PowerShell 会话的环境，不修改系统 PATH。编译脚本会把所需 DLL 复制到 `bin/`，可执行程序与 DLL 一起部署。作为库导入时，宿主程序同样需要这些 DLL；编译器 ABI 必须与 OpenCV 匹配。

如果从源码仓库取回项目而没有模型或 native 文件，或需要用自己的 MinGW 重建：

```powershell
.\scripts\setup-windows.ps1 -CompilerBin 'C:\path\to\mingw64\bin' -Jobs 4
# 强制重新编译 OpenCV：在上面命令末尾加 -Force。
```

此脚本下载固定版本 OpenCV 源码、ONNX Runtime 和检测模型，只写入库目录。模型从固定 RapidOCR wheel 中提取，不需要安装 Python。首次编译 OpenCV 需要时间，后续无需重复。

CLI 输出：

- `no_text.png`：自动去字结果。
- `mask.png`：可以检查和修改的遮罩。
- `detections.png`：检测框预览。
- `regions.json`：坐标、尺寸、各阶段耗时。

只定位：`-detect-only`。关闭分块检测：`-tile-size 0`。调整检测阈值：`-box-threshold 0.5`。

## Linux / macOS

需要支持 cgo 的 C/C++ 编译器、`pkg-config`、OpenCV 4（含 core/imgproc/photo），以及 ONNX Runtime 1.22 的 CPU 动态库。桥接层通过 `pkg-config opencv4` 查找 OpenCV。把 ONNX Runtime 动态库的绝对路径传给 `Config.RuntimePath`。

Linux 示例依赖：`libopencv-dev pkg-config g++`。macOS 可用 Homebrew 安装 `opencv pkg-config`。这两个平台的构建声明已提供，本次只实测 Windows amd64。

## 参数、并发和限制

- `Config.Detection`、`Config.Mask` 为 nil 时使用默认值。自定义时从 `DefaultDetectionOptions()`、`DefaultMaskOptions()` 开始修改。
- 默认整图检测加重叠分块检测，合并重复区域。`LongSide` 为推理缩放上限，默认 1920；`TileScale=1.5` 可帮助检测小字。
- 默认先提取文字笔画，再扩边并修复；有纹理的区域可能退回整个文字框遮罩。
- 图形保护是针对文档的启发式规则，可能与特殊字体冲突。可设置 `PreserveGraphics=false`；CLI 对应 `-preserve-graphics=false`。
- 同一个 Engine 的检测请求串行执行，`Close` 等待正在进行的检测并可重复调用。多个 Engine 共享同一 ONNX Runtime 生命周期；最后一个关闭时释放本库创建的环境。
- 如果宿主已通过 `onnxruntime_go` 初始化环境，`RuntimePath` 留空可复用；宿主须在所有 Engine 关闭后再释放环境。
- `context` 取消在推理分块和处理阶段之间生效，无法中断正在执行的单次原生调用。
- 输入最多 6400 万像素；PNG/JPEG CLI 不自动应用 JPEG EXIF 旋转。透明图片会铺白底。
- 独立页码、极小字、装饰字可能漏检；复杂照片背景可能修复不自然。自动结果需要检查，接口保留了人工增删坐标和修改遮罩的入口。

## 目录

```text
textinpaint/
  go.mod, go.sum         Go 模块
  textinpaint.go         公共类型、遮罩、修复接口
  engine.go             ONNX 会话与自动检测流程
  geometry.go           四边形合并与分块
  internal/cv/          OpenCV 桥接实现
  cmd/textinpaint/       CLI 示例
  cmd/textinpaint-server/ HTTP 服务入口
  httpapi/              HTTP 接口与内嵌网页测试页
  examples/http-client/  只依赖标准库的 Go HTTP 客户端
  scripts/              Windows 环境、依赖与构建脚本
  models/               检测模型
  native/windows-amd64/  Windows 头文件、导入库和 DLL
  third_party/          第三方说明与许可证
```

`native/`、模型二进制、`build/`、`bin/` 和本地测试图片默认不进 Git；发布源码后可用 setup 脚本恢复依赖。实际验证记录见 `VALIDATION.md`。
