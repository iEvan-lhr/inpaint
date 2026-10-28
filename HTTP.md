# 内网图片去字服务

服务运行在一台电脑上，其他程序通过 HTTP 上传图片，服务返回处理结果。模型在启动时加载一次。运行期间不访问外网，也不把图片写入服务器磁盘。

## 现在如何测试

完整 Windows 包已包含可执行文件、DLL 和模型，无需安装 Go 或 Python。打开 PowerShell，进入库目录：

```powershell
.\bin\textinpaint-server.exe -listen 0.0.0.0:8080
```

保持窗口运行。浏览器打开 **http://127.0.0.1:8080**，选择 PNG/JPEG 后点击「开始去字」。可对照原图检查效果、下载 PNG。按 Ctrl+C 停止服务。

需要文字位置时，选择图片后点击「获取位置 JSON」。页面直接展示格式化的区域坐标、检测分数、图片尺寸和耗时，并提供「复制 JSON」「下载 JSON」。这一步使用现有 `/v1/detect` 接口，不需要先去字，也没有增加文字内容识别模型。换一张图片时会清空上一张的 JSON。同一张图片再次点击会显示已有结果。

通过内网 HTTP 地址访问时，页面也支持复制回退；若浏览器仍限制自动复制，会选中内容并提示手动复制，下载不受此限制。

如果启动提示端口已占用，改为 `-listen 0.0.0.0:8086`，并把下面 URL 的端口对应改为 8086。

从源码运行前，先按 README 执行 `setup-windows.ps1` 和 `build.ps1`。服务也可以用 `scripts/start-server.ps1` 启动。模型默认位于可执行文件上一级的 `models/`；Windows DLL 与可执行文件放在同一 `bin/` 中。

## 内网访问

在服务电脑运行 `ipconfig` 查看当前网络适配器的 IPv4 地址。例如地址为 `192.168.1.50`，其他电脑访问 `http://192.168.1.50:8080`。

客户端和服务电脑需要网络互通。若本机可以访问而其他电脑不能访问，检查 Windows 防火墙是否允许此程序在所用内网访问 TCP 8080，以及无线网络是否开启了客户端隔离。启动脚本不会修改系统防火墙。`127.0.0.1` 仅指向调用程序所在电脑。

此命令启动一个前台 HTTP 进程，关闭进程或电脑关机后服务停止；它尚未注册为 Windows 开机自启服务。

## 接口

| 方法 / 路径 | 请求 | 成功返回 |
| --- | --- | --- |
| `GET /` | 无 | 浏览器测试页 |
| `GET /healthz` | 无 | `{"status":"ok"}`，表示模型已加载且服务可响应 |
| `POST /v1/remove` | multipart，文件字段 `image` | 去字后的 PNG |
| `POST /v1/remove?format=zip` | 同上 | ZIP：`no_text.png`、`mask.png`、`detections.png`、`regions.json` |
| `POST /v1/detect` | 同上 | JSON：文字四边形、分数、尺寸和检测耗时 |

每次请求只能提交一张 PNG/JPEG。不按文件扩展名判断格式。检测接口只定位文字，不识别文字内容。

两个 POST 接口均支持查询参数 `min_text_height`，范围 `0..256`，单位为原图像素。默认值为 12；0 关闭过滤。高度根据原图笔画估算，检测框扩大和推理时放大不会把极小字直接当成大字。`regions` 只返回达到阈值的区域，并增加 `text_height` 字段。检测、PNG 和 ZIP 使用同一过滤规则。

默认图形保护同时保留以小字为主、带深色完整边框的界面示意图；这些界面中的区域也不进入 `regions`。`min_text_height=0` 关闭这项随小字过滤启用的整体保护；`-preserve-graphics=false` 也可关闭图形保护。

```powershell
curl.exe --fail-with-body -F "image=@input.png" `
  "http://127.0.0.1:8080/v1/remove?min_text_height=16" -o no_text.png
curl.exe --fail-with-body -F "image=@input.png" `
  "http://127.0.0.1:8080/v1/detect?min_text_height=16" -o regions.json
```

页面可以直接调整这个值；修改图片或阈值后会清空旧预览和 JSON，避免使用不匹配的缓存。查询参数仅对当前请求生效，不改变其他调用方的配置。服务端和 CLI 都可用 `-min-text-height 16` 更改默认值；页面发送用户选择的值并覆盖服务默认值。自行实现 HTTP `Processor` 时，可同时实现 `OptionsProcessor` 接口以支持这个参数；旧的处理器仍可处理不带该参数的请求。

去字 / 检测成功响应附带 `X-Region-Count` 和 `X-Processing-Ms`。后者包括读取上传内容、解码、处理和输出编码的时间，不包含客户端下载结果的时间。`regions.json` 中的 `timings` 记录各算法阶段耗时。

### curl / PowerShell

以下命令使用 Windows 自带 `curl.exe`，避免 PowerShell 的 `curl` 别名差异。Linux/macOS 使用 `curl`。

```powershell
Invoke-RestMethod http://127.0.0.1:8080/healthz

curl.exe --fail-with-body -F "image=@C:\images\input.png" `
  http://127.0.0.1:8080/v1/remove -o no_text.png

curl.exe --fail-with-body -F "image=@C:\images\input.png" `
  "http://127.0.0.1:8080/v1/remove?format=zip" -o result.zip

curl.exe --fail-with-body -F "image=@C:\images\input.png" `
  http://127.0.0.1:8080/v1/detect -o regions.json
```

请检查 HTTP 状态或 curl 退出码；失败时返回的是 JSON 错误，不是图片。

### Go 调用

可运行的客户端位于 `examples/http-client/main.go`，仅依赖 Go 标准库，不导入图像处理库，不需要 cgo：

```powershell
go run ./examples/http-client -input C:\images\input.png -output no_text.png `
  -url http://127.0.0.1:8080/v1/remove
```

### Python 调用

调用程序安装 `requests` 后：

```python
from pathlib import Path
import requests

with open("input.png", "rb") as image:
    response = requests.post(
        "http://127.0.0.1:8080/v1/remove",
        files={"image": ("input.png", image, "image/png")},
        timeout=(10, 180),
    )
response.raise_for_status()
Path("no_text.png").write_bytes(response.content)
```

使用其他语言时，发送相同 multipart 请求并把成功响应体保存为 PNG 即可。网页测试页与服务同源；默认不开放跨域 CORS，需要不同网站的前端调用时，可由该网站的后端代理请求。

## 可选 API Key

默认无需密钥。如果要限制调用，在启动服务的 PowerShell 中设置环境变量：

```powershell
$env:TEXTINPAINT_API_KEY = '替换成你自己的密钥'
.\bin\textinpaint-server.exe -listen 0.0.0.0:8080
```

客户端为两个 POST 接口增加 `Authorization: Bearer <密钥>`：

```powershell
curl.exe --fail-with-body -H "Authorization: Bearer $env:TEXTINPAINT_API_KEY" `
  -F "image=@input.png" http://127.0.0.1:8080/v1/remove -o no_text.png
```

网页测试页展开「访问密钥」填写即可；Go 示例也读取 `TEXTINPAINT_API_KEY`。首页和健康检查不要求鉴权。密钥不保存到网页本地存储。内置监听为 HTTP；如需 HTTPS，可在服务前配置反向代理。

## 参数和错误

默认请求上限 20 MiB（包含 multipart 包装）、最多 1600 万像素、一次处理一个请求、处理期限 2 分钟。超出并发数返回 503 和 `Retry-After: 2`，客户端稍后重试；不会无限堆积待处理图片。

```powershell
.\bin\textinpaint-server.exe -listen 0.0.0.0:8080 `
  -max-upload-mb 20 -max-pixels 16000000 -max-concurrent 1 -timeout 2m
```

可用 `-tile-size`、`-box-threshold`、`-preserve-graphics` 调整处理参数，与 CLI 含义相同。默认启用页边放大补检，`-detect-margins=false` 可关闭以减少推理次数。可用 `-model`、`-runtime` 指定其他部署路径。单个模型的检测仍然串行，增大 HTTP 并发数不保证加速。超时和客户端断开会在处理阶段之间检查，无法中断正在执行的单次原生推理或编码。

POST 接口的处理错误为 JSON：

```json
{"error":{"code":"missing_image","message":"请选择一张图片，上传字段名为 image"}}
```

常见状态：400 表单/参数错误，401 密钥不正确，413 上传或像素超限，415 图片或请求格式不支持，408 请求取消，503 服务忙，504 处理超时，500 处理失败。500 的详细错误只写服务日志。

## 嵌入现有 Go HTTP 程序

除了独立可执行文件，还可以把 `httpapi.New(engine, httpapi.Options{})` 返回的 handler 交给 `http.Server`。先创建模型，等 HTTP 服务停止并完成请求后再关闭模型。生产部署时应像 `cmd/textinpaint-server` 一样设置 HTTP 读写超时。

Linux/macOS 的 OpenCV 与 ONNX Runtime 准备方式见 README，使用 `go build -o bin/textinpaint-server ./cmd/textinpaint-server` 构建。本次实测平台是 Windows amd64。
