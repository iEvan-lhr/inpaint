# 第三方来源

| 组件 | 版本 / 来源 | 许可证文件 |
| --- | --- | --- |
| OpenCV core/imgproc/photo | 官方 4.12.0 源码，本机 MinGW-w64 GCC 14.2 构建 | OpenCV.LICENSE |
| ONNX Runtime CPU | 1.22.0；交付 DLL 来自 onnxruntime_go v1.22.0 的测试运行库 | ONNXRuntime.LICENSE、ONNXRuntime.ThirdPartyNotices.txt |
| onnxruntime_go | github.com/yalue/onnxruntime_go v1.22.0 | onnxruntime_go.LICENSE |
| PP-OCRv4 mobile 模型 | rapidocr-onnxruntime 1.4.4 包内模型 | RapidOCR.LICENSE、PaddleOCR.LICENSE |
| libstdc++ / libgcc | 本机 MinGW-w64 GCC 14.2 运行库 | GCC.COPYING3、GCC.COPYING.RUNTIME |
| MCF Gthread | 本机 MinGW-w64 运行库 | MCFGThread.LICENSE.md、MCFGThread.LGPL3 |
| winpthreads | 本机 MinGW-w64 运行库 | winpthreads.COPYING |

检测的归一化和 DB 后处理参考 PaddleOCR/RapidOCR。PaddleOCR 相关代码版权声明：Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 相关实现按 Apache License 2.0 提供。

OpenCV 下载地址：
https://github.com/opencv/opencv/archive/refs/tags/4.12.0.zip

该源码归档 SHA256：`fa3faf7581f1fa943c9e670cf57dd6ba1c5b4178f363a188a2c8bff1eb28b7e4`

ONNX Runtime 官方重新下载地址：
https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip

本次交付的 `onnxruntime.dll` SHA256：`579b636403983254346a5c1d80bd28f1519cd1e284cd204f8d4ff41f8d711559`

GCC 源码及运行库许可：https://gcc.gnu.org/releases.html

MCF Gthread 源码及运行库许可：https://github.com/lhmouse/mcfgthread
