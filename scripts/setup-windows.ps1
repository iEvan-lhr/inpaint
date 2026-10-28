param([string]$CompilerBin, [int]$Jobs = 4, [switch]$Force)
$ErrorActionPreference = 'Stop'
$libraryRoot = Split-Path $PSScriptRoot -Parent
if ($CompilerBin) { $env:PATH = $CompilerBin + ';' + $env:PATH }
foreach ($command in @('cmake','gcc','g++','mingw32-make')) {
    if (!(Get-Command $command -ErrorAction SilentlyContinue)) { throw "$command is required; install MinGW-w64 and CMake or pass -CompilerBin" }
}
$nativeRoot = Join-Path $libraryRoot 'native\windows-amd64'
$buildRoot = Join-Path $libraryRoot 'build\opencv'
New-Item -ItemType Directory -Path $buildRoot,(Join-Path $nativeRoot 'bin'),(Join-Path $libraryRoot 'models'),(Join-Path $libraryRoot 'third_party') -Force | Out-Null
$ready = (Test-Path (Join-Path $nativeRoot 'lib\libopencv_photo4120.dll.a')) -and
         (Test-Path (Join-Path $nativeRoot 'lib\libopencv_imgproc4120.dll.a')) -and
         (Test-Path (Join-Path $nativeRoot 'lib\libopencv_core4120.dll.a'))
if ($Force -or !$ready) {
    $archive = Join-Path $buildRoot 'opencv-4.12.0.zip'
    if (!(Test-Path $archive)) { Invoke-WebRequest 'https://github.com/opencv/opencv/archive/refs/tags/4.12.0.zip' -OutFile $archive }
    $expected = 'FA3FAF7581F1FA943C9E670CF57DD6BA1C5B4178F363A188A2C8BFF1EB28B7E4'
    if ((Get-FileHash $archive -Algorithm SHA256).Hash -ne $expected) { throw 'OpenCV archive SHA256 mismatch' }
    $source = Join-Path $buildRoot 'opencv-4.12.0'
    if (!(Test-Path $source)) { Expand-Archive -LiteralPath $archive -DestinationPath $buildRoot }
    $binary = Join-Path $buildRoot 'compiled'
    & cmake -S $source -B $binary -G 'MinGW Makefiles' `
        '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_INSTALL_PREFIX=$nativeRoot" `
        '-DOPENCV_INCLUDE_INSTALL_PATH=include/opencv4' '-DOPENCV_LIB_INSTALL_PATH=lib' '-DOPENCV_BIN_INSTALL_PATH=bin' `
        '-DBUILD_LIST=core,imgproc,photo' '-DBUILD_opencv_world=OFF' '-DBUILD_SHARED_LIBS=ON' `
        '-DBUILD_TESTS=OFF' '-DBUILD_PERF_TESTS=OFF' '-DBUILD_EXAMPLES=OFF' '-DBUILD_opencv_apps=OFF' `
        '-DBUILD_JAVA=OFF' '-DBUILD_opencv_python2=OFF' '-DBUILD_opencv_python3=OFF' `
        '-DWITH_IPP=OFF' '-DWITH_OPENCL=OFF' '-DWITH_OPENEXR=OFF' '-DWITH_PROTOBUF=OFF' `
        '-DWITH_FFMPEG=OFF' '-DWITH_MSMF=OFF' '-DWITH_DSHOW=OFF' '-DWITH_ITT=OFF' `
        '-DWITH_TBB=OFF' '-DWITH_LAPACK=OFF' '-DBUILD_ZLIB=ON' '-DCPU_DISPATCH=' `
        '-DOPENCV_ENABLE_NONFREE=OFF' '-DCMAKE_C_COMPILER=gcc' '-DCMAKE_CXX_COMPILER=g++'
    if ($LASTEXITCODE -ne 0) { throw 'OpenCV configuration failed' }
    & cmake --build $binary --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'OpenCV compilation failed' }
    & cmake --install $binary
    if ($LASTEXITCODE -ne 0) { throw 'OpenCV installation failed' }
    Copy-Item -LiteralPath (Join-Path $source 'LICENSE') -Destination (Join-Path $libraryRoot 'third_party\OpenCV.LICENSE')
}
$compilerDirectory = Split-Path (Get-Command g++).Source -Parent
foreach ($name in @('libstdc++-6.dll','libgcc_s_seh-1.dll','libwinpthread-1.dll','libmcfgthread-1.dll')) {
    $file = Join-Path $compilerDirectory $name
    if (Test-Path $file) { Copy-Item -LiteralPath $file -Destination (Join-Path $nativeRoot 'bin') }
}
if (!(Test-Path (Join-Path $nativeRoot 'bin\onnxruntime.dll'))) {
    $runtimeZip = Join-Path $buildRoot 'onnxruntime-win-x64-1.22.0.zip'
    Invoke-WebRequest 'https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-win-x64-1.22.0.zip' -OutFile $runtimeZip
    $runtimeDir = Join-Path $buildRoot 'onnxruntime-win-x64-1.22.0'
    if (!(Test-Path $runtimeDir)) { Expand-Archive -LiteralPath $runtimeZip -DestinationPath $buildRoot }
    Copy-Item -LiteralPath (Join-Path $runtimeDir 'lib\onnxruntime.dll') -Destination (Join-Path $nativeRoot 'bin')
    Copy-Item -LiteralPath (Join-Path $runtimeDir 'LICENSE') -Destination (Join-Path $libraryRoot 'third_party\ONNXRuntime.LICENSE')
    Copy-Item -LiteralPath (Join-Path $runtimeDir 'ThirdPartyNotices.txt') -Destination (Join-Path $libraryRoot 'third_party\ONNXRuntime.ThirdPartyNotices.txt')
}
$modelPath = Join-Path $libraryRoot 'models\ch_PP-OCRv4_det_infer.onnx'
if (!(Test-Path $modelPath)) {
    # Extract only the ONNX model from the pinned official RapidOCR wheel.
    # This does not install Python or execute Python code.
    $release = Invoke-RestMethod 'https://pypi.org/pypi/rapidocr-onnxruntime/1.4.4/json'
    $wheel = $release.urls | Where-Object filename -EQ 'rapidocr_onnxruntime-1.4.4-py3-none-any.whl' | Select-Object -First 1
    if (!$wheel) { throw 'Pinned RapidOCR wheel not found' }
    $wheelPath = Join-Path $buildRoot $wheel.filename
    Invoke-WebRequest $wheel.url -OutFile $wheelPath
    if ((Get-FileHash $wheelPath -Algorithm SHA256).Hash -ne $wheel.digests.sha256) { throw 'RapidOCR wheel checksum mismatch' }
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [IO.Compression.ZipFile]::OpenRead($wheelPath)
    try {
        $entry = $zip.GetEntry('rapidocr_onnxruntime/models/ch_PP-OCRv4_det_infer.onnx')
        if (!$entry) { throw 'Model is absent from the RapidOCR wheel' }
        [IO.Compression.ZipFileExtensions]::ExtractToFile($entry,$modelPath,$true)
        $licenseEntry = $zip.Entries | Where-Object { $_.FullName -match 'dist-info/(licenses/)?LICENSE$' } | Select-Object -First 1
        if ($licenseEntry) { [IO.Compression.ZipFileExtensions]::ExtractToFile($licenseEntry,(Join-Path $libraryRoot 'third_party\RapidOCR.LICENSE'),$true) }
    } finally { $zip.Dispose() }
}
if ((Get-FileHash $modelPath -Algorithm SHA256).Hash -ne 'D2A7720D45A54257208B1E13E36A8479894CB74155A5EFE29462512D42F49DA9') { throw 'Model SHA256 mismatch' }
Write-Host 'Native libraries and the detection model are ready.'
