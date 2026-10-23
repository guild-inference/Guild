# Guild One-Command Native Installer for Windows PowerShell (x86_64)
#
# Usage (latest preview):
#   irm https://raw.githubusercontent.com/guild-inference/Guild/main/install.ps1 | iex
#
# Version-pinned installation:
#   $env:GUILD_VERSION = "v0.2.0-preview.3"; irm https://raw.githubusercontent.com/guild-inference/Guild/main/install.ps1 | iex
#
# Direct script options:
#   .\install.ps1 [-Version <tag>] [-Channel <preview|stable>] [-Variant <auto|cuda12|cpu>]
#                 [-InstallRoot <dir>] [-NoPathUpdate] [-Rollback] [-Uninstall] [-DryRun]

param(
    [string]$Version = $env:GUILD_VERSION,
    [string]$Channel = $(if ($env:GUILD_CHANNEL) { $env:GUILD_CHANNEL } else { "preview" }),
    [string]$Variant = $(if ($env:GUILD_VARIANT) { $env:GUILD_VARIANT } else { "auto" }),
    [string]$InstallRoot = $(if ($env:GUILD_INSTALL_ROOT) { $env:GUILD_INSTALL_ROOT } else { Join-Path $env:LOCALAPPDATA "Guild" }),
    [string]$ReleaseBaseUrl = $env:GUILD_RELEASE_BASE_URL,
    [switch]$NoPathUpdate,
    [switch]$Rollback,
    [switch]$Uninstall,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Repo = if ($env:GUILD_GITHUB_REPO) { $env:GUILD_GITHUB_REPO } else { "guild-inference/Guild" }
$BinDir = Join-Path $InstallRoot "bin"
$VersionsDir = Join-Path $InstallRoot "versions"
$StateFile = Join-Path $InstallRoot "install-state.json"

function Write-Info([string]$Message) {
    Write-Host "==> $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Warning $Message
}

if ($Uninstall) {
    Write-Info "Uninstalling Guild binaries from $InstallRoot (preserving data directory)..."
    if (Test-Path $BinDir) { Remove-Item -Recurse -Force $BinDir }
    if (Test-Path $VersionsDir) { Remove-Item -Recurse -Force $VersionsDir }
    if (Test-Path $StateFile) { Remove-Item -Force $StateFile }
    Write-Info "Uninstalled Guild executable. User models and GPACK/GEXEC caches were preserved."
    return
}

if ($Rollback) {
    if (-not (Test-Path $StateFile)) {
        throw "No installation state found at $StateFile to roll back."
    }
    $state = Get-Content $StateFile -Raw | ConvertFrom-Json
    if (-not $state.previous_dir -or -not (Test-Path (Join-Path $state.previous_dir "guild.exe"))) {
        throw "No previous Guild version available to roll back to."
    }
    Write-Info "Rolling back Guild to $($state.previous_dir)..."
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    Copy-Item -Path (Join-Path $state.previous_dir "*") -Destination $BinDir -Recurse -Force
    $prev = $state.previous_dir
    $state.previous_dir = $state.current_dir
    $state.current_dir = $prev
    $state | ConvertTo-Json -Depth 5 | Set-Content -Path $StateFile -Encoding UTF8
    & (Join-Path $BinDir "guild.exe") --version
    return
}

# 1. Detect Architecture and GPU
$Arch = $env:PROCESSOR_ARCHITECTURE
if ($Arch -ne "AMD64" -and $Arch -ne "x86_64") {
    throw "Precompiled Windows binaries require x86_64 / AMD64 (detected: $Arch)."
}

$GpuName = $env:GUILD_MOCK_GPU_NAME
$GpuCc = $env:GUILD_MOCK_GPU_CC
$GpuDriver = $env:GUILD_MOCK_GPU_DRIVER

if (-not $GpuCc) {
    $nvidiaSmi = Get-Command "nvidia-smi.exe" -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        try {
            $smiOut = & $nvidiaSmi.Source --query-gpu=name,compute_cap,driver_version --format=csv,noheader,nounits 2>$null | Select-Object -First 1
            if ($smiOut) {
                $parts = $smiOut.Split(",")
                if ($parts.Count -ge 3) {
                    $GpuName = $parts[0].Trim()
                    $GpuCc = $parts[1].Trim()
                    $GpuDriver = $parts[2].Trim()
                }
            }
        } catch {}
    }
}

$ResolvedVariant = "windows-x86_64-cpu"
if ($Variant -ne "auto") {
    switch ($Variant) {
        "cuda12" { $ResolvedVariant = "windows-x86_64-cuda12" }
        "windows-x86_64-cuda12" { $ResolvedVariant = "windows-x86_64-cuda12" }
        "cpu" { $ResolvedVariant = "windows-x86_64-cpu" }
        "windows-x86_64-cpu" { $ResolvedVariant = "windows-x86_64-cpu" }
        default { throw "Unsupported -Variant '$Variant' (expected auto, cuda12, or cpu)." }
    }
} elseif ($GpuCc -and $GpuDriver) {
    $ccParts = $GpuCc.Split(".")
    $ccInt = [int]$ccParts[0] * 10 + [int]$ccParts[1]
    $drvMajor = [int]($GpuDriver.Split(".")[0])
    if ($ccInt -ge 60 -and $drvMajor -ge 525) {
        $ResolvedVariant = "windows-x86_64-cuda12"
    }
}

$ArchiveName = "guild-$ResolvedVariant.zip"
Write-Info "Detected host: Windows $Arch (GPU='$(if ($GpuName) { $GpuName } else { 'none' })', CC='$(if ($GpuCc) { $GpuCc } else { 'n/a' })', Driver='$(if ($GpuDriver) { $GpuDriver } else { 'n/a' })')"

if ($DryRun) {
    [PSCustomObject]@{
        os = "Windows"
        arch = $Arch
        gpu_name = $(if ($GpuName) { $GpuName } else { "none" })
        gpu_cc = $GpuCc
        gpu_driver = $GpuDriver
        variant = $ResolvedVariant
        archive = $ArchiveName
        install_root = $InstallRoot
        bin_dir = $BinDir
    } | ConvertTo-Json -Compress
    return
}

function Copy-OrDownload([string]$Url, [string]$Destination) {
    if ($Url.StartsWith("file://")) {
        $localPath = $Url.Substring(7)
        Copy-Item -Path $localPath -Destination $Destination -Force
    } else {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
    }
}

New-Item -ItemType Directory -Force -Path $InstallRoot | Out-Null
$StageRoot = Join-Path $InstallRoot ("staging-" + $PID + "-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force -Path $StageRoot | Out-Null

try {
    if ($ReleaseBaseUrl) {
        $AssetBase = $ReleaseBaseUrl.TrimEnd("/")
        Copy-OrDownload "$AssetBase/guild-release.json" (Join-Path $StageRoot "guild-release.json")
        $manifest = Get-Content (Join-Path $StageRoot "guild-release.json") -Raw | ConvertFrom-Json
        if (-not $Version) { $Version = $manifest.release_tag }
    } else {
        if (-not $Version) {
            Write-Info "Resolving latest $Channel release from $Repo..."
            $releases = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases?per_page=30" -Headers @{ "User-Agent" = "Guild-Installer" }
            foreach ($rel in $releases) {
                if ($rel.draft) { continue }
                if ($Channel -eq "stable" -and $rel.prerelease) { continue }
                $hasManifest = $false
                foreach ($a in $rel.assets) {
                    if ($a.name -eq "guild-release.json") { $hasManifest = $true; break }
                }
                if ($hasManifest) {
                    $Version = $rel.tag_name
                    break
                }
            }
            if (-not $Version) {
                throw "No compatible published release with guild-release.json found on $Repo."
            }
        }
        $AssetBase = "https://github.com/$Repo/releases/download/$Version"
        Copy-OrDownload "$AssetBase/guild-release.json" (Join-Path $StageRoot "guild-release.json")
        $manifest = Get-Content (Join-Path $StageRoot "guild-release.json") -Raw | ConvertFrom-Json
    }

    if ($manifest.release_tag -ne $Version) {
        throw "Release manifest tag mismatch: expected $Version, got $($manifest.release_tag)"
    }

    # Fall back to windows-x86_64-cpu if windows-x86_64-cuda12 is not in this release's variants
    if ($Variant -eq "auto" -and $ResolvedVariant -eq "windows-x86_64-cuda12") {
        if (-not $manifest.variants -or -not $manifest.variants.PSObject.Properties[$ResolvedVariant]) {
            Write-Warn "Release $Version does not include $ResolvedVariant; falling back to windows-x86_64-cpu."
            $ResolvedVariant = "windows-x86_64-cpu"
            $ArchiveName = "guild-$ResolvedVariant.zip"
        }
    }

    $ArchivePath = Join-Path $StageRoot $ArchiveName
    $ShaPath = Join-Path $StageRoot "$ArchiveName.sha256"
    Write-Info "Downloading Guild $Version ($ArchiveName)..."
    Copy-OrDownload "$AssetBase/$ArchiveName" $ArchivePath
    Copy-OrDownload "$AssetBase/$ArchiveName.sha256" $ShaPath

    $ExpectedArchiveSha = ((Get-Content $ShaPath -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
    $ActualArchiveSha = (Get-FileHash -Path $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualArchiveSha -ne $ExpectedArchiveSha) {
        throw "Archive SHA-256 mismatch for ${ArchiveName}: expected $ExpectedArchiveSha, got $ActualArchiveSha"
    }

    $ExpectedExecSha = $manifest.executable_sha256
    if ($manifest.variants -and $manifest.variants.PSObject.Properties[$ResolvedVariant]) {
        $vEntry = $manifest.variants.PSObject.Properties[$ResolvedVariant].Value
        if ($vEntry.sha256.ToLowerInvariant() -ne $ActualArchiveSha) {
            throw "Manifest archive SHA-256 mismatch for $ResolvedVariant"
        }
        $ExpectedExecSha = $vEntry.executable_sha256
    }

    $ExtractDir = Join-Path $StageRoot "extracted"
    Expand-Archive -Path $ArchivePath -DestinationPath $ExtractDir -Force
    $InnerDir = Get-ChildItem -Path $ExtractDir -Directory | Select-Object -First 1
    $StagedBinDir = if ($InnerDir) { $InnerDir.FullName } else { $ExtractDir }
    $StagedExe = Join-Path $StagedBinDir "guild.exe"
    if (-not (Test-Path $StagedExe)) {
        throw "Extracted archive is missing guild.exe"
    }

    $ActualExecSha = (Get-FileHash -Path $StagedExe -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ExpectedExecSha -and ($ActualExecSha -ne $ExpectedExecSha.ToLowerInvariant())) {
        throw "Executable SHA-256 mismatch: expected $ExpectedExecSha, got $ActualExecSha"
    }

    # Smoke test staged binary before touching installation
    $SmokeData = Join-Path $StageRoot "smoke-data"
    New-Item -ItemType Directory -Force -Path $SmokeData | Out-Null
    $oldDataDir = $env:GUILD_DATA_DIR
    try {
        $env:GUILD_DATA_DIR = $SmokeData
        & $StagedExe --version | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Staged guild.exe --version failed with exit code $LASTEXITCODE" }
        & $StagedExe --help | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Staged guild.exe --help failed with exit code $LASTEXITCODE" }
    } finally {
        $env:GUILD_DATA_DIR = $oldDataDir
    }

    # Install into versioned directory and preserve previous for rollback
    New-Item -ItemType Directory -Force -Path $VersionsDir | Out-Null
    $VersionTarget = Join-Path $VersionsDir "$Version-$ResolvedVariant"
    if (Test-Path $VersionTarget) { Remove-Item -Recurse -Force $VersionTarget }
    Move-Item -Path $StagedBinDir -Destination $VersionTarget

    $PrevDir = $null
    if (Test-Path $StateFile) {
        try {
            $oldState = Get-Content $StateFile -Raw | ConvertFrom-Json
            if ($oldState.current_dir -and ($oldState.current_dir -ne $VersionTarget)) {
                $PrevDir = $oldState.current_dir
            } else {
                $PrevDir = $oldState.previous_dir
            }
        } catch {}
    }

    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    Copy-Item -Path (Join-Path $VersionTarget "*") -Destination $BinDir -Recurse -Force

    @{
        version = $Version
        variant = $ResolvedVariant
        current_dir = $VersionTarget
        previous_dir = $PrevDir
    } | ConvertTo-Json -Depth 5 | Set-Content -Path $StateFile -Encoding UTF8

    if (-not $NoPathUpdate) {
        $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
        if (-not $UserPath) { $UserPath = "" }
        $PathParts = $UserPath -split ";" | Where-Object { $_ -ne "" }
        if ($PathParts -notcontains $BinDir) {
            $NewUserPath = if ($UserPath) { "$BinDir;$UserPath" } else { $BinDir }
            [Environment]::SetEnvironmentVariable("Path", $NewUserPath, "User")
            $env:Path = "$BinDir;$env:Path"
            Write-Info "Added $BinDir to User PATH."
        }
    }

    $InstalledExe = Join-Path $BinDir "guild.exe"
    $VerLine = (& $InstalledExe --version | Select-Object -First 1)
    & $InstalledExe --help | Out-Null
    Write-Info "Installed $VerLine ($ResolvedVariant)"
    Write-Host "  Executable : $InstalledExe"
    Write-Host "  Launch TUI : guild tui"
    Write-Host "  Pull model : guild pull qwen3.8-flash-next"
    Write-Host "  Rollback   : .\install.ps1 -Rollback"
} finally {
    if (Test-Path $StageRoot) {
        Remove-Item -Recurse -Force $StageRoot -ErrorAction SilentlyContinue
    }
}
