param(
    [string]$StartUrl = "https://putusan3.mahkamahagung.go.id/direktori/index/kategori/perdagangan-orang-1.html",
    [string]$TargetDownloads = 264,
    [string]$OutDir = "downloads/TPPO",
    [int]$TimeoutSeconds = 120,
    [int]$ManualClearanceTimeoutSeconds = 120,
    [int]$MaxCandidates = 0,
    [switch]$RestartListing,
    [switch]$NewTarget
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$CaseTitlePrefix = "Putusan PN"

$CrawlArgs = @(
    "sinergi",
    "crawl",
    "--start-url", $StartUrl,
    "--out-dir", $OutDir,
    "--timeout-seconds", "$TimeoutSeconds",
    "--manual-clearance-timeout-seconds", "$ManualClearanceTimeoutSeconds",
    "--case-title-prefix", $CaseTitlePrefix,
    "--no-refresh-profile-snapshot"
)

if ($MaxCandidates -gt 0) {
    $CrawlArgs += @("--max-candidates", "$MaxCandidates")
}

if ($RestartListing) {
    $CrawlArgs += "--restart-listing"
}

if ($NewTarget) {
    $CrawlArgs += "--new-target"
}

$Selection = $TargetDownloads.Trim()
while ([string]::IsNullOrWhiteSpace($Selection) -or $Selection -ieq "ask") {
    $Selection = Read-Host "Download how many new PDFs? Enter a number or 'all'"
}

if ($Selection -ieq "all") {
    Write-Host "Starting resumable download for all matching new PDFs."
    & uv run @CrawlArgs --download-all
    exit $LASTEXITCODE
}

$DownloadTarget = 0
if (-not [int]::TryParse($Selection, [ref]$DownloadTarget) -or $DownloadTarget -le 0) {
    Write-Error "TargetDownloads must be a positive number, 'all', or 'ask'."
    exit 1
}

Write-Host "Starting or resuming target of $DownloadTarget new PDF(s)."
& uv run @CrawlArgs --target-downloads "$DownloadTarget"

exit $LASTEXITCODE
