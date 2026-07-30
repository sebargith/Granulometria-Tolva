$ErrorActionPreference = "Stop"

$toolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $toolDir "..\..")).Path
$shortcutPath = Join-Path $projectRoot "Abrir Webapp Granulometria de Carga.lnk"
$targetPath = Join-Path $toolDir "abrir_etiquetador.cmd"
$datasetPath = Join-Path $projectRoot "outputs\patches\grid96_batch003_data3_mixto_ai"
$iconPath = Join-Path $toolDir "assets\patch_labeler.ico"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.Arguments = "`"$datasetPath`""
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = "$iconPath,0"
$shortcut.Description = "Abrir la aplicacion de granulometria en tolva"
$shortcut.Save()

Write-Host "Acceso directo creado: $shortcutPath"
