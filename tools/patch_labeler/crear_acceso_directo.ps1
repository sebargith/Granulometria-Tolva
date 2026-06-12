$ErrorActionPreference = "Stop"

$toolDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = (Resolve-Path (Join-Path $toolDir "..\..")).Path
$shortcutPath = Join-Path $projectRoot "Etiquetar parches.lnk"
$targetPath = Join-Path $toolDir "abrir_etiquetador.cmd"
$iconPath = Join-Path $toolDir "assets\patch_labeler.ico"

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $targetPath
$shortcut.WorkingDirectory = $projectRoot
$shortcut.IconLocation = "$iconPath,0"
$shortcut.Description = "Abrir etiquetador de patches de granulometria"
$shortcut.Save()

Write-Host "Acceso directo creado: $shortcutPath"
