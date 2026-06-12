# Etiquetador de patches

Herramienta local autocontenida para generar y etiquetar patches de
granulometria visual.

## Abrir con doble clic

En la raiz del proyecto, abrir:

```text
Etiquetar parches.lnk
```

Por defecto carga `Data parches test`. La ventana de PowerShell/CMD debe
permanecer abierta mientras se utiliza la interfaz.

Tambien se puede arrastrar otra carpeta de imagenes sobre
`abrir_etiquetador.cmd`.

Si el proyecto cambia de ubicacion, el acceso directo se recrea con:

```powershell
powershell -ExecutionPolicy Bypass -File tools\patch_labeler\crear_acceso_directo.ps1
```

## Abrir desde PowerShell

```powershell
conda activate memoria-dexined
python tools\patch_labeler\launcher.py
```

Para usar otro dataset:

```powershell
python tools\patch_labeler\launcher.py "D:\ruta\a\patches"
```

## Generar patches

```powershell
python tools\patch_labeler\generate_patches.py `
  --source "D:\ruta\a\recortes_de_carga" `
  --run-name prueba_patches `
  --patch-size 256 `
  --stride 128 `
  --min-valid-fraction 0.8
```

La salida se guarda en `outputs\patches\<run-name>` con `images`,
`patches.csv` y `params.json`.

## Archivos

```text
tools/patch_labeler/
|-- app.py
|-- generate_patches.py
|-- patches.py
|-- launcher.py
|-- abrir_etiquetador.cmd
|-- crear_acceso_directo.ps1
|-- README.md
`-- assets/patch_labeler.ico
```
