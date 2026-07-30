# Tarea: etiquetado de granulometria por grilla

Vas a recibir imagenes de carga de tolva normalizadas por ancho. Cada imagen
tiene una version sin grilla y una version con grilla numerada. Cada celda de
la grilla corresponde a una zona de material que debe clasificarse.

## Clases permitidas

Usa solo estas clases:

- `fino`: material visualmente fino, textura granular pequena, sin rocas grandes dominantes.
- `medio`: rocas medianas o mezcla donde domina granulometria intermedia.
- `grueso`: rocas grandes/bolones dominantes o zona claramente mas gruesa que el resto.
- `invalido`: celda mal ubicada, contiene principalmente borde/fondo/tolva/sombra extrema/no material.
- `ambiguo`: no se puede decidir de forma confiable entre clases validas.

No inventes otras clases.

## Referencias

Dentro del zip hay una carpeta `referencias` con ejemplos de carga fina, media
y gruesa. Primero observa esas referencias para calibrar la escala visual.

La escala relevante es relativa a la carga completa, no a una celda aislada.
Si una roca parece grande dentro de una celda pero no es representativa frente
a la carga completa, evita sobreclasificarla como `grueso`.

## Instrucciones

Para cada frame:

1. Abre `01_carga_normalizada.jpg` para entender el contexto general.
2. Abre `02_grilla_numerada.jpg` para ver los IDs de celda.
3. Completa el archivo `cells_template.csv`.
4. Devuelve una sola respuesta consolidada por ZIP, no un bloque separado por frame.

El archivo/respuesta debe llamarse igual que el ZIP, cambiando el prefijo
`batch` por `labels` y manteniendo el numero de parte.

Ejemplos:

- si recibes `batch001_part001.zip`, responde con `labels_batch001_part001.csv`
- si recibes `batch001_part007.zip`, responde con `labels_batch001_part007.csv`

El CSV consolidado debe tener estas columnas exactas:

```csv
frame_folder,image_name,cell_id,label,confidence,notes
```

Valores permitidos para `confidence`:

- `alta`
- `media`
- `baja`

## Criterio

- Clasifica segun la zona visible dentro de cada celda.
- Usa el contexto global de la carga para decidir si una zona es fina, media o gruesa.
- Si una celda mezcla dos clases, elige la clase dominante.
- Si la mezcla es fuerte y no hay dominante claro, usa `ambiguo`.
- Si la celda contiene principalmente fondo, tolva, borde blanco, sombra sin detalle o zona fuera del material, usa `invalido`.
- No omitas celdas.
- No agregues IDs que no esten en `cells_template.csv`.
- Cada fila debe corresponder a una celda valida de un frame.
- Cada combinacion `frame_folder,cell_id` debe aparecer exactamente una vez.
- `frame_folder` debe ser exactamente el nombre de la carpeta del frame dentro de `frames_a_etiquetar`.
- `image_name` debe ser el nombre original indicado en `metadata.json`.
- `cell_id` debe conservar ceros a la izquierda, por ejemplo `001`, `002`, `003`.
- Si `notes` contiene comas, encierra el texto entre comillas dobles.

## Formato de respuesta

Formato ideal: adjunta un archivo CSV llamado `labels_batch001_partXXX.csv`.

Si no puedes adjuntar archivo, pega una unica tabla CSV en un bloque de codigo.
No mezcles texto explicativo dentro del CSV.

Ejemplo:

```csv
frame_folder,image_name,cell_id,label,confidence,notes
45109_2025-10-17_18-59-30.268895+00_8_0,45109_2025-10-17_18-59-30.268895+00_8_0.jpg,001,medio,alta,textura intermedia dominante
45109_2025-10-17_18-59-30.268895+00_8_0,45109_2025-10-17_18-59-30.268895+00_8_0.jpg,002,grueso,media,rocas grandes visibles
45109_2025-10-17_18-59-30.268895+00_8_0,45109_2025-10-17_18-59-30.268895+00_8_0.jpg,003,invalido,alta,borde de mascara/fondo
```

No incluyas explicaciones largas fuera del CSV. Si necesitas advertir un
problema, agregalo en `notes` de las filas afectadas.
