from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import (
    WD_ALIGN_PARAGRAPH,
    WD_BREAK,
    WD_COLOR_INDEX,
    WD_LINE_SPACING,
    WD_TAB_ALIGNMENT,
)
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "informe_final" / "Capitulo_Granulometria_Informe_Final.docx"
ASSET_DIR = ROOT / "docs" / "informe_final" / "assets"
FIGURE_ASSETS = [
    ASSET_DIR / "fig01_salida_conceptual.jpg",
    ASSET_DIR / "fig02_comparacion_metodos.png",
    ASSET_DIR / "fig03_pipeline.png",
    ASSET_DIR / "fig04_segmentacion.jpg",
    ASSET_DIR / "fig05_normalizacion.jpg",
    ASSET_DIR / "fig06_grid_comparison.jpg",
    ASSET_DIR / "fig07_referencias.jpg",
    ASSET_DIR / "fig08_grilla_csv.jpg",
    ASSET_DIR / "fig09_revision_ui.jpg",
    ASSET_DIR / "fig10_dexined.jpg",
    ASSET_DIR / "fig11_resnet_schematic.png",
    ASSET_DIR / "fig12_training_curves.png",
    ASSET_DIR / "fig13_confusion_matrix.png",
    ASSET_DIR / "fig14_frame_overlay.jpg",
    ASSET_DIR / "fig15_webapp.png",
    ASSET_DIR / "fig16_prueba_tecnica.png",
    ASSET_DIR / "fig17_coverage_accuracy.png",
    ASSET_DIR / "fig18_errores.jpg",
]
FIGURE_ASSET_INDEX = 0

FONT = "Times New Roman"
INK = "000000"
ACCENT = "1F4E79"
ACCENT_DARK = "17365D"
HEADER_FILL = "D9EAF7"
SUBHEADER_FILL = "EAF2F8"
LIGHT_FILL = "F4F6F7"
GRID = "A6A6A6"
PLACEHOLDER_RED = RGBColor(192, 0, 0)
CONTENT_WIDTH_DXA = 9684  # Letter with 2.5 cm left and 2.0 cm right margins.


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=90, start=80, bottom=90, end=80) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_cell_no_wrap(cell) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    if tc_pr.find(qn("w:noWrap")) is None:
        tc_pr.append(OxmlElement("w:noWrap"))


def set_table_borders(table, color=GRID, size="6") -> None:
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:space"), "0")
        tag.set(qn("w:color"), color)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_table_geometry(table, widths_dxa: list[int]) -> None:
    if sum(widths_dxa) != CONTENT_WIDTH_DXA:
        raise ValueError(f"Table widths must sum to {CONTENT_WIDTH_DXA}, got {sum(widths_dxa)}")
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(CONTENT_WIDTH_DXA))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "0")
    tbl_ind.set(qn("w:type"), "dxa")

    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)

    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            set_cell_width(cell, widths_dxa[idx])
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def prevent_row_split(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def style_cell_text(cell, bold=False, color=INK, size=9.5, align=WD_ALIGN_PARAGRAPH.LEFT) -> None:
    for paragraph in cell.paragraphs:
        paragraph.alignment = align
        paragraph.paragraph_format.space_before = Pt(0)
        paragraph.paragraph_format.space_after = Pt(0)
        paragraph.paragraph_format.line_spacing = 1.05
        for run in paragraph.runs:
            run.font.name = FONT
            run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = RGBColor.from_string(color)


def unnumber_caption(caption: str) -> str:
    if caption.startswith(("Tabla 6.", "Figura 6.")):
        return caption.split(". ", 1)[1]
    return caption


def add_table(doc: Document, caption: str, headers: list[str], rows: list[list[str]], widths: list[int],
              center_cols: set[int] | None = None, font_size: float = 9.5,
              no_wrap_cols: set[int] | None = None) -> None:
    cap = doc.add_paragraph(style="Caption")
    forced_breaks = ("Tabla 6.1.", "Tabla 6.2.", "Tabla 6.4.", "Tabla 6.8.",
                     "Tabla 6.9.", "Tabla 6.13.", "Tabla 6.15.", "Tabla 6.18.")
    cap.paragraph_format.page_break_before = caption.startswith(forced_breaks)
    cap.paragraph_format.keep_with_next = True
    cap.add_run(unnumber_caption(caption))
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    set_table_borders(table)
    set_repeat_table_header(table.rows[0])
    for idx, header in enumerate(headers):
        table.rows[0].cells[idx].text = header
        set_cell_shading(table.rows[0].cells[idx], HEADER_FILL)
        style_cell_text(table.rows[0].cells[idx], bold=True, color=ACCENT_DARK, size=font_size,
                        align=WD_ALIGN_PARAGRAPH.CENTER)
    for row_idx, values in enumerate(rows):
        cells = table.add_row().cells
        prevent_row_split(table.rows[-1])
        if row_idx % 2 == 1:
            for cell in cells:
                set_cell_shading(cell, LIGHT_FILL)
        for col_idx, value in enumerate(values):
            cells[col_idx].text = str(value)
            if no_wrap_cols and col_idx in no_wrap_cols:
                set_cell_no_wrap(cells[col_idx])
            align = WD_ALIGN_PARAGRAPH.CENTER if center_cols and col_idx in center_cols else WD_ALIGN_PARAGRAPH.LEFT
            style_cell_text(cells[col_idx], size=font_size, align=align)
    # Rows are appended after the initial grid is defined, so widths and cell
    # margins must be materialized again for every body cell.
    set_table_geometry(table, widths)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_placeholder(doc: Document, description: str, caption: str) -> None:
    global FIGURE_ASSET_INDEX
    asset = FIGURE_ASSETS[FIGURE_ASSET_INDEX] if FIGURE_ASSET_INDEX < len(FIGURE_ASSETS) else None
    FIGURE_ASSET_INDEX += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    if asset is not None and asset.exists():
        p.add_run().add_picture(str(asset), width=Cm(16.0))
    else:
        run = p.add_run(f"[AQUÍ COLOCAR IMAGEN DE {description.upper()}]")
        run.font.name = FONT
        run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = PLACEHOLDER_RED
        run.font.highlight_color = WD_COLOR_INDEX.YELLOW
    cap = doc.add_paragraph(unnumber_caption(caption), style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_figure(doc: Document, image_path: Path, caption: str, width_cm: float = 16.0) -> None:
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(image_path), width=Cm(width_cm))
    cap = doc.add_paragraph(unnumber_caption(caption), style="Caption")
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER


def add_body(doc: Document, text: str, indent=True, keep=False) -> None:
    p = doc.add_paragraph(style="Normal")
    if not indent:
        p.paragraph_format.first_line_indent = Cm(0)
    p.paragraph_format.keep_together = keep
    p.add_run(text)


def add_definition(doc: Document, term: str, definition: str) -> None:
    p = doc.add_paragraph(style="Definition")
    p.add_run(f"{term}. ").bold = True
    p.add_run(definition)


def add_heading(doc: Document, text: str, level: int) -> None:
    p = doc.add_paragraph(text, style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True


def math_run(text: str, plain: bool = False):
    run = OxmlElement("m:r")
    run_properties = OxmlElement("m:rPr")
    style = OxmlElement("m:sty")
    style.set(qn("m:val"), "p" if plain else "i")
    run_properties.append(style)
    run.append(run_properties)
    value = OxmlElement("m:t")
    value.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    value.text = text
    run.append(value)
    return run


def math_argument(tag: str, *children):
    argument = OxmlElement(tag)
    for child in children:
        argument.append(child)
    return argument


def math_subscript(base: str, subscript: str):
    node = OxmlElement("m:sSub")
    node.append(math_argument("m:e", math_run(base)))
    node.append(math_argument("m:sub", math_run(subscript, plain=True)))
    return node


def math_superscript(children: list, exponent: str):
    node = OxmlElement("m:sSup")
    node.append(math_argument("m:e", *children))
    node.append(math_argument("m:sup", math_run(exponent)))
    return node


def math_fraction(numerator: list, denominator: list):
    node = OxmlElement("m:f")
    node.append(math_argument("m:num", *numerator))
    node.append(math_argument("m:den", *denominator))
    return node


def math_radical(children: list):
    node = OxmlElement("m:rad")
    properties = OxmlElement("m:radPr")
    degree_hidden = OxmlElement("m:degHide")
    degree_hidden.set(qn("m:val"), "1")
    properties.append(degree_hidden)
    node.append(properties)
    node.append(OxmlElement("m:deg"))
    node.append(math_argument("m:e", *children))
    return node


def build_equation(number: str):
    equation = OxmlElement("m:oMath")

    if number == "(6.1)":
        numerator = [
            math_subscript("F1", "fino"),
            math_run(" + ", plain=True),
            math_subscript("F1", "medio"),
            math_run(" + ", plain=True),
            math_subscript("F1", "grueso"),
        ]
        parts = [
            math_subscript("F1", "macro"),
            math_run(" = ", plain=True),
            math_fraction(numerator, [math_run("3")]),
        ]
    elif number == "(6.2)":
        parts = [
            math_subscript("p", "c"),
            math_run(" = ", plain=True),
            math_fraction([math_subscript("N", "c")], [math_subscript("N", "válidos")]),
            math_run(",   c ∈ {fino, medio, grueso}", plain=True),
        ]
    elif number == "(6.3)":
        parts = [
            math_run("IG"),
            math_run(" = ", plain=True),
            math_subscript("p", "fino"),
            math_run(" + 2", plain=True),
            math_subscript("p", "medio"),
            math_run(" + 3", plain=True),
            math_subscript("p", "grueso"),
        ]
    elif number == "(6.4)":
        radicand = [
            math_subscript("p", "fino"),
            math_superscript([math_run("(1 − IG)", plain=True)], "2"),
            math_run(" + ", plain=True),
            math_subscript("p", "medio"),
            math_superscript([math_run("(2 − IG)", plain=True)], "2"),
            math_run(" + ", plain=True),
            math_subscript("p", "grueso"),
            math_superscript([math_run("(3 − IG)", plain=True)], "2"),
        ]
        parts = [math_run("D"), math_run(" = ", plain=True), math_radical(radicand)]
    elif number == "(6.5)":
        parts = [
            math_run("margen", plain=True),
            math_run(" = ", plain=True),
            math_subscript("p", "(1)"),
            math_run(" − ", plain=True),
            math_subscript("p", "(2)"),
        ]
    else:
        parts = [math_run("Ecuación", plain=True)]

    for part in parts:
        equation.append(part)
    return equation


def add_equation(doc: Document, text: str, number: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.first_line_indent = Cm(0)
    paragraph.paragraph_format.space_before = Pt(2)
    paragraph.paragraph_format.space_after = Pt(4)
    paragraph.paragraph_format.line_spacing = 1.0
    paragraph.paragraph_format.tab_stops.add_tab_stop(Cm(8.45), WD_TAB_ALIGNMENT.CENTER)
    paragraph.paragraph_format.tab_stops.add_tab_stop(Cm(16.9), WD_TAB_ALIGNMENT.RIGHT)
    paragraph.add_run("\t")
    paragraph._p.append(build_equation(number))
    number_run = paragraph.add_run(f"\t{number}")
    number_run.font.name = FONT
    number_run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    number_run.font.size = Pt(11)


def setup_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Cm(2.5)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    section.header_distance = Cm(1.25)
    section.footer_distance = Cm(1.25)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = FONT
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.first_line_indent = Cm(0.75)
    normal.paragraph_format.widow_control = True

    for level, size, before, after in ((1, 14, 12, 8), (2, 12, 12, 6), (3, 12, 8, 4)):
        style = styles[f"Heading {level}"]
        style.font.name = FONT
        style._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(INK)
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.0
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.keep_together = True
        style.paragraph_format.page_break_before = level == 1
    styles["Heading 3"].font.italic = True

    caption = styles["Caption"]
    caption.font.name = FONT
    caption._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    caption.font.size = Pt(10)
    caption.font.bold = True
    caption.font.italic = False
    caption.font.color.rgb = RGBColor.from_string(INK)
    caption.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(3)
    caption.paragraph_format.space_after = Pt(8)
    caption.paragraph_format.line_spacing = 1.0

    definition = styles.add_style("Definition", WD_STYLE_TYPE.PARAGRAPH)
    definition.base_style = normal
    definition.font.name = FONT
    definition._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    definition.font.size = Pt(12)
    definition.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    definition.paragraph_format.first_line_indent = Cm(0)
    definition.paragraph_format.left_indent = Cm(0.75)
    definition.paragraph_format.space_after = Pt(4)
    definition.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE

    bibliography = styles.add_style("Bibliography", WD_STYLE_TYPE.PARAGRAPH)
    bibliography.base_style = normal
    bibliography.font.name = FONT
    bibliography._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    bibliography.font.size = Pt(10.5)
    bibliography.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    bibliography.paragraph_format.left_indent = Cm(0.75)
    bibliography.paragraph_format.first_line_indent = Cm(-0.75)
    bibliography.paragraph_format.line_spacing = 1.0
    bibliography.paragraph_format.space_after = Pt(5)

    doc.core_properties.title = "Granulometría visual aparente en tolva"
    doc.core_properties.subject = "Capítulo autónomo para informe final"
    doc.core_properties.author = "Sebastián Rosas"
    doc.core_properties.keywords = "granulometría, CAEX, visión computacional, ResNet18, Random Forest"
    return doc


def build() -> Path:
    global FIGURE_ASSET_INDEX
    FIGURE_ASSET_INDEX = 0
    doc = setup_document()

    add_heading(doc, "6 Granulometría visual aparente en tolva", 1)
    add_body(
        doc,
        "Este capítulo presenta el desarrollo, la evaluación y la integración de un analítico de visión computacional orientado a estimar la granulometría visual aparente del material transportado en la tolva de camiones de extracción de alto tonelaje (CAEX). La solución recibe imágenes o secuencias de video, identifica la superficie visible de la carga, normaliza su escala, la divide en regiones locales y clasifica dichas regiones en tres categorías cualitativas: fino, medio y grueso. A partir de estas predicciones se construye una distribución espacial por imagen y un resumen por camión.",
        indent=False,
    )
    add_body(
        doc,
        "La palabra granulometría se utiliza en este capítulo en un sentido visual y operacional. El sistema no reemplaza un ensayo de tamizado ni entrega una distribución física de tamaños expresada en milímetros. Su salida describe la apariencia de la superficie visible de la carga bajo las condiciones de captura de las cámaras disponibles. Esta distinción limita el alcance de las conclusiones, pero permite desarrollar una herramienta de apoyo para seguimiento de tendencias, comparación entre cargas y priorización de casos que requieren revisión.",
    )
    add_placeholder(
        doc,
        "una imagen representativa de un camión CAEX con la carga segmentada y el mapa de clases fino, medio y grueso superpuesto",
        "Figura 6.1. Ejemplo conceptual de la salida del analítico de granulometría visual aparente.",
    )

    add_heading(doc, "6.1 Planteamiento del problema", 2)
    add_body(
        doc,
        "La fragmentación del material influye en operaciones posteriores como transporte, descarga, chancado y conminución. Sin embargo, la inspección visual manual de la carga es subjetiva, difícil de mantener de manera continua y dependiente de la experiencia del observador. Las imágenes obtenidas en terreno agregan variaciones de distancia, perspectiva, iluminación, sombras, polvo, oclusiones y proporción de material visible, por lo que una clasificación directa del frame completo puede aprender señales ajenas al tamaño aparente de las rocas.",
    )
    add_body(
        doc,
        "Los datos considerados provienen de secuencias capturadas en la faena Collahuasi mediante una cámara fija con vista posterior o superior-posterior del camión. Durante una secuencia, el vehículo se aproxima a la cámara y cambia de escala dentro del encuadre; también varían la perspectiva, las sombras y la presencia de polvo. Estas condiciones justifican normalizar la carga antes de extraer parches y evaluar el resultado sobre múltiples frames del mismo camión.",
    )
    add_body(
        doc,
        "El problema se formula como una estimación supervisada de composición superficial. Cada imagen se descompone en parches que conservan una escala relativa estable respecto del ancho de la carga. El modelo asigna una clase a cada parche y el resultado del frame corresponde a la proporción de parches de cada categoría. Este diseño permite representar cargas heterogéneas y evita reducir toda la imagen a una única etiqueta antes de observar su distribución interna.",
    )
    add_body(
        doc,
        "La principal dificultad no reside solamente en escoger un algoritmo, sino en construir etiquetas consistentes. Un parche aislado pierde el contexto necesario para decidir si un fragmento es fino, medio o grueso respecto de la carga completa. Por esta razón, la estrategia de datos evoluciona desde el etiquetado de recortes independientes hacia la clasificación sobre una imagen normalizada con grilla numerada, referencias visuales por clase y una etapa de revisión humana.",
    )

    add_heading(doc, "6.2 Objetivos y alcance", 2)
    add_heading(doc, "6.2.1 Objetivo general", 3)
    add_body(
        doc,
        "Desarrollar y validar un analítico de visión computacional que estime, a partir de imágenes y videos de camiones CAEX, la distribución visual aparente de material fino, medio y grueso sobre la superficie visible de la carga, manteniendo trazabilidad desde el frame de entrada hasta el resultado agregado por camión.",
        indent=False,
    )
    add_heading(doc, "6.2.2 Objetivos específicos", 3)
    add_definition(doc, "OE1", "Estandarizar la región de carga mediante segmentación, recorte y normalización geométrica, de modo que la escala relativa de los parches sea comparable entre frames capturados a distinta distancia.")
    add_definition(doc, "OE2", "Diseñar un procedimiento reproducible de generación, etiquetado y revisión de parches que preserve el vínculo con el frame original y evite contaminación entre entrenamiento, validación y prueba.")
    add_definition(doc, "OE3", "Implementar y comparar métodos basados en características de bordes y textura con un modelo convolucional de aprendizaje por transferencia.")
    add_definition(doc, "OE4", "Integrar el clasificador seleccionado con la segmentación de carga y producir resultados interpretables por frame y por video, incluyendo mapas espaciales, proporciones y medidas de confianza.")
    add_definition(doc, "OE5", "Evaluar el comportamiento del sistema en un conjunto técnico independiente, identificar sus errores dominantes y establecer las condiciones necesarias para avanzar hacia una validación operacional.")

    add_body(
        doc,
        "El alcance funcional comprende la segmentación de la superficie visible de la carga, la clasificación local en fino, medio y grueso, el cálculo de proporciones por frame y la agregación de múltiples observaciones de un mismo camión. El pipeline también conserva evidencia visual y archivos de resultados que permiten revisar el origen de cada estimación.",
    )
    add_body(
        doc,
        "Quedan fuera del alcance actual la medición de tamaños físicos en milímetros, la estimación del material oculto, el cálculo de masa o volumen por clase y la toma automática de decisiones de proceso. Tampoco se asume que el modelo pueda generalizar sin validación adicional a cámaras, faenas o condiciones ambientales no representadas en los datos utilizados.",
    )

    add_heading(doc, "6.3 Estado del arte y evaluación de alternativas", 2)
    add_body(
        doc,
        "El análisis de fragmentación a partir de imágenes se aborda mediante técnicas de realce de bordes, segmentación por regiones, redes de segmentación de instancias y, más recientemente, fusión de imágenes con información tridimensional. DexiNed aprende bordes multiescala y se utiliza como etapa previa a operaciones morfológicas o watershed [1], [5]. Estas técnicas son atractivas cuando el contorno de cada fragmento se observa con claridad, pero su rendimiento disminuye ante apilamiento, adherencia, sombras y textura interna, condiciones que producen bordes falsos o contornos incompletos [3].",
    )
    add_body(
        doc,
        "El trabajo de Li et al. [1], utilizado como referencia inicial, combina escala de grises, CLAHE, DexiNed y procesamiento morfológico para distinguir material fino, rocas conectadas y rocas individuales, y luego obtiene una distribución a partir de áreas de contornos que contrasta con harneo de laboratorio. Esa metodología opera sobre muestras más controladas. En las imágenes de tolva, la superposición, el movimiento, el polvo y las sombras impiden asumir que cada contorno cerrado corresponde a una roca, por lo que el enfoque se evalúa de manera exploratoria y no se adopta directamente como medición física.",
    )
    add_body(
        doc,
        "Los métodos supervisados permiten aprender representaciones directamente desde imágenes. Se han propuesto redes profundas para estimar fragmentación [2], segmentación con Mask R-CNN [4] y arquitecturas especializadas para separar partículas adheridas [9]. Los enfoques multimodales incorporan nubes de puntos, visión estéreo o fusión de modalidades para recuperar escala y geometría [6]-[8]. Estas alternativas pueden acercarse a una medición física, aunque requieren sensores adicionales, calibración geométrica o anotaciones de instancias que no están disponibles en la etapa actual del proyecto.",
    )
    add_body(
        doc,
        "Con los datos existentes se comparan dos familias. La primera utiliza características explícitas de intensidad, textura y bordes junto con Random Forest, algoritmo que combina múltiples árboles de decisión y reduce la varianza mediante agregación [10]. La segunda utiliza una red residual ResNet18 [11] con aprendizaje por transferencia, capaz de ajustar representaciones visuales al dominio minero. La clasificación por parches se selecciona como compromiso entre expresividad, costo de anotación y necesidad de representar mezclas dentro de una misma carga.",
    )
    add_table(
        doc,
        "Tabla 6.2. Comparación de alternativas técnicas consideradas.",
        ["Alternativa", "Fortaleza", "Limitación en este proyecto", "Decisión"],
        [
            ["Bordes DexiNed + morfología", "Entrega contornos interpretables y no exige etiquetar cada roca.", "La textura, las sombras y el contacto entre fragmentos generan bordes gruesos, falsos o incompletos.", "Se conserva como línea base experimental."],
            ["Características OpenCV + Random Forest", "Entrenamiento rápido, buena línea base y mayor interpretabilidad de variables.", "Depende de descriptores diseñados manualmente y presenta menor capacidad de adaptación visual.", "Se implementa como Baseline 1."],
            ["DexiNed + Random Forest/SVM", "Combina información de borde con un clasificador supervisado.", "El mapa de bordes no separa de forma estable las rocas en terreno.", "Queda definido como Baseline 2 pendiente de comparación sistemática."],
            ["CNN RGB por parches", "Aprende color, textura, forma y contexto local mediante transferencia.", "Requiere etiquetas consistentes y puede aprender sesgos de iluminación o cámara.", "Se implementa y selecciona como modelo actual."],
            ["Segmentación de instancias", "Podría contar y medir fragmentos individualmente.", "Demanda máscaras por roca, difíciles de obtener en material apilado.", "Se reserva para una etapa posterior."],
            ["Visión estéreo o fusión 2D-3D", "Permite recuperar escala y geometría física.", "Requiere sensores, sincronización y calibración no disponibles.", "Se propone como evolución para medición física."],
        ],
        [1900, 2450, 3300, 2034],
        font_size=8.8,
    )
    add_placeholder(
        doc,
        "un esquema comparativo de las alternativas: bordes, características OpenCV, CNN RGB, segmentación de instancias y fusión 2D-3D",
        "Figura 6.2. Familias de métodos consideradas para el análisis granulométrico visual.",
    )

    add_heading(doc, "6.4 Diseño de la solución", 2)
    add_body(
        doc,
        "La arquitectura separa la percepción de la carga de la estimación granulométrica. Un modelo YOLO de segmentación identifica la clase material; luego se forma la unión de sus máscaras y se descartan las regiones ajenas a la carga. Sobre esta superficie se ejecutan la normalización, la extracción de parches y la clasificación. Esta separación permite sustituir el clasificador granulométrico sin modificar el detector de carga, así como reutilizar la misma máscara para otros analíticos.",
    )
    add_heading(doc, "6.4.1 Arquitectura y secuencia de ejecución", 3)
    add_body(
        doc,
        "La arquitectura de integración separa el procesamiento visual de la interacción con el usuario. El video se convierte en frames, la carga se segmenta con YOLO, se normaliza y se divide en parches antes de aplicar ResNet18 y la agregación. La aplicación web local permite cargar el video, configurar y ejecutar la corrida, revisar resultados y descargar los artefactos generados. Tanto el pipeline como la aplicación escriben sobre una salida común que conserva CSV, máscaras, overlays y el resumen por video, lo que facilita la revisión técnica y la trazabilidad.",
    )
    add_placeholder(
        doc,
        "la arquitectura completa: video, extracción de frames, segmentación YOLO, normalización, grilla de parches, ResNet18, agregación, aplicación web, resultados e indicadores",
        "Arquitectura general del sistema de granulometría visual aparente.",
    )
    add_body(
        doc,
        "Durante una corrida, la aplicación web actúa como interfaz y delega la ejecución al orquestador local. Este componente extrae los frames, solicita a YOLO la segmentación de la carga y entrega las máscaras al analítico granulométrico. El analítico normaliza las cargas, genera los parches, ejecuta ResNet18 y agrega las predicciones. Finalmente, los resultados se escriben como archivos auditables y vuelven a la aplicación para su visualización, revisión y descarga.",
    )
    add_figure(
        doc,
        ASSET_DIR / "fig20_secuencia_ejecucion.png",
        "Secuencia de ejecución de una corrida completa desde la carga del video hasta la revisión de resultados.",
    )

    add_heading(doc, "6.4.2 Segmentación de la carga", 3)
    add_body(
        doc,
        "El modelo de percepción corresponde a haul_truck_segment_m_736_low_data_aug_07012025.pt. La inferencia se ejecuta con tamaño de entrada de 736 píxeles, confianza mínima de 0,25 e IoU de 0,70. El modelo distingue el camión y el material; para granulometría se utiliza la unión de las instancias de la clase material. El resultado define la única región en la que se permiten parches válidos.",
    )
    add_body(
        doc,
        "La calidad de esta máscara condiciona todo el pipeline. Una omisión elimina material que debería analizarse, mientras que una extensión sobre la tolva o el fondo introduce texturas que el clasificador puede confundir con roca. Por este motivo se conservan la máscara y la superposición de YOLO como evidencia visual de cada inferencia.",
    )
    add_body(
        doc,
        "La región válida se define como la superficie de material visible dentro de la tolva. Deben quedar fuera las paredes y bordes metálicos, ruedas, chasis, fondo de la escena, marcas de tiempo y sectores dominados por oclusión o polvo extremo. Los parches cercanos al límite solo se aceptan cuando cumplen la cobertura mínima de material; un frame cuya máscara sea insuficiente o claramente contaminada debe tratarse como no evaluable.",
    )
    add_placeholder(
        doc,
        "una secuencia de tres imágenes: frame original, máscara de material y recorte en fondo negro",
        "Figura 6.4. Segmentación y aislamiento de la superficie visible de la carga.",
    )

    add_heading(doc, "6.4.3 Normalización geométrica", 3)
    add_body(
        doc,
        "Los camiones se desplazan desde posiciones lejanas hacia la cámara, de modo que un parche definido directamente en coordenadas del frame representa áreas físicas diferentes. En los videos analizados, el material cubre prácticamente todo el ancho interior de la tolva; por ello se utiliza el ancho visible de la carga como referencia de escala. El rectángulo envolvente de la máscara se recorta y se redimensiona a 896 píxeles de ancho, manteniendo la relación de aspecto. Los píxeles fuera de la máscara se mantienen negros.",
    )
    add_body(
        doc,
        "La normalización no corrige por completo la perspectiva ni convierte píxeles en unidades físicas, pero reduce la variación dominante asociada a la distancia del camión. La elección de 896 píxeles permite distribuir parches de 96 por 96 píxeles a lo ancho de la carga, manteniendo contexto local suficiente para observar varios fragmentos dentro de cada ventana.",
    )
    add_placeholder(
        doc,
        "dos cargas del mismo camión a distinta distancia antes y después de normalizar el ancho a 896 píxeles",
        "Figura 6.5. Ilustración del efecto de la normalización por ancho de carga.",
    )

    add_heading(doc, "6.4.4 Extracción de parches", 3)
    add_body(
        doc,
        "La imagen normalizada se recorre mediante ventanas RGB de 96 por 96 píxeles con stride 96, por lo que los parches no se superponen. Cada registro conserva el nombre del frame, las coordenadas y la fila y columna de la grilla. Para los conjuntos de entrenamiento y para la prueba técnica se exige que al menos el 80 % del parche pertenezca a la máscara de material. Este criterio reduce la contaminación por fondo, bordes de la tolva y sectores vacíos.",
    )
    add_body(
        doc,
        "Durante el desarrollo se prueban ventanas de 128 y 64 píxeles. La ventana de 128 entrega mayor contexto, pero reduce la resolución espacial y mezcla con frecuencia materiales visualmente distintos. La ventana de 64 aumenta la resolución espacial, aunque un fragmento puede ocupar una fracción excesiva del parche y perderse el contexto relativo de la carga. La ventana de 96 ofrece el equilibrio utilizado en los experimentos finales.",
    )
    add_table(
        doc,
        "Tabla 6.3. Parámetros geométricos de la representación final.",
        ["Parámetro", "Valor", "Fundamento"],
        [
            ["Ancho normalizado", "896 px", "Reduce la variación de escala causada por la distancia a la cámara."],
            ["Tamaño de parche", "96 x 96 px", "Equilibrio entre detalle local y contexto de varios fragmentos."],
            ["Stride", "96 px", "Evita solapamiento y facilita interpretar porcentajes por conteo."],
            ["Cobertura mínima de material", "80 %", "Excluye parches dominados por fondo o estructura de la tolva."],
            ["Canales", "RGB", "Conserva información de color, iluminación y textura útil para transferencia."],
            ["Trazabilidad", "Frame, fila, columna y coordenadas", "Permite reconstruir la grilla y auditar cada predicción."],
        ],
        [2450, 1700, 5534],
        center_cols={1},
        font_size=9.2,
    )
    add_placeholder(
        doc,
        "la misma carga con grillas de 64, 96 y 128 píxeles para comparar la escala visual de los parches",
        "Figura 6.6. Comparación de tamaños de parche evaluados durante el desarrollo.",
    )

    add_heading(doc, "6.5 Construcción y evolución del conjunto de datos", 2)
    add_heading(doc, "6.5.1 Definición de clases", 3)
    add_body(
        doc,
        "Las clases fino, medio y grueso se definen mediante referencias visuales representativas del dominio. La etiqueta describe la granulometría dominante observada dentro del parche y se interpreta en relación con la escala normalizada de la carga. No se asocia a umbrales físicos de tamiz. Los parches mixtos, ambiguos o inválidos se mantienen disponibles para auditoría, pero se excluyen del entrenamiento final de tres clases.",
    )
    add_body(
        doc,
        "Cada etiqueta incluye un nivel de confianza alto, medio o bajo. La confianza funciona como criterio de selección de datos: los experimentos finales utilizan etiquetas de confianza alta y media. No se emplea como un peso continuo en la función de pérdida. Esta decisión evita introducir una interpretación probabilística que el procedimiento de anotación todavía no valida.",
    )
    add_placeholder(
        doc,
        "tres ejemplos de referencia, uno de carga fina, uno de carga media y uno de carga gruesa, todos segmentados y a escala comparable",
        "Figura 6.7. Referencias visuales utilizadas para definir las tres clases.",
    )

    add_heading(doc, "6.5.2 Estrategia de etiquetado", 3)
    add_body(
        doc,
        "La primera estrategia presenta cada parche como una imagen aislada. Este formato facilita la clasificación individual, pero elimina la relación con el resto de la carga y aumenta las confusiones entre medio y grueso. La estrategia final muestra la carga completa, ya segmentada y normalizada, sobre una grilla numerada. El anotador asigna una clase a cada celda usando las referencias visuales y entrega un archivo CSV que conserva identificadores, posición, clase, confianza y observaciones.",
    )
    add_body(
        doc,
        "El etiquetado se asiste mediante un modelo de inteligencia artificial y se somete posteriormente a revisión humana en una aplicación Streamlit. La interfaz presenta simultáneamente el parche y su frame original, destaca la celda activa, permite filtrar por clase y confianza y guarda las correcciones directamente en labels.csv. Esta etapa resulta crítica: los experimentos muestran que mejorar la consistencia de las etiquetas produce una ganancia mayor que agregar datos sin revisión.",
    )
    add_placeholder(
        doc,
        "una carga normalizada con grilla numerada y ejemplos de las columnas del CSV de etiquetado",
        "Figura 6.8. Unidad de etiquetado con contexto completo y salida tabular trazable.",
    )
    add_placeholder(
        doc,
        "la vista de revisión de la aplicación Streamlit mostrando parche, imagen original, clase, confianza y botones de corrección",
        "Figura 6.9. Aplicación de revisión humana de etiquetas.",
    )

    add_heading(doc, "6.5.3 Iteraciones del conjunto de datos", 3)
    add_body(
        doc,
        "El conjunto de datos se construye en iteraciones para evaluar el efecto de la escala, el volumen de datos y la calidad de anotación. Los Batches 001 y 002 amplían la diversidad, mientras que el Batch 003 se utiliza para medir explícitamente el efecto de la revisión humana. Los datos finales del Batch 003 contienen 2455 registros; después de excluir 5 parches inválidos y 22 etiquetas de confianza baja, quedan 2428 parches útiles de 90 frames.",
    )
    add_table(
        doc,
        "Tabla 6.4. Evolución de los conjuntos de parches utilizados.",
        ["Conjunto", "N.º frames", "Parches útiles", "Distribución F/M/G", "Propósito"],
        [
            ["Prueba inicial 128", "33", "181", "- / - / -", "Validar el flujo de características y Random Forest."],
            ["Batch 001", "50", "1159", "104 / 694 / 361", "Primera iteración de grilla normalizada de 96 px."],
            ["Batch 002", "90", "2305", "304 / 1160 / 841", "Ampliar diversidad y combinar cargas de distintas clases."],
            ["Batch 001 + 002", "140", "3464", "408 / 1854 / 1202", "Comparar el efecto de aumentar datos sin revisión completa común."],
            ["Batch 003 revisado", "90", "2428", "704 / 903 / 821", "Evaluar el efecto de etiquetas corregidas y mejor balanceadas."],
        ],
        [1700, 1100, 1300, 1900, 3684],
        center_cols={1, 2, 3},
        font_size=8.8,
        no_wrap_cols={1, 2, 3},
    )
    add_body(
        doc,
        "La distribución final no se fuerza a ser perfectamente uniforme. Se conserva una prevalencia moderada de la clase medio, compatible con la variabilidad observada en las cargas, y se compensan las diferencias dentro de los modelos mediante ponderación de clases. El objetivo no es replicar artificialmente una frecuencia operacional específica, sino evitar que una clase quede subrepresentada hasta el punto de impedir su aprendizaje.",
    )

    add_heading(doc, "6.5.4 Separación por frame", 3)
    add_body(
        doc,
        "Los parches de un mismo frame comparten iluminación, perspectiva, textura y composición. Si se distribuyen aleatoriamente entre entrenamiento y prueba, el modelo puede reconocer características del frame en lugar de generalizar a una carga nueva. Para evitar esta fuga de información, la separación se realiza por frame de origen: ningún frame aporta parches a más de una partición.",
    )
    add_table(
        doc,
        "Tabla 6.5. Partición final del Batch 003 revisado.",
        ["Partición", "Frames", "Parches", "Fino", "Medio", "Grueso"],
        [
            ["Entrenamiento", "63", "1706", "485", "641", "580"],
            ["Validación", "13", "365", "113", "131", "121"],
            ["Prueba interna", "14", "357", "106", "131", "120"],
            ["Total", "90", "2428", "704", "903", "821"],
        ],
        [2300, 1500, 1500, 1461, 1461, 1462],
        center_cols={1, 2, 3, 4, 5},
        font_size=9.4,
    )
    add_body(
        doc,
        "La separación por frame evita que los parches de una misma imagen se repartan entre conjuntos. Sin embargo, frames próximos de un mismo video todavía pueden ser altamente correlacionados. Por ello, una validación operacional posterior debe dividir los datos por video, secuencia o camión completo, de manera que ninguna observación temporalmente cercana aparezca en más de una partición.",
    )

    add_heading(doc, "6.6 Métodos implementados", 2)
    add_heading(doc, "6.6.1 Exploración con DexiNed", 3)
    add_body(
        doc,
        "DexiNed se implementa mediante el modelo disponible en Kornia como extractor de bordes preentrenado. La entrada se convierte a escala de grises, se realza con CLAHE y se normaliza antes de la inferencia. La salida continua se transforma con sigmoide y contraste por percentiles; luego se aplica un umbral por percentil, máscara de fondo, erosión de primer plano y apertura morfológica. El pipeline conserva mapas continuos, bordes binarios y superposiciones para comparar configuraciones.",
    )
    add_table(
        doc,
        "Tabla 6.6. Configuración representativa de DexiNed evaluada.",
        ["Etapa", "Configuración"],
        [
            ["Preprocesamiento", "Escala de grises + CLAHE, clip limit 2,0 y grilla 8 x 8."],
            ["Normalización", "BGR/255 con media del modelo preentrenado."],
            ["Salida", "Sigmoide y contraste entre percentiles 1 y 99."],
            ["Binarización", "Percentil 75."],
            ["Máscara", "Fondo negro, umbral de primer plano 8 y erosión de 12 píxeles."],
            ["Morfología", "Una iteración de apertura; sin cierre ni adelgazamiento en la ejecución representativa."],
            ["Región de interés", "Fracción relativa 0,12; 0,08; 0,88; 0,78."],
        ],
        [2800, 6884],
        font_size=9.2,
    )
    add_body(
        doc,
        "El método identifica con claridad algunos contornos de gran contraste, pero también responde a vetas, rugosidad, sombras y cambios de iluminación dentro de una misma roca. Los bordes resultan gruesos y con frecuencia no cierran objetos; en zonas con finos se obtiene una alta densidad de respuestas que no equivale a fragmentos individuales. Por lo tanto, DexiNed no se adopta como estimador final de tamaño. Se conserva como fuente potencial de características complementarias y como línea base para la comparación futura RGB + bordes.",
    )
    add_placeholder(
        doc,
        "un ejemplo de entrada, mapa continuo DexiNed, borde binario y superposición que muestre bordes internos y contornos incompletos",
        "Figura 6.10. Resultado representativo y limitaciones observadas con DexiNed.",
    )

    add_heading(doc, "6.6.2 Baseline de características OpenCV y Random Forest", 3)
    add_body(
        doc,
        "El primer clasificador supervisado utiliza descriptores explícitos calculados sobre cada parche. Se extraen estadísticas de intensidad después de ecualización local, magnitud y densidad de gradientes Sobel, densidad de Canny para distintos umbrales, cantidad y geometría de contornos, componentes conectados, histograma LBP de 16 intervalos, respuestas Gabor en varias orientaciones y coeficientes de baja frecuencia de la transformada discreta del coseno. Estas variables resumen contraste, rugosidad, frecuencia espacial y estructura de bordes.",
    )
    add_body(
        doc,
        "Random Forest se entrena con 500 árboles y ponderación balanced_subsample. Se comparan cuatro candidatos mediante F1 macro de validación. El modelo final selecciona max_features=sqrt y min_samples_leaf=3; posteriormente se reajusta con las particiones de entrenamiento y validación antes de evaluar una sola vez sobre prueba. Esta línea base establece cuánto puede obtenerse con características diseñadas y sirve como referencia para justificar el uso de una CNN.",
    )
    add_table(
        doc,
        "Tabla 6.7. Candidatos de Random Forest en el Batch 003 revisado.",
        ["Candidato", "Profundidad", "Mínimo por hoja", "Variables por división", "F1 macro validación"],
        [
            ["rf_leaf1_sqrt", "Sin límite", "1", "sqrt", "0,662"],
            ["rf_leaf2_sqrt", "Sin límite", "2", "sqrt", "0,662"],
            ["rf_leaf3_sqrt", "Sin límite", "3", "sqrt", "0,670"],
            ["rf_depth12_leaf2", "12", "2", "0,5", "0,669"],
        ],
        [2100, 1600, 1800, 2200, 1984],
        center_cols={1, 2, 3, 4},
        font_size=9.2,
    )

    add_heading(doc, "6.6.3 CNN RGB con ResNet18", 3)
    add_body(
        doc,
        "El segundo clasificador utiliza ResNet18 preentrenada en ImageNet. La última capa se reemplaza por un clasificador de tres salidas. El entrenamiento se divide en dos fases: durante ocho épocas se congela el backbone y se ajusta solamente la cabeza con tasa de aprendizaje 10^-3; luego se habilitan todos los parámetros y se realiza un fine-tuning de ocho épocas con tasa 10^-5. El mejor checkpoint se selecciona por F1 macro de validación, no por la última época.",
    )
    add_body(
        doc,
        "Los parches RGB se escalan a 96 por 96 píxeles y se normalizan con media [0,485; 0,456; 0,406] y desviación estándar [0,229; 0,224; 0,225], correspondientes al preentrenamiento. En entrenamiento se aplican volteos horizontal y vertical con probabilidad 0,5 y perturbaciones moderadas de brillo, contraste y saturación. Se utiliza entropía cruzada con pesos derivados de las clases, optimizador AdamW, weight decay 10^-4 y semilla 42. La arquitectura contiene aproximadamente 11,18 millones de parámetros.",
    )
    add_table(
        doc,
        "Tabla 6.8. Parámetros principales del modelo convolucional final.",
        ["Componente", "Configuración"],
        [
            ["Arquitectura", "ResNet18 con pesos ImageNet DEFAULT y cabeza de 3 clases."],
            ["Entrada", "Parche RGB de 96 x 96 píxeles."],
            ["Clases", "Fino, medio y grueso."],
            ["Fase 1", "Backbone congelado, 8 épocas, learning rate 1 x 10^-3."],
            ["Fase 2", "Fine-tuning completo, 8 épocas, learning rate 1 x 10^-5."],
            ["Optimización", "AdamW, weight decay 1 x 10^-4 y entropía cruzada ponderada."],
            ["Aumentación", "Volteos horizontal/vertical y ColorJitter moderado."],
            ["Selección", "Mayor F1 macro de validación; mejor época global: 15."],
            ["Semilla", "42."],
        ],
        [2600, 7084],
        font_size=9.2,
    )
    add_placeholder(
        doc,
        "un diagrama sencillo de ResNet18 con parche RGB de entrada y probabilidades fino, medio y grueso de salida",
        "Figura 6.11. Esquema del clasificador convolucional implementado.",
    )

    add_heading(doc, "6.7 Diseño experimental y métricas", 2)
    add_body(
        doc,
        "El ajuste de hiperparámetros se realiza únicamente con la partición de validación. La partición de prueba interna se mantiene separada hasta seleccionar el candidato. Se reportan exactitud y F1 macro. La exactitud mide la fracción total de predicciones correctas; F1 macro calcula el F1 de cada clase y luego les asigna igual importancia, por lo que es más informativo frente a distribuciones no perfectamente balanceadas.",
    )
    add_equation(doc, "F1_macro = (F1_fino + F1_medio + F1_grueso) / 3", "(6.1)")
    add_body(
        doc,
        "Además se inspeccionan precisión, recall y matrices de confusión por clase. La matriz de validación guía las decisiones de entrenamiento; la matriz de prueba interna estima el desempeño en frames no utilizados para ajustar el modelo. Finalmente, una prueba técnica sobre 90 imágenes adicionales evalúa el comportamiento del pipeline completo, incluyendo segmentación, normalización, clasificación y agregación por frame.",
    )
    add_body(
        doc,
        "La evaluación se organiza en distintos niveles. La validación por parches se utiliza para seleccionar hiperparámetros y la mejor época de entrenamiento. Una vez fijado el modelo, la prueba interna mide su generalización sobre parches pertenecientes a frames separados y permite comparar Random Forest con ResNet18.",
    )
    add_body(
        doc,
        "A nivel de imagen completa, la prueba técnica incorpora todas las etapas del pipeline y compara la clase dominante del resultado con una referencia visual independiente. Finalmente, la demostración sobre video comprueba la extracción temporal, la persistencia de resultados y la agregación de múltiples frames pertenecientes al mismo camión.",
    )

    add_heading(doc, "6.8 Resultados de clasificación por parches", 2)
    add_heading(doc, "6.8.1 Evolución entre iteraciones", 3)
    add_body(
        doc,
        "La comparación se concentra en tres iteraciones representativas, todas construidas con cargas normalizadas y parches de 96 píxeles. El Batch 001 establece el primer punto de referencia; la combinación de Batches 001 y 002 evalúa el efecto de aumentar el volumen de datos; y el Batch 003 revisado incorpora etiquetas corregidas, una distribución más equilibrada y una evaluación completa con Random Forest y ResNet18.",
    )
    add_table(
        doc,
        "Tabla 6.10. Resultados por iteración de datos y modelo.",
        ["Conjunto", "Modelo", "F1 val.", "F1 prueba", "Exactitud prueba"],
        [
            ["Batch 001", "Random Forest", "0,539", "0,654", "0,708"],
            ["Batch 001", "ResNet18 RGB", "0,590", "0,574", "0,602"],
            ["Batch 001 + 002", "Random Forest", "0,631", "0,621", "0,674"],
            ["Batch 001 + 002", "ResNet18 RGB", "0,638", "0,598", "0,624"],
            ["Batch 003 revisado", "Random Forest", "0,670", "0,697", "0,692"],
            ["Batch 003 revisado", "ResNet18 RGB", "0,730", "0,757", "0,754"],
        ],
        [2600, 2200, 1500, 1650, 1734],
        center_cols={2, 3, 4},
        font_size=9.0,
    )
    add_body(
        doc,
        "El aumento de datos del Batch 001 al conjunto 001 + 002 no produce una mejora en la prueba, lo que indica que la cantidad por sí sola no garantiza mayor generalización. El Batch 003 revisado obtiene los mejores resultados: Random Forest alcanza F1 macro 0,697 y ResNet18 llega a 0,757. Esta iteración confirma que la selección de frames, el equilibrio entre clases y la consistencia de las etiquetas son más relevantes que ampliar el dataset sin una revisión equivalente.",
    )

    add_heading(doc, "6.8.2 Resultado final de Random Forest", 3)
    add_body(
        doc,
        "El mejor Random Forest alcanza F1 macro 0,670 en validación y 0,697 en prueba interna, con exactitud de 0,692. La clase medio obtiene el menor F1 (0,633), mientras fino y grueso alcanzan 0,737 y 0,721. El patrón indica que las características manuales capturan extremos de textura, pero representan con menor precisión las transiciones intermedias.",
    )
    add_table(
        doc,
        "Tabla 6.11. Desempeño de Random Forest en prueba interna.",
        ["Clase", "Precisión", "Recall", "F1", "Soporte"],
        [
            ["Fino", "0,721", "0,755", "0,737", "106"],
            ["Medio", "0,599", "0,672", "0,633", "131"],
            ["Grueso", "0,798", "0,658", "0,721", "120"],
            ["Macro", "0,706", "0,695", "0,697", "357"],
        ],
        [2300, 1800, 1800, 1800, 1984],
        center_cols={1, 2, 3, 4},
        font_size=9.4,
    )

    add_heading(doc, "6.8.3 Resultado final de ResNet18", 3)
    add_body(
        doc,
        "ResNet18 alcanza su mejor validación en la época 15, durante el fine-tuning, con F1 macro 0,730 y exactitud 0,732. En prueba interna obtiene F1 macro 0,757 y exactitud 0,754. Supera a Random Forest en 0,060 puntos de F1 macro y 0,062 puntos de exactitud sobre la misma partición. La diferencia respalda la selección de la CNN RGB como modelo actual del pipeline.",
    )
    add_placeholder(
        doc,
        "las curvas de pérdida, exactitud y F1 macro de entrenamiento y validación de ResNet18, marcando la época 15",
        "Figura 6.12. Evolución del entrenamiento del modelo ResNet18 final.",
    )
    add_table(
        doc,
        "Tabla 6.12. Desempeño de ResNet18 en prueba interna.",
        ["Clase", "Precisión", "Recall", "F1", "Soporte"],
        [
            ["Fino", "0,710", "0,877", "0,785", "106"],
            ["Medio", "0,683", "0,641", "0,661", "131"],
            ["Grueso", "0,893", "0,767", "0,825", "120"],
            ["Macro", "0,762", "0,762", "0,757", "357"],
        ],
        [2300, 1800, 1800, 1800, 1984],
        center_cols={1, 2, 3, 4},
        font_size=9.4,
    )
    add_table(
        doc,
        "Tabla 6.13. Matriz de confusión de ResNet18 en prueba interna. Filas: clase real; columnas: predicción.",
        ["Clase real", "Fino", "Medio", "Grueso"],
        [
            ["Fino", "93", "12", "1"],
            ["Medio", "37", "84", "10"],
            ["Grueso", "1", "27", "92"],
        ],
        [3000, 2228, 2228, 2228],
        center_cols={1, 2, 3},
        font_size=9.6,
    )
    add_body(
        doc,
        "La confusión principal ocurre alrededor de la clase medio: 37 parches medios se clasifican como finos y 27 gruesos se clasifican como medios. Solo dos errores atraviesan directamente los extremos fino-grueso. Este comportamiento es coherente con categorías ordinales y con la existencia de transiciones visuales continuas, pero también señala que las referencias de la clase medio requieren mayor precisión.",
    )
    add_placeholder(
        doc,
        "la matriz de confusión de prueba de ResNet18 con valores y escala de color",
        "Figura 6.13. Matriz de confusión del clasificador ResNet18 final.",
    )

    add_heading(doc, "6.9 Inferencia sobre imágenes y videos", 2)
    add_heading(doc, "6.9.1 Salida por frame", 3)
    add_body(
        doc,
        "Para cada parche válido, la CNN entrega tres probabilidades y se asigna la clase de mayor valor. La proporción por conteo se calcula dividiendo la cantidad de parches de cada clase por el total de parches válidos del frame. También se calculan proporciones a partir del promedio de probabilidades. El conteo es la salida principal por su interpretación directa y su correspondencia con la grilla; el promedio de probabilidades se conserva como información complementaria.",
    )
    add_equation(doc, "p_c = N_c / N_válidos,   c ∈ {fino, medio, grueso}", "(6.2)")
    add_body(
        doc,
        "La clase dominante del frame corresponde a la mayor proporción. Para resumir la posición ordinal de la distribución se define un índice visual de granulometría IG, donde 1 representa fino, 2 medio y 3 grueso. El índice es adimensional y solo permite comparar resultados del mismo sistema; no equivale a un diámetro medio.",
    )
    add_equation(doc, "IG = 1·p_fino + 2·p_medio + 3·p_grueso", "(6.3)")
    add_body(
        doc,
        "El índice corresponde a un promedio ponderado: cada nivel ordinal se multiplica por su proporción y los productos se suman. Su rango es de 1 a 3; valores cercanos a 1 indican tendencia fina, cercanos a 2 tendencia media y cercanos a 3 tendencia gruesa. Para distinguir distribuciones con el mismo promedio, se incorpora la dispersión granulométrica visual D, calculada como la desviación estándar ponderada de esos tres niveles.",
    )
    add_equation(
        doc,
        "D = √[p_fino(1−IG)² + p_medio(2−IG)² + p_grueso(3−IG)²]",
        "(6.4)",
    )
    add_body(
        doc,
        "La dispersión toma valores entre 0 y 1. D = 0 significa que todos los parches pertenecen a una sola clase; valores crecientes indican una composición más repartida o heterogénea, y el máximo D = 1 ocurre cuando fino y grueso aportan proporciones iguales sin clase media. IG y D deben interpretarse conjuntamente: el primero indica la tendencia global y el segundo cuánto se distribuye la evidencia alrededor de esa tendencia. Ambos son indicadores visuales derivados del modelo y no sustituyen una curva granulométrica física.",
    )
    add_body(
        doc,
        "La confianza media de parches es el promedio del máximo softmax de cada parche. Esta variable expresa cuán concentrada se encuentra la salida del modelo, pero no debe interpretarse como probabilidad calibrada de que el frame sea correcto. Para una decisión por frame se utiliza además el margen entre la mayor y la segunda proporción; un margen pequeño indica una carga visualmente mixta o una decisión inestable.",
    )
    add_equation(doc, "margen = p_(1) − p_(2)", "(6.5)")
    add_placeholder(
        doc,
        "un frame con máscara blanca, grilla coloreada verde-amarillo-rojo, porcentajes por clase y clase dominante",
        "Figura 6.14. Visualización de la inferencia granulométrica por frame.",
    )
    add_body(
        doc,
        "Para conservar la trazabilidad, cada inferencia guarda la máscara binaria de material, la carga normalizada a 896 píxeles de ancho y dos visualizaciones sobre el frame original: una con la segmentación YOLO y otra con la grilla granulométrica coloreada. Estos archivos permiten comprobar si la región de carga fue detectada correctamente, si la normalización mantiene una escala adecuada y dónde se origina cada clasificación local.",
    )
    add_body(
        doc,
        "También se almacenan los parches RGB con sus coordenadas, un archivo de detalle con la clase y las probabilidades predichas para cada parche, y un resumen por frame con las proporciones, la clase dominante, la confianza, el índice granulométrico y la dispersión. De esta forma, el resultado visual puede relacionarse con los datos cuantitativos utilizados por la aplicación.",
    )

    add_heading(doc, "6.9.2 Agregación temporal por camión", 3)
    add_body(
        doc,
        "La entrada operacional prevista es un video que contiene múltiples observaciones del mismo camión. El pipeline extrae frames a un intervalo configurable, descarta aquellos sin una máscara de material válida y acumula las predicciones de sus parches. La distribución del camión se calcula agrupando todos los parches válidos, en lugar de votar solamente por la clase dominante de cada frame. De este modo, un frame con mayor superficie útil aporta más evidencia, aunque debe vigilarse que no domine por duplicación temporal.",
    )
    add_body(
        doc,
        "La agregación conserva número de frames evaluables, cantidad de parches, proporciones, clase dominante, confianza media, índice visual y dispersión granulométrica. Para una versión operacional se recomienda seleccionar frames suficientemente separados, evitar observaciones redundantes y descartar imágenes borrosas, con poco material visible, polvo extremo o perspectiva desfavorable. Cuando el margen agregado sea bajo o exista alta variabilidad entre frames, el sistema debe marcar el camión para revisión en lugar de forzar una conclusión categórica.",
    )
    add_body(
        doc,
        "En una demostración de integración se procesaron 10,87 segundos de video y se obtuvieron 11 frames válidos, que en conjunto aportaron 511 parches. La distribución agregada fue 33,66 % fino, 38,16 % medio y 28,18 % grueso, con clase dominante medio, confianza media de 67,00 %, índice granulométrico 1,945 y dispersión 0,784. El tiempo de inferencia registrado para la corrida fue de 28,88 segundos.",
    )
    add_body(
        doc,
        "En esta demostración, cinco frames tienen dominante fino, cuatro medio y dos grueso; sin embargo, la agrupación de los 511 parches produce una distribución agregada dominada por medio. El ejemplo confirma que la votación por frame y la composición superficial acumulada responden preguntas distintas. Para el objetivo del proyecto se prioriza la segunda, porque conserva información sobre mezclas y superficie analizada.",
    )
    add_placeholder(
        doc,
        "la vista de la webapp de video con selector de archivo, galería de overlays, resumen por frame y resultado agregado del camión",
        "Figura 6.15. Resumen agregado por video en la aplicación web.",
    )

    add_heading(doc, "6.10 Prueba técnica del pipeline completo", 2)
    add_heading(doc, "6.10.1 Preparación y criterio de referencia", 3)
    add_body(
        doc,
        "La prueba técnica utiliza 90 frames adicionales, organizados en tres grupos balanceados de 30 imágenes cuya carga se selecciona visualmente como mayoritariamente fina, media o gruesa. La carpeta de origen actúa como referencia de clase dominante. No existen coincidencias exactas de nombre con los frames de entrenamiento, validación o prueba interna. Todas las imágenes resultan evaluables por el pipeline.",
    )
    add_body(
        doc,
        "Esta referencia permite evaluar concordancia con una selección visual de la carga completa, pero no constituye verdad de terreno granulométrica. No se dispone de tamizado, fotogrametría calibrada ni medición física asociada a cada frame. En consecuencia, los resultados miden consistencia con el criterio visual definido para el proyecto.",
    )
    add_heading(doc, "6.10.2 Resultados", 3)
    add_body(
        doc,
        "El pipeline clasifica correctamente 69 de 90 imágenes, equivalente a una exactitud de 76,7 %. El F1 macro es 0,763 y la confianza media de los parches es 76,1 %. La clase fino obtiene 26 aciertos, medio 18 y grueso 25. La totalidad de los frames contiene una máscara y una cantidad de parches suficientes para emitir resultado.",
    )
    add_table(
        doc,
        "Tabla 6.16. Resumen de la prueba técnica por clase.",
        ["Clase esperada", "Correctas", "Precisión", "Recall", "F1"],
        [
            ["Fino", "26 / 30", "81,2 %", "86,7 %", "0,839"],
            ["Medio", "18 / 30", "69,2 %", "60,0 %", "0,643"],
            ["Grueso", "25 / 30", "78,1 %", "83,3 %", "0,806"],
            ["Macro / total", "69 / 90", "76,2 %", "76,7 %", "0,763"],
        ],
        [2600, 1900, 1700, 1700, 1784],
        center_cols={1, 2, 3, 4},
        font_size=9.4,
    )
    add_table(
        doc,
        "Tabla 6.17. Matriz de confusión de la prueba técnica. Filas: clase esperada; columnas: predicción.",
        ["Clase esperada", "Fino", "Medio", "Grueso", "No evaluable"],
        [
            ["Fino", "26", "4", "0", "0"],
            ["Medio", "5", "18", "7", "0"],
            ["Grueso", "1", "4", "25", "0"],
        ],
        [2800, 1700, 1700, 1700, 1784],
        center_cols={1, 2, 3, 4},
        font_size=9.5,
    )
    add_placeholder(
        doc,
        "el reporte gráfico actualizado de la prueba técnica con 90 imágenes, matriz de confusión, barras de acierto e indicadores IG y D por clase esperada",
        "Figura 6.16. Reporte consolidado de la prueba técnica del pipeline.",
    )
    add_body(
        doc,
        "La clase medio concentra la mayor dificultad. De sus 30 imágenes, cinco se desplazan hacia fino y siete hacia grueso. De los 21 errores totales, 20 ocurren entre clases adyacentes y solo uno corresponde a una confusión extrema entre grueso y fino. Este patrón sugiere que el modelo representa un continuo visual razonable, aunque la frontera de medio permanece insuficientemente definida.",
    )
    add_table(
        doc,
        "Tabla 6.18. Distribución e indicadores medios según la clase visual esperada.",
        ["Clase esperada", "Fino", "Medio", "Grueso", "IG medio", "D media"],
        [
            ["Fino", "68,96 %", "24,33 %", "6,72 %", "1,378", "0,543"],
            ["Medio", "26,66 %", "44,43 %", "28,91 %", "2,022", "0,681"],
            ["Grueso", "8,57 %", "28,94 %", "62,49 %", "2,539", "0,585"],
        ],
        [2200, 1450, 1450, 1450, 1550, 1584],
        center_cols={1, 2, 3, 4, 5},
        font_size=9.0,
    )
    add_body(
        doc,
        "Las cargas finas y gruesas presentan una concentración media superior al 62 % en su clase esperada. En cambio, las cargas medias distribuyen más de la mitad de la evidencia hacia los extremos. Esto explica su menor recall y confirma que una salida porcentual contiene más información que una única clase dominante. Para uso de monitoreo, una carga media con proporciones cercanas puede describirse como transición, aun cuando el sistema necesite escoger una clase para la métrica.",
    )
    add_body(
        doc,
        "Los valores de IG aumentan de forma ordenada entre las referencias fina, media y gruesa, lo que confirma que el indicador resume correctamente la dirección de la distribución predicha. La clase visual media obtiene la mayor dispersión (0,681), consistente con su composición más repartida y con la mayor confusión observada en la matriz. Estos valores provienen de predicciones del propio modelo y describen su salida; no constituyen una validación independiente de heterogeneidad física. Además, como la prueba contiene 30 imágenes de cada clase, un promedio global de sus indicadores no representa la prevalencia esperada en operación.",
    )

    add_heading(doc, "6.10.3 Política exploratoria de abstención", 3)
    add_body(
        doc,
        "Se analiza el margen entre las dos clases con mayor proporción como regla de abstención. Al exigir márgenes mayores, disminuye la cobertura y aumenta la exactitud de los casos aceptados. Por ejemplo, con margen mínimo 0,20 se emite resultado para 54 de 90 imágenes y la exactitud de ese subconjunto alcanza 92,6 %. Estos valores se calculan sobre la misma prueba técnica y, por lo tanto, son exploratorios; no deben utilizarse como umbral operacional sin un conjunto independiente adicional.",
    )
    add_table(
        doc,
        "Tabla 6.19. Relación exploratoria entre margen, cobertura y exactitud.",
        ["Margen mínimo", "Frames aceptados", "Cobertura", "Exactitud en aceptados"],
        [
            ["0,00", "90", "100,0 %", "76,7 %"],
            ["0,10", "69", "76,7 %", "79,7 %"],
            ["0,15", "58", "64,4 %", "87,9 %"],
            ["0,20", "54", "60,0 %", "92,6 %"],
            ["0,25", "47", "52,2 %", "95,7 %"],
            ["0,30", "40", "44,4 %", "97,5 %"],
        ],
        [2400, 2400, 2300, 2584],
        center_cols={0, 1, 2, 3},
        font_size=9.3,
    )
    add_placeholder(
        doc,
        "un gráfico de cobertura frente a exactitud para los distintos umbrales de margen",
        "Figura 6.17. Compromiso exploratorio entre cobertura y exactitud mediante abstención.",
    )

    add_heading(doc, "6.11 Discusión", 2)
    add_heading(doc, "6.11.1 Interpretación de los resultados", 3)
    add_body(
        doc,
        "La comparación controlada indica que ResNet18 ofrece la mejor representación para los parches RGB disponibles. Su ventaja sobre Random Forest aparece después de corregir las etiquetas, lo que demuestra que la mayor capacidad de la CNN solo se aprovecha cuando la supervisión es consistente. Antes de esa revisión, Random Forest iguala o supera a la CNN en algunas iteraciones, porque un modelo menos flexible puede ser más estable frente a datos escasos o ruidosos.",
    )
    add_body(
        doc,
        "La exactitud de 76,7 % en la prueba técnica es coherente con el F1 macro de 0,757 obtenido por parches, aunque las métricas no son directamente equivalentes. La primera incluye todo el pipeline y compara una clase dominante por frame; la segunda evalúa parches pertenecientes a frames separados del Batch 003. La similitud entre ambos resultados entrega evidencia favorable, pero no reemplaza una validación con referencias físicas ni una evaluación por videos completos de distintos periodos.",
    )
    add_body(
        doc,
        "El error de la clase medio se explica por tres factores. Primero, la granulometría visual forma un continuo y no presenta fronteras naturales entre categorías. Segundo, una misma celda puede contener fragmentos de escalas diferentes. Tercero, las variaciones de perspectiva e iluminación alteran la apariencia de textura. Por ello, el sistema debe exponer proporciones y márgenes, no solo una etiqueta categórica.",
    )
    add_body(
        doc,
        "El índice y la dispersión permiten resumir este continuo sin perder completamente la composición. Dos cargas pueden compartir un IG cercano a 2, pero representar situaciones distintas: una concentrada en parches medios presenta baja dispersión, mientras una mezcla de finos y gruesos presenta dispersión alta. Esta diferencia es relevante para comparar camiones y para identificar resultados que requieren inspección, siempre que se mantenga explícito su carácter visual y ordinal.",
    )

    add_heading(doc, "6.11.2 Validez y limitaciones", 3)
    add_body(
        doc,
        "La normalización por ancho mejora la consistencia de escala bajo el supuesto de que la carga alcanza los bordes de la tolva. Si esta condición no se cumple, el material puede ampliarse artificialmente y sesgar la clasificación. La corrección tampoco elimina el efecto de perspectiva vertical: un parche ubicado al fondo de la tolva no representa exactamente la misma superficie que uno cercano a la cámara. Una homografía o calibración por geometría de tolva constituye una mejora futura.",
    )
    add_body(
        doc,
        "La salida describe solamente la superficie visible. Los finos pueden ocultarse bajo fragmentos grandes y las rocas enterradas no aparecen en la imagen. La textura de la roca, humedad, polvo, sombras y compresión del video pueden modificar las señales aprendidas. Además, el etiquetado se basa en juicio visual asistido y no en una distribución granulométrica medida. Por consiguiente, el modelo es adecuado para clasificación visual aparente y análisis de tendencia, pero no para afirmar tamaños físicos absolutos.",
    )
    add_body(
        doc,
        "La confianza softmax no se encuentra calibrada. Un valor alto puede aparecer en un ejemplo fuera de distribución, por lo que debe combinarse con margen, calidad de máscara y detección de condiciones desconocidas. La prueba técnica está balanceada intencionalmente y no reproduce necesariamente la prevalencia real de cada clase en operación. Tampoco se utiliza para escoger el umbral final de abstención sin riesgo de sobreajuste.",
    )
    add_placeholder(
        doc,
        "una selección de errores representativos: medio confundido con fino, medio confundido con grueso y máscara defectuosa",
        "Figura 6.18. Casos de error que ilustran confusiones entre clases adyacentes.",
    )

    add_heading(doc, "6.11.3 Operación, trazabilidad e impactos", 3)
    add_body(
        doc,
        "La solución se integra en una aplicación web que permite cargar un video, configurar el intervalo de frames, ejecutar el pipeline y revisar overlays, estadísticas y archivos descargables. Cada corrida mantiene una carpeta propia con entradas, parámetros, modelo utilizado y resultados. Esta trazabilidad permite reproducir una inferencia, revisar fallas de segmentación y relacionar el resumen de un camión con los parches que lo originan.",
    )
    add_body(
        doc,
        "Desde una perspectiva operacional, el analítico puede apoyar el seguimiento de tendencias visuales y la comparación entre cargas, siempre que los casos con baja cobertura, margen reducido o máscaras deficientes permanezcan sujetos a revisión. Su posible aporte económico y ambiental se relaciona con la identificación de cambios que afecten el transporte, el chancado o la conminución; cualquier beneficio, ahorro o reducción de consumo debe comprobarse mediante una prueba operacional asociada a variables de proceso.",
    )
    add_body(
        doc,
        "La automatización también puede reducir la necesidad de inspecciones manuales repetitivas y conservar evidencia auditable. Sin embargo, el resultado no debe utilizarse como decisión de seguridad o evaluación de desempeño sin validación y gobernanza. Los datos y resultados requieren control de acceso, criterios de retención y monitoreo de posibles sesgos relacionados con la cámara o las condiciones ambientales.",
    )

    doc.add_page_break()
    add_heading(doc, "6.12 Reproducibilidad de la implementación", 2)
    add_body(
        doc,
        "Los artefactos principales se mantienen dentro del repositorio Granulometria-Tolva. A continuación se registran las rutas relativas que permiten localizar el conjunto final, los modelos seleccionados y las corridas de evaluación. Al fusionar este capítulo con el informe global, estas rutas pueden trasladarse a un anexo técnico para reducir la extensión del cuerpo principal.",
    )
    add_table(
        doc,
        "Tabla 6.21. Artefactos principales de la versión evaluada.",
        ["Artefacto", "Ruta relativa al repositorio"],
        [
            ["Frames de Batch 003", r"data\datasets\patch_frames\batch003"],
            ["Etiquetas revisadas", r"data\labeling\batches\batch003_grilla96_data3_mixto_20260702\labels_revisadas"],
            ["Dataset final", r"outputs\patches\grid96_batch003_data3_mixto_ai"],
            ["Modelo YOLO", r"modelos percepcion\carga_tolva_yolo\haul_truck_segment_m_736_low_data_aug_07012025.pt"],
            ["Modelo Random Forest", r"modelos granulometria\clasificador_patches\runs\rf_grid96_batch003_revisado_20260702\model.joblib"],
            ["Modelo ResNet18", r"modelos granulometria\clasificador_patches\runs\resnet18_rgb_grid96_batch003_revisado_cuda_20260702\model.pt"],
            ["Script de inferencia", r"modelos granulometria\inferencia_tolva\predict_granulometria_tolva.py"],
            ["Inferencia de 50 frames", r"outputs\granulometria_tolva\cnn_resnet18_batch003_revisado_prueba_cnn_20260703"],
            ["Frames de prueba tecnica", r"data\evaluation\prueba_tecnica"],
            ["Prueba técnica", r"outputs\prueba_tecnica\prueba_tecnica_ampliada_cnn_batch003_revisado_20260717_084113"],
            ["Aplicación de etiquetado", r"tools\patch_labeler\app.py"],
            ["Implementación DexiNed", r"src\granulometria\dexined.py"],
        ],
        [2700, 6984],
        font_size=8.8,
    )

    add_heading(doc, "6.13 Conclusiones y trabajo futuro", 2)
    add_heading(doc, "6.13.1 Conclusiones", 3)
    add_body(
        doc,
        "El trabajo permite construir un pipeline completo y trazable para estimar la granulometría visual aparente de la carga transportada por camiones CAEX. La solución integra la segmentación del material, la normalización por ancho, la división en parches RGB de 96 píxeles, la clasificación supervisada y la agregación de resultados por frame y por video. Como resultado, el sistema entrega porcentajes de material fino, medio y grueso, mapas espaciales, un índice ordinal de tendencia, una medida de dispersión y evidencia visual para auditoría.",
    )
    add_body(
        doc,
        "La comparación metodológica muestra que los bordes obtenidos con DexiNed no separan de manera estable los fragmentos en presencia de textura interna, sombras y superposición de rocas. Random Forest constituye una línea base competitiva y alcanza un F1 macro de 0,697 en la prueba interna. Sin embargo, ResNet18 obtiene el mejor resultado sobre la misma partición, con F1 macro de 0,757 y exactitud de 0,754, por lo que se selecciona como clasificador principal de la versión actual.",
    )
    add_body(
        doc,
        "La evolución de los experimentos confirma que la calidad de las etiquetas es determinante. El mejor resultado se obtiene con el Batch 003 revisado, mientras que el aumento de datos en la combinación de Batches 001 y 002 no mejora la generalización. Esto demuestra que agregar ejemplos sin un control equivalente de consistencia no garantiza un mejor modelo y que la revisión humana continúa siendo una etapa necesaria del proceso.",
    )
    add_body(
        doc,
        "En la prueba técnica de 90 frames, el pipeline alcanza 76,7 % de exactitud y un F1 macro de 0,763 respecto de una referencia visual balanceada. Las clases fino y grueso presentan mayor estabilidad que medio, y casi todos los errores se producen entre categorías vecinas. Los porcentajes por clase, junto con el índice granulométrico y la dispersión, describen mejor estas transiciones que una única etiqueta dominante. No obstante, la salida sigue representando la superficie visible de la carga y no equivale a una medición física en milímetros ni a una curva granulométrica de laboratorio.",
    )

    doc.add_page_break()
    add_heading(doc, "6.13.2 Trabajo futuro", 3)
    add_body(
        doc,
        "La primera línea de trabajo consiste en fortalecer la validación. Se requiere definir referencias visuales consensuadas con especialistas de operación y construir un conjunto independiente que incluya diferentes cámaras, horarios, niveles de polvo, distancias y tipos de material. La separación de entrenamiento, validación y prueba debe realizarse por video o camión completo para evitar que frames temporalmente cercanos inflen las métricas.",
    )
    add_body(
        doc,
        "La segunda línea corresponde a la confiabilidad operacional. Es necesario calibrar las probabilidades del clasificador, validar la política de abstención y establecer criterios objetivos para declarar un frame no evaluable. También se debe medir la estabilidad entre frames del mismo camión y estudiar una agregación temporal que reduzca la influencia de imágenes redundantes, borrosas o con máscaras deficientes.",
    )
    add_body(
        doc,
        "La tercera línea busca completar la comparación de modelos bajo un mismo conjunto de datos y particiones. Además de la línea base OpenCV + Random Forest y la CNN RGB ya evaluadas, se deben probar DexiNed + Random Forest/SVM, una CNN basada en escala de grises o mapas de bordes y una CNN multimodal que combine RGB con DexiNed. El detector de rocas sobredimensionadas puede incorporarse como una señal complementaria e independiente de la composición granulométrica general.",
    )
    add_body(
        doc,
        "La cuarta línea apunta a relacionar el indicador visual con tamaños físicos. Para ello se necesita incorporar una referencia geométrica mediante calibración de cámara y tolva, homografía sobre el plano de carga, visión estéreo, sensores de profundidad o fusión con nubes de puntos. Solo después de contrastar los resultados con tamizado u otra referencia física será posible evaluar una estimación de tamaños absolutos y no únicamente tendencias visuales.",
    )
    add_heading(doc, "Referencias del capítulo", 1)
    references = [
        "[1] H. Li, G. Asbjörnsson y M. Lindqvist, “Image Process of Rock Size Distribution Using DexiNed-Based Neural Network”, Minerals, vol. 11, no. 7, art. 736, 2021, doi: 10.3390/min11070736.",
        "[2] T. Bamford, K. Esmaeili y A. P. Schoellig, “A deep learning approach for rock fragmentation analysis”, International Journal of Rock Mechanics and Mining Sciences, vol. 145, art. 104839, 2021, doi: 10.1016/j.ijrmms.2021.104839.",
        "[3] Q. Guo, Y. Wang, S. Yang y Z. Xiang, “A method of blasted rock image segmentation based on improved watershed algorithm”, Scientific Reports, vol. 12, art. 7143, 2022, doi: 10.1038/s41598-022-11351-0.",
        "[4] S. Shrivastava, S. Bhattacharjee y D. Deb, “Segmentation of mine overburden dump particles from images using Mask R-CNN”, Scientific Reports, vol. 13, art. 2046, 2023, doi: 10.1038/s41598-023-28586-0.",
        "[5] X. Soria, A. Sappa, P. Humanante y A. Akbarinia, “Dense extreme inception network for edge detection”, Pattern Recognition, vol. 139, art. 109461, 2023, doi: 10.1016/j.patcog.2023.109461.",
        "[6] J. Peng, Y. Cui, Z. Zhong y Y. An, “Ore Rock Fragmentation Calculation Based on Multi-Modal Fusion of Point Clouds and Images”, Applied Sciences, vol. 13, no. 23, art. 12558, 2023, doi: 10.3390/app132312558.",
        "[7] Y. Tang, Y. Wang y G. Si, “Vision-based size distribution analysis of rock fragments using multi-modal deep learning and interactive annotation”, Automation in Construction, vol. 159, art. 105276, 2024, doi: 10.1016/j.autcon.2024.105276.",
        "[8] M. S. Bahraini e I. Atighi, “A novel intelligent stereo vision approach for blast-induced fragmentation size distribution”, Minerals Engineering, vol. 215, art. 108822, 2024, doi: 10.1016/j.mineng.2024.108822.",
        "[9] M. Li, M. Chen, W. Lu, F. Zhao, P. Yan y J. Liu, “RDT-FragNet: Rock fragment instance segmentation in complex conditions”, Computers and Geotechnics, vol. 177, art. 106809, 2025, doi: 10.1016/j.compgeo.2024.106809.",
        "[10] L. Breiman, “Random Forests”, Machine Learning, vol. 45, pp. 5-32, 2001, doi: 10.1023/A:1010933404324.",
        "[11] K. He, X. Zhang, S. Ren y J. Sun, “Deep Residual Learning for Image Recognition”, en Proc. IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770-778, doi: 10.1109/CVPR.2016.90.",
    ]
    for ref in references:
        doc.add_paragraph(ref, style="Bibliography")

    doc.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
