"""Building the PDF reports projects 3 and 4 have to hand to the reader.

Both briefs ask for a report that describes the experiments, justifies the design choices and
is downloadable from the project page. This wraps reportlab so those modules can write a
document — headings, paragraphs, tables, figures — instead of positioning things on a page.

Reports are rendered on demand into memory rather than committed. A committed PDF is another
build artefact that silently drifts from the results it describes.
"""

from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PAGE_WIDTH = A4[0] - 40 * mm


def _styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("title", parent=base["Title"], fontSize=20, spaceAfter=4),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=11, textColor=colors.grey,
            alignment=TA_CENTER, spaceAfter=18,
        ),
        "h1": ParagraphStyle("h1", parent=base["Heading1"], fontSize=14, spaceBefore=16, spaceAfter=6),
        "h2": ParagraphStyle("h2", parent=base["Heading2"], fontSize=11.5, spaceBefore=12, spaceAfter=4),
        "body": ParagraphStyle("body", parent=base["BodyText"], fontSize=9.5, leading=13.5, spaceAfter=6),
        "bullet": ParagraphStyle("bullet", parent=base["BodyText"], fontSize=9.5, leading=13.5,
                                 leftIndent=12, bulletIndent=2, spaceAfter=3),
        "formula": ParagraphStyle("formula", parent=base["BodyText"], fontName="Courier",
                                  fontSize=9, alignment=TA_CENTER, spaceBefore=6, spaceAfter=8),
        "caption": ParagraphStyle("caption", parent=base["BodyText"], fontSize=8.5,
                                  textColor=colors.grey, alignment=TA_CENTER, spaceAfter=10),
        # Table cells are paragraphs, not plain strings: reportlab draws a bare string on one
        # line and lets it run straight past the cell border rather than wrapping it.
        "cell": ParagraphStyle("cell", parent=base["BodyText"], fontSize=8.5, leading=10.5,
                               spaceBefore=0, spaceAfter=0),
        "cell_header": ParagraphStyle("cell_header", parent=base["BodyText"], fontSize=8.5,
                                      leading=10.5, fontName="Helvetica-Bold",
                                      spaceBefore=0, spaceAfter=0),
    }


class Report:
    """Collects flowables, then renders them to PDF bytes."""

    def __init__(self, title, subtitle=None):
        self.styles = _styles()
        self.flow = [Paragraph(title, self.styles["title"])]
        if subtitle:
            self.flow.append(Paragraph(subtitle, self.styles["subtitle"]))

    def heading(self, text, level=1):
        self.flow.append(Paragraph(text, self.styles["h1" if level == 1 else "h2"]))
        return self

    def paragraph(self, text):
        self.flow.append(Paragraph(text, self.styles["body"]))
        return self

    def bullets(self, items):
        for item in items:
            self.flow.append(Paragraph(item, self.styles["bullet"], bulletText="•"))
        self.flow.append(Spacer(1, 4))
        return self

    def formula(self, text):
        self.flow.append(Paragraph(text, self.styles["formula"]))
        return self

    def _cell(self, value, style):
        """Wrap a cell so its text wraps inside the column instead of overrunning it."""
        if hasattr(value, "wrap"):
            return value
        return Paragraph(str(value), self.styles[style])

    def table(self, rows, header=True, widths=None):
        """`rows` is a list of lists. Cells wrap; pass a flowable to keep your own formatting."""
        data = [
            [self._cell(cell, "cell_header" if header and index == 0 else "cell") for cell in row]
            for index, row in enumerate(rows)
        ]
        table = Table(data, colWidths=widths, hAlign="LEFT")
        style = [
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        if header:
            style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eeeeee")))
        table.setStyle(TableStyle(style))
        self.flow.extend([table, Spacer(1, 10)])
        return self

    def figure(self, path, caption=None, width=400):
        """Place an image scaled to `width`, keeping its aspect ratio."""
        image = Image(str(path))
        scale = min(width, PAGE_WIDTH) / image.imageWidth
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        image.hAlign = "CENTER"
        self.flow.append(image)
        if caption:
            self.flow.append(Paragraph(caption, self.styles["caption"]))
        else:
            self.flow.append(Spacer(1, 10))
        return self

    def page_break(self):
        self.flow.append(PageBreak())
        return self

    def render(self):
        buffer = BytesIO()
        SimpleDocTemplate(
            buffer,
            pagesize=A4,
            leftMargin=20 * mm,
            rightMargin=20 * mm,
            topMargin=18 * mm,
            bottomMargin=18 * mm,
        ).build(self.flow)
        return buffer.getvalue()
