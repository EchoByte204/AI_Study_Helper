from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from app.ingestion import _extract_pptx


def _slide_xml(title: str, include_chart: bool = False) -> str:
    chart = '<c:chart r:id="rId2"/>' if include_chart else ""
    return f'''<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
        xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
        xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart">
      <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>
      <a:tbl><a:tr><a:tc><a:txBody><a:p><a:r><a:t>Year</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>Output</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
      <a:tr><a:tc><a:txBody><a:p><a:r><a:t>2025</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>42</a:t></a:r></a:p></a:txBody></a:tc></a:tr></a:tbl>
      <p:pic><p:blipFill><a:blip r:embed="rId1"/></p:blipFill></p:pic>{chart}
      </p:spTree></p:cSld></p:sld>'''


def _rels(include_chart: bool = False) -> str:
    chart = '<Relationship Id="rId2" Target="../charts/chart1.xml"/>' if include_chart else ""
    return f'''<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
      <Relationship Id="rId1" Target="../media/logo.png"/>{chart}</Relationships>'''


def _pptx_fixture() -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("ppt/slides/slide1.xml", _slide_xml("Course results", True))
        archive.writestr("ppt/slides/slide2.xml", _slide_xml("Review"))
        archive.writestr("ppt/slides/_rels/slide1.xml.rels", _rels(True))
        archive.writestr("ppt/slides/_rels/slide2.xml.rels", _rels())
        archive.writestr("ppt/media/logo.png", b"\x89PNG" + b"x" * 3000)
        archive.writestr(
            "ppt/charts/chart1.xml",
            '''<c:chartSpace xmlns:c="http://schemas.openxmlformats.org/drawingml/2006/chart">
              <c:chart><c:plotArea><c:barChart><c:ser>
                <c:tx><c:strRef><c:strCache><c:pt><c:v>Series A</c:v></c:pt></c:strCache></c:strRef></c:tx>
                <c:cat><c:strRef><c:strCache><c:pt><c:v>2025</c:v></c:pt></c:strCache></c:strRef></c:cat>
                <c:val><c:numRef><c:numCache><c:pt><c:v>42</c:v></c:pt></c:numCache></c:numRef></c:val>
              </c:ser></c:barChart></c:plotArea></c:chart></c:chartSpace>''',
        )
    return buffer.getvalue()


def test_pptx_extraction_includes_table_and_chart_values_and_deduplicates_images():
    pages, visuals = _extract_pptx(_pptx_fixture(), "lecture.pptx")

    assert len(pages) == 2
    assert "Course results" in pages[0]
    assert "Table 1" in pages[0]
    assert "Year | Output" in pages[0]
    assert "2025 | 42" in pages[0]
    assert "Series A" in pages[0]
    assert "2025 = 42" in pages[0]
    assert len(visuals) == 1
    assert visuals[0].page_number == 1


def test_pptx_extraction_reports_slide_progress():
    updates = []
    _extract_pptx(_pptx_fixture(), "lecture.pptx", lambda value, message: updates.append((value, message)))

    assert updates[0][1] == "Reading slide 1 of 2"
    assert updates[-1][1] == "Reading slide 2 of 2"
