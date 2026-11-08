import pytest
from bs4 import BeautifulSoup

from chandra.output import get_image_name, parse_html, parse_markdown


@pytest.fixture
def invoice_html():
    return (
        '<div data-label="Page-Footer">Telefoon: 1200 - vodafone.nl</div>'
        '<div data-label="Page-Header"><strong>Factuurnummer XXXXXXXXXX</strong></div>'
        '<div data-label="Page-Header">Pagina 2 van 2</div>'
        '<div data-label="Section-Header"><strong>Rekeningdetails January 2026</strong></div>'
    )


def test_headers_and_footers_keep_block_boundaries(invoice_html):
    assert parse_markdown(invoice_html, include_headers_footers=True) == (
        "Telefoon: 1200 - vodafone.nl\n\n"
        "**Factuurnummer XXXXXXXXXX**\n\n"
        "Pagina 2 van 2\n\n"
        "**Rekeningdetails January 2026**"
    )


def test_headers_and_footers_are_excluded_by_default(invoice_html):
    assert parse_markdown(invoice_html) == "**Rekeningdetails January 2026**"
    html = parse_html(invoice_html)
    assert "Telefoon" not in html
    assert "Factuurnummer" not in html
    assert "Pagina" not in html


@pytest.mark.parametrize("label", ["Text", "Section-Header", "Caption", "Footnote"])
def test_inline_formatting_stays_within_its_layout_block(label):
    html = (
        f'<div data-label="{label}">First <strong>bold</strong> text.</div>'
        f'<div data-label="{label}">Second <em>italic</em> text.</div>'
    )
    assert parse_markdown(html) == "First **bold** text.\n\nSecond *italic* text."


def test_html_keeps_block_boundaries_without_layout_attributes(invoice_html):
    soup = BeautifulSoup(
        parse_html(invoice_html, include_headers_footers=True), "html.parser"
    )
    blocks = soup.find_all(recursive=False)
    assert len(blocks) == 4
    assert all(block.name in ["div", "p"] for block in blocks)
    assert all(not block.attrs for block in blocks)
    assert [block.get_text() for block in blocks] == [
        "Telefoon: 1200 - vodafone.nl",
        "Factuurnummer XXXXXXXXXX",
        "Pagina 2 van 2",
        "Rekeningdetails January 2026",
    ]


@pytest.mark.parametrize(
    "label, content, expected",
    [
        (
            "Text",
            "<p>First paragraph</p><p>Second paragraph</p>",
            "First paragraph\n\nSecond paragraph",
        ),
        ("Section-Header", "<h2>Heading</h2>", "## Heading"),
        ("List-Group", "<ul><li>First</li><li>Second</li></ul>", "- First\n- Second"),
        (
            "Table",
            "<table><tr><td>Cell</td></tr></table>",
            "<table><tr><td>Cell</td></tr></table>",
        ),
        ("Code-Block", "<pre>first\n  second</pre>", "```\nfirst\n  second\n```"),
        ("Equation-Block", '<math display="block">x + y</math>', "$$x + y$$"),
    ],
)
def test_existing_block_markup_is_preserved(label, content, expected):
    html = (
        '<div data-label="Caption">Before</div>'
        f'<div data-label="{label}">{content}</div>'
        '<div data-label="Footnote">After</div>'
    )
    assert parse_markdown(html) == f"Before\n\n{expected}\n\nAfter"
    assert content in parse_html(html)


def test_skipped_blocks_do_not_affect_image_names_or_spacing():
    html = (
        '<div data-label="Page-Header">Header</div>'
        '<div data-label="Text">Before</div>'
        '<div data-label="Blank-Page">Skipped</div>'
        '<div data-label="Figure"><img alt="Figure"/></div>'
        '<div data-label="Caption">After</div>'
    )
    image_name = get_image_name(html, 4)
    assert parse_markdown(html) == f"Before\n\n![Figure]({image_name})\n\nAfter"
    assert parse_markdown(html, include_images=False) == "Before\n\nAfter"
    assert "<img" not in parse_html(html, include_images=False)
