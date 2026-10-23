from PIL import Image

from chandra.input import load_pdf_images


class FakeRenderedPage:
    def __init__(self, size):
        self.size = size

    def to_pil(self):
        return Image.new("RGB", self.size, "white")


class FakePage:
    def __init__(self, width=200, height=300):
        self.width = width
        self.height = height

    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

    def render(self, scale):
        assert scale > 0
        return FakeRenderedPage((self.width, self.height))


class FakePdfDocument:
    def __init__(self, filepath):
        self.filepath = filepath
        self.pages = [FakePage(200, 300), FakePage(300, 400)]
        self.forms_initialized = False
        self.closed = False

    def init_forms(self):
        self.forms_initialized = True

    def __len__(self):
        return len(self.pages)

    def __getitem__(self, index):
        return self.pages[index]

    def close(self):
        self.closed = True


def test_load_pdf_images_processes_all_pages_when_page_range_omitted(monkeypatch):
    monkeypatch.setattr("chandra.input.pdfium.PdfDocument", FakePdfDocument)
    monkeypatch.setattr("chandra.input.flatten", lambda page: None)

    images = load_pdf_images("dummy.pdf")

    assert len(images) == 2
    assert [image.size for image in images] == [(200, 300), (300, 400)]


def test_load_pdf_images_respects_page_range(monkeypatch):
    monkeypatch.setattr("chandra.input.pdfium.PdfDocument", FakePdfDocument)
    monkeypatch.setattr("chandra.input.flatten", lambda page: None)

    images = load_pdf_images("dummy.pdf", page_range=[1])

    assert len(images) == 1
    assert images[0].size == (300, 400)
