from chandra import input as input_mod


def test_load_file_uses_pdf_loader_when_extension_pdf(monkeypatch):
    """Ensure load_file routes .pdf paths to load_pdf_images even if filetype.guess fails.

    This simulates a multi-page PDF where only the first page would be used
    if we treated the file as a single image.
    """

    calls = {}

    def fake_guess(_):
        # Simulate failure to detect PDF from file header.
        return None

    def fake_load_pdf_images(path, page_range):  # pragma: no cover - behavior verified via result
        calls["path"] = path
        calls["page_range"] = page_range
        # Pretend we decoded three pages
        return ["page0", "page1", "page2"]

    monkeypatch.setattr(input_mod.filetype, "guess", fake_guess)
    monkeypatch.setattr(input_mod, "load_pdf_images", fake_load_pdf_images)

    images = input_mod.load_file("dummy.pdf", {"page_range": "0-2"})

    assert images == ["page0", "page1", "page2"]
    assert calls["path"].endswith("dummy.pdf")
    # Parsed page range should be passed through as a list of ints
    assert calls["page_range"] == [0, 1, 2]
