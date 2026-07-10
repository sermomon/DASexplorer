# DASexplorer documentation skeleton

This is a starting structure for DASexplorer's documentation, built with MkDocs Material.

## How to use

1. Copy `mkdocs.yml`, `requirements-docs.txt`, `docs/`, and `.github/workflows/docs.yml` into the root of the `dasexplorer` repository.
2. Fill in the placeholder text (in parentheses) and add real screenshots to `docs/img/`.
3. Preview locally:

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

Then open http://127.0.0.1:8000

4. Enable GitHub Pages: in the repo settings, under **Pages**, set the source to the `gh-pages` branch (this branch is created automatically the first time the workflow in `.github/workflows/docs.yml` runs).
5. Push to `main` — the GitHub Action deploys automatically whenever `docs/`, `mkdocs.yml`, or `requirements-docs.txt` change. You can also trigger it manually from the Actions tab (`workflow_dispatch`).

## Structure

```
docs/
  index.md                 Home page
  installation.md
  getting-started.md
  configuration.md         Profile system, read_dmin_m / read_dmax_m
  interrogators/
    index.md
    optodas.md
    silixa.md
    das4whale.md
  annotation/
    index.md
    bbox.md
    obbox.md
    keypoints.md
    line.md
  export.md                YOLO / COCO / Raven
  faq.md
  changelog.md
  img/                     Put screenshots here
mkdocs.yml
requirements-docs.txt
.github/workflows/docs.yml
```

## Notes

- Screenshots referenced in the pages (e.g. `screenshot-main-gui.png`) are placeholders — replace with actual captures from the app.
- `mkdocs-glightbox` is included so screenshots get a click-to-zoom lightbox automatically.
- Once ready, you can add API reference pages generated from docstrings using `mkdocstrings` without changing the rest of the setup.
