# Configuration

DASexplorer stores its configuration in a single JSON file, `cfg/config.json`, inside the package directory. It contains two sections: **profiles** (reading and visualisation settings, see [What is a reading profile?](what-is-a-profile.md)) and **ui** (application-wide display settings).

To find the file on your system, run:

```bash
python -c "import dasexplorer, os; print(os.path.join(os.path.dirname(dasexplorer.__file__), 'cfg', 'config.json'))"
```

---

## UI settings

The `"ui"` section controls the appearance and behaviour of the application interface. These settings are particularly useful when running DASexplorer on screens with different resolutions or DPI scales.

```json
"ui": {
    "theme": "dark",
    "hidpi_scaling": true,
    "font_size_offset": -1,
    "left_panel_scroll": true,
    "compact_spinboxes": true
}
```

| Key | Type | Default | Description |
|---|---|---|---|
| `theme` | `string` | `"dark"` | Colour theme. `"dark"` or `"light"`. Can also be changed from **View → Theme**. |
| `hidpi_scaling` | `bool` | `true` | Enable automatic HiDPI / Retina display scaling. Set to `false` on fixed-DPI screens if the interface appears too large. |
| `font_size_offset` | `int` | `-1` | Adjusts the system font size by this many points at startup. `-1` makes the font slightly smaller, `0` uses the system default. |
| `left_panel_scroll` | `bool` | `true` | Show a vertical scrollbar in the left panel. Useful on small or low-resolution screens where not all controls fit without scrolling. Set to `false` on large screens. |
| `compact_spinboxes` | `bool` | `true` | Use a compact fixed height for spinboxes and Apply buttons. Set to `false` to revert to the Qt default size. |

!!! tip "Personal configuration"
    If you are working on a large high-resolution screen, you may prefer:
    ```json
    "ui": {
        "theme": "dark",
        "hidpi_scaling": false,
        "font_size_offset": 0,
        "left_panel_scroll": false,
        "compact_spinboxes": false
    }
    ```

---

## Reading profiles

Reading profiles are documented separately. See:

- [What is a reading profile?](what-is-a-profile.md)
- [Adding a reading profile](profiles.md)
- [Built-in reading profiles](readers/index.md)
