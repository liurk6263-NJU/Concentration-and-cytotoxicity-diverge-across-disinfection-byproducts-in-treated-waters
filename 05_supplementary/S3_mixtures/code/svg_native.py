from __future__ import annotations

from html import escape
from pathlib import Path

class NativeSVG:

    def __init__(self, width_mm: float = 100.0, height_mm: float = 75.0,
                 view_width: int = 600, view_height: int = 450) -> None:
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.view_width = view_width
        self.view_height = view_height
        self.items: list[str] = []
        self._counter = 0

    def _id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:04d}"

    @staticmethod
    def _attrs(attrs: dict[str, object]) -> str:
        return " ".join(f'{key.replace("_", "-")}="{escape(str(value))}"'
                        for key, value in attrs.items() if value is not None)

    def rect(self, x: float, y: float, width: float, height: float, *,
             fill: str, stroke: str = "#333333", stroke_width: float = 1.0,
             prefix: str = "rect") -> str:
        obj_id = self._id(prefix)
        attrs = {
            "id": obj_id, "x": f"{x:.3f}", "y": f"{y:.3f}",
            "width": f"{width:.3f}", "height": f"{height:.3f}",
            "fill": fill, "stroke": stroke, "stroke_width": stroke_width,
        }
        self.items.append(f"  <rect {self._attrs(attrs)} />")
        return obj_id

    def line(self, x1: float, y1: float, x2: float, y2: float, *,
             stroke: str = "#333333", stroke_width: float = 1.0,
             dash: str | None = None, prefix: str = "line") -> str:
        obj_id = self._id(prefix)
        attrs = {
            "id": obj_id, "x1": f"{x1:.3f}", "y1": f"{y1:.3f}",
            "x2": f"{x2:.3f}", "y2": f"{y2:.3f}",
            "stroke": stroke, "stroke_width": stroke_width,
            "stroke_dasharray": dash,
        }
        self.items.append(f"  <line {self._attrs(attrs)} />")
        return obj_id

    def text(self, x: float, y: float, value: str, *, size: float = 12.0,
             anchor: str = "middle", weight: str = "normal",
             style: str = "normal", rotate: float | None = None,
             fill: str = "#111111", prefix: str = "text") -> str:
        obj_id = self._id(prefix)
        transform = None if rotate is None else f"rotate({rotate:.1f} {x:.3f} {y:.3f})"
        attrs = {
            "id": obj_id, "x": f"{x:.3f}", "y": f"{y:.3f}",
            "font_family": "Times New Roman", "font_size": f"{size:.2f}",
            "font_weight": weight, "font_style": style,
            "text_anchor": anchor, "fill": fill, "transform": transform,
        }
        self.items.append(f"  <text {self._attrs(attrs)}>{escape(value)}</text>")
        return obj_id

    def hatch_rect(self, x: float, y: float, width: float, height: float, *,
                   spacing: float = 11.0, color: str = "#666666",
                   stroke_width: float = 0.8, prefix: str = "hatch") -> None:

        c_min = y - (x + width)
        c_max = y + height - x
        c = c_min
        while c <= c_max + 1e-9:
            candidates: list[tuple[float, float]] = []
            y_left = x + c
            if y <= y_left <= y + height:
                candidates.append((x, y_left))
            y_right = x + width + c
            if y <= y_right <= y + height:
                candidates.append((x + width, y_right))
            x_top = y - c
            if x <= x_top <= x + width:
                candidates.append((x_top, y))
            x_bottom = y + height - c
            if x <= x_bottom <= x + width:
                candidates.append((x_bottom, y + height))
            unique: list[tuple[float, float]] = []
            for point in candidates:
                if not any(abs(point[0] - q[0]) < 1e-6 and abs(point[1] - q[1]) < 1e-6
                           for q in unique):
                    unique.append(point)
            if len(unique) >= 2:
                self.line(*unique[0], *unique[1], stroke=color,
                          stroke_width=stroke_width, prefix=prefix)
            c += spacing

    def errorbar(self, x: float, low_y: float, high_y: float, *, cap: float = 5.0,
                 color: str = "#333333", stroke_width: float = 1.25,
                 prefix: str = "errorbar") -> None:
        self.line(x, low_y, x, high_y, stroke=color, stroke_width=stroke_width,
                  prefix=f"{prefix}-stem")
        self.line(x - cap, low_y, x + cap, low_y, stroke=color,
                  stroke_width=stroke_width, prefix=f"{prefix}-lowcap")
        self.line(x - cap, high_y, x + cap, high_y, stroke=color,
                  stroke_width=stroke_width, prefix=f"{prefix}-highcap")

    def frame(self, x0: float, y0: float, x1: float, y1: float,
              *, stroke: str = "#222222", stroke_width: float = 1.2) -> None:
        self.line(x0, y0, x1, y0, stroke=stroke, stroke_width=stroke_width,
                  prefix="frame-top")
        self.line(x1, y0, x1, y1, stroke=stroke, stroke_width=stroke_width,
                  prefix="frame-right")
        self.line(x1, y1, x0, y1, stroke=stroke, stroke_width=stroke_width,
                  prefix="frame-bottom")
        self.line(x0, y1, x0, y0, stroke=stroke, stroke_width=stroke_width,
                  prefix="frame-left")

    def save(self, path: Path, title: str, description: str) -> None:
        header = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg id="svg-root" xmlns="http://www.w3.org/2000/svg" '
            f'width="{self.width_mm:.1f}mm" height="{self.height_mm:.1f}mm" '
            f'viewBox="0 0 {self.view_width} {self.view_height}">\n'
            f'  <title id="svg-title">{escape(title)}</title>\n'
            f'  <desc id="svg-description">{escape(description)}</desc>\n'
            f'  <rect id="background" x="0" y="0" width="{self.view_width}" '
            f'height="{self.view_height}" fill="#ffffff" stroke="none" />\n'
        )
        path.write_text(header + "\n".join(self.items) + "\n</svg>\n", encoding="utf-8")
