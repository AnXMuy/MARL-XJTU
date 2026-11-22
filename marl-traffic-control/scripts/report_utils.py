"""Shared report serialization and SVG helpers."""
import csv
import json


def load(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def grouped(rows, keys):
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(row)
    return groups


def svg(path, title, width, height, parts):
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                    f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
                    f'<g font-family="sans-serif" font-size="13" fill="#192238">'
                    f'{text(24, 30, title, size=19)}{"".join(parts)}</g></svg>\n')


def text(x, y, label, *, size=13, color="#192238"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{label}</text>'


def line(x1, y1, x2, y2, color):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.4"/>'
