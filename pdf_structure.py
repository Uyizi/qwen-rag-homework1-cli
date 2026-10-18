import bisect
import re
from decimal import Decimal, InvalidOperation


def word_lines(words, tolerance=2.5):
    lines = []
    for word in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        center = (word[1] + word[3]) / 2
        if not lines or abs(center - lines[-1][0]) > tolerance:
            lines.append([center, [word]])
        else:
            lines[-1][1].append(word)
    return [(y, sorted(row, key=lambda w: w[0])) for y, row in lines]


def numeric_value(text):

    value = re.sub(r"\s+", "", text).replace(",", "").replace("−", "-")
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)(?:±(\d+(?:\.\d+)?))?%?", value)
    if not match:
        return None
    try:
        return {"mean": str(Decimal(match[1])), "std": str(Decimal(match[2])) if match[2] else None}
    except InvalidOperation:
        return None


def render_table(table):
    lines = [table["reference"] + ": " + table["caption"],
             "Columns: " + "; ".join(f"c{i}={name}" for i, name in enumerate(table["headers"]))]
    for row in table["rows"]:
        prefix = f"{row['id']}" + (f" [section={row['section']}]" if row["section"] else "")
        lines.append(prefix + ": " + "; ".join(f"c{i}={value}" for i, value in enumerate(row["cells"])))
    if table.get("notes"):
        lines.append("Notes: " + " ".join(table["notes"]))
    return "\n".join(lines)


def extract_tables(path, relative):
    import fitz
    units = []
    seen = set()
    with fitz.open(path) as doc:
        for page_number, page in enumerate(doc, 1):
            words = page.get_text("words")
            labels = []
            for word in words:
                if word[4].upper() != "TABLE":
                    continue
                following = [w for w in words if w[0] >= word[2] - 1 and w[0] < word[2] + 45
                             and abs((w[1] + w[3]) / 2 - (word[1] + word[3]) / 2) < 2.5]
                if following:
                    token = min(following, key=lambda w: w[0])[4].strip(".:")
                    if re.fullmatch(r"[IVX]+|\d+", token):
                        labels.append({"reference": "Table " + token, "word": word,
                                       "center": (word[0] + min(following, key=lambda w: w[0])[2]) / 2})
            rules = []
            for drawing in page.get_drawings():
                for item in drawing["items"]:
                    if item[0] == "l" and abs(item[1].y - item[2].y) < .5:
                        left, right = sorted((item[1].x, item[2].x))
                        if right - left > 70:
                            rules.append((left, item[1].y, right))
            for label in labels:
                candidates = [r for r in rules if r[1] > label["word"][3] and r[0] < label["center"] < r[2]]
                if not candidates:
                    continue
                first = min(candidates, key=lambda r: r[1])
                next_caption = min((l["word"][1] for l in labels if l["word"][1] > first[1]
                                    and first[0] < l["center"] < first[2]), default=page.rect.height)
                borders = sorted(set(r[1] for r in rules if abs(r[0] - first[0]) < 2 and abs(r[2] - first[2]) < 2
                                     and first[1] - .5 <= r[1] < next_caption))
                if len(borders) < 3:
                    continue
                left, top, right, bottom = first[0], borders[0], first[2], borders[-1]
                header_words = [w for w in words if left - 1 <= (w[0] + w[2]) / 2 <= right + 1
                                and top < (w[1] + w[3]) / 2 < borders[1]]
                header_lines = word_lines(header_words)
                if len(header_lines) != 1:
                    continue
                groups = []
                for word in header_lines[0][1]:
                    if not groups or word[0] - groups[-1][-1][2] >= 9:
                        groups.append([word])
                    else:
                        groups[-1].append(word)
                if len(groups) < 2:
                    continue
                headers = [" ".join(w[4] for w in group) for group in groups]
                cuts = [(a[-1][2] + b[0][0]) / 2 for a, b in zip(groups, groups[1:])]
                body_words = [w for w in words if left - 1 <= (w[0] + w[2]) / 2 <= right + 1
                              and borders[1] < (w[1] + w[3]) / 2 < bottom]
                lines = []
                for y, row in word_lines(body_words):
                    cells = [[] for _ in headers]
                    for word in row:
                        cells[bisect.bisect_right(cuts, (word[0] + word[2]) / 2)].append(word[4])
                    lines.append({"y": y, "words": row, "cells": [" ".join(c) for c in cells]})
                merged_labels = {}
                for band in range(1, len(borders) - 1):
                    group_lines = [line for line in lines if borders[band] < line["y"] < borders[band + 1]]
                    numerical = [line for line in group_lines if sum(numeric_value(c) is not None for c in line["cells"][1:]) >= 2]
                    labels_present = [line["cells"][0] for line in group_lines if line["cells"][0]]
                    if len(numerical) >= 2 and labels_present and sum(bool(line["cells"][0]) for line in numerical) < len(numerical) / 2:
                        merged_labels[band] = " ".join(labels_present)
                rows, notes, section = [], [], ""
                for line in lines:
                    cells = line["cells"]
                    raw_line = " ".join(w[4] for w in line["words"])
                    band = bisect.bisect_right(borders, line["y"]) - 1
                    if band in merged_labels:
                        if not cells[1]:
                            continue
                        cells[0] = merged_labels[band]
                        section = cells[0]
                    elif not cells[0] and not any(numeric_value(c) for c in cells):
                        section = raw_line
                        continue
                    elif raw_line.startswith(("†", "*")):
                        notes.append(raw_line)
                        continue
                    if not cells[0] or sum(bool(c) for c in cells) < 2:
                        notes.append(raw_line)
                        continue
                    rows.append({"id": f"r{len(rows) + 1}", "section": section, "cells": cells})
                if not rows or any(len(row["cells"]) != len(headers) for row in rows):
                    continue
                caption_area = max((r for r in rules if r[0] < label["center"] < r[2] and r[2] - r[0] < page.rect.width * .7),
                                   key=lambda r: r[2] - r[0], default=first)
                caption_words = [w for w in words if caption_area[0] - 1 <= (w[0] + w[2]) / 2 <= caption_area[2] + 1
                                 and label["word"][3] < (w[1] + w[3]) / 2 < top]
                caption = " ".join(" ".join(w[4] for w in row) for _, row in word_lines(caption_words))
                reference = label["reference"]
                suffix = reference.split()[1]
                unit_id = f"{relative}::t{suffix}"
                if unit_id in seen:
                    unit_id += f"-p{page_number}"
                seen.add(unit_id)
                table = {"reference": reference, "caption": caption, "headers": headers,
                         "rows": rows, "notes": notes, "bbox": [left, top, right, bottom]}
                units.append({"id": unit_id, "file": relative, "location": f"第 {page_number} 页，{reference}",
                              "chunk": suffix, "section": "body", "kind": "table", "table": table,
                              "text": render_table(table), "retrieval_text": render_table(table)})
    return units


def table_cells(unit):
    table = unit["table"]
    result = {}
    for row in table["rows"]:
        name = row["cells"][0]
        repeated = sum(r["cells"][0] == name for r in table["rows"]) > 1
        if repeated and len(row["cells"]) > 1 and numeric_value(row["cells"][1]) is None:
            name += " / " + row["cells"][1]
        for column, (name, value) in enumerate(zip(table["headers"], row["cells"])):
            result[f"{row['id']}:c{column}"] = {
                "label": f"{table['reference']} / {row['section'] + ' / ' if row['section'] else ''}{row['cells'][0]}"
                         + (" / " + row["cells"][1] if repeated and numeric_value(row["cells"][1]) is None else "") + f" / {name}",
                "value": value, "numeric": numeric_value(value), "section": row["section"], "column": column,
                "key_column": column == 0 or (column == 1 and repeated and numeric_value(value) is None)}
    return result


def format_decimal(value):
    return format(value.normalize(), "f")


def calculation_notes(units, question):

    notes = []
    if not re.search(r"相比|比较|比.{1,35}(?:提高|高|低|多|少)|差值|均值差|提高|优于|difference|compare|better|increase", question, re.I):
        return notes
    for unit in units:
        if unit.get("kind") != "table":
            continue
        table = unit["table"]
        if not re.search(r"accuracy|auc|acc\s*%", table["caption"] + " " + " ".join(r["section"] for r in table["rows"]), re.I):
            continue
        grouped = {}
        for row in table["rows"]:
            grouped.setdefault(row["section"], []).append(row)
        for section, rows in grouped.items():
            target_rows = [row for row in rows if re.search(r"(?<!\w)" + re.escape(row["cells"][0]) + r"(?!\w)", question, re.I)]
            for column, header in enumerate(table["headers"][1:], 1):
                named = [name for name in table["headers"][1:] if re.search(re.escape(name), question, re.I)]
                if named and header not in named:
                    continue
                if len(rows) == 2:
                    left, right = rows[1], rows[0]
                elif len(target_rows) == 1:
                    left = target_rows[0]
                    alternatives = [row for row in rows if row is not left and numeric_value(row["cells"][column])]
                    if not alternatives:
                        continue
                    right = max(alternatives, key=lambda row: Decimal(numeric_value(row["cells"][column])["mean"]))
                else:
                    continue
                a, b = numeric_value(left["cells"][column]), numeric_value(right["cells"][column])
                if a is None or b is None:
                    continue
                delta = format_decimal(Decimal(a["mean"]) - Decimal(b["mean"]))
                notes.append({"source_id": unit["id"], "section": section, "column": header,
                              "left": left["cells"][0], "left_cell": f"{left['id']}:c{column}", "left_value": left["cells"][column],
                              "right": right["cells"][0], "right_cell": f"{right['id']}:c{column}", "right_value": right["cells"][column],
                              "difference_percentage_points": delta})
    return notes
