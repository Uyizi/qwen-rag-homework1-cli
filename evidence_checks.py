import json
import re
from decimal import Decimal

from pdf_structure import table_cells, numeric_value


def requested_entity_guard(question, units):

    if not re.search(r"准确率|AUC|accuracy|结果|数值", question, re.I):
        return None
    requested = re.findall(r"在\s*([A-Za-z][A-Za-z0-9_.-]+)\s*(?:上|的)", question)
    text = " ".join(u["text"] for u in units if u.get("section") != "references")
    for entity in requested:
        if not re.search(r"(?<![A-Za-z0-9_])" + re.escape(entity) + r"(?![A-Za-z0-9_])", text, re.I):
            return {"status": "refused", "answer": "文档中没有相关信息。", "citations": [],
                    "reason": f"正文没有问题指定对象{entity}的资料，不能借用其他对象的数值。"}
    return None


def remap_evidence_ids(raw, aliases):

    try:
        obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
    except ValueError:
        return raw
    if isinstance(obj, dict):
        for field in ("citations", "facts"):
            if isinstance(obj.get(field), list):
                for item in obj[field]:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        item["id"] = aliases.get(item["id"], item["id"])
    return json.dumps(obj, ensure_ascii=False)


def explicit_context(units, question):

    requested_sections = set(re.findall(r"\b([IVX]+)[-–.]([A-Z])(?![A-Za-z])", question))
    symbols = set(re.findall(r"[\u0370-\u03ff]", question))
    main, subsection = {}, {}
    chosen, by_section = [], {}
    if re.search(r"作者|署名|标题|题名|机构|单位|所属|来自哪|\b(?:authors?|title|affiliations?|institution)\b|who\s+(?:wrote|authored)", question, re.I):
        documents = {}
        for unit in units:
            if unit.get("kind") != "table" and unit.get("section") != "references":
                documents.setdefault(unit["file"], []).append(unit)
        for rows in documents.values():
            first_page = rows[0].get("location")
            for unit in rows[:4]:
                if unit.get("location") != first_page:
                    break
                chosen.append(unit)
                if re.search(r"(?im)^\s*(?:abstract\b|摘要|(?:I|1)[.、]\s*(?:introduction|引言|绪论)|#{1,2}\s)", unit["text"]):
                    break
    for unit in units:
        if unit.get("kind") == "table" or unit.get("section") == "references":
            continue
        file = unit["file"]
        headings = list(re.finditer(r"(?m)^\s*([IVX]+|[A-Z])\.\s+([^\n]+)", unit["text"]))
        membership = {(main.get(file), subsection.get(file))} if not headings or headings[0].start() > 0 else set()
        for match in headings:
            number, title = match.groups()
            if re.fullmatch(r"[IVX]+", number) and title == title.upper():
                main[file], subsection[file] = number, None
            elif len(number) == 1:
                subsection[file] = number
            membership.add((main.get(file), subsection.get(file)))
        for section in requested_sections & membership:
            by_section.setdefault(section, []).append(unit)
    for section in sorted(by_section):
        chosen.extend(by_section[section][:8])
    for symbol in sorted(symbols):
        matches = [u for u in units if u.get("kind") != "table" and u.get("section") != "references" and symbol in u["text"]]
        # 公式条件优先定位
        matches.sort(key=lambda u: not bool(re.search(r"[≥≤<>]", u["text"])))
        chosen.extend(matches[:2])
    return list({u["id"]: u for u in chosen}.values())


def reference_requests(text):
    return {"Table " + match.upper() for match in re.findall(r"(?:Table|TABLE|表)\s*([IVX]+|\d+)(?![A-Za-z0-9])", text, re.I)}


def figure_statistics_guard(question, chunks):
    figures = re.findall(r"(?:Figure|Fig\.?|图)\s*(\d+)(?![A-Za-z0-9])", question, re.I)
    if not figures or not re.search(r"标准差|standard deviation|±", question, re.I):
        return None
    all_text = " ".join(c["text"] for c in chunks if c.get("kind") != "table")
    tables = {c["table"]["reference"]: c for c in chunks if c.get("kind") == "table"}
    for figure in figures:
        linked = set()
        for group in re.findall(r"\(([^()]{0,130})\)", re.sub(r"\s+", " ", all_text)):
            if re.search(r"(?:Figure|Fig\.?)\s*" + re.escape(figure) + r"\b", group, re.I):
                linked |= reference_requests(group)
        available = [tables[ref] for ref in linked if ref in tables]
        if not any(any("±" in cell for row in table["table"]["rows"] for cell in row["cells"]) for table in available):
            return {"status": "refused", "answer": "文档中没有相关信息。", "citations": [],
                    "reason": f"Figure {figure}的文字层未提取到精确均值与标准差，正文也未把统计表明确绑定到该图；不能用相邻表格冒充。"}
    return None


def monotonicity_notes(units, question):
    notes = []
    pattern = re.compile(r"([A-Za-z][A-Za-z0-9_]*(?:\s*[−+-]\s*[A-Za-z][A-Za-z0-9_]*)*)\s*(≥|≤|>=|<=|>|<)\s*([A-Za-z\u0370-\u03ff][A-Za-z0-9_\u0370-\u03ff]*)")
    seen = set()
    for unit in units:
        for match in pattern.finditer(unit["text"]):
            lhs, operator, parameter = match.groups()
            if parameter not in question or (lhs, operator, parameter) in seen:
                continue
            seen.add((lhs, operator, parameter))
            stricter = operator in {">", ">=", "≥"}
            notes.append({"source_id": unit["id"], "condition": match[0], "parameter": parameter,
                          "increasing_threshold": "stricter" if stricter else "looser",
                          "explanation": f"固定左侧量{lhs}时，提高右侧阈值{parameter}会使条件{'更严格，合法集合只会保持或缩小' if stricter else '更宽松，合法集合只会保持或扩大'}。"})
    return notes


def check_numeric_and_direction(answer, units, calculations, constraints):
    source_numbers = set()
    for text in [u["text"] for u in units] + [json.dumps(c, ensure_ascii=False) for c in calculations]:
        for number in re.findall(r"(?<![A-Za-z])[-−]?\d+(?:,\d{3})*(?:\.\d+)?", text):
            source_numbers.add(Decimal(number.replace(",", "").replace("−", "-")))
    differences = [Decimal(c["difference_percentage_points"]) for c in calculations]
    source_numbers.update(abs(value) for value in differences)
    if any(value < 0 for value in differences) and re.search(r"(?:均|都|全部|两项都)[^。！？\n]{0,12}(?:优于|领先|更好|更高|提高)", answer):
        return "全称领先结论与来源计算的负差值冲突"
    for number in re.findall(r"(?<![A-Za-z])[-−]?\d+\.\d+", answer):
        if Decimal(number.replace("−", "-")) not in source_numbers:
            return "回答中的数值没有本次证据或已验证计算支持：" + number
    for constraint in constraints:
        parameter = re.escape(constraint["parameter"])
        bad = "更可能保留|越可能保留|保留更多|更宽松|更容易通过" if constraint["increasing_threshold"] == "stricter" else "保留更少|更严格|更难通过"
        if re.search(parameter + r"[^。！？\n]{0,15}(?:越大|增加|增大|提高)[^。！？\n]{0,35}(?:" + bad + ")", answer):
            return "回答的条件方向与原文不等式冲突：" + constraint["condition"]
    return None


def selected_table_facts(raw, units, question):
    try:
        obj = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
    except ValueError:
        return [], None
    if not isinstance(obj, dict) or "facts" not in obj:
        return [], None
    facts = obj["facts"]
    if not isinstance(facts, list):
        return [], "facts必须为数组"
    mapping = {u["id"]: u for u in units if u.get("kind") == "table"}
    cited_tables = {c.get("id") for c in obj.get("citations", []) if isinstance(c, dict) and isinstance(c.get("id"), str)} if isinstance(obj.get("citations"), list) else set()
    verified = []
    for fact in facts:
        if isinstance(fact, dict) and "id" not in fact and isinstance(fact.get("cell"), str):
            candidates = [source for source in cited_tables if source in mapping and fact["cell"] in table_cells(mapping[source])]
            if len(candidates) == 1:
                fact = {**fact, "id": candidates[0], "source_resolved_from_unique_cited_table": True}
        if not isinstance(fact, dict) or not isinstance(fact.get("id"), str) or not isinstance(fact.get("cell"), str):
            return [], "表格单元格引用格式错误"
        unit = mapping.get(fact["id"])
        if unit is None or fact["cell"] not in table_cells(unit):
            return [], "表格或单元格不在本次证据中"
        cell = table_cells(unit)[fact["cell"]]
        headers = unit["table"]["headers"]
        requested_columns = [i for i, name in enumerate(headers[1:], 1) if re.search(re.escape(name), question, re.I)]
        if requested_columns and not cell["key_column"] and cell["column"] not in requested_columns:
            return [], "表格单元格属于问题未请求的列"
        verified.append({**fact, **cell})
    return verified, None
