#!/usr/bin/env python3
"""Sync competition docx + research report with fujian_hills charts."""
from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph

PROJECT = Path(__file__).resolve().parent.parent
CHARTS = PROJECT / "charts"
COMPETITION_DOC = PROJECT / "docs" / "[5]湖南科技学院第二十四届大学生课外学术科技作品竞赛.docx"
RESEARCH_DOC = PROJECT / "docs" / "参赛作品研究报告.docx"
RESEARCH_MD = PROJECT / "docs" / "参赛作品研究报告.md"

# Substring replacements (keep 辽宁海岸 in eight-scenario savings list).
TEXT_REPLACEMENTS: list[tuple[str, str]] = [
    ("图2 辽宁海岸场景：风场降尺度效果对比", "图2 福建丘陵场景：风场降尺度效果对比"),
    ("图5 辽宁海岸场景：信念预测与实际观测校准", "图5 福建丘陵场景：信念预测与实际观测校准"),
    ("图6 辽宁海岸场景：信念引导的路径规划四维决策视图", "图6 福建丘陵场景：信念引导的路径规划四维决策视图"),
    ("图7 辽宁海岸真实场景：无人机航迹与地形叠加（相对名义直线侧向借风）", "图7 福建丘陵场景：无人机航迹与地形叠加（相对名义直线侧向借风）"),
    ("图8 辽宁海岸场景：信念能量场与航迹演化", "图8 福建丘陵场景：信念能量场与航迹演化"),
    ("图9 辽宁海岸场景：任务执行剖面——电量、距离、高度与能量获取率", "图9 福建丘陵场景：任务执行剖面——电量、距离、高度与能量获取率"),
    ("图7以辽宁海岸真实DEM为背景", "图7以福建丘陵真实DEM为背景"),
    ("沿海岸地形与风场能量分布借势绕行", "沿丘陵沟壑地形与风场能量分布借势绕行"),
    ("图8对比辽宁海岸场景初始与最终信念能量热力图", "图8对比福建丘陵场景初始与最终信念能量热力图"),
    ("辽宁海岸场景的风场降尺度效果对比如图2所示", "福建丘陵场景的风场降尺度效果对比如图2所示"),
    ("辽宁海岸场景的信念预测与实际观测校准关系如图5所示", "福建丘陵场景的信念预测与实际观测校准关系如图5所示"),
    ("辽宁海岸场景的信念引导路径规划四维决策视图如图6所示", "福建丘陵场景的信念引导路径规划四维决策视图如图6所示"),
    ("图7 辽宁海岸场景——无人机航迹与地形叠加", "图7 福建丘陵场景——无人机航迹与地形叠加"),
    ("图8辽宁海岸场景——信念能量场与航迹演化", "图8 福建丘陵场景——信念能量场与航迹演化"),
    ("图8 辽宁海岸场景——信念能量场与航迹演化", "图8 福建丘陵场景——信念能量场与航迹演化"),
    ("辽宁海岸场景的无人机航迹与地形叠加如图7所示，信念能量场与航迹演化如图8所示", "福建丘陵场景的无人机航迹与地形叠加如图7所示，信念能量场与航迹演化如图8所示"),
    ("【图10位置：最初八场景真实地形与粗风场图（上排CORE：闽/京/青/辽；下排HOLDOUT：疆/川/晋/台）】\n\n", ""),
    ("【图11位置：最初八场景相对名义直线的闭环能耗节省对比图】\n\n", ""),
    ("风速预测RMSE降低\t9.8%", "风速预测RMSE降低\t69.4%"),
    ("相比直线路径能耗节省\t3.1%", "相比直线路径能耗节省\t4.0%"),
    ("额外路径距离\t3.0%", "额外路径距离\t约1%"),
    ("降低85.7%", "降低69.4%"),
    ("降低30.0%", "降低72.3%"),
    ("垂直风预测MAE降低\t30.0%", "垂直风预测MAE降低\t72.3%"),
    # 表1 / 正文：与 fujian_hills_viz 图表一致（scenarios/fujian_hills/metrics.json）
    ("风速预测RMSE从5.329 m/s降至0.763 m/s，降低85.7%", "风速预测RMSE从0.588 m/s降至0.180 m/s，降低69.4%"),
    ("风速预测RMSE从5.329 m/s降至0.763 m/s（降幅85.7%）", "风速预测RMSE从0.588 m/s降至0.180 m/s（降幅69.4%）"),
    ("垂直风预测MAE从0.233 m/s降至0.163 m/s，降低30.0%", "垂直风预测MAE从0.234 m/s降至0.065 m/s，降低72.3%"),
    ("垂直风预测MAE从0.233 m/s降至0.163 m/s（降幅30.0%）", "垂直风预测MAE从0.234 m/s降至0.065 m/s（降幅72.3%）"),
    ("风速预测RMSE降低85.7%", "风速预测RMSE降低69.4%"),
    ("垂直风预测MAE降低30.0%", "垂直风预测MAE降低72.3%"),
    ("垂直分量MAE降低30.0%", "垂直分量MAE降低72.3%"),
    ("风向角预测MAE\t—\t0.077 rad", "风向角预测MAE\t—\t0.036 rad"),
    ("5.329 m/s\t0.763 m/s\t降低85.7%", "0.588 m/s\t0.180 m/s\t降低69.4%"),
    ("0.233 m/s\t0.163 m/s\t降低30.0%", "0.234 m/s\t0.065 m/s\t降低72.3%"),
    (
        "在包含4,000条训练样本（实际拟合3,200条）、800条验证样本和1,800条测试样本的走廊合成风场数据集上进行了系统评估",
        "在福建丘陵真实地形—历史风场场景（SRTM1+Open-Meteo，2000条训练样本、400条验证样本、测试集抽样2000点）上进行了系统评估",
    ),
    (
        "测试集上风速RMSE从5.329 m/s降至0.763 m/s（降幅85.7%），垂直风MAE从0.233 m/s降至0.163 m/s（降幅30.0%）",
        "测试集上风速RMSE从0.588 m/s降至0.180 m/s（降幅69.4%），垂直风MAE从0.234 m/s降至0.065 m/s（降幅72.3%）",
    ),
    ("5.329 m/s", "0.588 m/s"),
    ("0.763 m/s", "0.180 m/s"),
    ("0.233 m/s", "0.234 m/s"),
    ("0.163 m/s", "0.065 m/s"),
    ("降幅85.7%", "降幅69.4%"),
    ("降幅30.0%", "降幅72.3%"),
    ("0.077 rad", "0.036 rad"),
]

# Image order in competition form: img0=图1(keep), then fig2..fig9, fig10, fig11.
CHART_BY_INDEX: list[Path | None] = [
    None,  # 图1 系统架构 — 不替换
    CHARTS / "05_wind_field_maps_1.png",
    CHARTS / "02_feature_importance.png",
    CHARTS / "03_model_performance.png",
    CHARTS / "23_belief_vs_reality_1.png",
    CHARTS / "24_belief_path_planning_1.png",
    CHARTS / "25_eight_scenarios_terrain_1.png",
    CHARTS / "26_eight_scenarios_energy.png",
    CHARTS / "08_trajectory_map.png",
    CHARTS / "22_belief_energy_path.png",
    CHARTS / "07_mission_profile_1.png",
]


def _apply_replacements(text: str) -> str:
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def _replace_in_paragraph(paragraph: Paragraph) -> bool:
    if not paragraph.text:
        return False
    text = paragraph.text.strip()
    exact_pct = {"85.7%": "69.4%", "30.0%": "72.3%", "9.8%": "69.4%", "3.1%": "4.0%"}
    if text in exact_pct:
        new_text = exact_pct[text]
    else:
        new_text = _apply_replacements(paragraph.text)
    if new_text == paragraph.text:
        return False
    for run in paragraph.runs:
        run.text = ""
    if paragraph.runs:
        paragraph.runs[0].text = new_text
    else:
        paragraph.add_run(new_text)
    return True


def _iter_all_paragraphs(doc: Document):
    def walk_table(table):
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:
                    yield p
                for nested in cell.tables:
                    yield from walk_table(nested)

    for p in doc.paragraphs:
        yield p
    for table in doc.tables:
        yield from walk_table(table)


def _paragraph_has_drawing(paragraph: Paragraph) -> bool:
    return bool(paragraph._element.findall(".//" + qn("w:drawing")))


def _replace_paragraph_picture(paragraph: Paragraph, image_path: Path, width_in: float = 6.2) -> None:
    for run in list(paragraph.runs):
        if run._element.findall(".//" + qn("w:drawing")):
            run._element.getparent().remove(run._element)
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(width_in))


def sync_text(doc: Document) -> int:
    changed = 0
    for paragraph in _iter_all_paragraphs(doc):
        if _replace_in_paragraph(paragraph):
            changed += 1
    return changed


def replace_chart_images(doc: Document) -> int:
    image_paragraphs = [p for p in _iter_all_paragraphs(doc) if _paragraph_has_drawing(p)]
    replaced = 0
    for idx, paragraph in enumerate(image_paragraphs):
        chart = CHART_BY_INDEX[idx] if idx < len(CHART_BY_INDEX) else None
        if chart is None or not chart.exists():
            continue
        _replace_paragraph_picture(paragraph, chart)
        replaced += 1
    return replaced


def sync_docx(path: Path) -> tuple[int, int]:
    doc = Document(path)
    text_n = sync_text(doc)
    img_n = replace_chart_images(doc)
    doc.save(path)
    return text_n, img_n


def _set_run_font(run, name: str, size_pt: float, bold: bool = False) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size_pt)
    run.bold = bold


def _add_body(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Pt(24)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(text)
    _set_run_font(run, "宋体", 12)


def _add_heading1(doc: Document, text: str) -> None:
    p = doc.add_paragraph()
    run = p.add_run(text)
    _set_run_font(run, "黑体", 12, bold=True)


def build_research_report_docx(path: Path) -> None:
    """Build research report docx from synchronized markdown manuscript."""
    md_text = RESEARCH_MD.read_text(encoding="utf-8")
    doc = Document()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    tr = title.add_run("基于风场感知与智能路径规划的无人机导航系统")
    _set_run_font(tr, "黑体", 14, bold=True)
    doc.add_paragraph()

    for line in md_text.splitlines():
        line = line.strip()
        if not line:
            doc.add_paragraph()
            continue
        if line.startswith("# "):
            continue
        if line.startswith("## "):
            _add_heading1(doc, line[3:].strip())
            continue
        if line.startswith("|") and "---" in line:
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            _add_body(doc, " | ".join(cells))
            continue
        if line.startswith("**") and line.endswith("**"):
            p = doc.add_paragraph()
            run = p.add_run(line.strip("*"))
            _set_run_font(run, "宋体", 12, bold=True)
            continue
        _add_body(doc, line)

    doc.save(path)


def main() -> None:
    if COMPETITION_DOC.exists():
        text_n, img_n = sync_docx(COMPETITION_DOC)
        print(f"synced competition doc: {text_n} text blocks, {img_n} images replaced")
    build_research_report_docx(RESEARCH_DOC)
    print(f"written: {RESEARCH_DOC}")
    print(f"source md: {RESEARCH_MD}")


if __name__ == "__main__":
    main()
