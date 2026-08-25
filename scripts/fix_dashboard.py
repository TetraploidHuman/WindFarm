"""Fix dashboard_parts.py: legend overlap, DEM labels, HiDPI canvas."""
from pathlib import Path

target = Path(__file__).resolve().parent.parent / "src" / "windfarm" / "dashboard_parts.py"
data = target.read_bytes()
text = data.decode("utf-8")

# Fix 1: Remove per-line label in drawLine
old1 = 'ctx.fillText(label, pad, 16 + coarseData.length * 0);'
new1 = '// label rendered in legend below chart'
if old1 in text:
    text = text.replace(old1, new1)
    print("[1] Removed per-line labels from drawLine")
else:
    print("[1] drawLine label not found (may already be fixed)")

# Fix 2: Replace legend at y=16 (overlapping with chart) with proper legend below chart
old2 = (
    'const labels = ["u_km", "v_km", "w_km", "|风速|"];\n'
    '      const colors = ["#115d6b", "#b53a2d", "#2f7d55", "rgba(216,116,32,0.85)"];\n'
    '      labels.forEach((lbl, i) => {\n'
    '        ctx.fillStyle = colors[i];\n'
    '        ctx.font = "bold 12px Microsoft YaHei";\n'
    '        ctx.fillText(lbl, pad + i * 68, 16);\n'
    '      });'
)
new2 = (
    '// Legend below chart\n'
    '      const legendY = pad + h + 22;\n'
    '      const legendItems = [\n'
    '        { color: "#115d6b", label: "u_km (东→)" },\n'
    '        { color: "#b53a2d", label: "v_km (北↑)" },\n'
    '        { color: "#2f7d55", label: "w_km (上升)" },\n'
    '        { color: "rgba(216,116,32,0.85)", label: "|风速|" }\n'
    '      ];\n'
    '      let lx = pad;\n'
    '      legendItems.forEach((item) => {\n'
    '        ctx.fillStyle = item.color;\n'
    '        ctx.font = "bold 11px Microsoft YaHei";\n'
    '        ctx.fillRect(lx, legendY - 10, 14, 3);\n'
    '        ctx.fillText(item.label, lx + 18, legendY);\n'
    '        lx += ctx.measureText(item.label).width + 30;\n'
    '      });'
)

# Try with \\r\\n line endings
old2_crlf = old2.replace('\n', '\r\n')
if old2 in text:
    text = text.replace(old2, new2)
    print("[2] Fixed legend with LF line endings")
elif old2_crlf in text:
    text = text.replace(old2_crlf, new2)
    print("[2] Fixed legend with CRLF line endings")
else:
    print("[2] Legend pattern not found - searching...")
    # Search for approximate match
    idx = text.find('const labels = ["u_km"')
    if idx > 0:
        snippet = text[idx:idx+250]
        print(f"    Found at {idx}: {repr(snippet[:120])}")

# Fix 3: DEM gap (already done but let's ensure it's 30 for extra safety)
old3 = 'const gap = 28;'
new3 = 'const gap = 30;'
if old3 in text:
    text = text.replace(old3, new3)
    print("[3] DEM gap 28 -> 30")

# Fix 4: DEM labels - move inside panel, away from next row title
old4 = (
    'ctx.fillText(`min ${fmt(min)}`, x0, y0 + panelH + 14);\n'
    '        ctx.fillText(`max ${fmt(max)}`, x0 + panelW - 60, y0 + panelH + 14);'
)
new4 = (
    'ctx.fillText(`min ${fmt(min)}`, x0 + 2, y0 + panelH - 6);\n'
    '        ctx.fillText(`max ${fmt(max)}`, x0 + panelW - 60, y0 + panelH - 6);'
)
old4_crlf = old4.replace('\n', '\r\n')
if old4 in text:
    text = text.replace(old4, new4)
    print("[4] DEM labels moved inside panels")
elif old4_crlf in text:
    text = text.replace(old4_crlf, new4)
    print("[4] DEM labels moved inside panels (CRLF)")
else:
    print("[4] DEM labels not found in expected format, searching...")
    idx = text.find('panelH + 14')
    if idx > 0:
        print(f"    Found at {idx}: ...{repr(text[idx-40:idx+40])}...")

# Fix 5: Canvas size references - replace el.coarseWindCanvas.width/height with cw/ch
# (already done, but verify)
if 'cw - pad * 2' in text:
    print("[5] Canvas size already using cw/ch (HiDPI)")
else:
    # Apply canvas size fix
    text = text.replace('el.coarseWindCanvas.width - pad * 2', 'cw - pad * 2')
    text = text.replace('el.coarseWindCanvas.height - pad * 2', 'ch - pad - pad')
    print("[5] Fixed canvas size refs to use cw/ch")

target.write_text(text, encoding="utf-8")
print("Done.")
