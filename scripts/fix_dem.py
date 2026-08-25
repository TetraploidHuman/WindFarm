"""Fix DEM labels — move out of panel, add background for readability."""
from pathlib import Path

target = Path(__file__).resolve().parent.parent / "src" / "windfarm" / "dashboard_parts.py"
data = target.read_bytes()
text = data.decode("utf-8")

# Fix: DEM labels - move from inside panel to below, with white background
# Find the DEM section
old_dem_labels = (
    'ctx.fillText(`min ${fmt(min)}`, x0 + 2, y0 + panelH - 6);\n'
    '        ctx.fillText(`max ${fmt(max)}`, x0 + panelW - 60, y0 + panelH - 6);'
)
new_dem_labels = (
    'const minText = `min ${fmt(min)}`;\n'
    '        const maxText = `max ${fmt(max)}`;\n'
    '        const minW = ctx.measureText(minText).width + 6;\n'
    '        const labelY = y0 + panelH + 16;\n'
    '        ctx.fillStyle = "rgba(255,255,255,0.75)";\n'
    '        ctx.fillRect(x0, labelY - 11, minW, 16);\n'
    '        ctx.fillRect(x0 + panelW - 62, labelY - 11, 62, 16);\n'
    '        ctx.fillStyle = "#172026";\n'
    '        ctx.fillText(minText, x0 + 3, labelY);\n'
    '        ctx.fillText(maxText, x0 + panelW - 60, labelY);'
)

if old_dem_labels in text:
    text = text.replace(old_dem_labels, new_dem_labels)
    print("[1] DEM labels moved below panels with background")
else:
    # Try CRLF
    old_crlf = old_dem_labels.replace('\n', '\r\n')
    if old_crlf in text:
        text = text.replace(old_crlf, new_dem_labels)
        print("[1] DEM labels moved below panels (CRLF)")
    else:
        print("[1] DEM labels not found - may need manual fix")

# Fix gap 30 -> 40
if 'const gap = 30;' in text:
    text = text.replace('const gap = 30;', 'const gap = 40;')
    print("[2] DEM gap 30 -> 40")
elif 'const gap = 28;' in text:
    text = text.replace('const gap = 28;', 'const gap = 40;')
    print("[2] DEM gap 28 -> 40")
else:
    print("[2] DEM gap not found")

# Fix title position: y0 - 6 -> y0 - 10
old_title = 'ctx.fillText(panel.title, x0, y0 - 6);'
new_title = 'ctx.fillText(panel.title, x0, y0 - 10);'
if old_title in text:
    text = text.replace(old_title, new_title)
    print("[3] Title moved up slightly")
else:
    print("[3] Title position not found")

target.write_text(text, encoding="utf-8")
print("Done")
