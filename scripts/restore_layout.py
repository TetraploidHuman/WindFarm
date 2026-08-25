"""Restore working HTML layout with cream CSS style."""
from pathlib import Path

# Fix dashboard.py - restore mini-grid + features-row, remove broken card-rows
dash = Path(__file__).resolve().parent.parent / "src" / "windfarm" / "dashboard.py"
text = dash.read_text(encoding="utf-8")

# Find the broken section and replace
legend_end = text.find('</span></div>\n      </div>')
if legend_end < 0:
    legend_end = text.find('class="legend"')
    legend_end = text.find('</div>', legend_end) + 6

side_grid = text.find('<div class="side-grid">')

# Build the replacement content
new_content = """</span></div>
      </div>
      <div class="mini-grid">
        <div class="panel"><h3>电量与能量</h3><canvas id="batteryCanvas" class="chart-canvas" width="640" height="360"></canvas></div>
        <div class="panel"><h3>残差热力图</h3><canvas id="residualCanvas" class="mini-canvas" width="640" height="360"></canvas></div>
        <div class="panel"><h3>风矢量场</h3><canvas id="vectorCanvas" class="mini-canvas" width="640" height="360"></canvas></div>
        <div class="panel"><h3>三维地形</h3><canvas id="terrainCanvas" class="mini-canvas" width="640" height="360"></canvas></div>
        <div class="panel"><h3>物理 vs 学习 vs 真值</h3><canvas id="compareCanvas" class="mini-canvas" width="640" height="360"></canvas></div>
        <div class="panel"><h3>训练曲线</h3><canvas id="trainingCanvas" class="mini-canvas" width="640" height="360"></canvas></div>
        <div class="panel"><h3>垂直风剖面</h3><canvas id="profileCanvas" class="mini-canvas" width="640" height="360"></canvas></div>
        <div class="panel"><h3>高度轨迹</h3><canvas id="altitudeCanvas" class="mini-canvas" width="640" height="360"></canvas></div>
      </div>
    </div>
    <div class="features-row">
      <div class="panel"><h3>粗尺度风场</h3><canvas id="coarseWindCanvas" class="chart-canvas" width="640" height="360"></canvas></div>
      <div class="panel"><h3>DEM 地形特征</h3><canvas id="demCanvas" class="chart-canvas" width="640" height="360"></canvas></div>
      <div class="panel"><h3>下推高分辨率风场</h3><canvas id="highResWindCanvas" class="chart-canvas" width="640" height="360"></canvas></div>
    </div>
    <div class="side-grid">"""

# Find the exact section to replace
old_start = text.find('<div class="legend">')
old_start = text.find('</span></div>', old_start)
old_start = text.find('\n', old_start)  # newline after legend div
old_end = text.find('<div class="side-grid">')

if old_start > 0 and old_end > old_start:
    text = text[:old_start] + new_content
    # Now remove everything between new_content's side-grid and old side-grid content
    # The new_content already has <div class="side-grid"> at the end
    # Find the old side-grid content after the replacement point
    remaining = text[old_start + len(new_content):]
    old_side_end = remaining.find('<div class="side-grid">')
    if old_side_end >= 0:
        # Skip the duplicate side-grid
        text = text[:old_start + len(new_content)] + remaining[old_side_end + len('<div class="side-grid">'):]
    print("Layout restored")
else:
    print(f"Could not find markers: old_start={old_start}, old_end={old_end}")

# Remove broken split-canvas JS references
text = text.replace('physicsErrorCanvas: document.getElementById("physicsErrorCanvas"),\n', '')
text = text.replace('predErrorCanvas: document.getElementById("predErrorCanvas"),\n', '')
text = text.replace('truthWindCanvas: document.getElementById("truthWindCanvas"),\n', '')

# Restore drawCompareChart render call
old_split = 'renderStep("drawPhysicsError", () => drawPhysicsError(frame));\n        renderStep("drawPredError", () => drawPredError(frame));\n        renderStep("drawTruthWindChart", () => drawTruthWindChart(frame));'
restored = 'renderStep("drawCompareChart", () => drawCompareChart(frame));'
if old_split in text:
    text = text.replace(old_split, restored)
    print("Render calls restored")

# Update CSS for features-row and mini-grid (cream compatible)
text = text.replace(
    '.mini-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }',
    '.mini-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }'
)
text = text.replace(
    '.features-row',
    '.features-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; } .features-row'
)
# Remove duplicate features-row declarations
text = text.replace(
    '.features-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; } .features-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }',
    '.features-row { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 14px; }'
)

dash.write_text(text, encoding="utf-8")
print("Done")
