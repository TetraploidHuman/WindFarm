from pathlib import Path

text = Path("src/windfarm/dashboard.py").read_text(encoding="utf-8")

# Find setupAltitudeSelect function and fix it
old_empty = "function setupAltitudeSelect() {{\n    init();"
new_setup = """function setupAltitudeSelect() {{
      if (!state.report || !state.report.mission) return;
      const levelMin = state.report.mission.altitude_levels ? state.report.mission.altitude_levels[0] : 0;
      const levelMax = state.report.mission.altitude_levels ? state.report.mission.altitude_levels[1] : 0;
      el.altitudeSelect.innerHTML = Array.from({{ length: levelMax - levelMin + 1 }}, (_, idx) => {{
        const level = levelMin + idx;
        return `<option value="${{level}}">高度层 z=${{level}}</option>`;
      }}).join("");
      el.altitudeSelect.value = String(state.altitudeLayer);
      el.altitudeSelect.addEventListener("change", (event) => {{
        state.altitudeLayer = Number(event.target.value);
        render();
      }});
      el.clearRouteBtn.addEventListener("click", () => {{
        state.routeDraftNodes = [];
        render();
      }});
      el.stepRange.addEventListener("input", (event) => {{
        state.idx = Number(event.target.value);
        if (LIVE_MODE) state.followLive = false;
        render();
      }});
      el.layerSelect.addEventListener("change", (event) => {{
        state.layer = event.target.value;
        render();
      }});
      window.setInterval(tick, 700);
    }}
    function init()"""

if old_empty in text:
    text = text.replace(old_empty, new_setup)
    print("setupAltitudeSelect populated")
else:
    print("Could not find empty setupAltitudeSelect")
    idx = text.find("function setupAltitudeSelect()")
    if idx > 0:
        print("Found at", idx, ":", repr(text[idx:idx+100]))

# Also remove the old duplicate init() at the end (keep only the one in setupAltitudeSelect)
# Find "init();" that should be at the end
init_call = text.rfind("init();")
if init_call > 0:
    # Check if it's after setupAltitudeSelect
    before = text[init_call-50:init_call]
    if "function" in before:
        # This init(); is inside setupAltitudeSelect - remove the standalone one
        print("Removing standalone init() call")

Path("src/windfarm/dashboard.py").write_text(text, encoding="utf-8")
print("Done")
