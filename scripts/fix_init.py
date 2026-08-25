"""Fix the restored dashboard.py init function for live mode."""
from pathlib import Path

text = Path("src/windfarm/dashboard.py").read_text(encoding="utf-8")

# Fix 1: Replace broken async/await init with promise-based init
old_init_start = 'function init() {{\n      if (LIVE_MODE) {{\n        const response = await fetch("/api/bootstrap"'
if old_init_start in text:
    # Find the full init function
    init_start = text.find(old_init_start)
    # Find the closing of the if/else block (before event listeners)
    search_from = init_start
    # Find "} else {" for live/non-live split
    else_pos = text.find('} else {', search_from)
    # Find where init function ends
    init_call = text.find('    init();', else_pos)

    # Build replacement
    new_init = '''function init() {
      if (LIVE_MODE) {
        el.modeBadge.textContent = "加载中...";
        fetch("/api/bootstrap", { cache: "no-store" }).then(r => r.json()).then(data => {
          state.report = data.report;
          state.summary = data.summary;
          state.trace = data.trace;
          state.interactiveGoal = Boolean(data.capabilities && data.capabilities.interactive_goal);
          state.latestSeq = data.latest_seq ?? (state.trace.length - 1);
          if (state.trace.length) state.idx = state.trace.length - 1;
          el.modeBadge.textContent = state.interactiveGoal ? "交互式导航" : "实时";
          el.playBtn.textContent = "跟随实时";
          setupAltitudeSelect();
          render();
        }).catch(e => { el.modeBadge.textContent = "加载失败"; });
        window.setInterval(refreshLive, 100);
      } else {
        el.modeBadge.textContent = "回放";
        el.playBtn.textContent = "播放";
        setupAltitudeSelect();
        render();
      }'''

    # Find the end of the if/else block and start of event listeners
    # The pattern after if/else is: el.playBtn.addEventListener
    addEventListener_pos = text.find('el.playBtn.addEventListener("click"', else_pos)

    # Replace from init start to addEventListener
    text = text[:init_start] + new_init + '\n' + text[addEventListener_pos:]
    print("Init function replaced")
else:
    print("Could not find init function start")

# Fix 2: Move altitude select code to a function
# Find the altitude setup code that's in the middle of init
old_alt = 'const levelMin = state.report.mission.altitude_levels ? state.report.mission.altitude_levels[0] : 0;'
if old_alt in text:
    alt_start = text.find(old_alt)
    # Find where this block ends (before el.clearRouteBtn)
    btn_pos = text.find('el.clearRouteBtn', alt_start)

    # Build the setupAltitudeSelect function
    alt_func = """function setupAltitudeSelect() {
      if (!state.report || !state.report.mission) return;
      const levelMin = state.report.mission.altitude_levels ? state.report.mission.altitude_levels[0] : 0;
      const levelMax = state.report.mission.altitude_levels ? state.report.mission.altitude_levels[1] : 0;
      el.altitudeSelect.innerHTML = Array.from({ length: levelMax - levelMin + 1 }, (_, idx) => {
        const level = levelMin + idx;
        return "<option value=\\"" + level + "\\">高度层 z=" + level + "</option>";
      }).join("");
      el.altitudeSelect.value = String(state.altitudeLayer);
    }

"""
    # Remove the old altitude code from init and replace with just the event listener
    text = text[:alt_start] + alt_func + text[btn_pos:]
    print("Altitude setup extracted to function")
else:
    print("Altitude code not found")

Path("src/windfarm/dashboard.py").write_text(text, encoding="utf-8")
print("Done")
