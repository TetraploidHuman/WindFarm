from pathlib import Path

text = Path("src/windfarm/dashboard.py").read_text(encoding="utf-8")

init_start = text.find("async function init()")
init_call = text.find("    init();", init_start)

if init_start < 0 or init_call < 0:
    print("Could not find init function")
    exit(1)

# Build new init — must use {{ and }} because it's inside Python f-string template
new_init = r"""function init() {{
      el.playBtn.addEventListener("click", () => {{
        if (LIVE_MODE) {{
          state.followLive = !state.followLive;
          el.playBtn.textContent = state.followLive ? "暂停跟随" : "跟随实时";
          if (state.followLive && state.trace.length) {{
            state.idx = state.trace.length - 1;
            render();
          }}
          return;
        }}
        state.playing = !state.playing;
        el.playBtn.textContent = state.playing ? "暂停" : "播放";
      }});
      if (LIVE_MODE) {{
        el.modeBadge.textContent = "加载中...";
        fetch("/api/bootstrap", {{ cache: "no-store" }}).then(r => r.json()).then(data => {{
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
        }}).catch(e => {{ el.modeBadge.textContent = "加载失败"; }});
        window.setInterval(refreshLive, 100);
      }} else {{
        el.modeBadge.textContent = "回放";
        el.playBtn.textContent = "播放";
        setupAltitudeSelect();
        render();
      }}
    }}
    function setupAltitudeSelect() {{

# Find the old altitude select code and remove it (it's now in setupAltitudeSelect)
alt_start = text.find("const levelMin = state.report.mission.altitude_levels")
clearRoute = text.find("el.clearRouteBtn", alt_start)
clearRoute_end = text.find("});", clearRoute)

if alt_start > 0 and clearRoute_end > 0:
    # Replace the altitude init + clearRoute code, keeping clearRoute handler in setupAltitudeSelect
    old_alt_block = text[alt_start:clearRoute_end+3]
    new_alt_block = "// altitude select and clearRoute moved to setupAltitudeSelect()"
    text = text.replace(old_alt_block, new_alt_block)
    print("Altitude code moved to function")
else:
    print(f"Could not find altitude code: alt_start={alt_start}, clearRoute_end={clearRoute_end}")

# Replace init function
text = text[:init_start] + new_init + text[init_call:]
Path("src/windfarm/dashboard.py").write_text(text, encoding="utf-8")
print("Done")
