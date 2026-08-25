function init() {{
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
