import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

start_marker = 'if page == "🏠 Command Center":'
end_marker = '# ██████████████████  PAGE: ZONE MAPS'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Could not find markers.")
    exit(1)

new_command_center = """if page == "🏠 Command Center":
    last_updated = df["created_datetime_ist"].max()[:10] if "created_datetime_ist" in df.columns else "2024-05-18"

    st.markdown(f'''
    <div style="display:flex;justify-content:space-between;align-items:flex-end;">
        <div>
            <div class="page-title" style="font-weight: 700; letter-spacing: -0.5px;">BTP Intelligence Operations</div>
            <div class="page-sub" style="color:{T['neutral']};">Bengaluru Traffic Police Command Center</div>
        </div>
        <div style="color:{T['neutral']}; font-size:0.8rem; padding-bottom:5px;">DATA CURRENT AS OF: <strong>{last_updated}</strong></div>
    </div>
    ''', unsafe_allow_html=True)
    
    # ── 1. Top-of-page Status Strip ──
    v_pct = validation_data.get("approved_pct", 0) if validation_data else 0
    s_pct = scita_data.get("sync_pct", 0) if scita_data else 0
    has_anom = anomaly_data.get("has_anomalies", False) if anomaly_data else False
    anom_msg = anomaly_data.get("warning_message", "1 Alert") if anomaly_data else ""

    st.markdown(f'''
    <div class="status-strip">
        <div class="status-pill">
            <span style="color:{'#52b788' if v_pct>=80 else '#e94560'}">{'🟢' if v_pct>=80 else '🔴'}</span> 
            Data Health: {v_pct:.1f}% Valid
        </div>
        <div class="status-pill">
            <span style="color:{'#52b788' if s_pct>=80 else '#e94560'}">{'🟢' if s_pct>=80 else '🔴'}</span> 
            SCITA Sync: {s_pct:.1f}%
        </div>
        <div class="status-pill">
            <span style="color:{'#e94560' if has_anom else '#52b788'}">{'🔴' if has_anom else '🟢'}</span> 
            {'Anomaly: ' + anom_msg if has_anom else 'System Normal'}
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # ── Split KPI Row ──
    col_left, col_right = st.columns([5, 6])
    
    with col_left:
        sec("Data Health & Integrity")
        dh1, dh2, dh3 = st.columns(3)
        kpi("Total Records", f"{len(dff):,}", dh1)
        kpi("Approval Rate", f"{v_pct:.1f}%", dh2, alert=(v_pct < 80))
        kpi("Anomalies Detected", "Yes" if has_anom else "None", dh3, alert=has_anom)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        sec("Operations Panel")
        # SCITA Card Overhaul -> Standardized
        scita_synced = scita_data.get('sent_to_scita', 0) if scita_data else 0
        scita_pend = scita_data.get('not_sent', 0) if scita_data else 0
        scita_alert = scita_pend > 0
        
        o1, o2 = st.columns(2)
        kpi("SCITA Synced", f"{scita_synced:,}", o1)
        kpi("SCITA Pending", f"{scita_pend:,}", o2, alert=scita_alert)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        hvy_c = int(dff['is_heavy_vehicle'].sum()) if "is_heavy_vehicle" in dff.columns else 0
        hvy_pct = (hvy_c / len(dff) * 100) if len(dff) > 0 else 0
        k_mv = multi_summary.get('total_multi_violation_records',0) if multi_summary else 0
        
        o3, o4 = st.columns(2)
        kpi("Heavy Vehicles",   f"{hvy_c:,} <span style='font-size:0.5em;color:{T['neutral']}'>({hvy_pct:.1f}%)</span>", o3)
        kpi("Multi-Violation",  f"{k_mv:,}", o4)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("System Exports")
        ex1, ex2 = st.columns(2)
        with ex1:
            if not priority_df.empty:
                st.download_button("Download Priority CSV", priority_df.to_csv(index=False).encode(),
                                   "priority_ranked.csv", "text/csv", use_container_width=True)
        with ex2:
            if not hotspot_df.empty:
                st.download_button("Download Hotspots CSV", hotspot_df.to_csv(index=False).encode(),
                                   "hotspot_clusters.csv", "text/csv", use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sec("AI Enforcement Directives")
        if isinstance(recommendations, list) and recommendations:
            for rec in recommendations[:5]:
                pri   = rec.get("priority", "")
                stn   = rec.get("station", "")
                direc = rec.get("directive", "")
                
                u_cls = "urgent-card" if pri == 1 else ""
                tag = f"<b>[PRIORITY {pri}]</b> {stn}" if pri else f"{stn}"
                st.markdown(f'<div class="dir-card {u_cls}" style="padding:10px 14px;"><span class="dir-rank" style="color:{T["text"]}; font-size:0.8rem;">{tag}</span><br><span style="color:{T["neutral"]};font-size:0.9rem;">{direc}</span></div>',
                            unsafe_allow_html=True)

    with col_right:
        sec("AI & Operations Overview")
        ao1, ao2, ao3 = st.columns(3)
        
        acc = model_summary.get('ensemble_accuracy', 0) if model_summary else 0
        uniq_veh = dff["vehicle_number"].nunique() if "vehicle_number" in dff.columns else 1
        hab_c = int(dff["is_habitual_offender"].sum()) if "is_habitual_offender" in dff.columns else 0
        hab_pct = (hab_c / uniq_veh * 100) if uniq_veh else 0
        
        ao1.markdown(f'''
        <div class="kpi" title="Note: Class performance is uneven. Minor offences exhibit lower recall rates.">
            <div class="kpi-lbl">Model Accuracy *</div>
            <div class="kpi-val">{acc:.1%}</div>
        </div>
        ''', unsafe_allow_html=True)
        
        n_clus = model_summary.get('n_clusters','—') if model_summary else '—'
        kpi("Hotspot Clusters", f"{n_clus}", ao2)
        kpi("Habitual Offenders", f"{hab_c:,} <span style='font-size:0.5em;color:{T['neutral']}'>({hab_pct:.1f}%)</span>", ao3, alert=hab_c > 0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # ── Glance Insights ──
        reac = reactive_data.get("reactive_pct", 0) if reactive_data else 0
        proa = reactive_data.get("proactive_pct", 0) if reactive_data else 0
        g_zone = patrol_gap_df.iloc[0]["geohash5"] if not patrol_gap_df.empty else "N/A"
        p_hour = global_peak_df.iloc[0]["hour"] if not global_peak_df.empty else "N/A"
        
        gi1, gi2, gi3 = st.columns(3)
        with gi1:
            st.markdown(f'<div class="glance-card"><div class="glance-text">Enforcement:<br><b>{proa:.1f}% proactive vs {reac:.1f}% reactive</b></div></div>', unsafe_allow_html=True)
            if st.button("Shift & Timing", key="g_shift1", use_container_width=True): navigate_to("⏰ Shift & Timing")
        with gi2:
            st.markdown(f'<div class="glance-card urgent-card"><div class="glance-text">Patrol Gap:<br><b>Highest enforcement gap in {g_zone}</b></div></div>', unsafe_allow_html=True)
            if st.button("Zone Maps", key="g_zone1", use_container_width=True): navigate_to("🗺️ Zone Maps")
        with gi3:
            st.markdown(f'<div class="glance-card"><div class="glance-text">Peak Window:<br><b>City-wide violations peak at {p_hour}:00 IST</b></div></div>', unsafe_allow_html=True)
            if st.button("Operations", key="g_shift2", use_container_width=True): navigate_to("⏰ Shift & Timing")

        st.markdown("<br>", unsafe_allow_html=True)
        
        # Mini Map Viewer Carousel
        sec("Operations Map Viewer")
        
        if "mini_map_idx" not in st.session_state:
            st.session_state.mini_map_idx = 0
            
        mini_maps = [
            {"title": "Congestion Heatmap", "file": "01_congestion_heatmap.html"},
            {"title": "Hotspot Clusters & Priority", "file": "02_hotspot_clusters_priority.html"},
            {"title": "Night vs Day Patrol", "file": "03_night_vs_day.html"}
        ]
        
        col_prev, col_map, col_next = st.columns([1, 10, 1])
        
        with col_prev:
            st.markdown("<br>"*12, unsafe_allow_html=True)
            if st.button("◄", key="btn_prev_map", use_container_width=True):
                st.session_state.mini_map_idx = (st.session_state.mini_map_idx - 1) % len(mini_maps)
                st.rerun()
                
        with col_map:
            current_map = mini_maps[st.session_state.mini_map_idx]
            st.markdown(f"<div style='text-align:center; font-weight:600; color:{T['text']};'>{current_map['title']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center; font-size:0.75rem; color:{T['neutral']}; padding-bottom:10px;'>Map {st.session_state.mini_map_idx + 1} of {len(mini_maps)}</div>", unsafe_allow_html=True)
            embed_map(MAP_DIR / current_map["file"], height=400)
                
        with col_next:
            st.markdown("<br>"*12, unsafe_allow_html=True)
            if st.button("►", key="btn_next_map", use_container_width=True):
                st.session_state.mini_map_idx = (st.session_state.mini_map_idx + 1) % len(mini_maps)
                st.rerun()
                
        st.markdown("<br>", unsafe_allow_html=True)
        
        sec("Quick Vehicle Search")
        vcols = st.columns([3, 1])
        cmd_search = vcols[0].text_input("Enter Plate No.", placeholder="KA01AB1234", key="cmd_search_input", label_visibility="collapsed")
        if vcols[1].button("Search", use_container_width=True, type="primary"):
            if cmd_search:
                st.session_state.quick_search_vnum = cmd_search.strip().upper()
            navigate_to("🚨 Offender Registry")
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        sec("Quick Navigation")
        qcols1, qcols2 = st.columns(2), st.columns(2)
        if qcols1[0].button("Zone Maps", use_container_width=True): navigate_to("🗺️ Zone Maps")
        if qcols1[1].button("Offender Registry", use_container_width=True): navigate_to("🚨 Offender Registry")
        if qcols2[0].button("Priority Board", use_container_width=True): navigate_to("📊 Priority Board")
        if qcols2[1].button("Shift & Timing", use_container_width=True): navigate_to("⏰ Shift & Timing")

"""

new_content = content[:start_idx] + new_command_center + "\n# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n" + content[end_idx:]

with open("app.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("Professional Patch applied.")
