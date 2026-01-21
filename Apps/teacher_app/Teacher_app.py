def practice_dialogues_section():
    """Dialogue practice section for teachers."""
    st.header("💭 Dialogue Practice")
    
    if df_dialogues.empty:
        st.info("No dialogues available. Add some content first!")
        return
    
    categories = df_dialogues["Category"].dropna().unique().tolist()
    selected_category = st.selectbox("Filter by Category:", ["All"] + categories, key="dialog_cat")
    
    display_dialogues = df_dialogues.copy()
    if selected_category != "All":
        display_dialogues = display_dialogues[display_dialogues["Category"] == selected_category]
    
    if display_dialogues.empty:
        st.info("No dialogues in selected category.")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Start Dialogue Practice", use_container_width=True, key="dialog_practice"):
            st.session_state.practice_mode = True
            st.session_state.practice_items = display_dialogues.to_dict('records')
            st.session_state.practice_index = 0
            st.rerun()
    with col2:
        if st.button("📊 Browse Dialogues One by One", use_container_width=True, key="dialog_view"):
            st.session_state.practice_mode = False
            st.rerun()
    
    # PRACTICE MODE
    if st.session_state.practice_mode and st.session_state.practice_items:
        item = st.session_state.practice_items[st.session_state.practice_index]
        
        st.subheader("Dialogue Line")
        st.markdown(f"### {item['Bulgarian']}")
        
        if pd.notna(item.get('Pronunciation')):
            st.write(f"*({item['Pronunciation']})*")
        
        audio_path = tts_audio(item['Bulgarian'])
        if audio_path:
            with open(audio_path, 'rb') as f:
                st.audio(f.read(), format='audio/mp3')
        
        answer = st.text_input("Type the English translation:", key="dialog_answer")
        if st.button("Check Answer", use_container_width=True, key="dialog_check"):
            if answer.lower().strip() == item['English'].lower().strip():
                st.success("✅ Correct!")
            else:
                st.error(f"❌ Correct answer: {item['English']}")
            
            if pd.notna(item.get('Grammar_Notes')):
                st.info(f"📝 Note: {item['Grammar_Notes']}")
        
        st.markdown("---")
        col_prev, col_next, col_finish = st.columns([1, 1, 2])
        
        with col_prev:
            if st.button("⬅️ Previous", disabled=st.session_state.practice_index == 0, key="dialog_prev"):
                st.session_state.practice_index -= 1
                st.rerun()
        
        with col_next:
            if st.button("Next ➡️", disabled=st.session_state.practice_index >= len(st.session_state.practice_items) - 1, key="dialog_next"):
                st.session_state.practice_index += 1
                st.rerun()
        
        with col_finish:
            if st.button("Finish Practice", type="primary", use_container_width=True, key="dialog_finish"):
                st.session_state.practice_mode = False
                st.success("Dialogue practice session completed!")
                st.rerun()
        
        progress = (st.session_state.practice_index + 1) / len(st.session_state.practice_items)
        st.progress(progress)
        st.caption(f"Dialogue {st.session_state.practice_index + 1} of {len(st.session_state.practice_items)}")
    
    # BROWSER MODE
    else:
        st.subheader(f"📚 Dialogue Browser ({len(display_dialogues)} items)")
        
        browser_key = f"dialog_browser_{selected_category}"
        if browser_key not in st.session_state:
            st.session_state[browser_key] = 0
        
        idx = st.session_state[browser_key]
        idx = max(0, min(idx, len(display_dialogues) - 1))
        st.session_state[browser_key] = idx
        
        row = display_dialogues.iloc[idx]
        
        st.markdown("---")
        st.write(f"**Bulgarian:** {row['Bulgarian']}")
        if pd.notna(row.get('Pronunciation')):
            st.write(f"**Pronunciation:** {row['Pronunciation']}")
        st.write(f"**English:** {row['English']}")
        if pd.notna(row.get('Grammar_Notes')):
            st.write(f"**Notes:** {row['Grammar_Notes']}")
        
        audio_path = tts_audio(row['Bulgarian'])
        if audio_path:
            with open(audio_path, 'rb') as f:
                st.audio(f.read(), format='audio/mp3')
        
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("◀ Previous", disabled=idx == 0, key=f"{browser_key}_prev"):
                st.session_state[browser_key] = idx - 1
                st.rerun()
        with c2:
            st.markdown(f"**{idx + 1} / {len(display_dialogues)}**")
        with c3:
            if st.button("Next ▶", disabled=idx >= len(display_dialogues) - 1, key=f"{browser_key}_next"):
                st.session_state[browser_key] = idx + 1
                st.rerun()
