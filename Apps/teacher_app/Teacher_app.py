def practice_words_section():
    """Word practice section for teachers."""
    st.header("📝 Word Practice")
    
    if df_words.empty:
        st.info("No words available. Add some content first!")
        return
    
    # Filter options
    categories = df_words["Category"].dropna().unique().tolist()
    selected_category = st.selectbox("Filter by Category:", ["All"] + categories)
    
    # Filter words
    display_words = df_words.copy()
    if selected_category != "All":
        display_words = display_words[display_words["Category"] == selected_category]
    
    if display_words.empty:
        st.info("No words in selected category.")
        return
    
    # Practice mode toggle
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎯 Start Practice Session", use_container_width=True):
            st.session_state.practice_mode = True
            st.session_state.practice_items = display_words.to_dict('records')
            st.session_state.practice_index = 0
            st.rerun()
    with col2:
        if st.button("📊 View Single Word Browser", use_container_width=True):
            st.session_state.practice_mode = False
            st.rerun()
    
    # PRACTICE MODE (unchanged behaviour)
    if st.session_state.practice_mode and st.session_state.practice_items:
        item = st.session_state.practice_items[st.session_state.practice_index]
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("Bulgarian Word")
            st.markdown(f"### {item['Bulgarian']}")
            
            if pd.notna(item.get('Pronunciation')):
                st.write(f"*({item['Pronunciation']})*")
            
            audio_path = tts_audio(item['Bulgarian'])
            if audio_path:
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format='audio/mp3')
        
        with col_right:
            st.subheader("Your Turn")
            answer = st.text_input("Type the English translation:", key="practice_answer")
            
            if st.button("Check Answer", use_container_width=True):
                if answer.lower().strip() == item['English'].lower().strip():
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Correct answer: {item['English']}")
                
                if pd.notna(item.get('Grammar_Notes')):
                    st.info(f"📝 Note: {item['Grammar_Notes']}")
        
        st.markdown("---")
        col_prev, col_next, col_finish = st.columns([1, 1, 2])
        
        with col_prev:
            if st.button("⬅️ Previous", disabled=st.session_state.practice_index == 0):
                st.session_state.practice_index -= 1
                st.rerun()
        
        with col_next:
            if st.button("Next ➡️", disabled=st.session_state.practice_index >= len(st.session_state.practice_items) - 1):
                st.session_state.practice_index += 1
                st.rerun()
        
        with col_finish:
            if st.button("Finish Practice", type="primary", use_container_width=True):
                st.session_state.practice_mode = False
                st.success("Practice session completed!")
                st.rerun()
        
        progress = (st.session_state.practice_index + 1) / len(st.session_state.practice_items)
        st.progress(progress)
        st.caption(f"Word {st.session_state.practice_index + 1} of {len(st.session_state.practice_items)}")
    
    # SINGLE-WORD BROWSER (replaces giant list)
    else:
        st.subheader(f"📚 Word Browser ({len(display_words)} words)")
        
        # Use a separate index for browsing so it doesn't clash with practice mode
        browser_key = f"words_browser_{selected_category}"
        if browser_key not in st.session_state:
            st.session_state[browser_key] = 0
        
        idx = st.session_state[browser_key]
        idx = max(0, min(idx, len(display_words) - 1))
        st.session_state[browser_key] = idx
        
        row = display_words.iloc[idx]
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Bulgarian:** {row['Bulgarian']}")
            if pd.notna(row.get('Pronunciation')):
                st.write(f"**Pronunciation:** {row['Pronunciation']}")
        with col2:
            st.write(f"**English:** {row['English']}")
            if pd.notna(row.get('Grammar_Notes')):
                st.write(f"**Notes:** {row['Grammar_Notes']}")
        
        audio_path = tts_audio(row['Bulgarian'])
        if audio_path:
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            st.audio(audio_bytes, format='audio/mp3')
        
        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("◀ Previous word", disabled=idx == 0, key=f"{browser_key}_prev"):
                st.session_state[browser_key] = idx - 1
                st.rerun()
        with c2:
            st.markdown(f"**{idx + 1} / {len(display_words)}**")
        with c3:
            if st.button("Next word ▶", disabled=idx >= len(display_words) - 1, key=f"{browser_key}_next"):
                st.session_state[browser_key] = idx + 1
                st.rerun()
