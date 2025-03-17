with open("src/services/proxy_service.py", "r") as f: lines = f.readlines(); for i in range(len(lines)): if "return None" in lines[i] and i > 340 and i < 360: lines.insert(i+1, "            else:
                # Final progress update for non-duration case
                if actual_progress_placeholder:
                    try:
                        progress_bar.progress(1.0)
                        actual_progress_placeholder.text(
                            \"Proxy video created successfully!\"
                        )
                    except Exception:
                        # Ignore Streamlit context errors
                        pass

"); break; with open("src/services/proxy_service.py", "w") as f: f.writelines(lines)
