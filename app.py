import streamlit as st
from resume_ranker import ResumeScorer
import os
import tempfile

st.set_page_config(page_title="Resume Ranker", page_icon=":briefcase:", layout="centered")
st.title("📄 Resume Ranker")

jd_text = st.text_area("📝 Paste the Job Description", height=200)

# Input: Upload resumes
uploaded_files = st.file_uploader("📤 Upload Resumes (.pdf, .docx, .txt)", accept_multiple_files=True)

if st.button("🔍 Rank Resumes") and jd_text and uploaded_files:
    scorer = ResumeScorer()

    # Save resumes to temp folder
    with tempfile.TemporaryDirectory() as temp_dir:
        file_paths = []
        for uploaded_file in uploaded_files:
            path = os.path.join(temp_dir, uploaded_file.name)
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(path)

        # Score resumes
        scores = []
        for path in file_paths:
            try:
                result = scorer.score_resume(path, jd_text)
                scores.append(result)
            except Exception as e:
                st.error(f"Failed to process {os.path.basename(path)}: {e}")

        # Display results
        scores = sorted(scores, key=lambda x: x.overall_score, reverse=True)
        st.success(f"Ranked {len(scores)} resume(s):")

        for i, s in enumerate(scores, 1):
            st.markdown(f"""
            **{i}. {s.filename}**  
            🔹 Score: `{s.overall_score:.2f}`  
            ✅ Matched Keywords: {len(s.matched_keywords)}  
            💬 Semantic Similarity: `{s.semantic_similarity:.2f}`  
            🧠 Skills Score: `{s.skill_match_score:.2f}`
            """)

else:
    st.info("Please paste a JD and upload at least one resume.")
