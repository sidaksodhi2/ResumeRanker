import os
import json
import pandas as pd 
import numpy as np
from typing import List, Dict, Optional
import re
from datetime import datetime
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import fitz
import docx2txt
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResumeScore:
    resume_id: str
    filename: str
    overall_score: float
    semantic_similarity: float
    keyword_match_score: float
    skill_match_score: float
    experience_match_score: float
    education_match_score: float
    section_scores: Dict[str, float]
    matched_keywords: List[str]
    missing_keywords: List[str]
    extracted_skills: List[str]
    extracted_experience: List[str]
    extracted_education: List[str]
    timestamp: str

class ResumeParser:
    SECTION_SYNONYMS = {
        "experience": r"(experience|work\s+history|employment\s+history|professional\s+experience|career\s+summary)",
        "education": r"(education|academic\s+background|qualifications|academics|educational\s+background)"
    }
    def __init__(self):
        try:
            self.nlp=spacy.load("en_core_web_sm")
        except OSError:
            logger.error("Spacy model 'en_core_web_sm' not found. Please download it using 'python -m spacy download en_core_web_sm'")
            self.nlp = None
    
    def extract_text_from_pdf(self, pdfpath: str) -> str:
        try:
            doc = fitz.open(pdfpath)
            return " ".join([page.get_text() for page in doc])
        except Exception as e:
            logger.error(f"Error reading PDF file {pdfpath}: {e}")
            return ""
    
    def extract_text_from_docx(self, docxpath: str) -> str:
        try:
            return docx2txt.process(docxpath)
        except Exception as e:
            logger.error(f"Error reading DOCX file {docxpath}: {e}")
            return ""
        
    def extract_text_from_txt(self, txtpath: str) -> str:
        try:
            with open(txtpath, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error reading TXT file {txtpath}: {e}")
            return ""
    
    def extract_resume_text(self,filepath: str) -> str:
        ext=os.path.splitext(filepath)[1].lower()
        if(ext=='.pdf'):
            return self.extract_text_from_pdf(filepath)
        elif(ext=='.docx'):
            return self.extract_text_from_docx(filepath)
        elif(ext=='.txt'):
            return self.extract_text_from_txt(filepath)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
    def clean_text(self,text: str) -> str:
        text=re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text=re.sub(r'[^\w\s\-\.,\(\)\[\]\+#]', '', text) # Remove special characters except common punctuation
        return text.strip()
    
    def extract_skills(self, text: str) -> List[str]:
        patterns = [r'(?i)(skills?|technologies|tools|expertise|proficiencies|technical skills?)[:\-\s]*([\w\s,;•\|]+)'
        ]
        skills=[]
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for _, match in matches:
                skills.extend([s.strip() for s in re.split(r'[;,•|]', match) if s.strip()])
        return list(set(skills))  # Remove duplicates
    
    def extract_section(self, text:str, section_name:str) -> str:
     if section_name.lower() in self.SECTION_SYNONYMS:
        section_title_pattern = self.SECTION_SYNONYMS[section_name.lower()]
     else:
        section_title_pattern = re.escape(section_name)

     pattern = re.compile(
        rf"{section_title_pattern}\s*[:\-]?\s*([\s\S]*?)(?=\n\s*[A-Z][A-Za-z\s]{{2,}}:|\Z)",
        re.IGNORECASE
    )
     match = pattern.search(text)
     return match.group(1).strip() if match else ""
 
    def extract_experience(self, text: str) -> str:
        return self.extract_section(text, 'experience')

    def extract_education(self, text: str) -> str:
        return self.extract_section(text, 'education')

class SemanticMatcher:
    def __init__(self,model_name: str = 'all-MiniLM-L6-v2'):
        try:
          self.model = SentenceTransformer(model_name)
          logger.info(f"Loaded Sentence Bert model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading Sentence Bert model {model_name}: {e}")
            self.model = None
    
    def compute_semantic_similarity(self, text1: str, text2: str) -> float:
        if not self.model:
            logger.error("Semantic model not initialized.")
            return 0.0
        try:
            embeddings = self.model.encode([text1,text2],convert_to_tensor=True)
            return float(cosine_similarity([embeddings[0].cpu().numpy()],[embeddings[1].cpu().numpy()])[0][0])
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def calculate_section_similarity(self, resume_sections:Dict[str,str], jd_sections:Dict[str,str]) -> Dict[str,float]:
        section_scores = {}
        for section in['skills', 'experience', 'education']:
              section_scores[section] = self.compute_semantic_similarity(
                resume_sections.get(section, ""), jd_sections.get(section, "")
            )
        return section_scores
    
class KeywordMatcher:
    def calculate_keyword_match(self, resume_text: str, jd_keywords: List[str]) -> Dict[str,any]:
       resume_lower = resume_text.lower()
       matched_keywords = [kw for kw in jd_keywords if kw.lower() in resume_lower]
       missing_keywords = [kw for kw in jd_keywords if kw.lower() not in resume_lower]
       score= len(matched_keywords) / len(jd_keywords) if jd_keywords else 0
       return {
              'matched_keywords': matched_keywords,
              'missing_keywords': missing_keywords,
              'score': score
         }
    
    def calculate_skills_match(seld,resume_skills: List[str], jd_skills: List[str]) -> float:
       resume_skills=[s.lower() for s in resume_skills]
       jd_skills=[s.lower() for s in jd_skills]
       matched=sum(1 for js in jd_skills if any(js in rs or rs in js for rs in resume_skills))
       return matched / len(jd_skills) if jd_skills else 0.0
    
class ResumeScorer:
    def __init__(self):
        self.parser = ResumeParser()
        self.semantic_matcher = SemanticMatcher()
        self.keyword_matcher = KeywordMatcher()
        self.weights={
            'semantic_similarity': 0.35,
            'keyword_match_score': 0.45,
            'skill_match_score': 0.25,
            'experience_match_score': 0.1,
            'education_match_score': 0.05
        }
         # Normalize weights to sum = 1
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}


    def extract_keywords_from_jd(self, jd: str) -> List[str]:
        return re.findall(r'\b[\w\-\+\.]+\b', jd.lower())
   
    def score_resume(self, file_path: str, job_desctiption: str) -> ResumeScore:
      text = self.parser.extract_resume_text(file_path)
      clean = self.parser.clean_text(text)
      skills = self.parser.extract_skills(clean)
      experience = self.parser.extract_experience(clean)
      education = self.parser.extract_education(clean)

      # Prepare section dictionaries
      resume_sections = {
        'skills': ' '.join(skills),
        'experience': experience,
        'education': education
       }
      jd_sections = {
        'skills': job_desctiption,
        'experience': job_desctiption,
        'education': job_desctiption
    }

      section_scores = self.semantic_matcher.calculate_section_similarity(resume_sections, jd_sections)
      semantic_score = sum(section_scores.values()) / len(section_scores) if section_scores else 0.0

      keyword_results = self.keyword_matcher.calculate_keyword_match(clean, job_desctiption.split())
      skills_score = self.keyword_matcher.calculate_skills_match(skills, job_desctiption.split())

      final_score = (
        self.weights['semantic_similarity'] * semantic_score +
        self.weights['keyword_match_score'] * keyword_results['score'] +
        self.weights['skill_match_score'] * skills_score +
        self.weights['experience_match_score'] * section_scores['experience'] +
        self.weights['education_match_score'] * section_scores['education']
    )

      return ResumeScore(
        resume_id=os.path.basename(file_path),
        filename=os.path.basename(file_path),
        overall_score=final_score,
        semantic_similarity=semantic_score,
        keyword_match_score=keyword_results['score'],
        skill_match_score=skills_score,
        experience_match_score=section_scores['experience'],
        education_match_score=section_scores['education'],
        section_scores=section_scores,
        matched_keywords=keyword_results['matched_keywords'],
        missing_keywords=keyword_results['missing_keywords'],
        extracted_skills=skills,
        extracted_experience=experience,
        extracted_education=education,
        timestamp=datetime.now().isoformat()
    )

    
    def score_folder(self, folder: str, jd: str) -> List[ResumeScore]:
        scores = []
        for f in os.listdir(folder):
            path = os.path.join(folder, f)
            if os.path.isfile(path) and f.lower().endswith(('.pdf', '.docx', '.txt')):
                try:
                    score = self.score_resume(path, jd)
                    scores.append(score)
                except Exception as e:
                    logger.error(f"Failed to score {f}: {e}")
        return sorted(scores, key=lambda s: s.overall_score, reverse=True)
    
    def save_scores_to_json(self, scores: List[ResumeScore], output_file: str):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(score) for score in scores], f, indent=4)
        logger.info(f"Scores saved to {output_file}")

    def load_scores_from_json(self, input_file: str) -> List[ResumeScore]:
        if not os.path.exists(input_file):
            logger.error(f"Input file {input_file} does not exist.")
            return []
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [ResumeScore(**item) for item in data]
    
if __name__ == "__main__":
    jd = """React, NodeJS, HTML ,WEbd, ML, data Structures, Cloud"""
    scorer = ResumeScorer()
    scores = scorer.score_folder("resumes", jd)
    for s in scores:
        print(f"{s.filename}: {s.overall_score:.2f} (Matched: {len(s.matched_keywords)})")



    


