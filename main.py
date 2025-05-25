from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import os
from google.cloud import vision
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO
from pdf2image import convert_from_bytes
from PIL import Image
import tempfile
from docx import Document
import re

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"D:\user\source\repos\OCR_Project\google-credentials.json"

app = FastAPI()

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
vision_client = vision.ImageAnnotatorClient()

def preprocess_image_bytes(image_bytes: bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

def extract_text_from_image(image_np: np.ndarray):
    success, encoded_image = cv2.imencode('.jpg', image_np)
    if not success:
        raise ValueError("Image encoding failed.")
    image = vision.Image(content=encoded_image.tobytes())
    response = vision_client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f"Vision API error: {response.error.message}")
    return response.full_text_annotation.text

def extract_text_from_docx(docx_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_docx:
        tmp_docx.write(docx_bytes)
        tmp_docx.flush()
        doc = Document(tmp_docx.name)
        os.unlink(tmp_docx.name)
        return "\n".join([p.text for p in doc.paragraphs])

def extract_text_pages_from_pdf(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    all_text = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format='JPEG')
        image_np = preprocess_image_bytes(buf.getvalue())
        page_text = extract_text_from_image(image_np)
        if not page_text.strip():
            page_text = "[No readable text found on this page.]"
        all_text.append(page_text)
    return all_text

def semantic_similarity(model_answer, student_answer):
    embeddings = semantic_model.encode([model_answer, student_answer], convert_to_tensor=True)
    return util.cos_sim(embeddings[0], embeddings[1]).item()

def tfidf_cosine_similarity(model_answer, student_answer):
    vectors = TfidfVectorizer().fit_transform([model_answer, student_answer]).toarray()
    return cosine_similarity([vectors[0]], [vectors[1]])[0][0]

def extract_answers_by_question(text: str):
    pattern = r"^(Q[^\n]*|\d+[\.\)]?.*)"
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    qa_pairs = {}

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        answer_text = text[start:end].strip()
        question_id = f"Q{i+1}"
        qa_pairs[question_id] = answer_text

    return qa_pairs

def get_letter_grade(score: float) -> str:
    if score >= 90:
        return "A+"
    elif score >= 80:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 50:
        return "D"
    else:
        return "F"

def calculate_final_grade(total_semantic_grade, total_tfidf_grade):
    final_numeric = (total_semantic_grade + total_tfidf_grade) / 2
    return final_numeric

@app.post("/evaluate")
async def evaluate_answer_sheet(
    total_marks: float = Form(...),
    studentId: str = Form(...),
    categoryId: str = Form(...),
    subjectId: str = Form(...),
    student_file: UploadFile = File(...),
    model_file: UploadFile = File(...)
):
    try:
        student_ext = student_file.filename.split(".")[-1].lower()
        model_ext = model_file.filename.split(".")[-1].lower()

        student_content = await student_file.read()
        model_content = await model_file.read()

        if model_ext == "txt":
            model_answer = model_content.decode("utf-8")
        elif model_ext == "docx":
            model_answer = extract_text_from_docx(model_content)
        elif model_ext == "pdf":
            model_answer = "\n".join(extract_text_pages_from_pdf(model_content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported model answer file type.")

        if student_ext == "pdf":
            student_pages = extract_text_pages_from_pdf(student_content)
        elif student_ext in ("jpg", "jpeg", "png"):
            image_np = preprocess_image_bytes(student_content)
            student_pages = [extract_text_from_image(image_np)]
        elif student_ext == "docx":
            student_pages = [extract_text_from_docx(student_content)]
        elif student_ext == "txt":
            student_pages = [student_content.decode("utf-8")]
        else:
            raise HTTPException(status_code=400, detail="Unsupported student file type.")

        combined_student_text = "\n".join(student_pages)

        student_answers = extract_answers_by_question(combined_student_text)
        model_answers = extract_answers_by_question(model_answer)

        question_count = len(model_answers)
        if question_count == 0:
            raise HTTPException(status_code=400, detail="No questions detected in model answer.")

        marks_per_question = total_marks / question_count

        results = []
        total_semantic_score = 0
        total_tfidf_score = 0
        evaluated_questions = 0

        for qid, model_text in model_answers.items():
            student_text = student_answers.get(qid)
            if not student_text:
                continue

            sem_score = semantic_similarity(model_text, student_text)
            tfidf_score = tfidf_cosine_similarity(model_text, student_text)

            semantic_grade = sem_score * marks_per_question
            tfidf_grade = tfidf_score * marks_per_question

            total_semantic_score += semantic_grade
            total_tfidf_score += tfidf_grade
            evaluated_questions += 1

            results.append({
                "question": qid,
                "semantic_score": round(sem_score, 2),
                "semantic_grade": round(semantic_grade, 2),
                "tfidf_score": round(tfidf_score, 2),
                "tfidf_grade": round(tfidf_grade, 2),
                "student_answer": student_text.strip()
            })

        final_numeric_grade = calculate_final_grade(total_semantic_score, total_tfidf_score)
        percentage = (final_numeric_grade / total_marks) * 100 if total_marks > 0 else 0
        final_letter_grade = get_letter_grade(percentage)
        status = "Pass" if percentage >= 40 else "Fail"

        return JSONResponse({
            "studentId": studentId,
            "categoryId": categoryId,
            "subjectId": subjectId,
            "questions_evaluated": evaluated_questions,
            "total_semantic_grade": round(total_semantic_score, 2),
            "total_tfidf_grade": round(total_tfidf_score, 2),
            "final_numeric_grade": round(final_numeric_grade, 2),
            "final_letter_grade": final_letter_grade,
            "status": status,
            "per_question_results": results
        },  headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")
