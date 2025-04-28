import streamlit as st
import pickle
import docx  # For extracting text from Word files
import PyPDF2  # For extracting text from PDF files
import re
import nltk

# Download NLTK data (only runs once)
nltk.download('punkt')
nltk.download('stopwords')

# Load trained models and tools
with open('clf.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

with open('tfidf.pkl', 'rb') as tfidf_file:
    tfidf = pickle.load(tfidf_file)

with open('encoder.pkl', 'rb') as enc_file:
    le = pickle.load(enc_file)

# Category ID to Name mapping
category_mapping = {
    15: "Java Developer", 23: "Testing", 8: "DevOps Engineer", 20: "Python Developer",
    24: "Web Designing", 12: "HR", 13: "Hadoop", 3: "Blockchain",
    10: "ETL Developer", 18: "Operations Manager", 6: "Data Science", 22: "Sales",
    16: "Mechanical Engineer", 1: "Arts", 7: "Database", 11: "Electrical Engineering",
    14: "Health and fitness", 19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
    2: "Automation Testing", 17: "Network Security Engineer", 21: "SAP Developer",
    5: "Civil Engineer", 0: "Advocate"
}

# Clean and preprocess resume text
def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"RT|cc", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", text)
    text = re.sub(r"[^\x00-\x7f]", r" ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# File type-based text extraction
def extract_text(file, file_ext):
    if file_ext == "pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() or "" for page in reader.pages])
    elif file_ext == "docx":
        doc = docx.Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    elif file_ext == "txt":
        try:
            return file.read().decode('utf-8')
        except UnicodeDecodeError:
            return file.read().decode('latin-1')
    else:
        return ""

# Predict job category
def predict_resume_category(text):
    cleaned = clean_resume(text)
    vectorized = tfidf.transform([cleaned])
    prediction = clf.predict(vectorized)[0]
    return category_mapping.get(prediction, "Unknown")

# Streamlit UI
def main():
    st.set_page_config(page_title="Resume Category Prediction", page_icon="📄", layout="wide")
    st.title("📄 Resume Category Prediction")
    st.markdown("Upload a resume (PDF, DOCX, or TXT) to see the predicted job category.")

    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        resume_text = extract_text(uploaded_file, file_ext)

        if not resume_text.strip():
            st.warning("⚠️ The uploaded file appears to be empty or unreadable.")
            return

        st.success("✅ Resume text extracted successfully.")

        if st.checkbox("Show extracted text"):
            st.text_area("Extracted Resume Text", resume_text, height=300)

        predicted_category = predict_resume_category(resume_text)
        st.subheader("Predicted Category")
        st.write(f"🔍 **{predicted_category}**")

if __name__ == "__main__":
    main()

