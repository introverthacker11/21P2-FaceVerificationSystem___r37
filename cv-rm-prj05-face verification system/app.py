import streamlit as st
from deepface import DeepFace
import numpy as np
import cv2
from numpy import dot
from numpy.linalg import norm
from PIL import Image
import os

# 📌 Import metadata
from people_info import people_info

st.title("🛡️ Verification & Face Matching System ⚡")
st.subheader("🤖 Powered by DeepFace (Facenet) — Top Matches Displayed with Similarity > 0.5")
st.subheader("👨‍💻 Developed by: Rayyan Ahmed")

# ---------------- Background ----------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)),
                      url("https://t4.ftcdn.net/jpg/02/87/07/13/360_F_287071353_WXFljgcyA6kHEniBIKCyqRYaviBZTS4p.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
h1 { color: #FFD700; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ==================== Sidebar Styling ====================
st.markdown(
    """
    <style>
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: rgba(0, 0, 65, 0.3);
            color: white;
        }

        /* Sidebar headings */
        [data-testid="stSidebar"] h1,
        h2,
        h3 {
            color: #00BFFF;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar-thumb {
            background: #FFD700;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==================== Sidebar Sections ====================
with st.sidebar.expander("📌 Project Intro"):
    st.markdown(
        """
        ### 🎯 **Project Goal**
        - Upload a **target image** and compare it with a **folder/database of images**.
        - Compute **face embeddings** using Facenet and calculate **cosine similarity**.
        - Identify **Top Matches** using a configurable similarity threshold (e.g., > 0.5).
        - Display **side-by-side comparisons** of matched faces with confidence scores.
        - Show **all database images** for reference before verification.
        - Provide **real-time progress tracking** during folder scanning.
        - Fully interactive UI built with **Streamlit**.
        - 🔍 **Bonus Feature:** Automatically extract **metadata information** (EXIF) from uploaded images (camera model, date, location if available, etc.).
        """
    )

with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown(
        """
        - **Hi, I'm Rayyan Ahmed**  
        - Google Certified **AI Prompt Specialist**  
        - IBM Certified **Advanced LLM FineTuner**  
        - Hugging Face Certified: **Fundamentalist of LLMs**  
        - Expertise in **EDA, ML, RL, ANN, CNN, CV, RNN, NLP, LLMs**  
        - [💼 Visit LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)
        """
    )

with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown(
        """
        - 🧠 **DeepFace (Facenet)** → High-accuracy face verification & embeddings  
        - 🎥 **OpenCV** → Image preprocessing & efficient pixel operations  
        - 🖼️ **Pillow (PIL)** → Image loading, metadata extraction (EXIF), and conversions  
        - ⚙️ **NumPy** → Cosine similarity computation & fast array operations  
        - 🌐 **Streamlit** → Fully interactive web interface  
        - 💾 **Pickle / NumPy Save** → Optional embedding caching for large datasets  
        - 🚀 **Performance Optimizations** → Multi-detector support, batching & resizing
        """
    )

# ====================
# 🎯 HELPER FUNCTIONS
# ====================

def ensure_rgb(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def cosine_similarity(e1, e2):
    return dot(e1, e2) / (norm(e1) * norm(e2))

# Extract key → "putin", "khan", "pablo"
def get_person_key(filename):
    name = os.path.splitext(filename)[0].lower()
    return name.split()[0]   # part before any space or number

def get_embedding_from_array(np_img):
    detectors = ["mtcnn", "retinaface", "opencv"]
    for det in detectors:
        try:
            rep = DeepFace.represent(
                img_path=np_img,
                model_name="Facenet",
                detector_backend=det,
                enforce_detection=False
            )
            return rep[0]["embedding"]
        except:
            pass
    return None


# -------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------

file1 = st.file_uploader("Upload FIRST image (Target Person)", type=["jpg", "jpeg", "png"])

FOLDER = "pics"
x = 1

if file1:
    img1 = np.array(Image.open(file1))
    img1 = ensure_rgb(img1)

    st.image(img1, caption="Target Image", width=250)

    # SHOW ALL IMAGES IN FOLDER
    st.subheader("📁 Images Available in Database")
    folder_images = [f for f in os.listdir(FOLDER) if f.lower().endswith((".jpg",".jpeg",".png"))]

    cols = st.columns(4)
    for idx, filename in enumerate(folder_images):
        img_path = os.path.join(FOLDER, filename)
        img = Image.open(img_path)
        with cols[idx % 4]:
            st.image(img, width=120)
            st.caption(filename)

    # -----------------------------
    # Extract target embedding
    # -----------------------------
    st.subheader("🔍 Extracting Embedding of Target...")
    emb1 = get_embedding_from_array(img1)

    if emb1 is None:
        st.error("Could not extract face embedding from target image!")
    else:
        st.success("Embedding extracted!")

        st.subheader("📂 Scanning Database Folder for Matches...")
        progress_bar = st.progress(0, text="Processing images...")

        results = []
        total_files = len(folder_images)

        for idx, filename in enumerate(folder_images):

            img_path = os.path.join(FOLDER, filename)
            img2 = np.array(Image.open(img_path))
            img2 = ensure_rgb(img2)

            emb2 = get_embedding_from_array(img2)

            if emb2 is not None:
                sim = cosine_similarity(emb1, emb2)
                if sim < 0.999:  
                    results.append((filename, sim, img2))

            progress_bar.progress((idx + 1) / total_files, 
                                  text=f"Processing: {idx+1}/{total_files}")

        # Sort results
        results.sort(key=lambda x: x[1], reverse=True)
        filtered = [r for r in results if r[1] > 0.50]

        st.success("✅ Face Verification Completed")
        st.subheader("🏆 Matches (Similarity > 0.50)")

        if len(filtered) == 0:
            st.error("❌ No match found above 0.50 threshold.")
        else:
            for (filename, score, image) in filtered:

                colA, colB = st.columns(2)

                with colA:
                    st.write("### 🎯 Target Image")
                    st.image(img1, width=180)

                with colB:
                    st.write(f"### 🟢 Matched Image {x}")
                    st.image(image, width=180)
                    x += 1

                st.write(f"**Similarity Score:** {score:.4f}")

                # -------------------------------
                # ⭐ GET & DISPLAY PERSON METADATA
                # -------------------------------
                person_key = get_person_key(filename)
                info = people_info.get(person_key)

                if info:
                    st.write("### 📌 **Person Info**")
                    st.write(f"**🧑 Name:** {info['name']}")
                    st.write(f"**🎂 Date of Birth:** {info['dob']}")
                    st.write(f"**💼 Occupation:** {info['occupation']}")
                    st.write(f"**🌍 Country of Birth:** {info['country_of_birth']}")
                    st.write(f"**🛂 Nationality:** {info['nationality']}")

                else:
                    st.write("No metadata found for this person.")

                st.write("---")
