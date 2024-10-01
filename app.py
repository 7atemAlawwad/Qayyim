import streamlit as st
import os
import io
import re
from PIL import Image
import openai

# For OCR
from google.cloud import vision

# For RAG and PDF processing
from langchain.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set up API keys (ensure they are stored in Streamlit's secrets)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# Streamlit app
st.title("تقرير الحادث المروري")

st.write("الرجاء تحميل الصور وإدخال وصف الحادث.")

# Image uploader for the accident image
accident_image_file = st.file_uploader("تحميل صورة الحادث", type=["jpg", "jpeg", "png"])

# Image uploaders for vehicle registrations
vehicle_reg_image1_file = st.file_uploader("تحميل استمارة تسجيل السيارة الأولى", type=["jpg", "jpeg", "png"])
vehicle_reg_image2_file = st.file_uploader("تحميل استمارة تسجيل السيارة الثانية", type=["jpg", "jpeg", "png"])

# Text inputs for descriptions
FirstPartyDescription = st.text_input("وصف الحادث من الطرف الأول:")
SecondPartyDescription = st.text_input("وصف الحادث من الطرف الثاني:")

# Button to generate report
if st.button("توليد تقرير الحادث"):

    if accident_image_file and vehicle_reg_image1_file and vehicle_reg_image2_file:
        # Display uploaded images
        st.image(accident_image_file, caption='صورة الحادث', use_column_width=True)
        st.image(vehicle_reg_image1_file, caption='تسجيل السيارة الأولى', use_column_width=True)
        st.image(vehicle_reg_image2_file, caption='تسجيل السيارة الثانية', use_column_width=True)

        # Process the accident image to get AccidentDescription
        # Placeholder for accident description
        AccidentDescription = "وصف الحادث بناءً على الصورة المقدمة."

        # Function to detect text using Google Vision API
        def detect_text(image_file):
            """Detects text in the uploaded image file."""
            client = vision.ImageAnnotatorClient()
            content = image_file.read()
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            texts = response.text_annotations

            if response.error.message:
                raise Exception(f'{response.error.message}')

            full_text = texts[0].description if texts else ''
            return full_text

        # Process vehicle registration images using OCR
        def extract_vehicle_registration_info(image_file):
            detected_text = detect_text(image_file)
            # Here you should include your extraction logic
            # For the sake of example, we'll return the detected text
            return detected_text

        VehicleRegistration1 = extract_vehicle_registration_info(vehicle_reg_image1_file)
        VehicleRegistration2 = extract_vehicle_registration_info(vehicle_reg_image2_file)

        # Display extracted registration info
        st.write("**معلومات تسجيل السيارة الأولى:**")
        st.write(VehicleRegistration1)
        st.write("**معلومات تسجيل السيارة الثانية:**")
        st.write(VehicleRegistration2)

        # Load and process the traffic laws PDF for RAG
        @st.cache_data
        def load_traffic_laws_pdf():
            file_path = "Traffic_Laws.pdf"  # Update with your PDF file path
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            retriever = vectorstore.as_retriever()
            return retriever

        retriever = load_traffic_laws_pdf()

        # Generate accident report using OpenAI GPT-4
        def generate_accident_report_with_fault(FirstPartyDescription, SecondPartyDescription, AccidentDescription, VehicleRegistration1, VehicleRegistration2):
            prompt = (
                f"""
يوجد حادث مروري لسيارتين:
وصف الحادث بناءً على الصورة المقدمة: {AccidentDescription}

تسجيل السيارة الأولى: {VehicleRegistration1}
تسجيل السيارة الثانية: {VehicleRegistration2}

وصف الطرف الأول: {FirstPartyDescription}
وصف الطرف الثاني: {SecondPartyDescription}

اكتب تقريرًا كاملاً عن الحادث، متضمنًا:
- وصف الحادث بالتفصيل بناءً على المعلومات المتاحة.
- تقييم نسبة الخطأ لكل طرف بناءً على البيانات المتوفرة (النسب المحتملة: [100%, 75%, 50%, 25%, 0%]).
- تقييم الأضرار المادية لكل سيارة.

يرجى عدم كتابة توصيات. وفي نهاية التقرير، اكتب أنه "قيد المراجعة".
                """
            )

            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "أنت مساعد في كتابة تقرير حادث مروري"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.1
            )

            return response.choices[0].message['content'].strip()

        # Generate the accident report
        with st.spinner('جاري توليد التقرير...'):
            accident_report = generate_accident_report_with_fault(
                FirstPartyDescription,
                SecondPartyDescription,
                AccidentDescription,
                VehicleRegistration1,
                VehicleRegistration2
            )

        # Display the accident report
        st.write("### تقرير الحادث:")
        st.write(accident_report)

    else:
        st.error("الرجاء تحميل جميع الصور المطلوبة وإدخال الوصف.")
