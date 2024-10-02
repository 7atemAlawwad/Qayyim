import os
import io
import re
import streamlit as st
from PIL import Image
import google.generativeai as genai
from google.cloud import vision

# Set up API keys from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
google_creds = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# Save the Google credentials to a temporary file
with open("/tmp/google-credentials.json", "w") as f:
    f.write(google_creds)

# Set the environment variable for Google Cloud
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/google-credentials.json"

# Initialize Google Cloud Vision Client
client = vision.ImageAnnotatorClient()

# Streamlit app UI
st.title("تقرير الحادث المروري")

st.write("الرجاء تحميل الصور وإدخال وصف الحادث.")

# Upload accident image
accident_image_file = st.file_uploader("تحميل صورة الحادث", type=["jpg", "jpeg", "png"])

# Upload vehicle registration images
vehicle_reg_image1_file = st.file_uploader("تحميل استمارة تسجيل السيارة الأولى", type=["jpg", "jpeg", "png"])
vehicle_reg_image2_file = st.file_uploader("تحميل استمارة تسجيل السيارة الثانية", type=["jpg", "jpeg", "png"])

# Input descriptions for each party
FirstPartyDescription = st.text_input("وصف الحادث من الطرف الأول:")
SecondPartyDescription = st.text_input("وصف الحادث من الطرف الثاني:")

# Function to detect text using Google Vision API
def detect_text(image_file):
    """Detects text in the uploaded image file."""
    content = image_file.read()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    full_text = texts[0].description if texts else ''
    return full_text

# Function to process vehicle registration images using OCR
def format_vehicle_registration_text(detected_text):
    lines = [line.strip() for line in detected_text.split('\n') if line.strip()]
    label_indices, label_set = create_label_indices(lines)

    output_lines = [
        extract_owner_name(lines, label_indices),
        extract_owner_id(lines, label_indices),
        extract_chassis_number(lines, label_indices),
        extract_plate_number(lines, label_indices, label_set),
        extract_vehicle_brand(lines, label_indices),
        extract_vehicle_weight(lines, label_indices),
        extract_registration_type(lines, label_indices),
        extract_vehicle_color(lines, label_indices),
        extract_year_of_manufacture(lines, label_indices)
    ]

    return "\n".join(output_lines)

# Helper functions to extract fields
def create_label_indices(lines):
    label_indices = {}
    label_set = set()
    labels = [
        'المالك', 'هوية المالك', 'رقم الهوية', 'رقم الهيكل', 'رقم اللوحة',
        'ماركة المركبة', 'الماركة', 'الوزن', 'نوع التسجيل', 'طراز المركبة',
        'الموديل', 'حمولة المركبة', 'اللون', 'سنة الصنع', 'اللون الأساسي'
    ]
    for idx, line in enumerate(lines):
        for label in labels:
            if label in line.strip(':').strip():
                label_indices[label] = idx
                label_set.add(label)
    return label_indices, label_set

def extract_owner_name(lines, label_indices):
    label = 'المالك'
    if label in label_indices:
        return f"المالك: {lines[label_indices[label]].split(':')[-1].strip()}"
    return "المالك: غير متوفر"

def extract_owner_id(lines, label_indices):
    label = 'هوية المالك'
    if label in label_indices:
        return f"هوية المالك: {lines[label_indices[label]].split(':')[-1].strip()}"
    return "هوية المالك: غير متوفر"

def extract_chassis_number(lines, label_indices):
    label = 'رقم الهيكل'
    if label in label_indices:
        return f"رقم الهيكل: {lines[label_indices[label]].split(':')[-1].strip()}"
    return "رقم الهيكل: غير متوفر"

def extract_plate_number(lines, label_indices, label_set):
    label = 'رقم اللوحة'
    if label in label_indices:
        return f"رقم اللوحة: {lines[label_indices[label]].split(':')[-1].strip()}"
    return "رقم اللوحة: غير متوفر"

def extract_vehicle_brand(lines, label_indices):
    label = 'ماركة المركبة'
    if label in label_indices:
        return f"ماركة المركبة: {lines[label_indices[label]].split(':')[-1].strip()}"
    return "ماركة المركبة: غير متوفر"

def extract_vehicle_weight(lines, label_indices):
    label = 'الوزن'
    if label in label_indices:
        return f"وزن المركبة: {lines[label_indices[label]].split(':')[-1].strip()}"
    return "وزن المركبة: غير متوفر"

def extract_registration_type(lines, label_indices):
    label = 'نوع التسجيل'
    if label in label_indices:
        return f"نوع التسجيل: {lines[label_indices[label]].split(':')[-1].strip()}"
    return "نوع التسجيل: غير متوفر"

def extract_vehicle_color(lines, label_indices):
    label = 'اللون'
    if label in label_indices:
        return f"اللون: {lines[label_indices[label]].split(':')[-1].strip()}"
    return "اللون: غير متوفر"

def extract_year_of_manufacture(lines, label_indices):
    label = 'سنة الصنع'
    if label in label_indices:
        return f"سنة الصنع: {lines[label_indices[label]].split(':')[-1].strip()}"
    return "سنة الصنع: غير متوفر"

# Function to process two vehicle registration images
def process_vehicle_registrations(image1, image2):
    reg1 = detect_text(image1)
    reg2 = detect_text(image2)
    formatted_text1 = format_vehicle_registration_text(reg1)
    formatted_text2 = format_vehicle_registration_text(reg2)

    return formatted_text1, formatted_text2

# Function to generate accident description using Gemini
def generate_accident_description(img_file):
    img = Image.open(img_file)
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    prompt_arabic = """
    انت محقق حوادث مروريه او شرطي مرور , سيتم تزويدك بصوره ان وجدت بها حادث قم بتحديد الطرف الاول على يسار الصوره والطرف الثاني على يمين الصوره.
    واريد منك وصف للحادث وتحديد الاضرار ان وجدت فقط. وإن لم يكن هناك حادث في الصوره قم بكتابة 'لم يتم العثور على حادث في الصوره'.
    """
    generation_config = {
        'temperature': 0.2  # Adjust this based on your needs
    }

    response = model.generate_content([prompt_arabic, img], generation_config=generation_config)
    return response.text

# Function to generate an accident report using OpenAI GPT-4
def generate_accident_report(FirstPartyDescription, SecondPartyDescription, AccidentDescription, VehicleRegistration1, VehicleRegistration2):
    prompt = f"""
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

# Button to generate accident report
if st.button("توليد تقرير الحادث"):
    if accident_image_file and vehicle_reg_image1_file and vehicle_reg_image2_file:
        # Process the accident image
        with st.spinner('جاري توليد وصف الحادث...'):
            AccidentDescription = generate_accident_description(accident_image_file)

        # Process vehicle registration images
        with st.spinner('جاري استخراج معلومات تسجيل السيارة...'):
            VehicleRegistration1, VehicleRegistration2 = process_vehicle_registrations(vehicle_reg_image1_file, vehicle_reg_image2_file)

        # Display registration info
        st.write("### معلومات تسجيل السيارة:")
        st.write(VehicleRegistration1)
        st.write(VehicleRegistration2)

        # Generate accident report
        with st.spinner('جاري توليد التقرير...'):
            accident_report = generate_accident_report(
                FirstPartyDescription,
                SecondPartyDescription,
                AccidentDescription,
                VehicleRegistration1,
                VehicleRegistration2
            )

        st.write("### تقرير الحادث:")
        st.write(accident_report)
    else:
        st.error("الرجاء تحميل جميع الصور المطلوبة وإدخال الوصف.")
