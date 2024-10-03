import requests
import google.generativeai as genai
import pathlib
import textwrap
import PIL.Image
import os
from google.cloud import vision
import re
from langchain.document_loaders import CSVLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.prompts import ChatPromptTemplate
import openai
from io import BytesIO
import streamlit as st
import pandas as pd

# Set API keys
openai.api_key = st.secrets["OPENAI_API_KEY"]
google_creds = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
genai.configure(api_key=st.secrets["GOOGLE_GENERATIVE_AI_API_KEY"])

# Save Google credentials to a temporary file for Vision API
with open("/tmp/google-credentials.json", "w") as f:
    f.write(google_creds)
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

# Process accident image
if accident_image_file is not None:
    # Load image with PIL
    img = PIL.Image.open(accident_image_file)

    # Display the uploaded image
    st.image(img, caption="صورة الحادث", use_column_width=True)

    # Define the prompt
    prompt_arabic = """
    انت محقق حوادث مرورية أو شرطي مرور. سيتم تزويدك بصورة؛ إذا وجدت بها حادث، قم بتحديد الطرف الأول على يسار الصورة والطرف الثاني على يمين الصورة.
    أريد منك وصفًا للحادث وتحديد الأضرار إن وجدت فقط.
    وإن لم يكن هناك حادث في الصورة، قم بكتابة "لم يتم العثور على حادث في الصورة".
    """

    # Generate the response using the Gemini model
    # Note: Ensure that the model supports image input if applicable
    response = genai.generate_text(
        prompt=prompt_arabic,
        max_tokens=500,
        temperature=0.2
    )
    AccidentDescription = response.result

    # Check and display the accident description
    if AccidentDescription.strip() != "لم يتم العثور على حادث في الصورة":
        st.write("وصف الحادث:")
        st.write(AccidentDescription)
        st.write("يرجى ملاحظة أن هذا الوصف يعتمد على الصورة فقط وقد لا يعكس بدقة تفاصيل الحادث. من المهم جمع معلومات إضافية من أطراف الحادث لمعرفة تفاصيل الحادث بشكل دقيق.")
    else:
        st.write("لم يتم العثور على حادث في الصورة.")

# Function to detect text using Google Vision API
def detect_text(image_file):
    content = image_file.getvalue()
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations

    if response.error.message:
        raise Exception(f'{response.error.message}')

    full_text = texts[0].description if texts else ''
    return full_text

# Functions to extract specific fields from the detected text
def create_label_indices(lines):
    label_indices = {}
    label_set = set()
    for idx, line in enumerate(lines):
        line_clean = line.strip(':').strip()
        labels = [
            'المالك', 'هوية المالك', 'رقم الهوية', 'رقم الهيكل', 'رقم اللوحة',
            'ماركة المركبة', 'الماركة', 'الوزن', 'نوع التسجيل', 'طراز المركبة',
            'الموديل', 'حمولة المركبة', 'اللون', 'سنة الصنع', 'اللون الأساسي'
        ]
        for label in labels:
            if label in line_clean:
                label_indices[label] = idx
                label_set.add(label)
    return label_indices, label_set

def extract_owner_name(lines, label_indices):
    label = 'المالك'
    if label in label_indices:
        idx = label_indices[label]
        name = lines[idx].split(':')[-1].strip()
        if not name and idx + 1 < len(lines):
            name = lines[idx + 1].strip()
        if name:
            return f"المالك: {name}"
    return "المالك: غير متوفر"

def extract_owner_id(lines, label_indices):
    labels = ['هوية المالك', 'رقم الهوية', 'رقم السجل']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            id_line = lines[idx]
            owner_id = re.search(r'\b\d{10}\b', id_line)
            if not owner_id and idx + 1 < len(lines):
                id_line_next = lines[idx + 1]
                owner_id = re.search(r'\b\d{10}\b', id_line_next)
            if owner_id:
                return f"هوية المالك: {owner_id.group()}"
    return "هوية المالك: غير متوفر"

def extract_chassis_number(lines, label_indices):
    labels = ['رقم الهيكل']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            chassis_line = lines[idx]
            match = re.search(r'رقم الهيكل[:\s]*(\S+)', chassis_line)
            if match:
                chassis_number = match.group(1)
                return f"رقم الهيكل: {chassis_number}"
            elif idx + 1 < len(lines):
                chassis_number = lines[idx + 1].strip()
                return f"رقم الهيكل: {chassis_number}"
        else:
            for line in lines:
                if re.match(r'^[A-HJ-NPR-Z0-9]{17}$', line):
                    return f"رقم الهيكل: {line.strip()}"
    return "رقم الهيكل: غير متوفر"

def extract_plate_number(lines, label_indices, label_set):
    labels = ['رقم اللوحة']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            plate_line = lines[idx]
            match = re.search(r'رقم اللوحة[:\s]*(.*?)(?:رقم|$)', plate_line)
            if match:
                plate_info = match.group(1).strip()
                plate_info = re.split(r'\s*رقم', plate_info)[0].strip()
                return f"رقم اللوحة: {plate_info}"
    return "رقم اللوحة: غير متوفر"

def extract_vehicle_brand(lines, label_indices):
    labels = ['ماركة المركبة', 'الماركة']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            brand = lines[idx].split(':')[-1].strip()
            return f"ماركة المركبة: {brand}"
    return "ماركة المركبة: غير متوفر"

def extract_vehicle_weight(lines, label_indices):
    label = 'الوزن'
    if label in label_indices:
        idx = label_indices[label]
        weight_match = re.search(r'الوزن[:\s]*(\d+)', lines[idx])
        if weight_match:
            return f"وزن المركبة: {weight_match.group(1)}"
    return "وزن المركبة: غير متوفر"

def extract_registration_type(lines, label_indices):
    label = 'نوع التسجيل'
    if label in label_indices:
        idx = label_indices[label]
        reg_type = lines[idx].split(':')[-1].strip()
        return f"نوع التسجيل: {reg_type}"
    return "نوع التسجيل: غير متوفر"

def extract_vehicle_color(lines, label_indices):
    labels = ['اللون', 'اللون الأساسي']
    for label in labels:
        if label in label_indices:
            idx = label_indices[label]
            color = lines[idx].split(':')[-1].strip()
            return f"اللون: {color}"
    return "اللون: غير متوفر"

def extract_year_of_manufacture(lines, label_indices):
    label = 'سنة الصنع'
    if label in label_indices:
        idx = label_indices[label]
        year_match = re.search(r'سنة الصنع[:\s]*(\d{4})', lines[idx])
        if year_match:
            return f"سنة الصنع: {year_match.group(1)}"
    return "سنة الصنع: غير متوفر"

# Main function to process and format the vehicle registration text
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

# Process vehicle registration images
if vehicle_reg_image1_file and vehicle_reg_image2_file:
    # Detect and format text for both images
    with st.spinner("Processing vehicle registration images..."):
        detected_text1 = detect_text(vehicle_reg_image1_file)
        formatted_text1 = format_vehicle_registration_text(detected_text1)

        detected_text2 = detect_text(vehicle_reg_image2_file)
        formatted_text2 = format_vehicle_registration_text(detected_text2)

    # Display the formatted text for both vehicle registrations
    st.write("### معلومات تسجيل السيارة الأولى:")
    st.text(formatted_text1)

    st.write("### معلومات تسجيل السيارة الثانية:")
    st.text(formatted_text2)
else:
    st.write("يرجى تحميل صور تسجيل المركبتين لاستخراج التفاصيل.")

# URL of the CSV file on GitHub
csv_url = 'https://raw.githubusercontent.com/7atemAlawwad/Qayyim/main/Traffic_Laws.csv'

# Function to download CSV from GitHub
def download_csv_from_github(csv_url):
    response = requests.get(csv_url)
    if response.status_code == 200:
        return BytesIO(response.content)  # Returns file-like object
    else:
        st.error("Failed to download CSV from GitHub.")
        return None

# Load the CSV file and process it with LangChain
@st.cache_data
def load_traffic_laws_csv(csv_url):
    csv_file = download_csv_from_github(csv_url)
    if csv_file is None:
        st.error("Could not download the CSV file.")
        return None
    df = pd.read_csv(csv_file)
    text_data = df.to_csv(index=False)
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_text(text_data)
    
    # Create documents from splits
    from langchain.schema import Document
    docs = [Document(page_content=chunk) for chunk in splits]
    
    # Embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

# Load retriever
retriever = load_traffic_laws_csv(csv_url)
if retriever is None:
    st.error("Unable to load traffic laws data.")
else:
    # Setup GPT model
    llm = OpenAI(model_name="gpt-4")

    # Define the system prompt for RAG processing
    system_prompt = (
        "أنت مساعد متخصص في تلخيص والإجابة على الأسئلة المتعلقة بقوانين المرور. "
        "من المتوقع أن تحلل وتقيّم الخطأ في الحوادث المرورية بناءً على المعلومات المقدمة. "
        "إذا كان الإدخال لا يتعلق بالوثيقة أو لا يمكنك تحديد الإجابة، فقم بالرد بـ 'ليس لدي أي فكرة'."
    )

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Create the RAG chain using ConversationalRetrievalChain
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=prompt_template,
        return_source_documents=False
    )

    # Function to generate the accident report
    def generate_accident_report_with_fault(FirstPartyDescription, SecondPartyDescription, AccidentDescription, VehicleRegistration1, VehicleRegistration2, retriever):
        # Retrieve relevant traffic laws or information using RAG
        query = AccidentDescription + FirstPartyDescription + SecondPartyDescription
        retrieved_context = rag_chain({'question': query})['answer']

        # Create the prompt dynamically by including the vehicle information and accident description
        prompt = (
            f"""
            يوجد حادث مروري لسيارتين:
            وصف الحادث بناءً على الصورة المقدمة: {AccidentDescription}

            تسجيل السيارة الأولى:
            {VehicleRegistration1}

            تسجيل السيارة الثانية:
            {VehicleRegistration2}

            وصف الطرف الأول: {FirstPartyDescription}
            وصف الطرف الثاني: {SecondPartyDescription}

            بناءً على القوانين المرورية والمعلومات التالية التي تم استرجاعها:
            {retrieved_context}
            مع كتابة البند المستخرج منه الحالة.

            أريد منك أن تكتب تقريرًا كاملاً عن الحادث، متضمنًا:
            - وصف الحادث بالتفصيل بناءً على المعلومات المتاحة.
            - تقييم نسبة الخطأ لكل طرف بناءً على البيانات المتوفرة (النسب المحتملة: [100%, 75%, 50%, 25%, 0%]).
            - تقييم الأضرار المادية لكل سيارة.

            يرجى عدم كتابة توصيات. وفي نهاية التقرير، اكتب أنه "قيد المراجعة".
            """
        )

        # Call the OpenAI API to generate the accident report based on the prompt
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "أنت مساعد في كتابة تقرير حادث مروري."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.1
        )

        # Return the generated report
        return response.choices[0].message['content'].strip()

    # Button to generate accident report
    if st.button("توليد تقرير الحادث"):
        if (
            accident_image_file and
            vehicle_reg_image1_file and
            vehicle_reg_image2_file and
            retriever is not None
        ):
            # Call RAG-based accident report generation
            with st.spinner("جاري توليد تقرير الحادث..."):
                try:
                    accident_report = generate_accident_report_with_fault(
                        FirstPartyDescription,
                        SecondPartyDescription,
                        AccidentDescription,
                        formatted_text1,
                        formatted_text2,
                        retriever
                    )
                    # Display the accident report
                    st.subheader("تقرير الحادث:")
                    st.write(accident_report)
                except Exception as e:
                    st.error(f"حدث خطأ أثناء توليد التقرير: {e}")
        else:
            st.error("الرجاء تحميل جميع الصور.")
