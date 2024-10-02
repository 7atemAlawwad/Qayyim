import google.generativeai as genai
import pathlib
import textwrap
from IPython.display import display, Markdown
from google.colab import userdata
import PIL.Image
import os
from google.cloud import vision
import io
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import openai
from io import BytesIO

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

# Upload PDF with traffic laws
traffic_law_pdf = st.file_uploader("تحميل ملف قوانين المرور (PDF)", type="pdf")

# Input descriptions for each party
FirstPartyDescription = st.text_input("وصف الحادث من الطرف الأول:")
SecondPartyDescription = st.text_input("وصف الحادث من الطرف الثاني:")

model = genai.GenerativeModel('gemini-1.5-pro-latest')
if accident_image_file is not None:
    # Load image with PIL
    img = Image.open(accident_image_file)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Define the prompt
    prompt_arabic = """
    انت محقق حوادث مروريه او شرطي مرور , سيتم تزيدك بصوره ان وجدت بها حادث قم بتحديد الطرف الاول على يسار الصوره و الطرف الثاني على يمين الصوره
    واريد منك وصف للحادث وتحديد الاضرار ان وجدت فقط
    وان لم يكن هناك حادث في الصوره قم بكتابة لم يتم العثور على حادث في الصوره
    """

    # Set generation configuration
    generation_config = {
        'temperature': 0.2  # You can adjust temperature based on your needs
    }

    # Generate the response using the Gemini model
    response = model.generate_content([prompt_arabic, img], generation_config=generation_config)
    AccidentDescription = response.text

    # Check and display the accident description
    if AccidentDescription.strip() != "لم يتم العثور على حادث في الصورة":
        st.write("Accident Description:")
        st.write(AccidentDescription)
        st.write(".يرجى ملاحظة أن هذا الوصف يعتمد على الصورة فقط وقد لا يعكس بدقة تفاصيل الحادث ومن المهم جمع معلومات إضافية من اطراف الحادث لمعرفة تفاصيل الحادث بشكل دقيق")
    else:
        st.write("لم يتم العثور على حادث في الصورة")

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

# Set up Google Vision credentials
st.title("Vehicle Registration Extraction")

# Set up Google Cloud credentials (from Streamlit secrets)
google_creds = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]

# Save the credentials to a temporary file for the Vision API
with open("/tmp/google-credentials.json", "w") as f:
    f.write(google_creds)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/tmp/google-credentials.json"

client = vision.ImageAnnotatorClient()

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

# Functions to extract specific fields from the detected text
def create_label_indices(lines):
    label_indices = {}
    label_set = set()
    for idx, line in enumerate(lines):
        line_clean = line.strip(':').strip()
        # Map labels to their indices
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
            # Fallback: look for a line with 17-character alphanumeric string
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

# Streamlit UI for uploading images
st.write("Upload two vehicle registration images to extract details.")

vehicle_reg_image1 = st.file_uploader("Upload first vehicle registration", type=["jpg", "jpeg", "png"])
vehicle_reg_image2 = st.file_uploader("Upload second vehicle registration", type=["jpg", "jpeg", "png"])

if vehicle_reg_image1 and vehicle_reg_image2:
    # Detect and format text for both images
    with st.spinner("Processing images..."):
        detected_text1 = detect_text(vehicle_reg_image1)
        formatted_text1 = format_vehicle_registration_text(detected_text1)

        detected_text2 = detect_text(vehicle_reg_image2)
        formatted_text2 = format_vehicle_registration_text(detected_text2)

    # Display the formatted text for both vehicle registrations
    st.write("### Vehicle Registration 1 Information:")
    st.text(formatted_text1)

    st.write("### Vehicle Registration 2 Information:")
    st.text(formatted_text2)
else:
    st.write("Please upload both vehicle registration images to proceed.")


# URL of the PDF on GitHub
pdf_url = 'https://github.com/7atemAlawwad/Qayyim/blob/main/Traffic_Laws.pdf'

def download_pdf_from_github(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        return BytesIO(response.content)  # Returns file-like object
    else:
        st.error("Failed to download PDF from GitHub.")
        return None

# Load the PDF and process it with LangChain
@st.cache_data
def load_traffic_laws_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Split documents
    text_splitter = TextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    
    # Embeddings
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

retriever = load_traffic_laws_pdf(temp_pdf_path)

# Setup GPT model
llm = OpenAI(model="gpt-4")

# Define the system prompt for RAG processing
system_prompt = (
    "You are an assistant for summarizing and answering questions about traffic laws. "
    "You are expected to analyze and assess fault in traffic accidents based on the provided information. "
    "If the input doesn't relate to the document or you can't determine the answer, respond with 'I don't have any idea'."
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the RAG chain
question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Function to generate the accident report
def generate_accident_report_with_fault(FirstPartyDescription, SecondPartyDescription, AccidentDescription, VehicleRegistration, rag_chain):
    # Retrieve relevant traffic laws or information using RAG
    retrieved_context = rag_chain.invoke({"input": AccidentDescription + FirstPartyDescription + SecondPartyDescription})

    # Create the prompt dynamically by including the vehicle information and accident description
    prompt = (
        f"""
        يوجد حادث مروري لسيارتين:
        وصف الحادث بناءً على الصورة المقدمة: {AccidentDescription}

        تسجيل السيارة الأولى: {VehicleRegistration}
        تسجيل السيارة الثانية: {VehicleRegistration}

        وصف الطرف الأول: {FirstPartyDescription}
        وصف الطرف الثاني: {SecondPartyDescription}

        بناءً على القوانين المرورية والمعلومات التالية التي تم استرجاعها:
        {retrieved_context}
        مع كتابة البند المستخرج منه الحاله

        أريد منك أن تكتب تقريرًا كاملاً عن الحادث، متضمناً:
        - وصف الحادث بالتفصيل بناءً على المعلومات المتاحة.
        - تقييم نسبة الخطأ لكل طرف بناءً على البيانات المتوفرة (النسب المحتملة: [100%, 75%, 50%, 25%, 0%]).
        - تقييم الأضرار المادية لكل سيارة.

        يرجى عدم كتابة توصيات. وفي نهاية التقرير، اكتب أنه "قيد المراجعة".
        """
    )

    # Call the OpenAI API to generate the accident report based on the prompt
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "أنت مساعد في كتابة تقرير حادث مروري"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        temperature=0.1
    )

    # Return the generated report
    return response.choices[0].message['content'].strip()

# Button to generate accident report
if st.button("توليد تقرير الحادث"):
    if accident_image_file and vehicle_reg_image1_file and vehicle_reg_image2_file and traffic_law_pdf:
        # Call RAG-based accident report generation
        with st.spinner("Generating accident report..."):
            try:
                accident_report = generate_accident_report_with_fault(
                    FirstPartyDescription,
                    SecondPartyDescription,
                    AccidentDescription,
                    VehicleRegistration1,
                    VehicleRegistration2,
                    retriever
                )
                # Display the accident report
                st.subheader("تقرير الحادث:")
                st.write(accident_report)
            except Exception as e:
                st.error(f"Error generating report: {e}")
    else:
        st.error("الرجاء تحميل جميع الصور وملف قوانين المرور.")



