import os
import smtplib
from typing import TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # Changed
from langgraph.graph import StateGraph, END
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# Cargar variables de entorno
load_dotenv()

# --- DEFINICIÓN DEL ESTADO DEL GRAFO ---
class AppState(TypedDict):
    """Define el estado que se pasará entre los nodos del grafo."""
    question: str
    email: str
    intention: str
    answer: str
    error: str

# --- FUNCIONES AUXILIARES ---
def create_pdf(title: str, content: str, filename: str) -> str:
    """Crea un archivo PDF con un título y contenido."""
    try:
        doc = SimpleDocTemplate(filename)
        styles = getSampleStyleSheet()
        story = []

        # Título
        story.append(Paragraph(title, styles['h1']))
        story.append(Spacer(1, 0.2*inch))

        # Contenido
        # Reemplazar saltos de línea para que se muestren correctamente en el PDF
        formatted_content = content.replace('\n', '<br/>')
        story.append(Paragraph(formatted_content, styles['BodyText']))

        doc.build(story)
        print(f"PDF '{filename}' creado exitosamente.")
        return filename
    except Exception as e:
        print(f"[ERROR] No se pudo crear el PDF: {e}")
        return None

def send_email_with_attachment(recipient: str, subject: str, body: str, attachment_path: str):
    """Envía un correo electrónico con un archivo adjunto."""
    sender_email = os.getenv("EMAIL_HOST_USER")
    password = os.getenv("EMAIL_HOST_PASSWORD")
    smtp_server = os.getenv("EMAIL_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("EMAIL_PORT", 587))

    if not all([sender_email, password]) or sender_email == "tu_correo@gmail.com":
        print("\n[ERROR] Credenciales de correo no configuradas en .env.")
        return

    # Crear el mensaje
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain", "utf-8"))

    # Adjuntar el archivo PDF
    try:
        with open(attachment_path, "rb") as attachment:
            part = MIMEApplication(attachment.read(), Name=os.path.basename(attachment_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(attachment_path)}"'
        message.attach(part)
    except FileNotFoundError:
        print(f"[ERROR] Archivo adjunto no encontrado en: {attachment_path}")
        return
    except Exception as e:
        print(f"[ERROR] No se pudo adjuntar el archivo: {e}")
        return

    # Enviar el correo
    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, recipient, message.as_string())
        print(f"Correo enviado exitosamente a {recipient}.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un error al enviar el correo: {e}")

# --- NODOS DEL GRAFO ---
github_token = os.getenv("GITHUB_TOKEN") # Changed
if not github_token:
    raise ValueError("GITHUB_TOKEN no encontrado en el archivo .env")

llm = ChatOpenAI(
    model="openai/gpt-4o",
    base_url="https://models.github.ai/inference",
    api_key=github_token
) # Changed

def get_user_data(state: AppState) -> AppState:
    """Nodo para solicitar los datos iniciales al usuario."""
    print("--- Asistente de IA v2 (con LangGraph y PDF) ---")
    question = input("Por favor, introduce tu pregunta: ")
    email = input("Ahora, introduce tu correo para recibir la respuesta: ")
    return {"question": question, "email": email}

def analyze_intention(state: AppState) -> AppState:
    """Nodo que analiza la intención de la pregunta del usuario."""
    print("Analizando intención de la pregunta...")
    prompt = f"Analiza y resume en una sola frase la intención principal de la siguiente pregunta: '{state['question']}'"
    try:
        response = llm.invoke(prompt)
        intention = response.content
        print(f"Intención detectada: {intention}")
        return {"intention": intention}
    except Exception as e:
        return {"error": f"Error analizando la intención: {e}"}

def generate_answer(state: AppState) -> AppState:
    """Nodo que genera la respuesta a la pregunta del usuario."""
    print("Generando respuesta principal...")
    try:
        response = llm.invoke(state['question'])
        answer = response.content
        print("Respuesta generada.")
        return {"answer": answer}
    except Exception as e:
        return {"error": f"Error generando la respuesta: {e}"}

def process_outputs(state: AppState) -> AppState:
    """Nodo final que crea el PDF y envía el correo."""
    print("Procesando salidas finales...")
    if state.get("error"):
        print(f"\n[PROCESO INTERRUMPIDO] {state['error']}")
        return {{}}

    answer = state["answer"]
    question = state["question"]
    email = state["email"]

    # Crear PDF
    pdf_filename = "respuesta_ia.pdf"
    pdf_path = create_pdf(f"Respuesta a: {question[:40]}...", answer, pdf_filename)

    # Enviar Correo
    if pdf_path:
        email_subject = f"Respuesta a tu pregunta: '{question[:30]}...'"
        email_body = "Hola,\n\nAdjunto encontrarás la respuesta generada por la IA a tu pregunta.\n\nSaludos."
        send_email_with_attachment(email, email_subject, email_body, pdf_path)

        # Limpiar el archivo PDF después de enviarlo
        os.remove(pdf_path)
        print(f"Archivo temporal '{pdf_path}' eliminado.")

    return {{}}

# --- CONSTRUCCIÓN DEL GRAFO ---
workflow = StateGraph(AppState)

# Añadir nodos
workflow.add_node("get_user_data", get_user_data)
workflow.add_node("analyze_intention", analyze_intention)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("process_outputs", process_outputs)

# Definir el flujo
workflow.set_entry_point("get_user_data")
workflow.add_edge("get_user_data", "analyze_intention")
workflow.add_edge("analyze_intention", "generate_answer")
workflow.add_edge("generate_answer", "process_outputs")
workflow.add_edge("process_outputs", END)

# Compilar el grafo
app = workflow.compile()

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    app.invoke({{}})
    print("\n--- Proceso del grafo finalizado. ---")