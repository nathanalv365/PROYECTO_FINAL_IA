import os
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from transformers import pipeline
from tkinter import Tk, Label, Button, Entry, messagebox, filedialog, Canvas
from tkinter.ttk import Combobox, Notebook
from tkinterdnd2 import TkinterDnD, DND_FILES
import tkinter as tk
from docx import Document  # Biblioteca para leer documentos Word

# Configurar RoBERTa para resumen de texto
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Ruta del archivo CSV
file_path = r"C:\Users\ioliv\OneDrive\Documentos\Universidad\Inteligencia artificial\proyecto_ final _2\Vinculaci_n_de_personas_con_discapacidad_20241122(1).csv"

# Funciones para preprocesamiento y entrenamiento del modelo
def preprocess_data(data):
    data['discapacidad'] = data[
        [
            'Tipo de discapacidad: Intelectual', 'Tipo de discapacidad: Fisica',
            'Tipo de discapacidad: Visual', 'Tipo de discapacidad: Psicosocial',
            'Tipo de discapacidad: Múltiple', 'Tipo de discapacidad: Auditiva',
            'Tipo de discapacidad: Otra'
        ]
    ].idxmax(axis=1).str.replace('Tipo de discapacidad: ', '', regex=False)

    data['ubicación'] = data['Municipio']
    data['Salario mensual promedio'] = (
        data['Salario mensual promedio']
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )
    data['Salario mensual promedio'] = pd.to_numeric(data['Salario mensual promedio'], errors='coerce')

    data = data[['discapacidad', 'ubicación', 'Salario mensual promedio']].dropna()

    label_encoders = {}
    for column in ['discapacidad', 'ubicación']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    X = data[['discapacidad', 'ubicación']]
    y = data['Salario mensual promedio']
    return X, y, label_encoders

data = pd.read_csv(file_path)
X, y, label_encoders = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f"MSE del modelo: {mean_squared_error(y_test, y_pred):.2f}")

# Función para leer documentos Word
def read_word_file(file_path):
    doc = Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

# Función para analizar el archivo con RoBERTa y leerlo con la codificación correcta
def analyze_file(file_path):
    if not os.path.exists(file_path):
        messagebox.showerror("Error", "Archivo no encontrado.")
        return

    try:
        # Leer el contenido según la extensión del archivo
        if file_path.endswith(".docx") or file_path.endswith(".doc"):
            content = read_word_file(file_path)
        else:
            # Detectar la codificación del archivo
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            # Abrir el archivo con la codificación detectada
            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

        # Usar RoBERTa para generar un resumen
        summary = summarizer(content, max_length=130, min_length=30, do_sample=False)[0]["summary_text"]

        # Crear perfil basado en el resumen
        discapacidad = "Otra"  # Default (puedes modificarlo según el análisis)
        ubicacion = "Sin especificar"
        sueldo_estimado = model.predict([[0, 0]])[0]  # Predicción genérica, ajustar si necesario

        # Crear y guardar el perfil
        output_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"Perfil del Trabajador:\n")
                f.write(f"Resumen del archivo:\n{summary}\n\n")
                f.write(f"Discapacidad: {discapacidad}\n")
                f.write(f"Ubicación: {ubicacion}\n")
                f.write(f"Sueldo estimado: ${sueldo_estimado:.2f}\n")
            messagebox.showinfo("Éxito", f"Perfil guardado en: {output_path}")
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo procesar el archivo: {e}")

# Función para abrir un archivo desde cualquier ubicación
def open_file_from_anywhere():
    file_path = filedialog.askopenfilename(
        title="Selecciona un archivo",
        filetypes=[
            ("Archivos de texto", "*.txt"),
            ("Documentos Word", "*.docx *.doc"),
            ("Todos los archivos", "*.*"),
        ],
    )
    if file_path:  # Verificar si el usuario seleccionó un archivo
        analyze_file(file_path)

# Interfaz gráfica para cargar y arrastrar archivos
def drop_file(event):
    file_path = event.data.strip("{}")  # Eliminar caracteres adicionales en la ruta
    analyze_file(file_path)

# Función para predecir salario a partir de los datos introducidos manualmente
def predict_salary():
    discapacidad_input = discapacidad_var.get()
    ubicacion_input = ubicacion_var.get()

    if not discapacidad_input or not ubicacion_input:
        messagebox.showerror("Error", "Por favor, ingresa todos los datos.")
        return

    discapacidad_encoded = label_encoders['discapacidad'].transform([discapacidad_input])[0]
    ubicacion_encoded = label_encoders['ubicación'].transform([ubicacion_input])[0]
    
    salary_prediction = model.predict([[discapacidad_encoded, ubicacion_encoded]])[0]
    messagebox.showinfo("Predicción de Sueldo", f"Sueldo estimado: ${salary_prediction:.2f}")

# Crear la ventana principal
root = TkinterDnD.Tk()
root.title("Predicción de Sueldo y Perfil")

# Crear un Notebook para las pestañas
notebook = Notebook(root)
notebook.grid(row=0, column=0, padx=10, pady=10)

# Pestaña 1: Predicción desde archivo
tab_file = tk.Frame(notebook)
notebook.add(tab_file, text="Predicción desde Archivo")

# Canvas para arrastrar archivos
canvas = Canvas(tab_file, width=400, height=200, bg="lightblue")
canvas.grid(row=0, column=0, columnspan=2)
canvas.create_text(200, 100, text="Arrastra aquí tu archivo .txt o .doc/.docx", font=("Arial", 12))

# Permitir arrastrar archivos
canvas.drop_target_register(DND_FILES)
canvas.dnd_bind('<<Drop>>', drop_file)

# Botón para abrir un archivo
Button(tab_file, text="Seleccionar Archivo", command=open_file_from_anywhere).grid(row=1, column=0, columnspan=2)

# Pestaña 2: Introducción manual de datos
tab_manual = tk.Frame(notebook)
notebook.add(tab_manual, text="Introducción de Datos")

# Campos de entrada manual para discapacidad y ubicación
Label(tab_manual, text="Discapacidad:").grid(row=0, column=0)
discapacidad_var = Combobox(tab_manual, values=label_encoders['discapacidad'].classes_.tolist())
discapacidad_var.grid(row=0, column=1)

Label(tab_manual, text="Ubicación:").grid(row=1, column=0)
ubicacion_var = Combobox(tab_manual, values=label_encoders['ubicación'].classes_.tolist())
ubicacion_var.grid(row=1, column=1)

Button(tab_manual, text="Predecir Sueldo", command=predict_salary).grid(row=2, column=0, columnspan=2)

root.mainloop()
