
# Face-AI Attendance  
_Sistema de asistencia por reconocimiento facial en tiempo real_

![python][badge-python] ![pytorch][badge-pytorch] ![opencv][badge-opencv]

> Proyecto desarrollado por **Matías Andrés Toribio Clark** para automatizar el control de asistencia en laboratorios, salas de clase o espacios de trabajo mediante visión por computador y deep‑learning con **FaceNet**.

---

## ✨ Características principales
| Función | Detalle |
| ------- | ------- |
| 📸 **Detección & alineamiento** | MTCNN localiza y corrige la cara antes de inferir. |
| 🧠 **Embeddings FaceNet** | Inception Resnet V1 (facenet‑pytorch) genera vectores de 512 D por cada rostro. |
| 🗂 **Gestión de perfiles** | Alta/baja de personas desde la GUI; el dataset se guarda en `dataset_ai/`. |
| 🏷 **Reconocimiento en vivo** | Umbrales configurables (`TH_PROFILE`, `TH_KEEP`) para decidir si la cara coincide con alguna registrada. |
| 🧾 **Asistencia automática** | Registra fecha y hora en `attendance_ai.csv`; muestra histórico y exporta a Excel/CSV. |
| 🙈 **Unknowns** | Rostros no identificados se almacenan en `unknowns/` para revisión posterior. |
| 🎛 **Soporte multicámara** | Hasta 10 cámaras USB/IP de forma simultánea. |
| 🖥 **Interfaz amigable** | GUI en Tkinter con logs, vista previa y controles rápidos. |

---

## 📂 Estructura del repositorio

```
Proyecto_face_detection/
├─ face.py                    # Aplicación principal (GUI + backend)
├─ face-ai-environment.yml    # Entorno Conda reproducible
├─ dataset_ai/                # Se crea automáticamente (rostros etiquetados)
├─ unknowns/                  # Caras desconocidas almacenadas
├─ attendance_ai.csv          # Registro de asistencia
└─ profiles.json              # Metadatos de usuarios
```

---

## ⚙️ Instalación

1. **Clonar el repositorio**

   ```bash
   git clone https://github.com/Z4kkeNNN/face-ai-attendance.git
   cd face-ai-attendance
   ```

2. **Crear el entorno**

   > Recomendado: _Miniconda/Anaconda_ con Python ≥ 3.10

   ```bash
   conda env create -f face-ai-environment.yml
   conda activate face-ai
   ```

   _Sin Conda_:  
   ```bash
   pip install -r <(conda env export -f face-ai-environment.yml | grep "^- ")
   ```

3. **Ejecutar**

   ```bash
   python face.py
   ```

---

## 🚀 Uso rápido

| Paso | Acción |
| ---- | ------ |
| **Añadir perfil** | Presiona **“New Profile”**, escribe el nombre y deja que la cámara capture ≥ `MIN_SAMPLES` imágenes. |
| **Tomar asistencia** | Haz clic en **“Start Attendance”**. El sistema marcará la hora de entrada automáticamente. |
| **Revisar registros** | Abre `attendance_ai.csv` o usa la pestaña **“History”** de la GUI. |
| **Depurar desconocidos** | Revisa la carpeta `unknowns/` y, si corresponde, conviértelos en nuevos perfiles. |

_Pista_: Ajusta los umbrales dentro de `face.py` si tu iluminación o cámaras varían mucho.

---

## 🛠 Personalización

* **Cambiar modelo**: sustituye `InceptionResnetV1` por otro backbone en `face.py`.
* **Persistencia**: integra una base de datos SQL alterando las funciones `load_profiles()` y `save_att()`.
* **UI**: Tkinter es modular; puedes migrar a PyQt5/Qt6 si necesitas una interfaz más moderna.

---

## 🌱 Roadmap

- [ ] Exportar reporte PDF directamente desde la app  
- [ ] Panel web (Flask/FastAPI) para monitoreo remoto  
- [ ] Entrenamiento incremental on‑device  
- [ ] Soporte GPU CUDA y CPU ONNX (opcional)

¡Se aceptan _issues_ y **pull requests** con gusto! 🇨🇱

---

## 🤝 Contribuir

1. Haz un _fork_ del repo  
2. Crea tu rama `feature/tu-fork`  
3. _Commit_ y abre un **PR** explicando tu cambio  
4. Verifica que `pre-commit` y `pytest` pasen sin errores

---

## 📜 Licencia

Este proyecto se publica bajo la licencia **MIT**.  
Consulta el archivo [`LICENSE`](LICENSE) para más detalles.

---

## 🙌 Agradecimientos

- [facenet‑pytorch](https://github.com/timesler/facenet-pytorch) por simplificar embeddings de rostros  
- Comunidad **OpenCV** por sus utilidades de visión  
- Inspiración de múltiples ejemplos _attendance‑systems_ en GitHub

---

![banner](docs/demo.gif)

[badge-python]: https://img.shields.io/badge/Python-3.10%2B-informational
[badge-pytorch]: https://img.shields.io/badge/PyTorch-2.0-red
[badge-opencv]: https://img.shields.io/badge/OpenCV-4.9-blue
