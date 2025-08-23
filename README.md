# MIT Project Banking

Proyecto corresponde a un caso de estudio en el sector banca.

## Flujo de trabajo

Exploración y análisis de datos (EDA).

Construcción de notebooks para experimentos.

Implementación de código productivo en la carpeta scr/.

Pruebas unitarias y de integración en la carpeta tests/.

Gestión de dependencias y entorno reproducible con Poetry.

## Arquitectura

Pendiente

## Estructura del repositorio

MIT-PROJECT-BANKING
├── Data/                  # Diccionario de datos y datasets (no versionados)
│   └── Diccionario.xlsx
├── Notebooks/             # Notebooks de EDA y experimentación
│   ├── notebook1.ipynb
│   └── notebook2.ipynb
├── scr/                   # Código listo para producción
├── tests/                 # Scripts de prueba antes de pase a producción
├── .gitignore             # Archivos y carpetas ignoradas por git
├── pyproject.toml          # Definición de dependencias y configuración del proyecto
├── poetry.lock             # Archivo de bloqueo de dependencias (Poetry)
└── README.md               # Documentación del proyecto

Dentro de la carpeta notebooks guardar los notebooks que se usan para EDA, experimentos.

## INSTALACION Y CONFIGURACION

1. Clonar repositorio ya sea en local con jupyter o en Colab

'''
bash
git clone https://github.com/Leonel481/MIT-project-banking.git
'''

2. Cuando se genere un nuevo archivo subir los cambios al repositorio

'''
bash
# 1. Verificar qué archivos cambiaron
git status

# 2. Agregar los archivos modificados al área de staging para que el repo rastree nuevos archivos
git add .

# 3. Crear un commit con un mensaje descriptivo
git commit -m "Ejemplo Mensaje: feat: agregar notebook de EDA inicial"

# 4. Enviar los cambios al repositorio en GitHub
git push origin main
'''
