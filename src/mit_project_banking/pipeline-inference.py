import google.cloud.aiplatform as aiplatform
import kfp
from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component, ClassificationMetrics, Markdown


@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def data_process(
    input_data_path: str,
    processed_data: Output[Dataset],
):
    import pandas as pd
    import numpy as np
    import os

    # Cargar los datos
    data = pd.read_csv(input_data_path)

    # Definiendo variables nan y categoricas
    var_nan = ['prev_address_months_count','current_address_months_count','intended_balcon_amount',
               'bank_months_count','session_length_in_minutes','device_distinct_emails_8w']

    # Ingenieria de variables
    data[var_nan] = data[var_nan].replace(-1, np.nan).astype('float')

    data['prev_address_valid'] = np.where(data['prev_address_months_count'] > 0,1,0)
    data['velocity_6h'] = np.where(data['velocity_6h'] <= 0,data["velocity_6h"].quantile(0.25),data["velocity_6h"])
    data['ratio_velocity_6h_24h'] = data['velocity_6h']/data['velocity_24h']
    data['ratio_velocity_24h_4w'] = data['velocity_24h']/data['velocity_4w']
    data['log_bank_branch_count_8w'] = np.log1p(data['bank_branch_count_8w'])
    data['log_days_since_request'] = np.log1p(data['days_since_request'])
    data['prev_bank_months_count'] = np.where(data['bank_months_count'] <=0, 0, 1)
    data['income_risk_score'] = data['income']*data['credit_risk_score']

    data = data.drop(columns = ['device_fraud_count','month','prev_address_months_count','intended_balcon_amount', 'source'])

    # Guardar los datos procesados
    os.makedirs(processed_data.path, exist_ok=True)
    data_file_path = f"{processed_data.path}/processed_data.csv"
    data.to_csv(data_file_path, index=False)
    print(f"Datos procesados guardados en: {data_file_path}")


@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def model_inference(
    model_resource_name: str,
    processed_data: Input[Dataset],
    predictions: Output[Dataset],
):
    import pandas as pd
    import os
    from google.cloud import aiplatform
    from pathlib import Path
    import gcsfs
    import json
    import joblib

    fs = gcsfs.GCSFileSystem()

    # Cargar los datos procesados
    data = pd.read_csv(f"{processed_data.path}/processed_data.csv")

    # Artefactos del modelo final      
    temp_download_dir = Path("/gcs_downloads")
    temp_download_dir.mkdir(exist_ok=True)

    LOCAL_MODEL_PATH = temp_download_dir / "final_model.joblib"
    LOCAL_ENCODER_PATH = temp_download_dir / "encoder.joblib"
    LOCAL_METRICS_PATH = temp_download_dir / "model_metrics.json"

    try:
        model = aiplatform.Model(model_resource_name)
        # Descarga todos los artefactos del modelo registrado a la carpeta local
        model.download_artifacts(artifact_directory=str(temp_download_dir))
        print("Artefactos descargados exitosamente con Model.download_artifacts().")
    except Exception as e:
        print(f"Error al descargar artefactos del modelo registrado: {e}")
        # Es crucial relanzar la excepción para que el componente falle
        raise

    
    # Cargar el Modelo (joblib)
    tuned_model = joblib.load(LOCAL_MODEL_PATH)
    print("Modelo cargado.")
    
    # Cargar el Encoder (joblib)
    encoder = joblib.load(LOCAL_ENCODER_PATH)
    print("Encoder cargado.")
    
    # Cargar las Métricas y extraer el threshold (json)
    with open(LOCAL_METRICS_PATH, "r") as f:
        metrics = json.load(f)

    # Realizar inferencias
    y_pred_proba = tuned_model.predict_proba(data)[:, 1]
    
    t_low_opt = metrics['t_low_opt']
    t_high_opt = metrics['t_high_opt']
    
    data['proba'] = y_pred_proba
    data['category'] = "NO_FRAUDE" 
    data.loc[(y_pred_proba < t_high_opt) & (y_pred_proba >= t_low_opt), 'category'] = "REVISIÓN"
    data.loc[y_pred_proba >= t_high_opt, 'category'] = "FRAUDE"

    results_df = pd.DataFrame({
        'proba': data['proba'],
        'category': data['category'],
        't_high_opt': t_high_opt,
        't_low_opt': t_low_opt
    })

    # Guardar las predicciones
    os.makedirs(predictions.path, exist_ok=True)
    predictions_file_path = f"{predictions.path}/predictions.csv"
    results_df.to_csv(predictions_file_path, index=False)
    print(f"Predicciones guardadas en: {predictions_file_path}")

@dsl.pipeline(
    name='mit-project-banking-pipeline-inference',
    description='Pipeline inferencia de modelos para el proyecto de detección de fraude bancario'
)
def pipeline(
    input_data_path: str,
    model_resource_name: str,
):
    data_process_task = data_process(
        input_data_path = input_data_path,
    )

    split_data_task = model_inference(
        processed_data=data_process_task.outputs['processed_data'],
        model_resource_name=model_resource_name,
    )

if __name__ == '__main__':

    PROJECT_ID = 'projectstylus01'
    REGION = 'us-central1'
    SERVICE_ACCOUNT = 'workbench-sa@projectstylus01.iam.gserviceaccount.com'
    GCS_BUCKET = 'gs://mit-project-vertex-ai-artifacts'
    NAME_CONFIG = 'config.yaml'
    PIPELINE_ROOT = f'{GCS_BUCKET}/pipeline_root/'

    aiplatform.init(project=PROJECT_ID, location=REGION, service_account=SERVICE_ACCOUNT, staging_bucket=GCS_BUCKET)

    INPUT_DATA_URI = f'{GCS_BUCKET}/data_inference/data_inference1.csv'

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='mit_project_banking_pipeline_inference.yaml'
    )

    job = aiplatform.PipelineJob(
        display_name='mit-project-banking-pipeline-inference',
        template_path='mit_project_banking_pipeline_inference.yaml',
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            'input_data_path': INPUT_DATA_URI,
            'model_resource_name' : 'projects/435304534790/locations/us-central1/models/8550465423097724928'
        },
        # enable_caching=False
    )

    job.run()