import google.cloud.aiplatform as aiplatform
import kfp
from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component, ClassificationMetrics

@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def load_process_data(
    raw_data_path: str,
    processed_data_path: Output[Dataset]
):
    import pandas as pd
    import numpy as np
    import os

    # Cargar los datos
    data = pd.read_csv(raw_data_path)

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
    os.makedirs(processed_data_path.path, exist_ok=True)
    data_file_path = f"{processed_data_path.path}/processed_data.csv"
    data.to_csv(data_file_path, index=False)
    print(f"Datos procesados guardados en: {data_file_path}")


@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def split_data(
    processed_data_path: Input[Dataset],
    train_data_path: Output[Dataset],
    test_data_path: Output[Dataset],
    val_data_path: Output[Dataset]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import os

    # Cargar los datos procesados
    data = pd.read_csv(f"{processed_data_path.path}/processed_data.csv")

    # Diividir los datos en train, validation y test
    train, val_test = train_test_split(data, test_size=0.2, random_state=42)
    val, test = train_test_split(val_test, test_size=0.5, random_state=42)

    # Guardar los conjuntos de datos
    os.makedirs(train_data_path.path, exist_ok=True)
    os.makedirs(val_data_path.path, exist_ok=True)
    os.makedirs(test_data_path.path, exist_ok=True)

    train_data_path = train_data_path.path + "/train_data.csv"
    val_data_path = val_data_path.path + "/val_data.csv"
    test_data_path = test_data_path.path + "/test_data.csv"

    train.to_csv(train_data_path, index=False)
    val.to_csv(val_data_path, index=False)
    test.to_csv(test_data_path, index=False)

@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def train_models(
    train_data_path: Input[Dataset],
    models_path: Output[Model],
    encode_path: Output[Model]
):
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import joblib
    import os

    # Cargar los datos de entrenamiento
    data = pd.read_csv(f'{train_data_path.path}/train_data.csv')

    # Crear encoder, separar características y etiqueta
    cat_features = ['payment_type','employment_status','housing_status','device_os']
    target = ['fraud_bool']

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder_features = encoder.fit_transform(data[cat_features])
    encoded_df = pd.DataFrame(encoder_features, columns=encoder.get_feature_names_out(cat_features))

    X_train = pd.concat([data.drop(columns=cat_features + target), encoded_df], axis=1)
    y_train = data[target]

    # Balancear clases
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = neg / pos

    # Definir y entrenar modelos
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced', 
            random_state=42
            ),
        'XGBoost': XGBClassifier(
            eval_metric='logloss', 
            scale_pos_weight=scale_pos_weight, 
            random_state=42
            ),
        'LightGBM': LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
              random_state=42
              )
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    # Guardar los modelos y el encoder
    os.makedirs(models_path.path, exist_ok=True)
    os.makedirs(encode_path.path, exist_ok=True)
    models_path = models_path.path + "/trained_models.joblib"
    encode_path = encode_path.path + "/encoder.joblib"

    joblib.dump(trained_models, models_path)
    joblib.dump(encoder, encode_path)

@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def evaluate_models(
    val_data_path: Input[Dataset],
    models_path: Input[Model],
    encode_path: Input[Model],
    best_model_path: Output[Model],
    metrics_path: Output[Metrics],
    best_model_metrics_path: Output[Metrics],
    best_model_metrics_models: Output[ClassificationMetrics]
):
    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    import joblib
    import json
    import os

    # Cargar los datos de validación
    data = pd.read_csv(f'{val_data_path.path}/val_data.csv')

    # Cargar los modelos y el encoder
    trained_models = joblib.load(f"{models_path.path}/trained_models.joblib")
    encoder = joblib.load(f"{encode_path.path}/encoder.joblib")

    # Preparar los datos de validación
    cat_features = ['payment_type','employment_status','housing_status','device_os']
    target = ['fraud_bool']

    encoder_features = encoder.transform(data[cat_features])
    encoded_df = pd.DataFrame(encoder_features, columns=encoder.get_feature_names_out(cat_features))

    X_val = pd.concat([data.drop(columns=cat_features + target), encoded_df], axis=1)
    y_val = data[target]

    # Evaluar los modelos
    all_metrics = {}
    best_model_name = None
    best_model = None
    best_f1 = -1

    for name, model in trained_models.items():
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        all_metrics[name] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1,
            'roc_auc': roc_auc_score(y_val, y_pred)
        }

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model

    print(f'Resultados de la evaluación de modelos: {all_metrics}')

    # Crear directorios si no existen en el docker
    os.makedirs(metrics_path.path, exist_ok=True)
    os.makedirs(best_model_path.path, exist_ok=True)
    os.makedirs(best_model_metrics_path.path, exist_ok=True)

    # Guardar el mejor modelo y las métricas
    best_model_metrics = all_metrics[best_model_name]

    metrics_path = metrics_path.path + "/model_metrics.txt" 
    best_model_path = best_model_path.path + f"/best_model_{best_model_name}.joblib"
    best_model_metrics_path = best_model_metrics_path.path + f"/best_model_{best_model_name}_metrics.json"

    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)

    print(f'Mejor modelo: {best_model_name} con métricas: {best_model_metrics}')

    joblib.dump(best_model, best_model_path)
    with open(best_model_metrics_path, 'w') as f:
        json.dump(best_model_metrics, f, indent=4)

    
    # log the confusion matrix
    labels = ['No Fraude', 'Fraude']

    y_pred_best = best_model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred_best)

    confusion_matrix_data = []
    for i, row in enumerate(cm):
        row_dict = {"label": labels[i], "row_metrics": [{"label": labels[j], "value": int(val)} for j, val in enumerate(row)]}
        confusion_matrix_data.append(row_dict)

    best_model_metrics_models.log_confusion_matrix(
        labels=labels,
        confusion_matrix=confusion_matrix_data
    )

    # log roc auc
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

    best_model_metrics_models.log_roc_auc(
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        thresholds=thresholds.tolist()
    )


@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def upload_model_to_vertex(
    best_model_path: Input[Model],
    best_model_metrics_path: Input[Metrics],
    model_display_name: str,
    experiment_name: str = 'fraud_detection_experiment'
):
    import google.cloud.aiplatform as aiplatform
    import json
    import os

    # Inicializar Vertex AI
    aiplatform.init()

    metrics_file = os.path.join(best_model_metrics_path.path, os.listdir(best_model_metrics_path.path)[0])
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    
    # Crear Experimento
    aiplatform.init(experiment=experiment_name)
    run = aiplatform.start_run(run = f"run_{model_display_name}")

    # Subir el modelo a Vertex AI
    artifact = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=best_model_path.path.rsplit('/', 1)[0],
        serving_container_image_uri='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest'
    )

    for k, v in metrics.items():
        aiplatform.log_metric(k, v)
    
    aiplatform.end_run()

    print(f'Modelo subido a Vertex AI con ID: {artifact.resource_name}')

@dsl.pipeline(
    name='mit-project-banking-pipeline-training',
    description='Pipeline entrenamiento de modelos para el proyecto de detección de fraude bancario'
)
def pipeline(
    raw_data_path: str,
    model_display_name: str = 'fraud_detection_model'
):
    load_process_task = load_process_data(
        raw_data_path=raw_data_path,
    )

    split_data_task = split_data(
        processed_data_path=load_process_task.outputs['processed_data_path'],
    )

    train_models_task = train_models(
        train_data_path=split_data_task.outputs['train_data_path'],
    )

    evaluate_models_task = evaluate_models(
        val_data_path=split_data_task.outputs['val_data_path'],
        models_path=train_models_task.outputs['models_path'],
        encode_path=train_models_task.outputs['encode_path'],
    )

    upload_model_task = upload_model_to_vertex(
        best_model_path=evaluate_models_task.outputs['best_model_path'],
        best_model_metrics_path=evaluate_models_task.outputs['best_model_metrics_path'],
        model_display_name=model_display_name
    )

if __name__ == '__main__':

    PROJECT_ID = 'projectstylus01'
    REGION = 'us-central1'
    SERVICE_ACCOUNT = 'workbench-sa@projectstylus01.iam.gserviceaccount.com'
    GCS_BUCKET = 'gs://mit-project-vertex-ai-artifacts'
    PIPELINE_ROOT = f'{GCS_BUCKET}/pipeline_root/'

    aiplatform.init(project=PROJECT_ID, location=REGION, service_account=SERVICE_ACCOUNT, staging_bucket=GCS_BUCKET)

    INPUT_DATA_URI = f'{GCS_BUCKET}/data/Base.csv'

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='mit_project_banking_pipeline.yaml'
    )

    job = aiplatform.PipelineJob(
        display_name='mit-project-banking-pipeline-training',
        template_path='mit_project_banking_pipeline.yaml',
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            'raw_data_path': INPUT_DATA_URI,
            'model_display_name': 'fraud_detection_model'
        }
    )

    job.run()