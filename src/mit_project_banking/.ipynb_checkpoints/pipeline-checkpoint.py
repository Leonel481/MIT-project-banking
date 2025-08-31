import kfp
from kfp import dsl
from google.cloud import aiplatform
import pandas as pd
from datetime import datetime


# componenetes del pipeline: procesamiento -> entrenamiento -> experiment params, logs, artifacts

@dsl.component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def load_prepare(data_raw_path: str, processed_csv: dsl.OutputPath(dsl.Dataset)):
    
    # subprocess.check_call(["pip", "install", "pandas","numpy","google-cloud-aiplatform"])
    
    import pandas as pd
    import numpy as np
    from datetime import datetime

    df = pd.read_csv(data_raw_path) 
    
    var_nan = ['prev_address_months_count','current_address_months_count','intended_balcon_amount',
               'bank_months_count','session_length_in_minutes','device_distinct_emails_8w']
    
    var_cat = ['payment_type','employment_status','housing_status','source','device_os']
    
    # Reemplazando para obtener los vacios
    df[var_nan] = df[var_nan].replace(-1, np.nan).astype('float')
    
    #Transformacion de variables
    df['prev_address_valid'] = np.where(df['prev_address_months_count'] > 0,1,0)
    df['velocity_6h'] = np.where(df['velocity_6h'] <= 0,df["velocity_6h"].quantile(0.25),df["velocity_6h"])
    df['ratio_velocity_6h_24h'] = df['velocity_6h']/df['velocity_24h']
    df['ratio_velocity_24h_4w'] = df['velocity_24h']/df['velocity_4w']
    df['log_bank_branch_count_8w'] = np.log1p(df['bank_branch_count_8w'])
    df['log_days_since_request'] = np.log1p(df['days_since_request'])
    df['prev_bank_months_count'] = np.where(df['bank_months_count'] <=0, 0, 1)
    df['income_risk_score'] = df['income']*df['credit_risk_score']
    
    # Eliminando variables que no son consideradas relevantes para el modelo o variables que ya fueron tratadas
    df = df.drop(columns = ['device_fraud_count','month','prev_address_months_count','intended_balcon_amount', 'source'])
    
    df.to_csv(output_processed_csv.path, index=False)
    
    print(f"Datos preprocesados guardados en: {output_processed_csv.path}")

@dsl.component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def preprocess_and_split_data(processed_csv: dsl.InputPath(dsl.Dataset),train_data_csv: dsl.OutputPath(dsl.Dataset),test_data_csv: dsl.OutputPath(dsl.Dataset)):
    
    # subprocess.check_call(["pip", "install", "pandas","scikit-learn","google-cloud-aiplatform"])
    
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(processed_csv.path)
    
    print("Información del DataFrame después de leer el archivo procesado:")
    print(df.info())
    print("\nNúmero de filas en el DataFrame:", len(df))
    
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    train.to_csv(train_data_csv.path, index=False)
    test.to_csv(test_data_csv.path, index=False)

    print(f"Data entrenamiento: {train_data_csv.path}")
    print(f"Data test: {test_data_csv.path}")
    
@dsl.component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def train_models(train_data_csv: dsl.InputPath(dsl.Dataset), all_models_and_encoder_path: dsl.OutputPath(str)):
    
    # subprocess.check_call(["pip", "install", "pandas","scikit-learn","xgboost","lightgbm","joblib","google-cloud-aiplatform"])
    
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    import lightgbm as lgb
    import joblib
    
    df = pd.read_csv(train_data_csv.path)
    
    cat_features = ['payment_type','employment_status','housing_status','device_os']
    target = 'fraud_bool'
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = encoder.fit_transform(df[cat_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(cat_features))
    
    df_final = pd.concat([df.drop(columns=cat_features), encoded_df], axis=1)
    
    X_train_full = df_final.drop(columns = [target])
    y_train_full = df[target]
       
    # Balancear
    neg, pos = y_train_full.value_counts()[0], y_train_full.value_counts()[1]
    scale_pos_weight = neg / pos
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
            ),
        'XGBoost': xgb.XGBClassifier(
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=42
            ),
        'LightGBM': lgb.LGBMClassifier(
            random_state=42,
            scale_pos_weight=scale_pos_weight
            )
        }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_full, y_train_full)
        trained_models[name] = model
        
    models_artifacts = {
                'models': trained_models,
                'encoder': encoder
                }
    
    joblib.dump(models_artifacts, all_models_and_encoder_path)
    print(f"Path models y encoder: {all_models_and_encoder_path}")
    
@dsl.component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def evaluate_models(all_models_and_encoder_path: dsl.InputPath(str), test_data_csv: dsl.InputPath(dsl.Dataset), metrics_path: dsl.OutputPath(str), best_model_path: dsl.OutputPath(str)):
    
    # subprocess.check_call(["pip", "install", "pandas","scikit-learn","xgboost","lightgbm","joblib","google-cloud-aiplatform"])
    
    import pandas as pd
    import joblib
    import json
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import xgboost as xgb
    import lightgbm as lgb
    
    combined_artifact = joblib.load(all_models_and_encoder_path)
    models = combined_artifact['models']
    encoder = combined_artifact['encoder']
    
    df_test = pd.read_csv(test_data_csv)
    cat_features = ['payment_type','employment_status','housing_status','device_os']
    target = 'fraud_bool'
    
    X_test = df_test.drop(columns = [target])
    y_test = df_test[target]
    
    X_test_encoded = pd.DataFrame(encoder.transform(X_test[cat_features]), columns=encoder.get_feature_names_out(cat_features), index=X_test.index)
    X_test_final = pd.concat([X_test.drop(columns=cat_features), X_test_encoded], axis=1)
    
    all_metrics = {}
    best_model_name = None
    best_roc_auc_score = -1
    
    for name, model in models.items():
        y_pred = model.predict(X_test_final)
        
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_final)[:, 1])
        
        all_metrics[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc_score': roc_auc
        }
    
        if roc_auc > best_roc_auc_score:
            best_roc_auc_score = roc_auc
            best_model_name = name
            
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    
    print(f'Path metricas: {metrics_path}')
    print(f'Resultados de los modelos entrenados: {all_metrics}')
    
    best_model_artifact = {
        'model': models[best_model_name],
        'encoder': encoder
        }
    joblib.dump(best_model_artifact, best_model_path)
    
    print(f'Mejor modelo {best_model_name}, path: {best_model_path}')
    
@dsl.component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def upload_model(model_path: dsl.InputPath(str), model_display_name: str, model_version: str):
    
    # subprocess.check_call(["pip", "install", "pandas","scikit-learn","xgboost","lightgbm","joblib","google-cloud-aiplatform"])
    
    from google.cloud import aiplatform
    
    combined_artifact = joblib.load(model_path)
    model = combined_artifact['model']
    
    temp_model_path = 'best_model.pkl'
    joblib.dump(model, temp_model_path)
    
    aiplatform.init()
    
    artifact = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=temp_model_path,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    )
    
    print(f"Modelo registrado con nombre: {artifact.display_name}")
    print(f"Versión del modelo: {model_version}")
    
    
@dsl.pipeline(
    name="end-to-end-training-pipeline",
    description="Pipeline de entrenamiento modelos base que carga datos de GCS, entrena, evalúa y registra un modelo.",
)
def training_pipeline(input_data_uri: str, model_name: str, model_version: str):
    
    # Tarea 1: Cargar y preprocesar los datos
    load_prepare_task = load_prepare(data_raw_path=input_data_uri)

    # Tarea 2: Dividir los datos en conjuntos de entrenamiento y prueba
    preprocess_and_split_data_task = preprocess_and_split_data(
        processed_csv=load_prepare_task.outputs["processed_csv"]
    )

    # Tarea 3: Entrenar todos los modelos
    train_models_task = train_models(
        train_data_csv=preprocess_and_split_data_task.outputs["train_data_csv"]
    )
    
    # Tarea 4: Evaluar todos los modelos y seleccionar el mejor
    evaluate_models_task = evaluate_models(
        all_models_and_encoder_path=train_models_task.outputs["all_models_and_encoder_path"],
        test_data_csv=preprocess_and_split_data_task.outputs["test_data_csv"]
    )

    # Tarea 5: Registrar el mejor modelo en el registro de modelos de Vertex AI
    upload_model_task = upload_model(
        model_path=evaluate_models_task.outputs["best_model_path"],
        model_display_name=model_name,
        model_version=model_version
    ).after(evaluate_models_task)

    
    
if __name__ == '__main__':
    
    # Configuracion de la isntancia
    PROJECT_ID = "projectstylus01"
    REGION = "us-central1"
    SERVICE_ACCOUNT = "workbench-sa@projectstylus01.iam.gserviceaccount.com"
    
    GCS_BUCKET = "gs://mit-project-vertex-ai-artifacts"
    PIPELINE_ROOT = f'{GCS_BUCKET}/pipeline_root'
    
    aiplatform.init(project=PROJECT_ID, location=REGION, service_account=SERVICE_ACCOUNT)
    
    INPUT_DATA_URI = f'{GCS_BUCKET}/data/Base.csv' # Asegúrate de que este archivo exista
    MODEL_DISPLAY_NAME = "mejor-modelo-base"
    MODEL_VERSION = "v1.0"
    
    kfp.compiler.Compiler().compile(
        pipeline_func=training_pipeline,
        package_path="training_pipeline.json",
    )
    
    job = aiplatform.PipelineJob(
        display_name="mit-fraud-detection-job",
        template_path="training_pipeline.json",
        pipeline_root=PIPELINE_ROOT,
        parameter_values={
            "input_data_uri": INPUT_DATA_URI,
            "model_name": MODEL_DISPLAY_NAME,
            "model_version": MODEL_VERSION
        },
        enable_caching=False
    )
    
    job.run()