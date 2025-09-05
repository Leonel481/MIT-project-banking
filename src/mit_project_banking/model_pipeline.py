import google.cloud.aiplatform as aiplatform
import kfp
from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component, ClassificationMetrics, Markdown

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
    val_data_path: Output[Dataset],
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    import os

    # Cargar los datos procesados
    data = pd.read_csv(f"{processed_data_path.path}/processed_data.csv")

    # Diividir los datos en train, validation y test
    train, val_test = train_test_split(data, test_size = (1 - train_size), random_state=42, stratify=data['fraud_bool'])
    val, test = train_test_split(val_test, test_size = (test_size / (val_size + test_size)), random_state=42, stratify=val_test['fraud_bool'])

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
    val_data_path: Input[Dataset],
    best_model_name: Output[str],
    encode_path: Output[Model],
    best_model_metrics: Output[ClassificationMetrics],
    metrics_models: Output[Markdown],
):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    import joblib
    import json
    import os

    # Cargar los datos de entrenamiento
    data = pd.read_csv(f'{train_data_path.path}/train_data.csv')

    # Crear encoder, separar características y etiqueta
    cat_features = ['payment_type','employment_status','housing_status','device_os']
    target = 'fraud_bool'

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder_features = encoder.fit_transform(data[cat_features])
    encoded_df = pd.DataFrame(encoder_features, columns=encoder.get_feature_names_out(cat_features))

    X_train = pd.concat([data.drop(columns=cat_features + [target]), encoded_df], axis=1)
    y_train = data[target]

    # Balancear clases
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = neg / pos

    # Definir y entrenar modelos
    models = {
        'RandomForestClassifier': RandomForestClassifier(
            n_estimators=10,
            class_weight='balanced', 
            random_state=42
            ),
        # 'XGBClassifier': XGBClassifier(
        #     eval_metric='logloss', 
        #     scale_pos_weight=scale_pos_weight, 
        #     random_state=42
        #     ),
        # 'LGBMClassifier': LGBMClassifier(
        #     scale_pos_weight=scale_pos_weight,
        #       random_state=42
        #       )
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    # Guardar los modelos y el encoder
    # os.makedirs(models_path.path, exist_ok=True)
    os.makedirs(encode_path.path, exist_ok=True)
    # models_path = models_path.path + "/trained_models.joblib"
    encode_path = encode_path.path + "/encoder.joblib"

    # joblib.dump(trained_models, models_path)
    joblib.dump(encoder, encode_path)

    
    # Evaluate models
    data_val = pd.read_csv(f'{val_data_path.path}/val_data.csv')
    encoder_features_val = encoder.transform(data_val[cat_features])
    encoded_df = pd.DataFrame(encoder_features_val, columns=encoder.get_feature_names_out(cat_features))

    X_val = pd.concat([data_val.drop(columns=cat_features + [target]), encoded_df], axis=1)
    y_val = data_val[target]

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

    # Table in Markdown
    markdown_table = "| Modelo | Accuracy | Precision | Recall | F1 Score | ROC AUC |\n"
    markdown_table += "|--------|----------|-----------|--------|----------|---------|\n"
    for model, metrics in all_metrics.items():
        markdown_table += f"| {model} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['roc_auc']:.4f} |\n"
    
    os.makedirs(metrics_models.path, exist_ok=True)
    markdown_path = os.path.join(metrics_models.path, "markdown.md")
    with open(markdown_path, "w") as f:
        f.write(markdown_table)

    # log the confusion matrix
    labels = ['No Fraude', 'Fraude']

    y_pred_best = best_model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred_best)
    confusion_matrix_data = cm.tolist()

    best_model_metrics.log_confusion_matrix(
        categories=labels,
        matrix=confusion_matrix_data
    )

    # log roc auc
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

    N_points = 200
    total_points = len(fpr)
    indices = np.linspace(0, total_points - 1, N_points, dtype = int)

    fpr = np.nan_to_num(fpr[indices], nan=0.0, posinf=1.0, neginf=0.0)
    tpr = np.nan_to_num(tpr[indices], nan=0.0, posinf=1.0, neginf=0.0)
    thresholds = np.nan_to_num(thresholds[indices], nan=0.0, posinf=1.0, neginf=0.0)

    best_model_metrics.log_roc_curve(
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        threshold=thresholds.tolist()
    )

    # Best model to output
    # with open(best_model_name_output.path + "/best_model.txt", "w") as f:
    #     f.write(best_model_name)

    return best_model_name, encode_path, best_model_metrics, metrics_models



# @component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
# def evaluate_models(
#     val_data_path: Input[Dataset],
#     models_path: Input[Model],
#     encode_path: Input[Model],
#     best_model_path: Output[Model],
#     metrics_path: Output[Metrics],
#     models_metrics: Output[Markdown],
#     # best_model_metrics_path: Output[Metrics],
#     # best_model_metrics_models: Output[ClassificationMetrics],
#     best_model_name_output: Output[str]
# ):
#     import pandas as pd
#     import numpy as np
#     from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
#     import joblib
#     import json
#     import os

#     # Cargar los datos de validación
#     data = pd.read_csv(f'{val_data_path.path}/val_data.csv')

#     # Cargar los modelos y el encoder
#     trained_models = joblib.load(f"{models_path.path}/trained_models.joblib")
#     encoder = joblib.load(f"{encode_path.path}/encoder.joblib")

#     # Preparar los datos de validación
#     cat_features = ['payment_type','employment_status','housing_status','device_os']
#     target = ['fraud_bool']

#     encoder_features = encoder.transform(data[cat_features])
#     encoded_df = pd.DataFrame(encoder_features, columns=encoder.get_feature_names_out(cat_features))

#     X_val = pd.concat([data.drop(columns=cat_features + target), encoded_df], axis=1)
#     y_val = data[target]

#     # Evaluar los modelos
#     all_metrics = {}
#     best_model_name = None
#     best_model = None
#     best_f1 = -1

#     for name, model in trained_models.items():
#         y_pred = model.predict(X_val)
#         f1 = f1_score(y_val, y_pred)

#         all_metrics[name] = {
#             'accuracy': accuracy_score(y_val, y_pred),
#             'precision': precision_score(y_val, y_pred),
#             'recall': recall_score(y_val, y_pred),
#             'f1_score': f1,
#             'roc_auc': roc_auc_score(y_val, y_pred)
#         }

#         if f1 > best_f1:
#             best_f1 = f1
#             best_model_name = name
#             best_model = model

#     print(f'Resultados de la evaluación de modelos: {all_metrics}')

#     # Crear directorios si no existen en el docker
#     os.makedirs(metrics_path.path, exist_ok=True)
#     os.makedirs(best_model_path.path, exist_ok=True)
#     # os.makedirs(best_model_metrics_path.path, exist_ok=True)

#     # Guardar el mejor modelo y las métricas
#     best_model_metrics = all_metrics[best_model_name]

#     metrics_path = metrics_path.path + "/model_metrics.txt" 
#     best_model_path = best_model_path.path + f"/best_model_{best_model_name}.joblib"
#     # best_model_metrics_path = best_model_metrics_path.path + f"/best_model_{best_model_name}_metrics.json"

#     with open(metrics_path, 'w') as f:
#         json.dump(all_metrics, f, indent=4)

#     print(f'Mejor modelo: {best_model_name} con métricas: {best_model_metrics}')

#     joblib.dump(best_model, best_model_path)

#     # with open(best_model_metrics_path, 'w') as f:
#     #     json.dump(best_model_metrics, f, indent=4)

    
#     # # log the confusion matrix
#     # labels = ['No Fraude', 'Fraude']

#     # y_pred_best = best_model.predict(X_val)
#     # cm = confusion_matrix(y_val, y_pred_best)
#     # confusion_matrix_data = cm.tolist()

#     # best_model_metrics_models.log_confusion_matrix(
#     #     categories=labels,
#     #     matrix=confusion_matrix_data
#     # )

#     # # log roc auc
#     # y_pred_proba = best_model.predict_proba(X_val)[:, 1]
#     # fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

#     # N_points = 200
#     # total_points = len(fpr)
#     # indices = np.linspace(0, total_points - 1, N_points, dtype = int)

#     # fpr = np.nan_to_num(fpr[indices], nan=0.0, posinf=1.0, neginf=0.0)
#     # tpr = np.nan_to_num(tpr[indices], nan=0.0, posinf=1.0, neginf=0.0)
#     # thresholds = np.nan_to_num(thresholds[indices], nan=0.0, posinf=1.0, neginf=0.0)

#     # best_model_metrics_models.log_roc_curve(
#     #     fpr=fpr.tolist(),
#     #     tpr=tpr.tolist(),
#     #     threshold=thresholds.tolist()
#     # )

#     # Metricas in Markdown
#     markdown_table = "| Modelo | Accuracy | Precision | Recall | F1 Score | ROC AUC |\n"
#     markdown_table += "|--------|----------|-----------|--------|----------|---------|\n"
#     for model, metrics in all_metrics.items():
#         markdown_table += f"| {model} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1_score']:.4f} | {metrics['roc_auc']:.4f} |\n"
    
#     os.makedirs(models_metrics.path, exist_ok=True)
#     markdown_path = os.path.join(models_metrics.path, "markdown.md")
#     with open(markdown_path, "w") as f:
#         f.write(markdown_table)

#     # Output del nombre del mejor modelo
#     with open(best_model_name_output.path, "w") as f:
#         f.write(best_model_name)

@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def tuning_model(
    train_data_path: Input[Dataset],
    val_data_path: Input[Dataset],
    encoder_path: Input[Model],
    best_model_name: str,
    params_config_path: str,
    tuned_model_path: Output[Model],
    tune_model_metrics: Output[ClassificationMetrics],
    n_trials: int = 50,
):
    import os
    import pandas as pd
    import numpy as np
    import joblib
    import optuna
    import yaml
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier  
    from sklearn.ensemble import RandomForestClassifier

    # Cargar los datos de validación
    data_train = pd.read_csv(f'{train_data_path.path}/train_data.csv')
    data_val = pd.read_csv(f'{val_data_path.path}/val_data.csv')

    # Cargar el encoder
    encoder = joblib.load(f"{encoder_path.path}/encoder.joblib")

    # Preparar los datos de validación
    cat_features = ['payment_type','employment_status','housing_status','device_os']
    target = ['fraud_bool']

    encoder_features_train = encoder.transform(data_train[cat_features])
    encoded_df_train = pd.DataFrame(encoder_features_train, columns=encoder.get_feature_names_out(cat_features))

    X_train = pd.concat([data_train.drop(columns=cat_features + target), encoded_df_train], axis=1)
    y_train = data_train[target]

    encoder_features_val = encoder.transform(data_val[cat_features])
    encoded_df_val = pd.DataFrame(encoder_features_val, columns=encoder.get_feature_names_out(cat_features))

    X_val = pd.concat([data_val.drop(columns=cat_features + target), encoded_df_val], axis=1)
    y_val = data_val[target]
        
    # Balancear clases
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = neg / pos

    # Cargar el yaml de hiperparámetros
    with open(params_config_path, 'r') as file:
        params_config = yaml.safe_load(file)

    # Definir hiperparámetros para ajustar
    def objective(trial, params_config=params_config):

        param_type = {
            'n_estimators': 'int', 'max_depth': 'int', 'min_samples_split': 'int',
            'min_samples_leaf': 'int', 'num_leaves': 'int',
            'learning_rate': 'float', 'subsample': 'float', 'colsample_bytree': 'float',
            'reg_alpha': 'float', 'reg_lambda': 'float', 'gamma': 'float',
            'feature_fraction': 'float', 'bagging_fraction': 'float', 'lambda_l1': 'float',
            'lambda_l2': 'float',
            'max_features': 'categorical', 'bootstrap': 'categorical', 'class_weight': 'categorical',
            'objective': 'constant'
        }

        params = {}

        model_config = params_config.get(best_model_name)

        for param_name, value in model_config.items():

            if param_type.get(param_name) == 'int':
                params[param_name] = trial.suggest_int(param_name, value[0], value[1])
            elif param_type.get(param_name) == 'float':
                params[param_name] = trial.suggest_float(param_name, value[0], value[1])
            elif param_type.get(param_name) == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, value)
            elif param_type.get(param_name) == 'constant':
                params[param_name] = value
            else:
                raise ValueError(f"Tipo de parámetro no soportado: {param_type[param_name]}")
        
        if best_model_name in ['LGBMClassifier', 'XGBClassifier']:
            params['scale_pos_weight'] = scale_pos_weight
        
        # Crear la instancia del modelo dinámicamente
        if best_model_name == 'RandomForestClassifier':
            model = RandomForestClassifier(**params)
        elif best_model_name == 'LGBMClassifier':
            model = LGBMClassifier(**params)
        elif best_model_name == 'XGBClassifier':
            model = XGBClassifier(**params)
        else:
            raise ValueError(f"Modelo no soportado para tuning: {best_model_name}")

        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False,
                  early_stopping_rounds=10 if best_model_name in ['LGBMClassifier', 'XGBClassifier'] else None)

        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred)

        return f1

    # Ejecutar la optimización de hiperparámetros
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Entrenar el modelo con los mejores hiperparámetros
    best_params = study.best_trial.params
    tuned_model = XGBClassifier(**best_params, random_state=42)
    tuned_model.fit(X_val, y_val)

    # Guardar el modelo ajustado
    os.makedirs(tuned_model_path.path, exist_ok=True)
    tuned_model_file = tuned_model_path.path + "/tuned_model.joblib"
    joblib.dump(tuned_model, tuned_model_file)

    # Metricas del modelo ajustado
    # log the confusion matrix
    labels = ['No Fraude', 'Fraude']

    y_pred_best = tuned_model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred_best)
    confusion_matrix_data = cm.tolist()

    tune_model_metrics.log_confusion_matrix(
        categories=labels,
        matrix=confusion_matrix_data
    )

    # log roc auc
    y_pred_proba = tuned_model.predict_proba(X_val)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)

    N_points = 200
    total_points = len(fpr)
    indices = np.linspace(0, total_points - 1, N_points, dtype = int)

    fpr = np.nan_to_num(fpr[indices], nan=0.0, posinf=1.0, neginf=0.0)
    tpr = np.nan_to_num(tpr[indices], nan=0.0, posinf=1.0, neginf=0.0)
    thresholds = np.nan_to_num(thresholds[indices], nan=0.0, posinf=1.0, neginf=0.0)

    tune_model_metrics.log_roc_curve(
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        threshold=thresholds.tolist()
    )

@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def upload_model_to_vertex(
    best_model_path: Input[Model],
    # best_model_metrics_path: Input[Metrics],
    model_display_name: str,
    experiment_name: str = 'fraud-detection-experiment'
):
    import google.cloud.aiplatform as aiplatform
    import datetime
    import json
    import os

    # Inicializar Vertex AI
    aiplatform.init()

    # metrics_file = os.path.join(best_model_metrics_path.path, os.listdir(best_model_metrics_path.path)[0])
    # with open(metrics_file, "r") as f:
    #     metrics = json.load(f)
    
    # Crear Experimento
    aiplatform.init(experiment=experiment_name)
    run_name = f"run-{model_display_name}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    run = aiplatform.start_run(run = run_name)

    # Subir el modelo a Vertex AI
    artifact = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=best_model_path.path.rsplit('/', 1)[0],
        serving_container_image_uri='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest'
    )

    # for k, v in metrics.items():
    #     aiplatform.log_metrics(k, v)
    
    # aiplatform.log_metrics(metrics)

    aiplatform.end_run()

    print(f'Modelo subido a Vertex AI con ID: {artifact.resource_name}')

@dsl.pipeline(
    name='mit-project-banking-pipeline-training',
    description='Pipeline entrenamiento de modelos para el proyecto de detección de fraude bancario'
)
def pipeline(
    raw_data_path: str,
    params_config_path: str,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    n_trials: int = 50,
    model_display_name: str = 'fraud-detection-model'
):
    load_process_task = load_process_data(
        raw_data_path=raw_data_path,
    )

    split_data_task = split_data(
        processed_data_path=load_process_task.outputs['processed_data_path'],
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    train_models_task = train_models(
        train_data_path=split_data_task.outputs['train_data_path'],
        val_data_path=split_data_task.outputs['val_data_path'],
    )

    # evaluate_models_task = evaluate_models(
    #     val_data_path=split_data_task.outputs['val_data_path'],
    #     models_path=train_models_task.outputs['models_path'],
    #     encode_path=train_models_task.outputs['encode_path'],
    # )
    
    tuning_model_task = tuning_model(
        train_data_path=split_data_task.outputs['train_data_path'],
        val_data_path=split_data_task.outputs['val_data_path'],
        encoder_path=train_models_task.outputs['encode_path'],
        best_model_name=train_models_task.outputs['best_model_name'],
        params_config_path=params_config_path,
        n_trials=n_trials,
    )
    

    upload_model_task = upload_model_to_vertex(
        best_model_path=tuning_model_task.outputs['tuned_model_path'],
        # best_model_metrics_path=evaluate_models_task.outputs['best_model_metrics_path'],
        model_display_name=model_display_name
    )

if __name__ == '__main__':

    PROJECT_ID = 'projectstylus01'
    REGION = 'us-central1'
    SERVICE_ACCOUNT = 'workbench-sa@projectstylus01.iam.gserviceaccount.com'
    GCS_BUCKET = 'gs://mit-project-vertex-ai-artifacts'
    NAME_CONFIG = 'config.yaml'
    PIPELINE_ROOT = f'{GCS_BUCKET}/pipeline_root/'

    aiplatform.init(project=PROJECT_ID, location=REGION, service_account=SERVICE_ACCOUNT, staging_bucket=GCS_BUCKET)

    INPUT_DATA_URI = f'{GCS_BUCKET}/data/Base.csv'
    PARAMS_CONFIG_PATH = f'{GCS_BUCKET}/config/{NAME_CONFIG}'

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
            'params_config_path': PARAMS_CONFIG_PATH,
            'model_display_name': 'fraud-detection-model',
            'train_size': 0.8,
            'val_size': 0.1,
            'test_size': 0.1,
            'n_trials': 50,
        }
    )

    job.run()