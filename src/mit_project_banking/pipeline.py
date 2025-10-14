import google.cloud.aiplatform as aiplatform
import kfp
from kfp import compiler, dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Model, Output, component, ClassificationMetrics, Markdown
from typing import NamedTuple

class UploadModelOutputs(NamedTuple):
    model_resource_name: str

@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def split_data(
    processed_data_path: str,
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
    data = pd.read_csv(processed_data_path)

    # Diividir los datos en train, validation y test
    train, val_test = train_test_split(
                                data, test_size = (1 - train_size), 
                                random_state=42, 
                                stratify=data['fraud_bool']
                                )
    val, test = train_test_split(
                                val_test, test_size = (test_size / (val_size + test_size)), 
                                random_state=42, 
                                stratify=val_test['fraud_bool']
                                )

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
    encode_path: Output[Model],
    best_model_metrics: Output[ClassificationMetrics],
    metrics_path: Output[Metrics],
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
    cat_features = data.select_dtypes(include=['object']).columns.tolist()
    target = 'fraud_bool'

    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder_features = encoder.fit_transform(data[cat_features])
    encoded_df = pd.DataFrame(encoder_features, 
                              columns=encoder.get_feature_names_out(cat_features),
                              index=data.index)

    X_train = pd.concat([data.drop(columns=cat_features + [target]), encoded_df], axis=1)
    y_train = data[target]

    # Balancear clases
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = neg / pos

    # Definir y entrenar modelos
    models = {
        'RandomForestClassifier': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced', 
            random_state=42
            ),
        'XGBClassifier': XGBClassifier(
            eval_metric='logloss', 
            scale_pos_weight=scale_pos_weight, 
            random_state=42
            ),
        'LGBMClassifier': LGBMClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42
            )
    }

    # Guardar el encoder
    os.makedirs(encode_path.path, exist_ok=True)
    encode_path = encode_path.path + "/encoder.joblib"
    joblib.dump(encoder, encode_path)

    
    # Evaluate models
    data_val = pd.read_csv(f'{val_data_path.path}/val_data.csv')
    encoder_features_val = encoder.transform(data_val[cat_features])
    encoded_df = pd.DataFrame(encoder_features_val, 
                              columns=encoder.get_feature_names_out(cat_features),
                              index=data_val.index)

    X_val = pd.concat([data_val.drop(columns=cat_features + [target]), encoded_df], axis=1)
    y_val = data_val[target]

    all_metrics = {}
    best_model = None
    best_f1 = -1

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_pred)

        all_metrics[name] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1,
            'roc_auc': roc_auc_score(y_val, y_proba)
        }

        print(f"{name} - F1 Score: {f1}")
        print(f"{name} - Accuracy: {all_metrics[name]}")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model

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

    # log param metric
    for name, metrics_dict in all_metrics.items():
        metrics_path.log_metric(f'{name}_f1_score', metrics_dict.get('f1_score'))
        metrics_path.log_metric(f'{name}_roc_auc', metrics_dict.get('roc_auc'))
    
    os.makedirs(metrics_path.path, exist_ok=True)
    metrics_file_path = metrics_path.path + "/models_metrics.json" 
    with open(metrics_file_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)


@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def tuning_model(
    train_data_path: Input[Dataset],
    val_data_path: Input[Dataset],
    encoder_path: Input[Model],
    metrics_path: Input[Metrics],
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
    import json
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier  
    from sklearn.ensemble import RandomForestClassifier
    import gcsfs
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    # Cargar los datos de validación
    data_train = pd.read_csv(f'{train_data_path.path}/train_data.csv')
    data_val = pd.read_csv(f'{val_data_path.path}/val_data.csv')

    # Cargar el encoder
    encoder = joblib.load(f"{encoder_path.path}/encoder.joblib")

    # Preparar los datos de validación
    cat_features = data_train.select_dtypes(include=['object']).columns.tolist()
    target = 'fraud_bool'

    encoder_features_train = encoder.transform(data_train[cat_features])
    encoded_df_train = pd.DataFrame(encoder_features_train, 
                                    columns=encoder.get_feature_names_out(cat_features),
                                    index=data_train.index)

    X_train = pd.concat([data_train.drop(columns=cat_features + [target]), encoded_df_train], axis=1)
    y_train = data_train[target]

    encoder_features_val = encoder.transform(data_val[cat_features])
    encoded_df_val = pd.DataFrame(encoder_features_val, 
                                  columns=encoder.get_feature_names_out(cat_features),
                                  index=data_val.index)

    X_val = pd.concat([data_val.drop(columns=cat_features + [target]), encoded_df_val], axis=1)
    y_val = data_val[target]
        
    # Balancear clases
    neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
    scale_pos_weight = neg / pos

    # Cargar el yaml de hiperparámetros
    def download_yaml_from_gcs(gcs_path: str, local_path: str = "config.yaml"):
        fs = gcsfs.GCSFileSystem()
        fs.get(gcs_path, local_path)

        with open(local_path, "r") as f:
            config = yaml.safe_load(f)
        return config

    params_config = download_yaml_from_gcs(params_config_path)

    # Load metrics json
    metrics_file = os.path.join(metrics_path.path, os.listdir(metrics_path.path)[0])
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    best_model = max(metrics.items(), key=lambda x: x[1]["f1_score"])
    best_model_name = best_model[0]

    # Definir hiperparámetros para ajustar
    def objective(trial, X: pd.DataFrame, y: pd.Series):

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
        
        # validacion cruzada estratificada

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = []

        for train_index, val_index in skf.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
            y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
            # Crear la instancia del modelo dinámicamente
            if best_model_name == 'RandomForestClassifier':
                model = RandomForestClassifier(**params, random_state=42)
            elif best_model_name == 'LGBMClassifier':
                model = LGBMClassifier(**params, random_state=42, verbose=-1)
            elif best_model_name == 'XGBClassifier':
                model = XGBClassifier(**params, random_state=42, verbose=0, use_label_encoder=False)
            else:
                raise ValueError(f"Modelo no soportado para tuning: {best_model_name}")

            if best_model_name == 'RandomForestClassifier':
                model.fit(X_train_fold, y_train_fold)
            elif best_model_name == 'LGBMClassifier':
                model.fit(X_train_fold, y_train_fold,
                            eval_set=[(X_val_fold, y_val_fold)],
                            )
            elif best_model_name == 'XGBClassifier':
                model.fit(X_train_fold, y_train_fold,
                            eval_set=[(X_val_fold, y_val_fold)],
                            verbose=False,
                            )

            # Evaluar el modelo  
            y_pred_proba_trial = model.predict_proba(X_val)[:, 1]
            precision_trial, recall_trial, _ = precision_recall_curve(y_val, y_pred_proba_trial)
            
            f1_scores_fold = 2 * (precision_trial * recall_trial) / (precision_trial + recall_trial + 1e-9)
            
            best_f1 = np.max(f1_scores_fold)
            f1_scores.append(best_f1)

        return np.mean(f1_scores)

    # Ejecutar la optimización de hiperparámetros
    study = optuna.create_study(direction='maximize')
    func = lambda trial: objective(trial, X_train, y_train)
    study.optimize(func, n_trials=n_trials)

    # Entrenar el modelo con los mejores hiperparámetros
    best_params = study.best_trial.params


    if best_model_name == 'RandomForestClassifier':
        tuned_model = RandomForestClassifier(**best_params, random_state=42)
        tuned_model.fit(X_train, y_train)
    elif best_model_name == 'LGBMClassifier':
        tuned_model = LGBMClassifier(**best_params, random_state=42, verboose=-1)
        tuned_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=10
        )
    elif best_model_name == 'XGBClassifier':
        tuned_model = XGBClassifier(**best_params, random_state=42, verbose=0, use_label_encoder=False)
        tuned_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        raise ValueError(f"Modelo no soportado: {best_model_name}")
    
    # Validacion del modelo ajustado

    y_pred = tuned_model.predict(X_val)
    y_pred_proba = tuned_model.predict_proba(X_val)[:, 1]    
    precision_final, recall_final, thresholds_final = precision_recall_curve(y_val, y_pred_proba)

    f1_scores = 2 * (precision_final * recall_final) / (precision_final + recall_final + 1e-9)

    best_threshold_index = np.argmax(f1_scores)

    if best_threshold_index == len(thresholds_final):
         optimal_threshold = thresholds_final[-1] # Último umbral
    else:
         optimal_threshold = thresholds_final[best_threshold_index]

    # Metricas del modelo ajustado
    # log the confusion matrix
    labels = ['No Fraude', 'Fraude']

    y_pred_best = (y_pred_proba >= optimal_threshold).astype(int)
    cm = confusion_matrix(y_val, y_pred_best)
    confusion_matrix_data = cm.tolist()

    tune_model_metrics.log_confusion_matrix(
        categories=labels,
        matrix=confusion_matrix_data
    )

    # log roc auc
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

    # Metadata del modelo ajustado
    tuned_model_path.metadata['model'] = best_model_name
    tuned_model_path.metadata['ROC-AUC'] = roc_auc_score(y_val, y_pred_proba)
    tuned_model_path.metadata['Optimal_Threshold'] = float(optimal_threshold)
    tuned_model_path.metadata['F1-score'] = f1_score(y_val, y_pred_best)
    tuned_model_path.metadata['Recall'] = recall_score(y_val, y_pred_best)
    tuned_model_path.metadata['Precision'] = precision_score(y_val, y_pred_best)

    # Guardar el modelo ajustado
    os.makedirs(tuned_model_path.path, exist_ok=True)
    tuned_model_file = tuned_model_path.path + "/tuned_model.joblib"
    joblib.dump(tuned_model, tuned_model_file)


@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def calibrate_model(
    val_data_path: Input[Dataset],
    best_model_path: Input[Model],
    encoder_path: Input[Model],
    scenery_metrics: Output[Metrics],
    tune_model_metrics: Output[ClassificationMetrics],
    c_fn: float = 20000,
    c_fp: float = 200,
    c_review: float = 30,
    human_hit_rate: float = 0.80,
):
    import pandas as pd
    import numpy as np
    from typing import Optional, Dict, Union
    from sklearn.metrics import roc_curve, roc_auc_score
    import joblib
    import json
    import os

    def cost_function(
            y_true: np.ndarray, 
            y_proba: np.ndarray, 
            t_low_grid: np.ndarray, 
            t_high_grid: np.ndarray,
            c_fn: float = 10000, 
            c_fp: float = 200, 
            c_review: float = 30, 
            h: float = 0.80,
            max_reviews: Optional[int] = None
        ) -> Dict[str, Union[float, int]]:
        """
        Calcula el Umbral Óptimo de Decisión de Tres Vías (3WD) que minimiza el costo operacional total.

        Esta función simula un sistema de detección de fraude con tres resultados (Auto-Aprobar,
        Revisión Humana, Auto-Rechazar) y encuentra la combinación de umbrales (t_low, t_high)
        que resulta en el menor costo de negocio, considerando los costos asimétricos y la
        capacidad de revisión humana.

        Args:
            y_true: Etiquetas reales de fraude (0 o 1).
            y_proba: Probabilidades predichas por el modelo para la clase positiva (fraude).
            t_low_grid: Array de umbrales inferiores a probar (p.ej., np.linspace(0.01, 0.4, 50)).
            t_high_grid: Array de umbrales superiores a probar (p.ej., np.linspace(0.4, 0.99, 50)).
            c_fn: Costo de un Falso Negativo (Fraude no detectado). Alto costo por defecto.
            c_fp: Costo de un Falso Positivo (Transacción legítima rechazada automáticamente). Costo moderado.
            c_review: Costo de procesar un solo caso por el equipo de revisión humana. Bajo costo.
            h: Eficacia (Hit Rate) del equipo de revisión humana (fracción de fraudes que atrapan).
            max_reviews: Límite superior opcional para la cantidad de casos que el equipo humano
                        puede revisar. Si se excede, la combinación de umbrales se ignora.

        Returns:
            Un diccionario que contiene el costo mínimo (`cost`), los umbrales óptimos (`t_low`, `t_high`)
            y métricas de desempeño detalladas para ese umbral (TP, FP, FN, Recall, etc.).
        """
        best = {'cost': np.inf}
        n = len(y_true)

        real_fraud = (y_true == 1)
        real_no_fraud = (y_true == 0)
        total_frauds = real_fraud.sum()

        for t_low in t_low_grid:
            for t_high in t_high_grid:

                if t_low >= t_high:
                    continue
                
                # Segmentacion Umbrales
                auto_decline_mask = (y_proba >= t_high)
                review_mask = (y_proba < t_high) & (y_proba > t_low)
                auto_approve_mask = (y_proba <= t_low)

                # Casos
                TP_auto = np.sum(auto_decline_mask & real_fraud)
                FP_auto = np.sum(auto_decline_mask & real_no_fraud)
                FN_auto = np.sum(auto_approve_mask & real_fraud)

                frauds_in_review = np.sum(review_mask & real_fraud)
                no_frauds_in_review = np.sum(review_mask & real_no_fraud)
                review_count = np.sum(review_mask)

                # Capacidad
                if (max_reviews is not None) and (review_count > max_reviews):
                    continue
                
                frauds_caught_by_review = h * frauds_in_review
                frauds_missed_by_review = (1 - h) * frauds_in_review
                FN_after = FN_auto + frauds_missed_by_review

                no_frauds_declined_by_review = (1 - h) * no_frauds_in_review
                FP_after = FP_auto + no_frauds_declined_by_review
                
                # Funcion de costo a minimizar 

                f_cost = c_fn * FN_after + c_fp * FP_after + c_review * review_count

                # Resultados

                if f_cost < best['cost']:
                    best = {
                        "cost": f_cost,
                        "t_low": t_low,
                        "t_high": t_high,
                        "TP_auto": int(TP_auto),
                        "FP_auto": int(FP_auto),
                        "frauds_in_review": int(frauds_in_review),
                        "legit_in_review": int(no_frauds_in_review),
                        "review_count": int(review_count),
                        "FN_after": float(FN_after),
                        "FP_after": float(FP_after), # Añadido para seguimiento
                        "recall_overall": (TP_auto + frauds_caught_by_review) / (total_frauds + 1e-9),
                        "precision_auto_decline": TP_auto / (TP_auto + FP_auto + 1e-9),
                        "frac_review": review_count / n
                    }
        return best

    def analyze_cost_function(best_results, total_samples, human_hit_rate):

        TP_auto = best_results['TP_auto']
        frauds_in_review = best_results['frauds_in_review']
        FN_after = best_results['FN_after']
        FP_after = best_results['FP_after']
        FP_auto = best_results['FP_auto']
        h = human_hit_rate

        FN_auto = FN_after - (1-h) * frauds_in_review
        

        Total_Fraudes = TP_auto + frauds_in_review + FN_auto
        legit_in_review = best_results['legit_in_review']

        if total_samples is None:
            total_samples = round(best_results['review_count'] / best_results['frac_review'])

        Total_Fraudes = TP_auto + frauds_in_review + FN_auto
        Total_Legitimos = total_samples - Total_Fraudes
        TN_auto = Total_Legitimos - legit_in_review - FP_auto

        confusion_matrix_data = [[TN_auto, legit_in_review, FP_auto],
                                 [0,0,0],
                                 [FN_auto, frauds_in_review, TP_auto]]
        
        frauds_caught_by_review = h * frauds_in_review
        TP_after = TP_auto + frauds_caught_by_review

        recall_fraude = TP_after / (TP_after + FN_after + 1e-9)
        precision_fraude = TP_after / (TP_after + FP_after + 1e-9)
        f1_fraude = 2 * (precision_fraude * recall_fraude) / (precision_fraude + recall_fraude + 1e-9)
        
        return confusion_matrix_data, recall_fraude, precision_fraude, f1_fraude

    # Cargar los datos de validación
    data_val = pd.read_csv(f'{val_data_path.path}/val_data.csv')

    # Cargar el encoder
    encoder = joblib.load(f"{encoder_path.path}/encoder.joblib")

    # Preparar los datos de validación
    cat_features = data_val.select_dtypes(include=['object']).columns.tolist()
    target = 'fraud_bool'

    encoder_features_val = encoder.transform(data_val[cat_features])
    encoded_df_val = pd.DataFrame(encoder_features_val, 
                                  columns=encoder.get_feature_names_out(cat_features),
                                  index=data_val.index)

    X_val = pd.concat([data_val.drop(columns=cat_features + [target]), encoded_df_val], axis=1)
    y_val = data_val[target]

    # Cargar el modelo
    tuned_model = joblib.load(f"{best_model_path.path}/tuned_model.joblib")

    y_pred_proba = tuned_model.predict_proba(X_val)[:, 1]

    t_low_grid = np.linspace(0.01, 0.2, 40)
    t_high_grid = np.linspace(0.05, 0.5, 60)

    results = cost_function(y_val, y_pred_proba,
                                t_low_grid, t_high_grid,
                                c_fn = c_fn,
                                c_fp = c_fp,
                                c_review = c_review,
                                h = human_hit_rate,
                                max_reviews=2000)
    
    # Treholds óptimos
    t_low_opt = results['t_low']
    t_high_opt = results['t_high']

    # Metricas del modelo ajustado con la calibracion de la funcion de costo
    # log the confusion matrix
    labels = ['No Fraude', 'Observado', 'Fraude']

    # y_pred_best = (y_pred_proba >= optimal_threshold).astype(int)
    cm, recall, precision, f1 = analyze_cost_function(results, total_samples=None, human_hit_rate=human_hit_rate)

    tune_model_metrics.log_confusion_matrix(
        categories=labels,
        matrix=cm
    )

    # log roc auc
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

    calibrated_metrics = {
            'recall': recall,
            'precision': precision,
            'f1_score': f1,
    }

    calibrated_metrics['roc_auc'] = float(roc_auc_score(y_val, y_pred_proba))
    calibrated_metrics['t_low_opt'] = float(t_low_opt)
    calibrated_metrics['t_high_opt'] = float(t_high_opt)
    calibrated_metrics['cost_fn'] = float(c_fn)
    calibrated_metrics['cost_fp'] = float(c_fp)
    calibrated_metrics['cost_review'] = float(c_review)
    calibrated_metrics['human_hit_rate'] = float(human_hit_rate)

    # log param metric
    for metric_name, metric_value in calibrated_metrics.items():
        scenery_metrics.log_metric(f'Calibrated_{metric_name}', metric_value)
    
    os.makedirs(scenery_metrics.path, exist_ok=True)
    metrics_file_path = scenery_metrics.path + "/scenery_metrics.json" 
    with open(metrics_file_path, 'w') as f:
        json.dump(calibrated_metrics, f, indent=4)


@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def evaluate_model(
    test_data_path: Input[Dataset],
    best_model_path: Input[Model],
    encode_path: Input[Model],
    scenery_metrics: Input[Metrics],
    final_tuned_model_path: Output[Model],
    evaluate_metrics: Output[ClassificationMetrics],
    evaluate_metrics_path: Output[Metrics],
    human_hit_rate: float = 0.80
):

    import pandas as pd
    import numpy as np
    from sklearn.metrics import roc_auc_score, roc_curve
    import joblib
    import json
    import os

    # Cargar los datos de test
    data_test = pd.read_csv(f'{test_data_path.path}/test_data.csv')

    # Cargar el encoder y modelo
    encoder = joblib.load(f"{encode_path.path}/encoder.joblib")
    best_model = joblib.load(f"{best_model_path.path}/tuned_model.joblib")

    # Preparar los datos de test
    cat_features = data_test.select_dtypes(include=['object']).columns.tolist()
    target = 'fraud_bool'

    encoder_features_test = encoder.transform(data_test[cat_features])
    encoded_df_test = pd.DataFrame(encoder_features_test, columns=encoder.get_feature_names_out(cat_features))

    X_test = pd.concat([data_test.drop(columns=cat_features + [target]), encoded_df_test], axis=1)
    y_test = data_test[target]

    # Evaluar el modelo
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    # log the confusion matrix
    labels = ['No Fraude', 'Observado', 'Fraude']

    # Load optimal treshold json
    metrics_file = os.path.join(scenery_metrics.path, os.listdir(scenery_metrics.path)[0])
    with open(metrics_file, "r") as f:
        opt_tresholds = json.load(f)

    def results_model(y_test, y_pred_proba, opt_tresholds, human_hit_rate = 0.8):

        t_low_opt = opt_tresholds['t_low_opt']
        t_high_opt = opt_tresholds['t_high_opt']

        n = len(y_test)

        real_fraud = (y_test == 1)
        real_no_fraud = (y_test == 0)
        total_frauds = real_fraud.sum()

        fraud_mask = (y_pred_proba >= t_high_opt)
        review_mask = (y_pred_proba < t_high_opt) & (y_pred_proba > t_low_opt)
        no_fraud_mask = (y_pred_proba <= t_low_opt)

        TP_auto = int(np.sum(fraud_mask & real_fraud))
        FP_auto = int(np.sum(fraud_mask & real_no_fraud))
        FN_auto = int(np.sum(no_fraud_mask & real_fraud))
        TN_auto = int(np.sum(no_fraud_mask & real_no_fraud))

        frauds_in_review = int(np.sum(review_mask & real_fraud))
        legit_in_review = int(np.sum(review_mask & real_no_fraud))
        review_count = int(np.sum(review_mask))

        # Fraudes que el humano atrapa
        frauds_caught_by_review = human_hit_rate * frauds_in_review
        # Fraudes que el humano pierde (se convierten en FN final)
        frauds_missed_by_review = (1 - human_hit_rate) * frauds_in_review
        # Legítimos que el humano rechaza (se convierten en FP final)
        no_frauds_declined_by_review = (1 - human_hit_rate) * legit_in_review

        FN_total = FN_auto + frauds_missed_by_review
        FP_total = FP_auto + no_frauds_declined_by_review
        TP_total = TP_auto + frauds_caught_by_review
        Frauds_total = TP_total + FN_total + frauds_in_review

        recall_final = float(TP_total / (Frauds_total + 1e-9))
        precision_final = float(TP_total / (TP_total + FP_total + 1e-9))
        f1_final = float(2 * (precision_final * recall_final) / (precision_final + recall_final + 1e-9))

        confusion_matrix_data = [
            [TN_auto, legit_in_review, FP_auto], # Real No Fraude
            [0,0,0],
            [FN_auto, frauds_in_review, TP_auto]  # Real Fraude
        ]

        results = {
            "recall": recall_final,
            "precision": precision_final,
            "f1_score": f1_final,
            "review_fraction": review_count / n
        }

        return confusion_matrix_data, results


    cm , results = results_model(
        y_test, 
        y_pred_proba,
        opt_tresholds,
        human_hit_rate = human_hit_rate
    )

    # log the confusion matrix
    evaluate_metrics.log_confusion_matrix(
        categories=labels,
        matrix=cm
    )

    # Model
    os.makedirs(final_tuned_model_path.path, exist_ok=True)
    final_model_file_path = final_tuned_model_path.path + "/final_tuned_model.joblib"
    joblib.dump(best_model, final_model_file_path)

    # log roc auc
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    N_points = 200
    total_points = len(fpr)
    indices = np.linspace(0, total_points - 1, N_points, dtype = int)

    fpr = np.nan_to_num(fpr[indices], nan=0.0, posinf=1.0, neginf=0.0)
    tpr = np.nan_to_num(tpr[indices], nan=0.0, posinf=1.0, neginf=0.0)
    thresholds = np.nan_to_num(thresholds[indices], nan=0.0, posinf=1.0, neginf=0.0)

    evaluate_metrics.log_roc_curve(
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        threshold=thresholds.tolist()
    ) 

    # log metric
    results['roc_auc'] = float(roc_auc_score(y_test, y_pred_proba))
    results['t_low_opt'] = opt_tresholds['t_low_opt']
    results['t_high_opt'] = opt_tresholds['t_high_opt']
    results['human_hit_rate'] = human_hit_rate
    results['cost_fn'] = opt_tresholds['cost_fn']
    results['cost_fp'] = opt_tresholds['cost_fp']
    results['cost_review'] = opt_tresholds['cost_review']
    
    for metric_name, metric_value in results.items():
        evaluate_metrics_path.log_metric(f'Calibrated_{metric_name}', metric_value)

    os.makedirs(evaluate_metrics_path.path, exist_ok=True)
    metrics_file_path = evaluate_metrics_path.path + "/model_metrics.json"
    with open(metrics_file_path, 'w') as f:
        json.dump(results, f, indent=4)

@component(base_image='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest')
def upload_model_to_vertex(
    best_model_path: Input[Model],
    best_model_metrics_path: Input[Metrics],
    encode_path: Input[Model],
    artifacts: Output[Artifact],
    model_display_name: str,
    experiment_name: str = 'fraud-detection-experiment'
) :
    
    import google.cloud.aiplatform as aiplatform
    import datetime
    import shutil
    import json
    import joblib
    import os
    from pathlib import Path


    # Artifact
    os.makedirs(artifacts.path, exist_ok = True)

    # Cargar encoder al artifact path
    encoder_src = Path(encode_path.path) / 'encoder.joblib'
    encoder_artifact = Path(artifacts.path) / 'encoder.joblib'
    shutil.copy(encoder_src, encoder_artifact)
    
    # Cargar Modelo
    model_source = Path(best_model_path.path)
    model_files = [f for f in  model_source.iterdir() if f.is_file()]

    if model_files:
        model_file_path = model_files[0]
        model_artifact = Path(artifacts.path) / 'final_model.joblib'
        shutil.copy(model_file_path, model_artifact)
        print(f'Modelo copiado: {model_file_path.name} -> final_tuned_model.joblib')
    else:
        raise FileNotFoundError("Error: No se encontró el archivo del modelo.")

    # Cargar metricas
    metrics_source_dir = Path(best_model_metrics_path.path)
    metrics_files = [f for f in metrics_source_dir.iterdir() if f.is_file()] 
    
    if metrics_files:
        metrics_file = metrics_files[0]
        # Cargamos las métricas para loguearlas en el experimento
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        shutil.copy(metrics_file, Path(artifacts.path) / "model_metrics.json")
    else:
        print("Advertencia: No se encontró el archivo de métricas. Logueando métricas vacías.")
        metrics = {}

    
    # Inicializar Vertex AI
    aiplatform.init()
   
    # Crear Experimento
    aiplatform.init(experiment=experiment_name)
    run_name = f"run-{model_display_name}-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    run = aiplatform.start_run(run = run_name)

    # Subir el modelo a Vertex AI
    artifact = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifacts.path,
        # artifact_uri=best_model_path.path.rsplit('/', 1)[0],
        serving_container_image_uri='us-central1-docker.pkg.dev/projectstylus01/vertex/mit-project-custom:latest'
    )
  
    aiplatform.log_metrics(metrics)

    aiplatform.end_run()

    print(f'Modelo subido a Vertex AI con ID: {artifact.resource_name}')

@dsl.pipeline(
    name='fraud-model-pipeline-experiments',
    description='Pipeline entrenamiento de modelos para el proyecto de detección de fraude bancario'
)
def pipeline(
    raw_data_path: str,
    params_config_path: str,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    n_trials: int = 50,
    c_fn: float = 20000,
    c_fp: float = 200,
    c_review: float = 30,
    human_hit_rate: float = 0.80,
    model_display_name: str = 'fraud-detection-model'
):

    split_data_task = split_data(
        processed_data_path=raw_data_path,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )

    train_models_task = train_models(
        train_data_path=split_data_task.outputs['train_data_path'],
        val_data_path=split_data_task.outputs['val_data_path'],
    )
 
    tuning_model_task = tuning_model(
        train_data_path=split_data_task.outputs['train_data_path'],
        val_data_path=split_data_task.outputs['val_data_path'],
        encoder_path=train_models_task.outputs['encode_path'],
        metrics_path=train_models_task.outputs['metrics_path'],
        params_config_path=params_config_path,
        n_trials=n_trials,
    )  

    calibrate_model_task = calibrate_model(
        val_data_path=split_data_task.outputs['val_data_path'],
        best_model_path=tuning_model_task.outputs['tuned_model_path'],
        encoder_path=train_models_task.outputs['encode_path'],
        c_fn=c_fn,
        c_fp=c_fp,
        c_review=c_review,
        human_hit_rate = human_hit_rate,
    )

    evaluate_model_task = evaluate_model(
            test_data_path=split_data_task.outputs['test_data_path'],
            best_model_path=tuning_model_task.outputs['tuned_model_path'],
            encode_path=train_models_task.outputs['encode_path'],
            scenery_metrics=calibrate_model_task.outputs['scenery_metrics'],
            human_hit_rate=human_hit_rate,
        )
     
    upload_model_task = upload_model_to_vertex(
        best_model_path=evaluate_model_task.outputs['final_tuned_model_path'],
        best_model_metrics_path=evaluate_model_task.outputs['evaluate_metrics_path'],
        encode_path=train_models_task.outputs['encode_path'],
        model_display_name=model_display_name
    )

if __name__ == '__main__':

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='fraud_model_pipeline_experiments.yaml'
    )