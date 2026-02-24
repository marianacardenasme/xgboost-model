import json
import os
import io
import joblib
import pandas as pd
import xgboost as xgb
import numpy as np

MODEL_PATH = "/opt/ml/model"

MODEL_FILE = os.path.join(MODEL_PATH, "xgboost-model")
ENCODER_FILE = os.path.join(MODEL_PATH, "onehot_encoder.joblib")
SCALER_FILE = os.path.join(MODEL_PATH, "scaler.joblib")
FEATURE_COLUMNS_FILE = os.path.join(MODEL_PATH, "feature_columns.joblib")


# -------------------------------------------------
# 1️⃣ Cargar modelo y artefactos
# -------------------------------------------------
def model_fn(model_dir):
    booster = xgb.Booster()
    booster.load_model(MODEL_FILE)

    encoder = joblib.load(ENCODER_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_columns = joblib.load(FEATURE_COLUMNS_FILE)

   # --- PARCHE DE COMPATIBILIDAD (Versiones de Scikit-Learn) ---
    
    # 1. Solución para .sparse vs .sparse_output
    if hasattr(encoder, 'sparse_output') and not hasattr(encoder, 'sparse'):
        encoder.sparse = encoder.sparse_output
    elif hasattr(encoder, 'sparse') and not hasattr(encoder, 'sparse_output'):
        encoder.sparse_output = encoder.sparse
    
    # Forzar densidad para evitar errores de tipo Matrix en XGBoost
    encoder.sparse = False
    if hasattr(encoder, 'sparse_output'):
        encoder.sparse_output = False

    # 2. Solución para get_feature_names_out (v1.0+) vs get_feature_names (v0.24-)
    if not hasattr(encoder, 'get_feature_names_out'):
        # Si no existe, creamos un alias al método antiguo
        if hasattr(encoder, 'get_feature_names'):
            encoder.get_feature_names_out = encoder.get_feature_names
        else:
            # Caso extremo: si ninguno existe (muy raro), definimos una función lambda básica
            encoder.get_feature_names_out = lambda: [f"cat_{i}" for i in range(encoder.categories_[0].size)]

    # --------------------------------------------------
    
    return {
        "model": booster,
        "encoder": encoder,
        "scaler": scaler,
        "feature_columns": feature_columns
    }


# -------------------------------------------------
# 2️⃣ Transformar input
# -------------------------------------------------
def input_fn(request_body, request_content_type):

    if request_content_type == 'application/json':
        data = json.loads(request_body)
        df = pd.DataFrame([data])
        df.columns = [
            "did_you_get_injured_byaslip_or_fall_accident",
            "did_you_have_an_accident_at_work",
            "how_you_were_involved",
            "days_since_accident",
            "state_accident_occur",
            "were_you_affected_by_possible_malpractice",
            "were_you_involved_in_an_automobile_accident"
        ]

        df = df.replace(r'^\s*$', np.nan, regex=True)
        df = df.replace('None', np.nan)

        return df

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


# -------------------------------------------------
# 3️⃣ Hacer predicción
# -------------------------------------------------
def predict_fn(input_data, model_artifacts):
    df = input_data.copy()
    print("Columns received:", df.columns)
    
    booster = model_artifacts["model"]
    encoder = model_artifacts["encoder"]
    scaler = model_artifacts["scaler"]
    feature_columns = model_artifacts["feature_columns"]

    #df['accident_date'] = pd.to_datetime(df['accident_date'])
    #df['accident_date'] = (pd.Timestamp.today().normalize() - df['accident_date']).dt.days
    #df = df.rename(columns={'accident_date': 'days_since_accident'})
    
    df["days_since_accident"] = scaler.transform(
        df[["days_since_accident"]]
    )
    
    # Convertir booleanos a 0/1
    boolean_cols = [
        "did_you_get_injured_byaslip_or_fall_accident",
        "did_you_have_an_accident_at_work",
        "were_you_affected_by_possible_malpractice",
        "were_you_involved_in_an_automobile_accident"
    ]

    for col in boolean_cols:
        df[col] = df[col].map({True: 1, False: 0})
    
   # -------------------------------------------------
    # Validar estado permitido
    # -------------------------------------------------
    valid_states = ["New York", "New Jersey"]
    
    state_value = df.loc[0, "state_accident_occur"]
    
    if state_value not in valid_states:
        raise ValueError(
            f"Invalid state_accident_occur: '{state_value}'. "
            f"Allowed values are {valid_states}."
        )

    df["how_you_were_involved"] = df["how_you_were_involved"].fillna("Not_involved")
    categorical_cols = ["how_you_were_involved", "state_accident_occur"]    
    encoded = encoder.transform(df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df.index
    )
    
    final_df = pd.concat(
        [df[["days_since_accident"]], encoded_df],
        axis=1
    )

    # -------------------------------------------------
    # 4️⃣ Alinear columnas con entrenamiento
    # -------------------------------------------------
    final_df = final_df.reindex(columns=feature_columns, fill_value=0)

    # -------------------------------------------------
    # 5️⃣ Convertir a DMatrix y predecir
    # -------------------------------------------------
    dmatrix = xgb.DMatrix(final_df.values)

    predictions = booster.predict(dmatrix)

    return predictions

# -------------------------------------------------
# 4️⃣ Formatear salida
# -------------------------------------------------
def output_fn(predictions, content_type):

    if content_type == "application/json":
        return json.dumps({
            "predictions": predictions.tolist()
        }), content_type

    else:
        raise ValueError(f"Unsupported content type: {content_type}")
