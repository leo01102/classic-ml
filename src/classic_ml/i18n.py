# src/classic_ml/i18n.py

import locale

# Default to system language if supported, otherwise spanish
try:
    _default_lang = locale.getdefaultlocale()[0][:2]
    _lang = "en" if _default_lang == "en" else "es"
except Exception:
    _lang = "es"

_translations = {
    "en": {
        "app_title": "CLASSIC-ML TUI",
        "select_algo": "Select Algorithm",
        "menu_perceptron": "Perceptron",
        "menu_linear": "Linear Regression",
        "menu_kmeans": "K-Means",
        "menu_exit": "Exit",
        "menu_lang": "Language",
        
        "label_path": "CSV File",
        "label_k": "Clusters (K)",
        "label_iters": "Max Iterations",
        "label_tol": "Tolerance",
        "label_lr": "Learning Rate",
        "label_epochs": "Epochs",
        "label_lambda": "Regularization (λ)",
        "label_mode": "Mode",
        "label_w_init": "Initial Weights",
        "label_predict": "Predict (comma sep)",
        "label_bias": "Bias",
        "label_range": "Range (min,max,step)",
        
        "mode_binary": "Binary",
        "mode_ovr": "One-vs-Rest",
        "mode_test": "Test Weights",
        "mode_grid": "Grid Search",

        "btn_load_iris": "Load Iris",
        "btn_run": "Run Algorithm",
        
        "msg_no_data": "Please load a dataset or type a file path",
        "msg_error_load": "Error loading file: {e}",
        "msg_converged": "Converged in {i} iterations",
        "msg_not_converged": "Did not converge",
        "msg_pred_result": "Prediction: {res}",
        "msg_grid_solutions": "Solutions found: {n}",
        "msg_accuracy": "Accuracy: {acc}%",
        "msg_results": "Results",
        
        "footer_quit": "Quit",
        "footer_lang": "Toggle Language",
        "footer_reset": "Reset",
        
        "kmeans_title": "Interactive K-means",
        "lr_title": "Linear Regression",
        "perceptron_title": "Perceptron",
        "title_config": "Configuration",
        "title_results": "Results",
        "title_viz": "Visualization",
    },
    "es": {
        "app_title": "CLASSIC-ML TUI",
        "select_algo": "Seleccionar Algoritmo",
        "menu_perceptron": "Perceptrón",
        "menu_linear": "Regresión Lineal",
        "menu_kmeans": "K-Means",
        "menu_exit": "Salir",
        "menu_lang": "Idioma",
        
        "label_path": "Archivo CSV",
        "label_k": "Clusters (K)",
        "label_iters": "Iteraciones Máx",
        "label_tol": "Tolerancia",
        "label_lr": "Tasa Aprendizaje",
        "label_epochs": "Épocas",
        "label_lambda": "Regularización (λ)",
        "label_mode": "Modo",
        "label_w_init": "Pesos Iniciales",
        "label_predict": "Predecir (sep coma)",
        "label_bias": "Bias",
        "label_range": "Rango (ini,fin,paso)",

        "mode_binary": "Binario",
        "mode_ovr": "One-vs-Rest",
        "mode_test": "Probar Pesos",
        "mode_grid": "Búsqueda en Grilla",
        
        "btn_load_iris": "Cargar Iris",
        "btn_run": "Ejecutar",
        
        "msg_no_data": "Por favor, cargue un conjunto de datos o escriba una ruta",
        "msg_error_load": "Error al cargar: {e}",
        "msg_converged": "Convergió en {i} iteraciones",
        "msg_not_converged": "No convergió",
        "msg_pred_result": "Predicción: {res}",
        "msg_grid_solutions": "Soluciones: {n}",
        "msg_accuracy": "Precisión: {acc}%",
        "msg_results": "Resultados",
        
        "footer_quit": "Salir",
        "footer_lang": "Cambiar Idioma",
        "footer_reset": "Reiniciar",
        
        "kmeans_title": "K-means Interactivo",
        "lr_title": "Regresión Lineal",
        "perceptron_title": "Perceptrón",
        "title_config": "Configuración",
        "title_results": "Resultados",
        "title_viz": "Visualización",
    }
}

def set_language(lang):
    global _lang
    if lang in _translations:
        _lang = lang

def t(key, **kwargs):
    # Fallback to English if key missing in current lang
    text = _translations.get(_lang, _translations["es"]).get(key, _translations["en"].get(key, key))
    if kwargs:
        try:
            return text.format(**kwargs)
        except KeyError:
            return text
    return text

def get_language():
    return _lang

def toggle_language():
    global _lang
    _lang = "es" if _lang == "en" else "en"
    return _lang
