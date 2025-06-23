import cv2
import os
import shutil
import gradio as gr
import requests
import base64
from ultralytics import YOLO
import numpy as np

# Charger modèle YOLO


model = YOLO("best.pt")

# API KEY Plant.id
PLANT_ID_API_KEY = "TNJE9RLrYvrMouBlZnnbtQnFBcoRM7ebwQ7BhK2tx7czLrHN6w"  # ← remplacez par votre vraie clé

def estimate_age(image_tronc, largeur_ref_txt, croissance_txt):

    try:
        largeur_ref_cm = float(largeur_ref_txt)
    except:
        largeur_ref_cm = 10.0  # défaut

    # Croissance
    croissance_map = {
        "Croissance très rapide (Peuplier, orme, saule, érable)": 1.5,
        "Croissance rapide (Arbres fruitiers, bouleau, pin, mélèze, tilleul)": 2.0,
        "Croissance lente (Sapin, hêtre, frêne)": 2.5,
        "Croissance très lente (Chêne, noyer, châtaignier)": 3.0,
    }
    facteur = croissance_map.get(croissance_txt, 2.5)

    # Nettoyage ancien dossier
    pred_dir = "runs/detect/predict"
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)

    # Sauver image
    img_path = "temp.jpg"
    cv2.imwrite(img_path, cv2.cvtColor(image_tronc, cv2.COLOR_RGB2BGR))

    results = model.predict(
        img_path, save=True, conf=0.4, project="runs/detect", name="predict", exist_ok=True
    )

    # Lire image annotée
    img_result_path = os.path.join("runs/detect/predict", os.path.basename(img_path))
    img_annot = cv2.imread(img_result_path)






    boxes = results[0].boxes
    names = model.names

    ref_px = tronc_px = None
    for box in boxes:
        cls_id = int(box.cls.cpu().item())
        label = names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        width = x2 - x1
        if label == "ref":
            ref_px = width
        elif label == "tronc":
            tronc_px = width

    if ref_px and tronc_px:
        ratio = largeur_ref_cm / ref_px
        diametre_cm = tronc_px * ratio
        age_estime = diametre_cm / facteur

        txt_diam = f"Diamètre estimé : {diametre_cm:.1f} cm"
        txt_age = f"Âge estimé : {age_estime:.1f} ans"

        cv2.putText(img_annot, txt_diam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img_annot, txt_age, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)
    else:
        txt_diam = "Diamètre non détecté"
        txt_age = "Âge non estimable"
        cv2.putText(img_annot, txt_diam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return cv2.cvtColor(img_annot, cv2.COLOR_BGR2RGB), txt_diam, txt_age

def identify_species(image_feuille):
    _, im_arr = cv2.imencode('.jpg', cv2.cvtColor(image_feuille, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(im_arr).decode('utf-8')

    url = "https://api.plant.id/v2/identify"
    headers = {"Content-Type": "application/json"}
    data = {
        "api_key": PLANT_ID_API_KEY,
        "images": [img_base64],
        "modifiers": ["crops_fast", "similar_images"],
        "plant_language": "fr",
        "plant_details": ["common_names", "url", "wiki_description"]
    }

    response = requests.post(url, json=data, headers=headers)

    if response.status_code != 200:
        return "Erreur API : clé invalide ou requête échouée."

    res_json = response.json()
    suggestions = res_json.get("suggestions", [])
    if not suggestions:
        return "Espèce non identifiée."

    best = suggestions[0]
    nom = best["plant_name"]
    noms_communs = ", ".join(best.get("plant_details", {}).get("common_names", []))
    conf = best["probability"] * 100

    return f"{nom} ({noms_communs}) - {conf:.1f}% de confiance"

# Interface Gradio
demo = gr.Interface(
    fn=lambda tronc, ref, croiss, feuille: (*estimate_age(tronc, ref, croiss), identify_species(feuille)),
    inputs=[
        gr.Image(type="numpy", label="Image du tronc avec feuille de référence"),
        gr.Textbox(label="Largeur réelle de la feuille blanche (cm)", placeholder="ex: 10"),
        gr.Radio(
            label="Vitesse de croissance estimée",
            choices=[
                "Croissance très rapide (Peuplier, orme, saule, érable)",
                "Croissance rapide (Arbres fruitiers, bouleau, pin, mélèze, tilleul)",
                "Croissance lente (Sapin, hêtre, frêne)",
                "Croissance très lente (Chêne, noyer, châtaignier)"
            ],
            value="Croissance rapide (Arbres fruitiers, bouleau, pin, mélèze, tilleul)"
        ),
        gr.Image(type="numpy", label="Image d'une feuille de l'arbre")
    ],
    outputs=[
        gr.Image(type="numpy", label="Image annotée"),
        gr.Textbox(label="Diamètre estimé"),
        gr.Textbox(label="Âge estimé"),
        gr.Textbox(label="Espèce identifiée via Plant.id")
    ],
    title="Estimation de l'âge + identification d'un arbre",
    description="Téléversez une photo du tronc avec un repère (ex: feuille A4), puis une autre photo de la feuille pour identifier l'espèce avec l'IA Plant.id."




)

if __name__ == "__main__":
    demo.launch()
