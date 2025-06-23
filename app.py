import gradio as gr
import cv2
import os
import shutil
import numpy as np
import requests
from ultralytics import YOLO

# Charger le modèle YOLO pour détection
model = YOLO("best.pt")

# Clé API Plant.id
PLANT_ID_API_KEY = "TNJE9RLrYvrMouBlZnnbtQnFBcoRM7ebwQ7BhK2tx7czLrHN6w"

# Dictionnaire espèce → coefficient de croissance
coefs_especes = {
    "quercus robur": 3.0,       # Chêne pédonculé
    "populus nigra": 1.5,       # Peuplier noir
    "fagus sylvatica": 2.5,     # Hêtre
    "acer platanoides": 1.5,    # Érable
    "betula pendula": 2.0,      # Bouleau
    "pinus sylvestris": 2.0,    # Pin sylvestre
    "castanea sativa": 3.0,     # Châtaignier
    "juglans regia": 3.0,       # Noyer
    "salix alba": 1.5           # Saule blanc
}

def predict_species(image_np):
    _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    image_bytes = img_encoded.tobytes()

    headers = {
        "Content-Type": "application/json",
        "Api-Key": PLANT_ID_API_KEY
    }

    payload = {
        "images": [image_bytes.hex()],
        "organs": ["leaf"],
        "modifiers": ["crops_fast", "similar_images"],
        "plant_language": "fr",
        "plant_details": ["common_names", "url", "wiki_description"]
    }

    response = requests.post("https://api.plant.id/v2/identify", headers=headers, json=payload)

    if response.status_code == 200:
        result = response.json()
        if result["suggestions"]:
            plant = result["suggestions"][0]
            name = plant["plant_name"].lower()
            desc = plant["plant_details"]["wiki_description"]["value"] if "plant_details" in plant else ""
            return name, f"Espèce : {name}\n\n{desc}"
        else:
            return None, "Aucune espèce identifiée"
    else:
        return None, f"Erreur API : {response.status_code}"

def detect_objects_and_species(image, ref_largeur_txt, croissance_txt, feuille_img):
    try:
        ref_cm = float(ref_largeur_txt)
    except:
        ref_cm = 10.0

    # Coefficient par défaut selon sélection utilisateur
    croissance_map = {
        "Croissance très rapide (Peuplier, orme, saule, érable)": 1.5,
        "Croissance rapide (Arbres fruitiers, bouleau, pin, mélèze, tilleul)": 2.0,
        "Croissance lente (Sapin, hêtre, frêne)": 2.5,
        "Croissance très lente (Chêne, noyer, châtaignier)": 3.0,
    }
    facteur = croissance_map.get(croissance_txt, 2.5)

    # Nettoyer ancien dossier
    if os.path.exists("runs/detect/predict"):
        shutil.rmtree("runs/detect/predict")

    cv2.imwrite("temp.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    results = model.predict("temp.jpg", save=True, conf=0.4,
                            project="runs/detect", name="predict", exist_ok=True)

    img_annot = cv2.imread("runs/detect/predict/temp.jpg")
    boxes = results[0].boxes
    names = model.names

    ref_w = tronc_w = None
    for box in boxes:
        cls_id = int(box.cls.cpu().item())
        label = names[cls_id]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        w = x2 - x1
        if label == "ref":
            ref_w = w
        elif label == "tronc":
            tronc_w = w

    species_detected = None
    if feuille_img is not None:
        species_detected, species_info = predict_species(feuille_img)
        if species_detected in coefs_especes:
            facteur = coefs_especes[species_detected]
            species_info += f"\n\n💡 Facteur de croissance défini automatiquement : {facteur}"
    else:
        species_info = "Aucune image de feuille fournie."

    if ref_w and tronc_w:
        diam = tronc_w * (ref_cm / ref_w)
        age = diam / facteur
        msg_diam = f"Diamètre estimé : {diam:.1f} cm"
        msg_age = f"Âge estimé : {age:.1f} ans"
        cv2.putText(img_annot, msg_diam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.putText(img_annot, msg_age, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,0), 2)
    else:
        msg_diam = "Impossible d’estimer le diamètre"
        msg_age = "Impossible d’estimer l’âge"
        cv2.putText(img_annot, msg_diam, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    return (
        cv2.cvtColor(img_annot, cv2.COLOR_BGR2RGB),
        msg_diam,
        msg_age,
        species_info
    )

# Interface Gradio
demo = gr.Interface(
    fn=detect_objects_and_species,
    inputs=[
        gr.Image(type="numpy", label="Image du tronc avec référentiel"),
        gr.Textbox(label="Largeur réelle du référentiel (cm)", placeholder="ex: 10"),
        gr.Radio(
            choices=[
                "Croissance très rapide (Peuplier, orme, saule, érable)",
                "Croissance rapide (Arbres fruitiers, bouleau, pin, mélèze, tilleul)",
                "Croissance lente (Sapin, hêtre, frêne)",
                "Croissance très lente (Chêne, noyer, châtaignier)"
            ],
            label="Vitesse de croissance estimée (utilisée si l’espèce n’est pas reconnue)"
        ),
        gr.Image(type="numpy", label="Image de la feuille (facultatif)")
    ],
    outputs=[
        gr.Image(type="numpy", label="Image annotée"),
        gr.Textbox(label="Diamètre estimé"),
        gr.Textbox(label="Âge estimé"),
        gr.Textbox(label="Résultat identification d’espèce")
    ],
    title="🌳 Estimation âge + espèce d’un arbre",
    description="Téléversez une image du tronc avec référentiel. Ajoutez une feuille (facultatif) pour identifier l’espèce et adapter automatiquement le calcul de l’âge."
)

if __name__ == "__main__":
    demo.launch()
