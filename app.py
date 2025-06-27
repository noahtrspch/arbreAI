import cv2
import os
import shutil
import gradio as gr
import requests
import base64
from ultralytics import YOLO
import numpy as np

# Modèle YOLO
model = YOLO("best.pt")

# Dictionnaire des facteurs de croissance par espèce
CROISSANCE_PAR_ESPECE = {
    # 🌱 Très rapide (1.3 – 1.5)
    "Albizia julibrissin": 1.5,
    "Paulownia tomentosa": 1.2,
    "Ailanthus altissima": 1.0,

    # 🌿 Croissance rapide (0.7 à 0.9 cm/an)
    "Populus nigra": 0.9,
    "Salix alba": 0.9,
    "Acer negundo": 0.8,
    "Catalpa bignonioides": 0.8,
    "Robinia pseudoacacia": 0.8,
    "Platanus × acerifolia": 0.8,
    "Malus sylvestris": 0.87,
    "Morus alba": 0.75,

    # 🌳 Croissance moyenne (0.5 à 0.7 cm/an)
    "Tilia cordata": 0.6,
    "Acer pseudoplatanus": 0.6,
    "Betula pendula": 0.6,
    "Pinus sylvestris": 0.5,
    "Prunus avium": 0.5,
    "Malus domestica": 0.5,
    "Celtis australis": 0.5,

    # 🍂 Croissance lente (0.3 à 0.45 cm/an)
    "Fagus sylvatica": 0.4,
    "Fraxinus excelsior": 0.4,
    "Carpinus betulus": 0.4,
    "Quercus petraea": 0.3,
    "Quercus robur": 0.35,
    "Castanea sativa": 0.35,
    "Juglans regia": 0.35,
    "Ginkgo biloba": 0.3,
    "Abies alba": 0.3,
    "Sequoiadendron giganteum": 0.3,

    # Valeur par défaut
    "Autre": 2.4
}

# Associations d’espèces équivalentes ou génériques
ESPECE_EQUIVALENTES = {
    "Malus sylvestris": "Malus domestica",
    "Malus sieversii": "Malus domestica",
    "Malus": "Malus domestica",

    "Pyrus pyraster": "Pyrus communis",
    "Pyrus": "Pyrus communis",

    "Prunus cerasus": "Prunus avium",
    "Prunus": "Prunus avium",

    "Betula pubescens": "Betula pendula",
    "Betula": "Betula pendula",

    "Quercus ilex": "Quercus robur",
    "Quercus": "Quercus robur",

    "Juglans nigra": "Juglans regia",
    "Juglans": "Juglans regia",

    "Tilia platyphyllos": "Tilia cordata",
    "Tilia": "Tilia cordata",

    "Acer campestre": "Acer pseudoplatanus",
    "Acer": "Acer pseudoplatanus",

    "Populus alba": "Populus nigra",
    "Populus": "Populus nigra",

    "Salix caprea": "Salix alba",
    "Salix fragilis": "Salix alba",
    "Salix": "Salix alba",

    "Fraxinus angustifolia": "Fraxinus excelsior",
    "Fraxinus": "Fraxinus excelsior",

    "Platanus orientalis": "Platanus × acerifolia",
    "Platanus": "Platanus × acerifolia",

    "Corylus maxima": "Corylus avellana",
    "Corylus": "Corylus avellana",
}

# Fonction de normalisation d’un nom scientifique
def normaliser_nom_espece(nom_scientifique):
    if nom_scientifique in CROISSANCE_PAR_ESPECE:
        return nom_scientifique
    if nom_scientifique in ESPECE_EQUIVALENTES:
        return ESPECE_EQUIVALENTES[nom_scientifique]
    genre = nom_scientifique.split(" ")[0]
    if genre in ESPECE_EQUIVALENTES:
        return ESPECE_EQUIVALENTES[genre]
    return nom_scientifique

# Fonction d’identification via Plant.id
def identify_species_from_api(image):
    _, im_arr = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(im_arr).decode('utf-8')

    url = "https://api.plant.id/v2/identify"
    headers = {"Content-Type": "application/json"}
    api_key = os.environ["API_KEY"]
    data = {
        "api_key": api_key,
        "images": [img_base64],
        "modifiers": ["crops_fast", "similar_images"],
        "plant_language": "fr",
        "plant_details": ["common_names"]
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        return "Espèce inconnue", 2.4, "", 0.0

    suggestions = response.json().get("suggestions", [])
    if not suggestions:
        return "Espèce non identifiée", 2.4, "", 0.0

    plant = suggestions[0]
    nom_scientifique = plant["plant_name"]
    noms_communs = ", ".join(plant.get("plant_details", {}).get("common_names", []))
    confiance = plant.get("probability", 0.0) * 100

    nom_normalise = normaliser_nom_espece(nom_scientifique)
    facteur = CROISSANCE_PAR_ESPECE.get(nom_normalise, 2.4)

    return nom_normalise, facteur, noms_communs, confiance


def estimate_age_and_species(tronc_img, largeur_txt, feuille_img, croissance_txt):
    # Gestion optionnelle de la feuille
    if feuille_img is not None and feuille_img.size != 0:
        espece, facteur_espece, noms_communs, conf = identify_species_from_api(feuille_img)
    else:
        espece, facteur_espece, noms_communs, conf = "Espèce non fournie", None, "", 0.0

    # Facteur croissance manuel selon choix utilisateur
    croissance_map = {
        "Croissance très rapide (Peuplier, orme, saule, érable)": 1.2,
        "Croissance rapide (Arbres fruitiers, bouleau, pin, mélèze, tilleul)": 0.8,
        "Croissance lente (Sapin, hêtre, frêne)": 0.5,
        "Croissance très lente (Chêne, noyer, châtaignier)": 0.3,
    }
    facteur_croissance = croissance_map.get(croissance_txt, 2.5)

    # Priorité au facteur espèce si identifié
    facteur = facteur_espece if facteur_espece is not None else facteur_croissance

    # Conversion largeur référentiel
    try:
        largeur_cm = float(largeur_txt)
    except:
        largeur_cm = 10.0

    # Nettoyage dossier YOLO temporaire
    if os.path.exists("runs/detect/predict"):
        shutil.rmtree("runs/detect/predict")

    cv2.imwrite("temp.jpg", cv2.cvtColor(tronc_img, cv2.COLOR_RGB2BGR))
    res = model.predict("temp.jpg", save=True, conf=0.4, project="runs/detect", name="predict", exist_ok=True)
    img_annot = cv2.imread("runs/detect/predict/temp.jpg")

    boxes = res[0].boxes
    names = res[0].names
    ref_w = tronc_w = None

    for b in boxes:
        cls = int(b.cls.cpu())
        label = names[cls]
        x1, y1, x2, y2 = b.xyxy[0].cpu().numpy()
        if label == "ref":
            ref_w = x2 - x1
        elif label == "tronc":
            tronc_w = x2 - x1

    if ref_w and tronc_w:
        ratio = largeur_cm / ref_w
        diam = tronc_w * ratio
        age = diam / facteur
        txt_b = f"ratio : {ratio:.1f} cm/pixel"
        txt_d = f"Diamètre estimé : {diam:.1f} cm"
        txt_a = f"Âge estimé : {age:.1f} ans"
        cv2.putText(img_annot, txt_d, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img_annot, txt_b, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img_annot, txt_a, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 0), 2)
    else:
        txt_d = "Diamètre non détecté"
        txt_a = "Âge non estimable"
        cv2.putText(img_annot, txt_d, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Texte infos avec indication si facteur par défaut ou non
    if facteur == facteur_croissance:
        info_txt = (
            f"Espèce estimée : {espece} ({noms_communs})\n"
            f"Confiance : {conf:.1f}%\n"
            f"Facteur de croissance : {facteur} (valeur manuelle)"
        )
    else:
        info_txt = (
            f"Espèce estimée : {espece} ({noms_communs})\n"
            f"Confiance : {conf:.1f}%\n"
            f"Facteur de croissance : {facteur}"
        )

    wiki_url = f"https://fr.wikipedia.org/wiki/{espece.replace(' ', '_')}"
    return cv2.cvtColor(img_annot, cv2.COLOR_BGR2RGB), txt_d, txt_a, info_txt, wiki_url

# Interface Gradio
with gr.Blocks() as demo:
    gr.Markdown("# Estimation d'âge + espèce d'un arbre")

    gr.Markdown(
        "### Instructions – Comment utiliser l'application\n\n"
        "Pour commencer, munnissez vous d'un carré blanc de et d'un téléphone\n\n"
        "1. Placez le référentiel **sur le tronc**\n"
        "- Prenez une photo du tronc de l’arbre\n"
        "- Assurez-vous que l’objet de référence est **bien visible** sur la photo.\n\n"
        "2. **(Optionnel) Prenez une photo d'une feuille de l'arbre**\n"
        "- Prenez la feuille en gros plan, bien nette.\n"
        "- Cela permet d’identifier automatiquement l’espèce de l’arbre.\n\n"
        "3. **Indiquez la largeur réelle de l’objet de référence**\n"
        "- Entrez cette valeur en centimètres (ex. : `10` pour un carré de 10x10).\n\n"
        "4. **Lancez l’analyse**\n"
        "- L’application détecte le tronc et l’objet de référence.\n"
        "- Elle estime le **diamètre du tronc**, puis l’**âge approximatif** de l’arbre.\n"
        "- Si une feuille est fournie, l’espèce est identifiée automatiquement."
    )

    with gr.Row():
        tronc_input = gr.Image(type="numpy", label="Tronc avec référentiel")
        feuille_input = gr.Image(type="numpy", label="Feuille de l'arbre (optionnel)")
        largeur_input = gr.Textbox(label="Largeur réelle du référentiel (cm)", placeholder="ex: 10")
        QCM_input = gr.Radio(
            choices=[
                "Croissance très rapide (Peuplier, orme, saule, érable)",
                "Croissance rapide (Arbres fruitiers, bouleau, pin, mélèze, tilleul)",
                "Croissance lente (Sapin, hêtre, frêne)",
                "Croissance très lente (Chêne, noyer, châtaignier)"
            ],
            label="Vitesse de croissance estimée"
        )

    bouton = gr.Button("Lancer l'analyse")

    image_out = gr.Image(type="numpy", label="Image annotée")
    diam_out = gr.Textbox(label="Diamètre estimé")
    age_out = gr.Textbox(label="Âge estimé")
    info_out = gr.Textbox(label="Informations espèce")
    lien_html = gr.HTML()

    def process_all(tronc, largeur, feuille, croissance_txt):
        img, diam, age, infos, wiki_url = estimate_age_and_species(tronc, largeur, feuille, croissance_txt)
        html_link = f'<a href="{wiki_url}" target="_blank"><button style="background-color:green;color:white;padding:10px;border:none;border-radius:5px;">🌿 En savoir plus sur Wikipedia</button></a>'
        return img, diam, age, infos, html_link

    bouton.click(
        fn=process_all,
        inputs=[tronc_input, largeur_input, feuille_input, QCM_input],
        outputs=[image_out, diam_out, age_out, info_out, lien_html]
    )

if __name__ == "__main__":
    demo.launch()
