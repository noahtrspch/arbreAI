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
    "Populus nigra": 1.5,              # Peuplier noir
    "Salix alba": 1.5,                 # Saule blanc
    "Albizia julibrissin": 1.5,        # Arbre de soie
    "Ailanthus altissima": 1.5,        # Ailante glanduleux
    "Acer negundo": 1.5,               # Érable négondo
    "Acer campestre": 1.5,             # Érable champêtre
    "Paulownia tomentosa": 1.5,        # Paulownia impérial
    "Eucalyptus globulus": 1.5,        # Eucalyptus
    "Catalpa bignonioides": 1.5,       # Catalpa commun
    "Bambusa vulgaris": 1.5,           # Bambou
    "Albizia julibrissin": 1.5,        # Arbre de soie
    "Melia azedarach": 1.5,               # Lilas de Perse
    "Leucaena leucocephala": 1.5,         # Leucaena
    "Sesbania grandiflora": 1.5,          # Sesbania
    "Erythrina crista-galli": 1.5,        # Érythrine crête-de-coq
    "Tamarix gallica": 1.5,               # Tamaris de France
    "Koelreuteria paniculata": 1.5,


                # 🌿 Rapide (2.0)
    "Betula pendula": 2.0,             # Bouleau verruqueux
    "Acer campestre": 2.0,             # Érable champêtre
    "Pinus sylvestris": 2.0,           # Pin sylvestre
    "Robinia pseudoacacia": 2.0,       # Robinier faux-acacia
    "Liriodendron tulipifera": 2.0,    # Tulipier de Virginie
    "Prunus avium": 2.0,               # Merisier
    "Tilia cordata": 2.0,              # Tilleul à petites feuilles
    "Acer pseudoplatanus": 2.0,        # Érable sycomore
    "Ulmus minor": 2.0,                # Orme champêtre
    "Larix decidua": 2.0,              # Mélèze d'Europe
    "Populus tremula": 2.0,            # Tremble
    "Acer platanoides": 2.0,           # Érable plane
    "Morus alba": 2.0,                 # Mûrier blanc
    "Cercis siliquastrum": 2.0,           # Arbre de Judée
    "Aesculus hippocastanum": 2.0,        # Marronnier d'Inde
    "Liquidambar styraciflua": 2.0,       # Copalme d'Amérique
    "Platanus × acerifolia": 2.0,         # Platane commun
    "Gleditsia triacanthos": 2.0,         # Févier d'Amérique


                    # 🍂 Lente (2.5)
    "Fagus sylvatica": 2.5,            # Hêtre commun
    "Fraxinus excelsior": 2.5,         # Frêne élevé
    "Carpinus betulus": 2.5,           # Charme commun
    "Magnolia grandiflora": 2.5,       # Magnolia à grandes fleurs
    "Malus domestica": 2.5,            # Pommier
    "Pyrus communis": 2.5,             # Poirier
    "Corylus avellana": 2.5,           # Noisetier
    "Amelanchier alnifolia": 2.5,      # Amélanchier
    "Sorbus aucuparia": 2.5,           # Sorbier des oiseleurs
    "Alnus glutinosa": 2.5,            # Aulne glutineux
    "Pseudotsuga menziesii": 2.5,      # Douglas vert
    "Zelkova serrata": 2.5,               # Zelkova du Japon
    "Cornus mas": 2.5,                    # Cornouiller mâle
    "Crataegus monogyna": 2.5,            # Aubépine monogyne
    "Prunus cerasifera": 2.5,             # Prunier-cerise
    "Celtis australis": 2.5,              # Micocoulier de Provence


    # 🌳 Très lente (3.0)
    "Quercus robur": 3.0,              # Chêne pédonculé
    "Castanea sativa": 3.0,            # Châtaignier
    "Juglans regia": 3.0,              # Noyer commun
    "Cedrus libani": 3.0,              # Cèdre du Liban
    "Taxus baccata": 3.0,              # If commun
    "Ginkgo biloba": 3.0,              # Arbre aux 40 écus
    "Abies alba": 3.0,                 # Sapin pectiné
    "Picea abies": 3.0,                # Épicéa commun
    "Quercus ilex": 3.0,               # Chêne vert
    "Buxus sempervirens": 3.0,         # Buis
    "Pinus pinea": 3.0,                # Pin parasol
    "Tsuga canadensis": 3.0,           # Pruche du Canada
    "Quercus suber": 3.0,                 # Chêne-liège
    "Sequoiadendron giganteum": 3.0,      # Séquoia géant
    "Taxodium distichum": 3.0,            # Cyprès chauve
    "Pinus nigra": 3.0,                   # Pin noir d'Autriche
    "Cedrus atlantica": 3.0,              # Cèdre de l'Atlas


    # Par défaut
    "Autre": 2.4                      # Espèce inconnue → croissance moyenne
}


def identify_species_from_api(image):
    """Utilise Plant.id pour identifier l'espèce et retourne toutes les infos utiles"""
    _, im_arr = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    img_base64 = base64.b64encode(im_arr).decode('utf-8')

    url = "https://api.plant.id/v2/identify"
    headers = {"Content-Type": "application/json"}
    data = {
        "api_key": "TNJE9RLrYvrMouBlZnnbtQnFBcoRM7ebwQ7BhK2tx7czLrHN6w",
        "images": [img_base64],
        "modifiers": ["crops_fast", "similar_images"],
        "plant_language": "fr",
        "plant_details": ["common_names"]
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        return "Espèce inconnue", 2.5, "", 0.0

    suggestions = response.json().get("suggestions", [])
    if not suggestions:
        return "Espèce non identifiée", 2.5, "", 0.0

    plant = suggestions[0]
    nom_scientifique = plant["plant_name"]
    noms_communs = ", ".join(plant.get("plant_details", {}).get("common_names", []))
    confiance = plant.get("probability", 0.0) * 100
    facteur = CROISSANCE_PAR_ESPECE.get(nom_scientifique, 2.4)

    return nom_scientifique, facteur, noms_communs, confiance

def estimate_age_and_species(tronc_img, largeur_txt, feuille_img):
    # 1. Identification de l’espèce
    if feuille_img is not None:
        espece, facteur, noms_communs, conf = identify_species_from_api(feuille_img)
    
    else:
        espece, facteur, noms_communs, conf = "Espèce non fournie", 2.5, "", 0.0

    # 2. Détection YOLO
    try:
        largeur_cm = float(largeur_txt)
    except:
        largeur_cm = 10.0

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

    # Texte d'infos
   

    if facteur == 2.4:
        info_txt = (
        f"Espèce estimée : {espece} ({noms_communs})\n"
        f"Confiance : {conf:.1f}%\n"
        f"Facteur de croissance : {facteur} (par défaut, espèce non répertoriée)"
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
        "1. **Prenez une photo du tronc de l’arbre**\n"
        "- Placez le référentiel **sur le tronc**.\n"
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
        
        
    

    bouton = gr.Button("Lancer l'analyse")

    image_out = gr.Image(type="numpy", label="Image annotée")
    diam_out = gr.Textbox(label="Diamètre estimé")
    age_out = gr.Textbox(label="Âge estimé")
    info_out = gr.Textbox(label="Informations espèce")
    lien_html = gr.HTML()


    def process_all(tronc, largeur, feuille):
        img, diam, age, infos, wiki_url = estimate_age_and_species(tronc, largeur, feuille)
        html_link = f'<a href="{wiki_url}" target="_blank"><button style="background-color:green;color:white;padding:10px;border:none;border-radius:5px;">🌿 En savoir plus sur Wikipedia</button></a>'
        return img, diam, age, infos, html_link

    bouton.click(fn=process_all, inputs=[tronc_input, largeur_input, feuille_input],
                 outputs=[image_out, diam_out, age_out, info_out, lien_html])



if __name__ == "__main__":
    demo.launch()
