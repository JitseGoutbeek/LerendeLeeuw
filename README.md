# LerendeLeeuw
Dit is de Nederlandse versie van het LearningLion-project. Voor de Engelse versie zie: https://github.com/SSC-ICT-Innovatie/LearningLion

## Beschrijving van het project
Het project is een studie naar het gebruik van generatieve AI om de diensten van SSC-ICT te verbeteren door medewerkers te ondersteunen en interne processen te optimaliseren. In het bijzonder ligt de focus op generatieve grote taalmodellen (LLM), in de vorm van Retrieval Augmented Generation (RAG), omdat ze de grootste impact kunnen hebben op het dagelijkse werk van SSC-ICT-medewerkers. De opdracht (Speech Recognition & AI) is goedgekeurd door de SpB en loopt sinds begin 2023.

## Repository

De huidige repository is gebaseerd op de repository gemaakt door PBL onder leiding van Stefan Troost (https://github.com/pbl-nl/appl-docchat), aangepast voor onze specifieke gebruiksscenario's. De demo is een voorbeeld van Retrieval Augmented Generation (RAG) en maakt het mogelijk om zowel Open als Closed-Source LLM's te gebruiken en te evalueren. Het maakt gebruik van Langchain, Chroma en Streamlit, onder andere, om documentvragen en -antwoorden uit te voeren. Op dit moment is het voornamelijk een kloon met enkele toevoegingen (om het lokaal uitvoeren ervan bijvoorbeeld gemakkelijker te maken) en bevat het aanvullende projectdocumenten (in project_docs) die achtergrondinformatie bieden over dit project, de beslissingen die we hebben genomen en het onderzoek dat we hebben uitgevoerd.


## Gebruik van de repository
Deze repo is getest op een Windows-platform.
De instructies zijn geschreven onder de veronderstelling dat u Anaconda en Python hebt geïnstalleerd. Zo niet, download Python (https://www.python.org/downloads/) en volg deze installatiegids voor Anaconda: https://docs.anaconda.com/free/anaconda/install/windows/.

## Voorbereiding
Open uw terminal (bijvoorbeeld Anaconda PowerShell-prompt) en open de map waarin u deze repository wilt installeren, maak bijvoorbeeld een map genaamd Repositories en open deze in uw terminal (u kunt het commando cd gebruiken om naar de benodigde map te gaan, bijvoorbeeld cd windows/users/repositories).
Clone deze repo met het commando <br><code>git clone https://github.com/JitseGoutbeek/LerendeLeeuw.git </code><br>
Maak een submap vector_stores in de hoofdmap van de gekloonde repo
Ga ook in deze omgeving met <br><code> cd Lerendeleeuw </code><br>

## Conda virtuele omgeving instellen
Open een Anaconda-prompt of andere opdrachtprompt
Ga naar de hoofdmap van het project en maak een Python-omgeving met conda met behulp van command-line commando<br>
<code>conda env create -f lerendeleeuw.yml</code><br>
NB: De naam van de omgeving is standaard lerendeleeuw. Het kan worden gewijzigd in een naam naar keuze in de eerste regel van het yml-bestand
Activeer deze omgeving met behulp van command-line commando <br><code>conda activate lerendeleeuw</code><br>
Alle benodigde pakketten kunnen nu worden geïnstalleerd met het command-line commando<br>
<code>pip install -r requirements.txt</code><br> (sla dit over als u liever in een virtuele omgeving werkt)

## Pip virtuele omgeving instellen
! U hoeft geen pip virtuele omgeving in te stellen als u uw conda-omgeving al hebt ingesteld.

Open een Anaconda-prompt of andere opdrachtprompt
Ga naar de hoofdmap van het project en maak een Python-omgeving met pip met behulp van command-line commando<br>
<code>python -m venv venv</code><br>
Dit maakt een basis virtuele omgevingsmap genaamd venv in de hoofdmap van uw project<br>
NB: De gekozen naam van de omgeving is hier venv. Het kan worden gewijzigd in een naam naar keuze.
Activeer deze omgeving met behulp van command-line commando<br>
<code>venv\Scripts\activate</code><br>
Alle benodigde pakketten kunnen nu worden geïnstalleerd met het command-line commando<br>
<code>pip install -r requirements.txt</code><br>

## Het kiezen van uw parameters
Het bestand settings_template.py bevat alle parameters die kunnen worden gebruikt en moet worden gekopieerd naar settings.py. Vul in settings.py de parameterwaarden in die u wilt gebruiken voor uw gebruikssituatie. Voorbeelden en beperkingen voor parameterwaarden worden gegeven in de commentaarl...

Als u dit lokaal wilt uitvoeren met behulp van Ollama, moet u het volgende doen:

1. Download Ollama: https://ollama.com/
2. Maak in de instellingen de keuze `LLM_TYPE = "local_llm"`
3. Ga naar modellen en kies een model dat u wilt gebruiken, we raden zephyr en mistral-openorca met name aan bij het werken in niet-Engelse talen, als u het geheugen heeft, kan Mixtral ook een goede keuze zijn (dit model is 26 GB vergeleken met 4 GB voor mistral-openorca, maar zou beter moeten presteren). U kunt zelf experimenteren met verschillende modellen. Stel `LLM_MODEL_TYPE = <model_name>` in. Dus in ons voorbeeld `LLM_MODEL_TYPE = "mistral-openorca"`.
4. Download het model op uw apparaat door <code> ollama pull model_name </code> in uw terminal te typen, bijvoorbeeld <code> ollama pull mistral-openorce </code>
5. In de instellingen, kies `EMBEDDING_PROVIDER = "local_embeddings"`, kies een embeddings-model van: https://huggingface.co/models?library=sentence-transformers&sort=trending. Stel in de instellingen het embedding-model in op die naam `EMBEDDINGS_MODEL = "jegormeister/bert-base-dutch-cased"`.

Als u het laatste wilt doen en de LLM's of Embedding-modellen van OpenAI (GPT-3.5 / GPT-4 / Text-Embeddings-Ada-002) wilt gebruiken, moet u het volgende doen:

1. Ga naar [https://platform.openai.com/docs/overview](https://auth0.openai.com/u/signup/identifier?state=hKFo2SAxWUNzRWVLbFJfWnFkYzAyNm5oTFRkbF8xZ2NJNkhSV6Fur3VuaXZlcnNhbC1sb2dpbqN0aWTZIDJBaFhUNTB6RWx1VkRaSXZ6U3JLQ2NDaUdwY255Mjlao2NpZNkgRFJpdnNubTJNdTQyVDNLT3BxZHR3QjNOWXZpSFl6d0Q) en log in of meld u aan, deze optie kost geld (een fractie van een cent per vraag).
2. Maak een bestand `.env` aan en voer uw OpenAI API-sleutel in op de eerste regel van dit bestand: <br>
<code>OPENAI_API_KEY="sk-....."</code><br>
Sla het `.env`-bestand op en sluit het. <br>
* Als u nog geen OpenAI API-sleutel heeft, kunt u er hier een verkrijgen: https://platform.openai.com/account/api-keys.
* Klik op + Nieuwe geheime sleutel maken.
* Voer een identificatienaam in (optioneel) en klik op Geheime sleutel maken.

Als u een van de vele open source modellen op Huggingface wilt gebruiken en hun hardware wilt gebruiken (dus extern uitvoeren):

* Registreer op https://huggingface.co/join.
* Wanneer u bent geregistreerd en ingelogd, kunt u uw API-sleutel van Hugging Face vinden in uw profielinstellingen.
* Voer uw Hugging Face API-sleutel in op de tweede regel van het `.env`-bestand:<br>
<code>HUGGINGFACEHUB_API_TOKEN="hf_....."</code><br>

U moet ook beslissen hoe u uw documenten wilt 'chunken'. Voordat een document wordt ingebed, wordt het opgedeeld in stukken tekst van een bepaalde lengte en deze stukken worden vervolgens gevectoriseerd en het meest relevante (hoogste gelijkenisscore) ervan wordt opgehaald en gebruikt als invoer door de LLM. In instellingen kunt u ongeveer kiezen hoe lang u de chunks wilt maken en hoeveel u er wilt ophalen.

## Gebruik van het repository

Om dit repository te gebruiken, moet u zich in de juiste map en het juiste virtuele omgeving bevinden. Als u zich niet in de juiste virtuele omgeving bevindt, moet u uw virtuele omgeving activeren:
<br><code>conda activate learninglion</code><br>
Als u zich niet in de juiste map bevindt (LearningLion), moet u naar die map gaan. Met het commando cd kunt u naar de juiste map gaan, bijvoorbeeld:
<br><code> cd windows/users/repositories/LearningLion </code><br> natuurlijk als u het LearningLion repository naar een andere locatie hebt gekloond, moet u aanpassen om naar de locatie te gaan waar u het hebt opgeslagen.

Dit maakt een paar functionaliteiten mogelijk:
* U kunt een gebruikersinterface starten waarin u een documentmap kunt selecteren en er vragen over kunt stellen
* U kunt ook uw documenten insluiten en vragen stellen via uw opdrachtterminal, daarnaast kunt u via de opdrachtterminal automatisch een grotere reeks vragen stellen en de antwoorden opslaan of automatisch evalueren. We zullen hieronder doorlopen hoe u deze verschillende opties kunt gebruiken, vergeet niet om de juiste instellingen te kiezen voordat u een functionaliteit gebruikt.

### Vragen stellen over uw documenten via een gebruikersinterface
Met het opdrachtregelcommando: `streamlit run streamlit_app.py` kunt u een interface starten. Wanneer dit commando wordt gebruikt, wordt automatisch een browsersessie geopend. In deze browser kunt u vragen stellen over documenten die u in de map docs heeft geplaatst, er staat een kleine uitleg aan de linkerkant van het scherm die de eerste keer moet worden gelezen bij het gebruik van de online interface.

### Documenten insluiten
We kunnen ook vragen stellen, één voor één of in meervouden, via de opdrachtterminal. In dit geval moeten we eerst de documenten vectoriseren waarover we vragen willen stellen.
Hiervoor wordt ingest.py gebruikt. Maak hiervoor eerst een submap in de map docs met de documenten waarover u vragen wilt stellen, geef de map een herkenbare naam met alleen kleine letters. Typ vervolgens `python ingest.py` in een geactiveerde virtuele omgeving. Dit vraagt u in welke map u wilt vectoriseren en u kunt de naam van de map met de relevante documenten typen.

### Documenten bevragen
Om vragen te stellen over documenten in uw virtuele omgeving kunt u het bestand query.py gebruiken (de benodigde map moet zijn ingesloten). Typ om dit te doen gewoon `python query.py` in uw geactiveerde virtuele omgeving, en typ de naam van de map die de documenten bevat waarover u vragen wilt stellen.

### Meerdere documenten bevragen met meerdere vragen in batch
U kunt ook meerdere vragen tegelijk stellen, de code zal elke vraag door uw RAG-pijplijn uitvoeren en de relevante vragen en antwoorden opslaan in een `.csv`-bestand. Om dit te doen, ga naar de map die de relevante documenten bevat in docs, maak een submap genaamd "review" en maak een `.txt`-bestand in die map met de relevante vragen. Typ nu in uw opdrachtvenster:
`python review.py` en typ de naam van de map die de documenten bevat waar uw vragen over gaan. Als u dit lokaal doet en het `.txt`-bestand bevat veel vragen, kan het even duren, maar aan het eind moet een `.csv`-bestand worden opgeslagen in de review map met de vragen en automatisch gegenereerde antwoorden.

### Evaluatie van Vraag- en Antwoordresultaten
Het bestand evaluate.py kan worden gebruikt om de gegenereerde antwoorden te evalueren voor een lijst met vragen, op voorwaarde dat het bestand eval.json bestaat, met daarin niet alleen de lijst met vragen maar ook de bijbehorende lijst met gewenste antwoorden (grondwaarheid).
Evaluatie wordt op mapniveau uitgevoerd in de geactiveerde virtuele omgeving met het commandoregelcommando: `python evaluate.py`. Het is ook mogelijk om een evaluatie uit te voeren over alle mappen met `python evaluate_all.py`. De resultaten worden gegenereerd in een `.tsv`-bestand. Dit bestand kan worden geopend in Microsoft Excel om een duidelijk overzicht van de resultaten te krijgen. Meer informatie over de metrieken die in de resultaten worden getoond, is te vinden op: https://docs.ragas.io/en/stable/getstarted/evaluation.html#metrics/.

#### Genereren van testgegevens voor evaluatie
TODO

### Monitoring van de evaluatieresultaten via een Streamlit-gebruikersinterface
Alle evaluatieresultaten kunnen worden bekeken door gebruik te maken van een speciale gebruikersinterface.
Deze evaluatie-UI kan worden gestart met het commandoregelcommando:<br>
<code>streamlit run streamlit_evaluate.py</code><br>
Wanneer dit commando wordt gebruikt, wordt automatisch een browsersessie geopend.


### Insluiten en bevragen van documenten via een Flask-gebruikersinterface
De hierboven beschreven functionaliteiten kunnen ook worden gebruikt via een Flask-gebruikersinterface.
De Flask-UI kan worden gestart in de geactiveerde virtuele omgeving met het commandoregelcommando:<br>
<code>python flask_app.py</code>
De Flask-UI is op maat gemaakt voor toekomstig gebruik in productie en bevat meer inzicht in de chunks (gebruikt) en bevat ook gebruikersbeheerfunctionaliteit, onder andere.<br>
Voor een meer gedetailleerde beschrijving en installatie, zie het readme-bestand in de flask_app-map

## Tools
- **LangChain**: Framework voor het ontwikkelen van applicaties aangedreven door taalmodellen
- **LlamaCPP**: Python-bindings voor de Transformer-modellen geïmplementeerd in C/C++
- **FAISS**: Open-source bibliotheek voor efficiënte gelijkeniszoekopdrachten en clustering van dichte vectoren.
- **Sentence-Transformers (all-MiniLM-L6-v2)**: Open-source voorgetraind transformer-model voor het insluiten van tekst in een 384-dimensionale dichte vectorruimte voor taken zoals clustering of semantisch zoeken.
- **Llama-2-7B-Chat**: Open-source fijnafgestemd Llama 2-model ontworpen voor chatdialogen. Maakt gebruik van openbaar beschikbare instructiedatasets en meer dan 1 miljoen menselijke annotaties. 

## Erkenningen
Dit is een fork van [appl-docchat van het Planbureau voor de Leefomgeving](https://github.com/pbl-nl/appl-docchat).

## Referenties
Dit repo is voornamelijk geïnspireerd door:
- https://docs.streamlit.io/
- https://docs.langchain.com/docs/
- https://blog.langchain.dev/tutorial-chatgpt-over-your-data/
- https://github.com/PatrickKalkman/python-docuvortex/tree/master
- https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/
- https://github.com/explodinggradients/ragas
