# A.I.D.A.
Artificially Intelligent Digital Assistant
Detaljan plan implementacije (Tvoja arhitektura)
Ovaj plan prati logičku strukturu foldera na tvom kompjuteru. Pravićete takozvani "Baby LLM" (model od oko 15 do 30 miliona parametara). Matematika je apsolutno ista kao kod velikih modela, ali ovaj ćeš moći da istreniraš i pokreneš na svom računaru.

📂 1_neural_core (Faze 1 i 2: Mozak)

config.py: Fajl u kom držiš hiperparametre (koliko slojeva ima model, veličina vektora, broj "glava" za pažnju). Ovo ti omogućava da lako smanjuješ i povećavaš model.

tokenizer.py: Skripta koja učitava tekst i pretvara slova/reči u brojeve (ID-jeve).

model.py: Srce projekta. Čist PyTorch kod. Ovde pišeš klasu za Multi-Head Self-Attention (matematiku koja modelu daje razumevanje konteksta), FeedForward mrežu i spajaš ih u Transformer blokove.

📂 2_training_loop (Faza 3: Učenje)

dataset.py: Kod koji secka ogromne tekstualne fajlove u male "prozore" koje grafička kartica može da svari.

pretrain.py: Petlja koja vrti model kroz tekst. Model uči da pogađa sledeće slovo i računa svoju grešku (Loss).

sft_trainer.py: Supervised Fine-Tuning. Ovde prebacuješ model iz "brbljivca" u alat. Praviš mu dataset gde ga strogo kažnjavaš ako na matematičko pitanje odgovori tekstom, i nagrađuješ ga ako ispiše savršen JSON (npr. {"alat": "kalkulator", "a": 5, "b": 10}).

📂 3_symbolic_engine (Faza 4: Ruke i Logika)

tools.py: Ovde nema AI-ja, samo determinizam. Tvoje Python funkcije: def saberi(a, b), def pretrazi_bazu(ime).

parser.py: Bezbednosni sloj. Ako LLM pogreši u kucanju JSON-a (npr. zaboravi navodnike), ovaj kod sprečava da tvoj program pukne i šalje modelu grešku nazad da je ispravi.

📂 4_agent_orchestration (Faza 5: Spajanje)

react_loop.py: Čuvena Reason and Act (ReAct) while petlja. Korisnik pita -> Model izbaci JSON -> Python ga hvata, zaustavlja model, pokreće funkciju iz tools.py -> Rezultat se vraća modelu -> Model izgovara tačan odgovor korisniku.