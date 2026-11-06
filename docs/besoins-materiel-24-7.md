# Besoins matériels théoriques : GLaDOS 24 h/24

Calcul établi le 2026-11-06 à partir des fiches techniques des modèles, des fichiers
du projet et de la config `configs/my_glados.yaml`. Aucune mesure n'a été faite sur le PC actuel.
Les seules mesures citées sont celles de l'auteur du projet amont (RTX 3080), dans `docs/benchmarks/`.

---

## 1. Ce qui tourne en permanence

| Composant | Modèle | Où il tourne | Rôle |
|---|---|---|---|
| LLM principal | `qwen3:4b-instruct` (Q4_K_M, 4,0 G paramètres) | GPU (Ollama) | Réponses, autonomie, Health, émotions |
| Vision | `qwen2.5vl:3b` (3,8 G paramètres, dont un encodeur vision d'environ 0,67 G) | GPU, chargé à la demande | `describe_screen`, `locate`/`click_on` |
| Worker | `gpt-5-mini` (API OpenAI) | Cloud | `delegate_task` (le repli local est `qwen3:8b`) |
| Base Nexus | `gpt-5-nano` (API) | Cloud | Bascule quand le local est lent |
| ASR | Parakeet TDT 0.6B v3 (ONNX FP32), nécessaire pour le français | CPU ou GPU | Transcription |
| VAD | Silero VAD | CPU | Détection de la parole |
| TTS | GLaDOS EN + `fr_FR-glados-medium` (VITS/Piper) et le phonémiseur | CPU | Voix |
| Appli web | Python 3.12, HTTPS sur le port 8050 | CPU | Console, API |
| Navigateur agent | Chromium (profil `~/.glados/browser`) | CPU | Outil `browser` |
| Tâches de fond | Autonomie toutes les 90 s, Health toutes les 300 s, Emotion à chaque événement | GPU (LLM) | Elles s'exécutent **en parallèle** des réponses |

---

## 2. Mémoire GPU (VRAM)

### Formules
- **Poids** = taille du fichier GGUF.
- **Cache KV (FP16) par token** = 2 × couches × têtes KV × dim_tête × 2 octets.
- **VRAM d'un modèle** ≈ poids + cache KV × contexte × slots + tampon de calcul (0,3 à 0,5 Go ; environ 1 Go pour un modèle vision).

### Par modèle

| Modèle | Poids | Architecture (couches / têtes KV / dim) | KV par token | Contexte retenu | KV total | **VRAM** |
|---|---:|---|---:|---|---:|---:|
| qwen3:4b-instruct | 2,50 Go | 36 / 8 / 128 | 144 Kio | 8192 × 2 slots | 2,4 Go | **≈ 5,2 Go** |
| qwen2.5vl:3b | 3,20 Go | 36 / 2 / 128 | 36 Kio | 4096 × 1 | 0,15 Go | **≈ 4,3–4,5 Go** |
| qwen3:8b (si le worker passe en local) | 5,23 Go | 36 / 8 / 128 | 144 Kio | 8192 × 1 (ou × 2) | 1,2 Go (2,4) | **≈ 6,8 Go (8,0)** |
| Parakeet 0.6B (si sur GPU) | 1,26 Go | — | — | — | — | **≈ 1,5 Go** |
| *Gemma 4 E4B (profil natif amont, llama.cpp)* | 5,15 Go | — | — | 16K × 4 slots | — | *4,9–8,9 Go selon le contexte et les slots (README)* |

Les 2 slots de 8192 tokens pour qwen3:4b correspondent à ce qu'Ollama lance aujourd'hui (`-c 16384 -np 2`).
Les prompts de GLaDOS font de 3 000 à 6 000 tokens, donc **8192 est le minimum** : avec 2048, Ollama renvoyait des erreurs HTTP 400.

### Profils

| Profil | Ce qui reste en VRAM | **VRAM minimum** | **VRAM confortable** |
|---|---|---:|---:|
| **A. Config actuelle** (worker sur l'API) | qwen3:4b, et la vision chargée quand on l'appelle | 6 Go (la vision évince le LLM, ce qui coûte un rechargement) | **10–12 Go** (les deux restent chargés, ASR sur le CPU) |
| **A+ et ASR sur le GPU** | idem + Parakeet | 12 Go | 12–16 Go |
| **B. Tout en local** (worker qwen3:8b) | 4b + vision + 8b | 16 Go (serré) | **20–24 Go** |
| **C. Profil amont** (llama.cpp + Gemma 4 E4B, qui fait chat, vision et routage) | E4B 4 slots + Parakeet | 8 Go | **10–12 Go** (l'amont mesure un pic de 8 976 / 10 240 Mio sur une RTX 3080, bureau compris) |

**Mémoire unifiée** (iGPU, APU, Mac) : ajoute les besoins RAM de la section 4 aux chiffres de VRAM.

---

## 3. Puissance de calcul GPU

### 3.1 Génération (vitesse de parole) : c'est la **bande passante mémoire** qui compte
Chaque token généré relit tous les poids, d'où :

```
tok/s ≈ 0,65 × bande passante (Go/s) / taille des poids (Go)
```
Le facteur 0,65 est l'efficacité typique de llama.cpp/Ollama. C'est une hypothèse.

- La TTS lit environ 3 tokens/s. Pour que la première phrase (15 à 20 tokens) soit prête en moins de 0,5 s, il faut **au moins 40 tok/s**.
- Pour `qwen3:4b` (2,5 Go), il faut une bande passante d'**au moins 160 Go/s**.
- Pour `qwen3:8b` en modèle principal (5,2 Go), il faut **au moins 330 Go/s**.

Ordres de grandeur constructeur, à vérifier sur la fiche technique :

| Mémoire | Bande passante |
|---|---:|
| DDR5-5600, double canal (iGPU de bureau) | ~90 Go/s |
| LPDDR5X-8000, bus 128 bits (iGPU de portable) | ~128 Go/s |
| LPDDR5X-8000, bus 256 bits (APU Ryzen AI Max) | ~256 Go/s |
| GDDR6, bus 128 bits (cartes d'entrée de gamme) | ~270–290 Go/s |
| GDDR6, bus 192 bits | ~360 Go/s |
| GDDR6X/GDDR7, bus 192 à 384 bits | 500–1000 Go/s |

### 3.2 Prefill (lecture du prompt) : c'est la **puissance de calcul** qui compte
Le prefill décide du temps avant le premier mot quand le cache ne sert pas : au démarrage, après des résultats d'outils, ou pour les tâches de fond sur l'autre slot.

```
FLOP ≈ 2 × paramètres hors embeddings × N  +  2 × couches × N² × (têtes × dim_tête)
```
Pour qwen3:4b (3,6 G paramètres hors embeddings, 36 couches, 32 × 128) et un prompt de **N = 5 000 tokens**, cela donne 36 + 7 = **≈ 43 TFLOP**.

| Objectif pour un prompt froid de 5 000 tokens | Débit effectif nécessaire |
|---|---:|
| ≤ 1 s | ~43 TFLOPS effectifs |
| **≤ 2 s (recommandé)** | **~22 TFLOPS effectifs** |
| ≤ 5 s | ~9 TFLOPS effectifs |

- **Conversion vers la fiche technique** : sur l'amont, la RTX 3080 fait environ 3 950 tok/s de prefill avec Gemma E4B, soit environ 32 TFLOPS effectifs, à peu près **50 % de la valeur « FP16 Tensor dense, accumulation FP32 »** de la carte. Il faut donc viser **≥ 45 TFLOPS FP16 Tensor (dense, accumulation FP32)** sur la fiche, ou **≥ 20 TFLOPS** si 5 s te conviennent.
- **Tour avec cache chaud** (100 à 300 nouveaux tokens) : moins de 3 TFLOP, ce que presque n'importe quel GPU traite en moins de 0,3 s.
- **Marge pour la concurrence** : l'autonomie, Health et Emotion tournent en parallèle des réponses. Prévois **×1,5 à ×2** sur le prefill pour que la réponse ne ralentisse pas pendant une tâche de fond.
- **Vision** : une capture 1920 × 1080 fait environ 2 700 tokens image plus environ 10 800 patchs dans l'encodeur, soit environ 30 TFLOP par capture. À 20 TFLOPS effectifs, cela fait environ 1,5 s.

### 3.3 Backend GPU
| Backend | Ollama | llama.cpp (logprobs) | Remarque |
|---|---|---|---|
| NVIDIA CUDA 12+ | oui | oui, le plus simple à compiler | C'est la référence de l'amont |
| AMD ROCm 7.1+ | oui | oui | Liste de GPU supportés limitée, à vérifier |
| Vulkan (Intel, AMD) | oui | oui, mais il faut les en-têtes Vulkan | La compilation a échoué sur ta machine actuelle |

**llama.cpp** débloque le routage (savoir à qui tu parles) et le vrai Emotion Core, qui ont besoin des logprobs et de `/tokenize`. Ollama ne les fournit pas.

---

## 4. Mémoire système (RAM)

| Poste | Estimation |
|---|---:|
| Linux sans écran (headless) | 1–2 Go |
| Appli GLaDOS (Python, ONNX Runtime, TTS FR+EN, VAD) | 1,5–2,5 Go |
| Parakeet 0.6B FP32 sur le CPU (poids et activations) | +1,5–2 Go |
| Chromium de l'agent | 0,5–1,5 Go |
| Ollama ou llama-server (hors VRAM) | 0,3–1 Go |
| Tests de self-improve (suite lancée 2 fois) | +1–2 Go en pointe |
| Cache disque des modèles (accélère les rechargements) | 5–15 Go si disponible |
| **Total avec GPU dédié** | **minimum 16 Go, confortable 32 Go** |
| **Total en mémoire unifiée** | **besoins RAM + VRAM du profil : 32 Go (A), 48–64 Go (B)** |

---

## 5. Processeur (CPU)

Le CPU s'occupe de l'orchestration Python, de la VAD, de la TTS, de l'ASR (si elle n'est pas sur le GPU), de Chromium, des commandes shell de l'agent et des tests.

- **ASR Parakeet 0.6B sur le CPU** : environ 2 × 0,6 G × 12,5 trames/s × durée, soit environ **75 GFLOP pour 5 s de parole**. Un CPU moderne à 6 ou 8 cœurs avec AVX2 (100 à 300 GFLOPS effectifs) transcrit en 0,25 à 0,75 s.
- **TTS VITS** (environ 30 M paramètres par voix) : très léger, plus rapide que le temps réel même sur 4 cœurs.
- **Exigences** : x86_64 **avec AVX2** (ONNX Runtime). AVX-512 et VNNI sont un plus pour l'ASR sur CPU.
- **Recommandé** : **6 à 8 cœurs**, Zen 3+ ou Intel 12e génération et suivantes. **4 cœurs suffisent** si l'ASR est sur le GPU.

---

## 6. Stockage

| Contenu | Taille |
|---|---:|
| Modèles Ollama (qwen3:4b + qwen2.5vl:3b + qwen3:8b) | 11 Go |
| `models/` (ASR + TTS) | 2,0 Go |
| `.venv` (CPU) | 0,75 Go (+3 à 5 Go avec ONNX Runtime GPU / CUDA) |
| Dépôt, `data/`, `~/.glados`, sauvegardes | ~0,2 Go |
| Gemma 4 E4B (si tu passes au profil C) | +5,2 Go |
| Toolchain llama.cpp + CUDA (compilation) | +5–8 Go |
| Linux et paquets | 15–25 Go |
| Marge pour tester d'autres modèles | 20–50 Go |
| **Total** | **minimum 64 Go, confortable 128–256 Go** |

- **Croissance** : moins de 1 Go par an (logs, mémoire, audit). Elle est négligeable.
- **Type de disque** : un SSD est indispensable. Le NVMe accélère seulement le chargement : 5 Go se chargent en environ 2 s en NVMe contre environ 10 s en SATA, et cela n'arrive qu'au démarrage grâce à `keep_alive: -1`.
- **Endurance** : les écritures sont faibles, n'importe quel SSD grand public convient. Évite les cartes SD et les clés USB.

---

## 7. Fonctionnement 24 h/24

- **Coût électrique** : `W au repos × 8,76 kWh/an × prix du kWh`. À 0,25 €/kWh, **1 W permanent coûte environ 2,2 €/an**. Au repos, la machine est inactive environ 95 % du temps, mais le modèle reste chargé en VRAM.

  | Machine au repos | €/an |
  |---|---:|
  | 10 W (mini PC, APU) | ~22 € |
  | 30 W | ~66 € |
  | 50 W (tour + GPU dédié) | ~110 € |

  Pour un GPU, regarde sa **consommation au repos** dans les tests, pas son TDP.
- **Charge réelle** : des pics GPU à chaque réponse, l'autonomie toutes les 90 s, Health toutes les 300 s. C'est du travail par à-coups, une alimentation de 80 PLUS Bronze ou mieux, avec 30 % de marge, suffit.
- **Bruit et chaleur** : la machine tourne en continu, donc un ventirad silencieux et un GPU avec arrêt des ventilateurs au repos (mode 0 dB) sont à privilégier.
- **Réseau** : Ethernet de préférence. Le débit est négligeable (API OpenAI, recherche Exa, console web), seule la latence compte.
- **Audio** : il faut un micro et un haut-parleur dans la pièce, par exemple une enceinte de conférence USB avec annulation d'écho. Sinon, la console web du téléphone suffit.
- **Agent** : les outils de bureau (capture d'écran, clics, Hyprland) agissent sur **la machine où tourne GLaDOS**. Sur un serveur séparé, piloter ton PC de jeu demandera un accès SSH ou à distance.
- **Optionnel** : un onduleur, si tu veux éviter les coupures de mémoire et de session.

---

## 8. Cahier des charges

| Critère | Minimum (profil A) | **Recommandé (A+ ou C, avec llama.cpp)** | Tout en local (B) |
|---|---|---|---|
| VRAM | 8 Go | **12 Go** | 20–24 Go |
| Bande passante VRAM | ≥ 160 Go/s | **≥ 300 Go/s** | ≥ 400 Go/s |
| GPU : FP16 Tensor dense (acc. FP32) | ≥ 20 TFLOPS | **≥ 45 TFLOPS** | ≥ 45 TFLOPS |
| Backend | Vulkan ou ROCm via Ollama | **CUDA 12+** (ou ROCm 7.1+) | CUDA 12+ |
| RAM | 16 Go | **32 Go** | 32 Go |
| CPU | 4 cœurs, AVX2 | **6 à 8 cœurs, AVX2** | 8 cœurs |
| SSD | 64 Go SATA | **256 Go NVMe** | 256–512 Go NVMe |
| Consommation au repos | — | **le plus bas possible (≤ 40 W pour la machine entière)** | — |

Si tu prends de la mémoire unifiée (APU, iGPU), il faut **≥ 32 Go** pour le profil A, **≥ 64 Go** pour le B, et une bande passante **≥ 200 Go/s**.
