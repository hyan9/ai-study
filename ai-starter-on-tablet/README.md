# AI Starter (Tablet-Friendly)

This repo is designed so you can work **entirely from a Galaxy Tab S9+** (or any tablet) using **Google Colab** or **GitHub Codespaces**. No local installs required.

## Option A — Google Colab (Recommended to start)
1. Open **https://colab.research.google.com**
2. Sign in with Google → *New Notebook*.
3. Run this cell to set up PyTorch/Transformers:
```python
!pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
!pip -q install transformers datasets accelerate sentencepiece faiss-cpu
```
4. Test:
```python
import torch, transformers
print('Torch:', torch.__version__)
print('CUDA available?', torch.cuda.is_available())
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained("gpt2")
print(tok("hello, ai")["input_ids"][:8])
```

> Tip: In Colab, **Runtime → Change runtime type → (T4/A100 if available)** for GPU. Free tier may vary by account/time.

## Option B — GitHub Codespaces (Browser VS Code)
1. Create a repo on GitHub and enable **Codespaces**.
2. Create a new Codespace → open Terminal and run:
```bash
pip install -r requirements.txt
```
3. Start a simple app:
```bash
python src/hello_ai.py
```

## Option C — (Optional) Android local via Termux
- For quick Python practice: install **Termux** and `pkg install python` (heavy ML libs like PyTorch may not work well on-device). Use cloud for GPU tasks.

---

### Folder Structure
```
ai-starter-on-tablet/
  ├── src/
  │   └── hello_ai.py
  ├── notebooks/
  ├── requirements.txt
  ├── .vscode/
  │   └── settings.json
  └── README.md
```

### Next Steps
- Open the `hello_ai.ipynb` example in Colab if you upload it later, or just use a fresh notebook.
- When ready, create a private GitHub repo and upload this folder, then try **Codespaces**.

### Voice TTS Fine-tuning (Colab)
- Use `notebooks/voice_tts_finetune_colab.ipynb` to fine-tune a base model with your own voice data.
- The notebook includes dataset formatting, training, and inference steps for swapping out ElevenLabs.

Good luck! 🚀
