from google.colab import drive
import os
from huggingface_hub import login,snapshot_download
from dotenv import load_dotenv


load_dotenv()
#HF Login
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if hf_token != "hf_YourTokenHere":
    login(token=hf_token)
else:
    print("Set HF token in .env or run login manually in Colab")
#download base model
snapshot_download(repo_id="gpt2",local_dir="models/base_model")
print("Colab setup complete! Models in /content/stealth-llm-evaer/models")