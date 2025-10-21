from google.colab import drive
import os
from huggingface_hub import login,snapshot_download
from dotenv import load_dotenv


load_dotenv()
#HF Login
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if hf_token:
    login(token=hf_token)
else:
    print("Token not found in.env - set manually")
#download base model
snapshot_download(repo_id="gpt-2",local_dir="models/base_model")
print("Colab setup complete! Models in /content/stealth-llm-evaer/models")