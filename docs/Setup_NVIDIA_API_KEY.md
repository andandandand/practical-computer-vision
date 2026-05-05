# NVIDIA Inference Microservices API Key Setup Instructions

## 1. Create / Verify Your NVIDIA Account

1. Navigate to the NVIDIA API Catalog at [https://build.nvidia.com](https://build.nvidia.com).
2. Sign in to your existing NVIDIA account, or create a new one using the email address you plan to use for the workshop.
3. Once logged in, verify that you can access the model pages in the catalog.

## 2. Generate Your NVIDIA API Key

During the lab, we will use the hosted NIM endpoint (OpenAI-style API). Each participant must generate their own API key.

1. While logged into [https://build.nvidia.com](https://build.nvidia.com), open the top-right profile menu and go to **Settings → API Keys**.
2. Click **Generate Key**.
3. Copy the key value and store it in a secure location, such as a password manager or a text file you will not lose.
4. Treat this key like a password. Do not share it publicly or commit it to any Git repositories.

## 3. Add the API Key as a Secret in Google Colab

To use your API key securely in Google Colab without hardcoding it into your notebook:

1. Open your notebook in [Google Colab](https://colab.research.google.com).
2. In the left sidebar, click the **🔑 key icon** to open the **Secrets** panel.
3. Click **Add new secret**.
4. Set the **Name** to `NVIDIA_API_KEY`.
5. Paste your NVIDIA API key into the **Value** field.
6. Toggle **Notebook access** to **ON** so your notebook can read the secret.
7. In your notebook, access the key with the following code:

   ```python
   from google.colab import userdata
   import os

   os.environ["NVIDIA_API_KEY"] = userdata.get("NVIDIA_API_KEY")
   ```

> **Note:** Secrets are tied to your Google account and are not shared when you share the notebook. Each workshop participant must add their own key.
