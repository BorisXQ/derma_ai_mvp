# Render — exact deployment

## 1. GitHub

Create a new repository, for example:

derma-ai-mvp

Upload the project files.

Do NOT upload:
- `.venv`
- `.env`
- private patient images
- future private datasets
- secrets

## 2. Render

Open:
https://render.com/

Choose:
Dashboard -> New -> Web Service

Connect your GitHub repository.

Settings:

Runtime:
Python

Build Command:
pip install -r requirements.txt

Start Command:
uvicorn app.main:app --host 0.0.0.0 --port $PORT

Plan:
Free

## 3. Result

Render gives a public HTTPS URL similar to:

https://derma-ai-mvp-xxxx.onrender.com

Open it in a browser and upload a lesion image.

## Free-plan limitation

Render Free Web Services spin down after 15 minutes without inbound traffic and need about a minute to wake up. Free services also have 750 instance-hours per workspace per month and ephemeral local storage. Do not store patient data on local disk.

Source:
https://render.com/docs/free
