# Render deployment

1. Replace the files in the existing GitHub repository with this archive's contents.
2. Commit and push. Render deploys automatically if the service is connected to that repository.
3. In Render → Environment, add `DERMA_ADMIN_TOKEN` with a long random secret. Do not commit the token to GitHub.
4. Keep the existing start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
5. Open `/admin`, enter the token, add class metadata, upload labeled examples, and export them as ZIP before redeploys.

Important: the free Render filesystem may be ephemeral. Do not rely on it as the only copy of uploaded examples or trained weights. Training is intended to run locally, not inside the web service. After training, deploy the matching custom checkpoint and ensure `configs/classes.json` has matching class metadata.

The fallback checkpoint is downloaded at first inference if `model/custom_model.pth` is absent. The service requires outbound network access unless the compatible fallback checkpoint is bundled.
