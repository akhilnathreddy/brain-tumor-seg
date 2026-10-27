# Deploying the demo on Streamlit Community Cloud

The app (`app.py`) runs on CPU, loads the half-precision weights committed in
`models/` and the example patients in `examples/`, so the repository is all it needs.

## Before you start

- The GitHub repository must be **public** (free tier).
- `requirements.txt` installs CPU-only PyTorch from the PyTorch index, so the build
  doesn't download multi-GB GPU packages.

## Steps

1. Go to **https://share.streamlit.io** and sign in with your GitHub account
   (allow Streamlit to access your repositories when asked).
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `akhilnathreddy/brain-tumor-seg`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick a name, e.g. `brain-tumor-seg`
4. Open **Advanced settings** and choose **Python 3.13** (the version the
   requirements were tested with). No secrets are needed. Save.
5. Click **Deploy**. The first build takes a few minutes (it installs PyTorch);
   you can watch the log in the panel on the right.
6. When the page loads, pick an example patient. The first segmentation takes
   ~10–30 s on the shared CPU; switching slices is instant afterwards.

## After it is live

- Put the URL in the README (`Live demo:` line at the top).
- Every `git push` to `main` redeploys the app automatically.
- If the app shows "Oh no" or runs out of memory: open it, click **Manage app**
  (bottom right) → **Reboot app**. Only one result is kept per session to stay
  inside the free tier's memory.
- Apps that get no visitors for a while go to sleep; the next visitor wakes them
  with one click (it takes ~30 s).

## Run the same thing locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
