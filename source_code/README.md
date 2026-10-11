# Demo 3.2 - reproduce Fig. 3 of the survey (Chefer et al., 2021a)

Run (from this folder):

    py -m pip install -r requirements.txt
    py .\3_2_paper.py

- `3_2_paper.py`  computes the 6 methods (rollout, raw-attention, GradCAM, LRP, partial LRP, Beyond Attention)
  with the OFFICIAL code in `official_repo/` (hila-chefer/Transformer-Explainability); only `.cuda()` calls were replaced to run on CPU.
- `images/`  input photos (dog_and_cat.png, elephant_and_zebra.png).
- `outputs/` result figure `output_paper_figure3.png`.
- ViT-B/16 weights are downloaded automatically on the first run (cached in `~/.cache/torch`).
