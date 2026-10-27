# Source audit

Reviewed on 2026-10-27.

1. [Pravaas project page](https://kirtiraj-chaudhari.vercel.app/projects/pravaas): the site served a JavaScript shell; project details were not available as readable page text. No claims were copied from an unrendered page.
2. [Rail-defect-detection](https://github.com/pradeepxkumar/Rail-defect-detection), commit `4beabd6f814b9c1aff7766a554e60f5de8fe7209`. README, source tree, annotated images and checkpoint inspected. The README limits intended use to academic/research and requests author permission for commercial/production use. Checkpoint labels were inspected directly: only `railway`.
3. [RailwayObjectDetectionProject](https://github.com/KirtirajChaudhari/RailwayObjectDetectionProject) redirects to [Pravaas](https://github.com/KirtirajChaudhari/Pravaas), reviewed at commit `888d10aafdd593747830b946f423755852bedc0c`. Two Colab notebooks, `best.pt`, `data.zip.zip`, and `project.zip` were inspected. The two annotation archives list six and seven classes respectively; the checkpoint itself exposes six.
4. [Railway Track Fault Detection](https://www.kaggle.com/datasets/salmaneunus/railway-track-fault-detection), Salman Ibne Eunus, version 2. Downloaded the requested full dataset, not a substitute resized dataset. Kaggle's API lists the license as “Data files © Original Authors”. The archive SHA-256 is in the data manifest.
5. [Roboflow railway search](https://universe.roboflow.com/search?q=railway%20track) is a discovery page. The reference README points to [defect-rexb3/rail-defect](https://universe.roboflow.com/defect-rexb3/rail-defect). Direct Universe access returned 403; the search result does not pin a dataset version or license. Roboflow-marked image filenames bundled in the reference repo were inspected. No fresh export or verified complete Roboflow lineage is claimed.
6. The user supplied `Lab_4.docx`, requiring a CNN and transfer-learning classification experiment, feature extraction/fine-tuning comparison, and measured evaluation. The document is mapped in `LAB_4.md` and is not redistributed.

Model source hashes and sample provenance are recorded in `dist/data/registry.json`. Demo images drawn from training/reference material are labeled as demonstrations. They are not used to claim test performance.
