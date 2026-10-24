# Provenance: experiments executed 2026-10-22 to 2026-10-24

The detector and inference sources in this branch were synchronized from the local project used for the reported runs. The local project directory was not a Git checkout, so the original experiment code could not be recovered from a local commit.

## Recorded execution hashes

From `results/_logs/run_manifest_10seeds.json`:

- `scripts/detectar.py`: SHA-256 `2981076f065a52090105684c1bfae63cde65c3dfc7f6f1eb3ea4b256bc75018e`
- `config/02_random_vs_chrono_10seeds.yaml`: SHA-256 `070c722f43bca8dd95792a854c4d5432634872c6ab5b5117fff5aec1ea3df37e`
- `config/04_ablation_logontype_10seeds.yaml`: SHA-256 `9069ce75a75fe64b2f4ee420182fed9d4994f4ec9f9a79d79a518af8142e5578`

The run manifest also records 80 expected rows for YAML02 and 40 new rows for YAML04.

## Experiment manifests

This branch includes the specialist brute-force manifest, CERT-Like and LANL-Like training manifests, and YAML12 inference manifest. In those manifests, machine-specific paths under the user's Desktop were replaced with `${CONTEXTOS_ROOT}`; configure that placeholder to the directory containing the corresponding datasets before reusing those manifests.

## Remote comparison

The original GitHub `main` at `8b599fd7c7f42c12d364f2fb606a32e6db495cd3` differed from the local execution sources. In particular, `scripts/batch_inferir.py` was absent remotely. This branch synchronizes the local `core/` and `scripts/` Python files and adds the missing inference runner.
