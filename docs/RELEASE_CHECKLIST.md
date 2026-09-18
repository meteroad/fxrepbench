# Public release checklist

## Required before v1.0.0

- [ ] Confirm the public GitHub repository URL.
- [x] Select code and benchmark-metadata licenses.
- [ ] Add third-party notices and verify every FMA license record.
- [x] Remove all internal paths and runtime-specific provenance.
- [x] Validate CounterFX-Dev80 and CounterFX-200 manifests.
- [ ] Reconstruct and hash-check a sample from each split on a clean machine.
- [x] Add the canonical output metric and submission format.
- [x] Add frozen candidate-budget and CMA-ES result curves.
- [ ] Add redistributable encoder adapters and approved head checkpoints.
- [ ] Add the paper and archival DOI to `CITATION.cff`.
- [x] Build a deterministic, audio-free release archive with `MANIFEST.sha256`.
- [ ] Tag `v1.0.0` and create a versioned Zenodo deposit.
- [ ] Link GitHub, Zenodo, project page, and arXiv in both directions.

## Must not be published

- internal training audio, filenames, or metadata;
- model checkpoints without confirmed redistribution rights;
- API keys, tokens, cookies, or `.env` files;
- usernames, server addresses, or absolute `/data` paths;
- unrestricted experiment logs and caches;
- source or reconstructed FMA audio under a blanket repository license.
