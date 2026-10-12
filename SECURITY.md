# Security and privacy

This is **research-only software**, not a medical device or clinical decision tool.

## Reporting vulnerabilities

Do not post passwords, access tokens, private EEG recordings, or exploit details in public GitHub issues. Use GitHub's **private vulnerability reporting** if it is enabled for this repository; otherwise contact the maintainer privately through their GitHub profile. Public issues are appropriate for ordinary non-sensitive defects.

## Supported scope

There is currently **no formally supported released version**. Research snapshots are provided without production service-level guarantees.

## Data handling

- Do not add raw PhysioNet recordings, private EEG datasets, subject-level caches, trained weight binaries, or credentials to Git.
- The demonstration accepts user-supplied arrays; never upload recordings containing private participant data to an untrusted hosted instance.
- Do not use `allow_pickle=True` to load untrusted NumPy files.
- Treat `torch.load` artifacts as untrusted unless weight-only loading and provenance have been checked.
- Keep local caches and generated outputs outside commits, and review any published metadata for identifiability.

## Known limits

Offline quality gates screen a small set of prohibited filename extensions and credential filenames. They do not replace secret scanning, malware analysis, dependency vulnerability review, or a clinical security assessment.
