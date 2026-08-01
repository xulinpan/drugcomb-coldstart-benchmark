# How to deposit this repository (required before acceptance)

BMC Bioinformatics requires analysis code to be openly available in a permanent repository with a
license and a citable archival identifier. The manuscript's *Availability of data and materials*
section currently promises deposition; these are the steps to fulfil it.

## 1. Push to GitHub

**Before you start:** create the repository on GitHub first (empty — do not let GitHub add a README,
license or .gitignore, since this folder already has them), and set your git identity, or the very
first `git commit` will fail and the later push will report
`error: src refspec main does not match any`.

```bash
git config --global user.email "xxp5066@psu.edu"
git config --global user.name  "Xulin Pan"
```

Then, replacing `YOUR-USERNAME` with your actual GitHub username (no angle brackets):

```bash
cd drugcomb-coldstart-benchmark
git init
git add -A
git commit -m "Leakage-controlled cold-start benchmark for drug-combination synergy classification"
git log --oneline -1        # must print a commit; if not, the commit failed - read its error
git branch -M main          # run this AFTER the first commit
git remote add origin https://github.com/YOUR-USERNAME/drugcomb-coldstart-benchmark.git
git remote -v               # confirm no angle brackets remain in the URL
git push -u origin main
```

When prompted for a password, GitHub requires a **personal access token**, not your account
password (Settings -> Developer settings -> Personal access tokens).

Note `.gitignore` deliberately excludes `*.csv` so the ~900 MB modeling table is not committed;
`src/drug_smiles_demo.csv` is whitelisted.

### If the push fails

| Error | Cause | Fix |
|---|---|---|
| `src refspec main does not match any` | No commit exists yet | Check `git log`; set identity and re-commit |
| `remote: Repository not found` | Placeholder username, or repo not created | `git remote set-url origin <real URL>`; create the repo |
| `failed to push some refs` / non-fast-forward | Remote already has commits | `git pull --rebase origin main`, then push |
| `Authentication failed` | Password used instead of token | Use a personal access token |

## 2. Archive to Zenodo and mint a DOI

1. Sign in at <https://zenodo.org> with GitHub and authorise the account.
2. In Zenodo → **GitHub**, toggle the repository **On**.
3. On GitHub create a release, e.g. tag `v1.0.0`, title "v1.0.0 - manuscript submission".
4. Zenodo archives the release automatically and mints a DOI. `.zenodo.json` supplies the metadata.

## 3. Update the manuscript and supporting files

Replace the placeholders in **both** `manuscript/main.tex` and `manuscript/main_article.tex`:

- Project home page: `https://github.com/<USERNAME>/drugcomb-coldstart-benchmark`
- Archived release: Zenodo DOI `10.5281/zenodo.XXXXXXX`
- License: MIT
- Programming language: Python (>= 3.10)

Then uncomment `repository-code` and `doi` in `CITATION.cff`, and update
`submission_checklist.txt` and `cover_letter_bmc_bioinformatics.txt`.

## 4. Verify before submitting

- [ ] Repository is public
- [ ] LICENSE present and stated in the manuscript
- [ ] Zenodo DOI resolves to the archived release
- [ ] README quick-start runs from a clean clone
- [ ] No large data files or credentials committed (`git ls-files | xargs du -ch | tail -1`)
