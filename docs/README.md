# Editing these docs

[docs.leeroo.com](https://docs.leeroo.com/docs) is built and served by Mintlify.
Push to `main` and the site rebuilds itself — there is no deploy step to run and
no machine of ours in the path. The only infrastructure we own is the DNS record
pointing `docs.leeroo.com` at Mintlify.

## Change a page

1. Edit the `.mdx` file.
2. Open a PR. CI runs `mint validate` and `mint broken-links` against it.
3. Merge. Mintlify rebuilds `main`; the change is live in a minute or two.

## Add a page

Create the `.mdx` file **and** add its path to `navigation` in `/docs.json`.

Both halves matter. A file that is not in the navigation is not published —
Mintlify serves 404 for it. A navigation entry with no file behind it puts a
dead link in the sidebar, which is exactly what CI is there to catch.

## Preview locally

```bash
npx mint@4 dev            # http://localhost:3000
npx mint@4 validate       # what CI runs
npx mint@4 broken-links   # what CI runs
```

## Worth knowing

- **CI fails on warnings, not just errors.** A page referenced in `docs.json`
  that does not exist is only a warning, and it is the bug this gate exists for.
- **Drafts do not belong in `docs/`.** Everything here is parsed by the build.
  Internal notes live under `docs/plans/` and `docs/research/*.md`, which
  `/.mintignore` excludes; anything else new is fair game for the build.
- **Patterns in `/.mintignore` must be anchored.** An unanchored `benchmarks/`
  also matches `docs/benchmarks/` and silently unpublishes those pages.
- **There are no PR preview deployments** on the current plan. CI passing is the
  only signal you get before a change is live, so read it.
