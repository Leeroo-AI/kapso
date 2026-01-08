# Weserv_Image_Proxy_Pattern

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Doc|weserv.nl API|https://images.weserv.nl/]]
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
|-
! Domains
| [[domain::Frontend]], [[domain::Optimization]]
|-
! Last Updated
| [[last_updated::2026-01-08 00:00 GMT]]
|}

== Overview ==

Use the weserv.nl image proxy service to transform GitHub avatars into circular, cached, and consistently-sized images for contributor displays.

=== Description ===

The weserv.nl service is a free image proxy that provides on-the-fly image transformation. For contributor avatars, it transforms rectangular GitHub profile pictures into circular thumbnails with consistent sizing and caching. This pattern ensures uniform visual presentation without requiring manual image processing or storage.

=== Usage ===

Use this heuristic when **generating contributor avatar displays** in README files or documentation. The proxy handles resizing, masking, and caching automatically.

== The Insight (Rule of Thumb) ==

* **Action:** Wrap avatar URLs with the weserv.nl proxy.
* **Value:** Use parameters: `fit=cover`, `mask=circle`, `maxage=7d`, `width=60`, `height=60`.
* **Trade-off:** External service dependency; images may fail if weserv.nl is unavailable.
* **URL Format:** `https://images.weserv.nl/?url={avatar_url}&fit=cover&mask=circle&maxage=7d`

== Reasoning ==

Using weserv.nl provides:
1. **Consistent presentation:** All avatars are uniformly circular and sized
2. **Performance:** 7-day caching reduces load on GitHub's servers
3. **Simplicity:** No server-side image processing required
4. **Bandwidth:** Proxy serves optimized images from CDN
5. **Free tier:** No cost for public usage

== Code Evidence ==

Avatar HTML generation from `.github/scripts/update_contributors.py:31-33`:
<syntaxhighlight lang="python">
    for contributor in contributors:
        avatar_url = contributor['avatar_url']
        new_block += f"<a href='https://github.com/{contributor['login']}'><img src='https://images.weserv.nl/?url={avatar_url}&fit=cover&mask=circle&maxage=7d' width='60' height='60' alt='{contributor['login']}'/></a> "
</syntaxhighlight>

Resulting HTML in `README.md:518`:
<syntaxhighlight lang="html">
<a href='https://github.com/0PandaDEV'><img src='https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/70103896?v=4&fit=cover&mask=circle&maxage=7d' width='60' height='60' alt='0PandaDEV'/></a>
</syntaxhighlight>

== Related Pages ==

=== Used By ===
This heuristic is referenced by:
* Implementation: update_readme_HTML_Block
* Principle: Avatar_HTML_Generation
* Workflow: Automated_Contributor_Update
