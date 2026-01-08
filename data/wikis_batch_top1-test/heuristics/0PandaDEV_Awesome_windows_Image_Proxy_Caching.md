# Heuristic: Image_Proxy_Caching

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|0PandaDEV/awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|weserv Images|https://images.weserv.nl/]]
|-
! Domains
| [[domain::Optimization]], [[domain::Web_Performance]]
|-
! Last Updated
| [[last_updated::2026-01-08 18:30 GMT]]
|}

== Overview ==
Use weserv.nl as an image proxy to cache and transform GitHub avatars for README display.

=== Description ===
This heuristic addresses the problem of displaying consistent, optimized user avatars in README files. GitHub avatar URLs can change, be slow to load, or have inconsistent sizing. By routing images through a proxy service like `images.weserv.nl`, you can apply transformations (circle masks, resizing), enable caching, and ensure consistent display across browsers.

=== Usage ===
Use this heuristic when you need to **display user avatars** or external images in README files and want:
- Consistent circular avatar display
- Cached images for faster page loads
- Resized images to reduce page weight
- Protection against broken images if source changes

== The Insight (Rule of Thumb) ==

* **Action:** Route avatar URLs through `images.weserv.nl` proxy instead of linking directly to GitHub avatars
* **Value:** Use parameters: `url={avatar_url}&fit=cover&mask=circle&maxage=7d`
* **Trade-off:** Adds dependency on third-party service; images may be briefly unavailable if weserv.nl is down
* **Benefit:** Consistent 60x60px circular avatars with 7-day caching

== Reasoning ==

GitHub avatar URLs (`avatars.githubusercontent.com`) are:
1. Square by default (require CSS for circular display in raw markdown)
2. Full resolution (often 400x400px, wasteful for 60px display)
3. Subject to rate limiting for unauthenticated requests

Using weserv.nl proxy provides:
- **Circle masking:** `mask=circle` creates circular avatars without CSS
- **Responsive sizing:** `fit=cover` with explicit width/height
- **CDN caching:** `maxage=7d` caches processed images for 7 days
- **Reliability:** Weserv.nl acts as a buffer against GitHub CDN issues

== Code Evidence ==

Image proxy usage from `update_contributors.py:33`:
<syntaxhighlight lang="python">
for contributor in contributors:
    avatar_url = contributor['avatar_url']
    new_block += f"<a href='https://github.com/{contributor['login']}'><img src='https://images.weserv.nl/?url={avatar_url}&fit=cover&mask=circle&maxage=7d' width='60' height='60' alt='{contributor['login']}'/></a> "
</syntaxhighlight>

=== URL Construction Pattern ===

<syntaxhighlight lang="text">
Base URL: https://images.weserv.nl/
Parameters:
  - url={source_url}      # Original image URL (GitHub avatar)
  - fit=cover             # Crop to fill dimensions
  - mask=circle           # Apply circular mask
  - maxage=7d             # Cache for 7 days
  - width=60              # (implicit from img tag)
  - height=60             # (implicit from img tag)
</syntaxhighlight>

== Related Pages ==

* [[used_by::Implementation:0PandaDEV_Awesome_windows_update_readme_generation]]
