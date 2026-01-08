# Principle: Avatar HTML Generation

{| class="wikitable" style="float:right; margin-left:1em; width:300px;"
|-
! Knowledge Sources
|
* [[source::Repo|awesome-windows|https://github.com/0PandaDEV/awesome-windows]]
* [[source::Doc|weserv.nl Image Proxy|https://images.weserv.nl/docs/]]
|-
! Domains
| [[domain::HTML_Generation]], [[domain::Image_Processing]], [[domain::Template_Rendering]]
|-
! Last Updated
| [[last_updated::2026-01-08 13:00 GMT]]
|}

== Overview ==

Principle for generating HTML markup displaying contributor avatars with circular masking via an image proxy service.

=== Description ===

Avatar HTML Generation creates the visual representation of contributors in the README's Backers section. It transforms raw GitHub avatar URLs into styled HTML elements using:

- '''weserv.nl:''' Image proxy for on-the-fly transformations
- '''Circular mask:''' Avatar images cropped to circles
- '''Caching:''' 7-day cache headers for performance
- '''Linking:''' Each avatar links to the contributor's GitHub profile

This creates a visually appealing contributor wall without requiring local image processing.

=== Usage ===

Apply this principle when:
- Displaying user avatars in documentation
- Generating contributor acknowledgment sections
- Creating visual team/backer displays
- Styling external images without processing them locally

== Theoretical Basis ==

'''Image Proxy Pattern:'''

<syntaxhighlight lang="text">
Original URL (GitHub Avatar)
https://avatars.githubusercontent.com/u/12345?v=4
                    │
                    ▼
          weserv.nl Proxy URL
https://images.weserv.nl/?url={original_url}&fit=cover&mask=circle&maxage=7d
                    │
                    ▼
            Transformed Image
        (Circular, cached, optimized)
</syntaxhighlight>

'''weserv.nl Parameters:'''

{| class="wikitable"
|-
! Parameter !! Value !! Effect
|-
| url || avatar_url || Source image to transform
|-
| fit || cover || Scale to fill dimensions
|-
| mask || circle || Apply circular mask
|-
| maxage || 7d || Cache for 7 days
|}

'''HTML Structure:'''

<syntaxhighlight lang="html">
<a href='https://github.com/{login}'>
  <img src='https://images.weserv.nl/?url={avatar_url}&fit=cover&mask=circle&maxage=7d'
       width='60'
       height='60'
       alt='{login}'/>
</a>
</syntaxhighlight>

'''Benefits of Image Proxy:'''
- No local image processing required
- Images served from CDN edge locations
- Transformations applied on-the-fly
- Consistent styling across all avatars
- Reduced README repository size

== Practical Guide ==

=== Template Structure ===

The Backers section follows this template:

<syntaxhighlight lang="markdown">
## Backers

Thanks to all contributors without you this project would not exist.

{avatar_html_block}

Please, consider supporting me as it is a lot of work to maintain this list!

<a href="https://buymeacoffee.com/pandadev_">
  <img src="https://img.shields.io/badge/Buy_Me_A_Coffee-..."/>
</a>
</syntaxhighlight>

=== Avatar Block Generation ===

For each contributor:
1. Extract `login` and `avatar_url`
2. Construct weserv.nl proxy URL
3. Generate `<a>` wrapper with profile link
4. Generate `<img>` with proxy URL and dimensions
5. Concatenate all avatars into single block

== Related Pages ==

=== Implemented By ===
* [[implemented_by::Implementation:0PandaDEV_awesome-windows_update_readme_HTML_Block]]

=== Part of Workflow ===
* [[workflow::Workflow:0PandaDEV_awesome-windows_Automated_Contributor_Update]]
