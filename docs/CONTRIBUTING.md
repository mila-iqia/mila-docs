# Contributing to the Mila Docs

Thank you for your interest into making a better documentation for all at Mila.

Here are some guidelines to help bring your contributions to life.

## What should be included in the Mila Docs

* Mila cluster usage
* Digital Research Alliance of Canada cluster usage
* Job management tips / tricks
* Research good practices
* Software development good practices
* Useful tools

**_NOTE_**: Examples should aim to not consume much more than 1 GPU/hour and 2 CPU/hour


## Issues

Issues can be used to report any error in the documentation, missing or 
unclear sections, broken tools or other suggestions to improve the 
overall documentation.

## Pull Requests

PRs are welcome and we value the contents of contributions over the 
appearance or functionality of the Pull Request. If you encounter problems 
with the Markdown formatting, simply provide the content you would like to 
add in the PR with instructions to format it. In the PR, reference the 
related issues like this:

```
Resolves: #123
See also: #456, #789
```

If you would like to contribute directly in the code of the documentation, 
keep the lines width to 80 characters or less. You can attempt to build 
the docs yourself to see if the formating is right. This is done using [`uv`](https://docs.astral.sh/uv):

## Building the docs

First, install `uv` if you don't have it yet, using the commands described in the [Getting Started section of the uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

You can use it to:

* build the documentation and view it by opening an HTML file, or
* serve the docs locally on localhost.

This command will build the documentation, which can be viewed by opening the local file `site/index.html`:

```console
uv run mkdocs build
```

## Serving the docs locally

You can also serve the site with a simple HTTP server with live reloading when a file changes. This is particularly useful if you want to improve the docs and see your changes in real time.

```console
uv run mkdocs serve --livereload
```

You can then access the local site through your browser at the URL
[http://127.0.0.1:8000/docs/](http://127.0.0.1:8000/docs/).

If you have any trouble building the docs, don't hesitate to open an issue to
request help.

## Markdown examples

The markup language used for the Mila Docs is Markdown.
The documentation framework used is [MkDocs](https://www.mkdocs.org/), with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

Here are some examples of the most common Markdown constructs used in the Mila Docs. We encourage you to take a look a the mkdocs and mkdocs-material documentation for more examples:

- [MkDocs documentation](https://www.mkdocs.org/user-guide/writing-your-docs/)
- [MkDocs-Material documentation](https://squidfunk.github.io/mkdocs-material/reference/)



### Inline markup

* one asterisk: `*text*` for *emphasis* (italics),
* two asterisks: `**text**` for **strong emphasis** (boldface), and
* backquotes: `` `text` `` for `code samples`, and
* external links: `` [Link text](http://target)` ``.

### Lists

```md
* this is
* a list
  * with a nested list
  * and some subitems

* and here the parent list continues
```

### Sections

```md
# This is one of the main headers

And this is its alternative
===========================

## This is a sub-header

And this is its alternative
---------------------------

### This is a sub-sub-header
#### Etc
```

### Note box
[Admonitions](https://squidfunk.github.io/mkdocs-material/reference/admonitions/) are used
to add panels. This is done through the `!!!` shortcut, depicted below. There are several types, such as `note`, `abstract`, `info`, `tip`,
`success`, `question`, `warning`, `failure`, `danger`, `bug`, `example` or `quote`.
```md
!!! note
   This is a note panel.
```

!!! note
    This is a note panel.

```md
!!! example
   This is an example panel, such as below.
```

!!! example
    This is an example panel.


Panels could also be collapsible by using `???` instead of `!!!`, such as:

```md
??? tip
    I am collapsible!
```

??? tip
    I am collapsible!

### Tables
Tables can be added by following the format below:
```
| Header 1      | Header 2         | Header 3 |
| ------------- | ---------------- | -------- |
| `First line`  | Hello world      |          |
| `Second line` | Juste a new line | The end  |
```

| Header 1      | Header 2         | Header 3 |
| ------------- | ---------------- | -------- |
| `First line`  | Hello world      |          |
| `Second line` | Juste a new line | The end  |


For more examples of what is possible with Markdown in MkDocs, please refer to the mkdocs documentation and mkdocs-material documentation pages:

- [MkDocs documentation](https://www.mkdocs.org/user-guide/writing-your-docs/)
- [MkDocs-Material documentation](https://squidfunk.github.io/mkdocs-material/reference/)

