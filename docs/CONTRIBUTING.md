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

## Issues / Pull Requests

### Issues

Issues can be used to report any error in the documentation, missing or 
unclear sections, broken tools or other suggestions to improve the 
overall documentation.

### Pull Requests

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
the docs yourself to see if the formating is right. This could be done:

* by using `uv`
* by using `pip`.


=== "with uv"

    `uv` simplifies the environment handling.
    ```console
    # Install uv
    python -m pip install uv
    ```

    You can use it to:

    * build the documentation and access it by opening a file
    * or set up a local version on localhost.

    #### Build then use a file
    ```console
    # Build the documentation using MKDocs
    build mkdocs
    ```

    This will produce a version of the documentation which you can navigate
    by opening the local file `site/index.html`.

    #### Access through [localhost](http://127.0.0.1:8000/docs/)
    You can also try it locally as follows:

    ```console
    mkdocs serve --livereload
    ```

    You can then access the local site through your browser at the URL
    [http://127.0.0.1:8000/docs/](http://127.0.0.1:8000/docs/).

=== "with pip"

    ```console
    # Set up the virtual environment
    python -m venv venv

    # Activate it
    source venv/bin/activate

    # Install the dependencies
    python -m pip install mkdocs
    python -m pip install mkdocs-material
    python -m pip install mkdocs-include-markdown-plugin
    python -m pip install mkdocs-literate-nav
    ```

    You can then either:

    * build the documentation and access it by opening a file
    * or set up a local version on localhost.

    #### Build then use a file
    ```console
    # Build the documentation using MKDocs
    build mkdocs
    ```

    This will produce a version of the documentation which you can navigate
    by opening the local file `site/index.html`.

    #### Access through [localhost](http://127.0.0.1:8000/docs/)
    You can also try it locally as follows:

    ```console
    mkdocs serve --livereload
    ```

    You can then access the local site through your browser at the URL
    [http://127.0.0.1:8000/docs/](http://127.0.0.1:8000/docs/).


If you have any trouble building the docs, don't hesitate to open an issue to
request help.

## Markdown (md)

The markup language used for the Mila Docs is Markdown.


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
| :----------   | :--------------- | :------- |
| `First line`  | Hello world      |          |
| `Second line` | Juste a new line | The end  |
```

| Header 1      | Header 2         | Header 3 |
| :----------   | :--------------- | :------- |
| `First line`  | Hello world      |          |
| `Second line` | Juste a new line | The end  |

