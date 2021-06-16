# Contributing to the Mila Docs

Thank you for your interest into making a better documentation for all at Mila.
Here are some guidelines to help bring your contributions to life.

## What should be included in the Mila Docs

* Mila cluster usage
* Compute Canada cluster usage
* Job management tips / tricks
* Research good practices
* Software development good practices
* Useful tools

## Issues / Pull Requests

### Issues

Issues can be used to report any error in the documentation, missing or unclear
sections, broken tools or other suggestions to improve the overall
documentation.

### Pull Requests

PRs are welcome and we value the contents of contributions over the appearance
or functionality of the pull request. If you don't know how to write the proper
markup in reStructuredText, simply provide the content you would like to add in
the PR text form which supports markdown or with instructions to format the
content. In the PR, reference the related issues like this:

```
Resolves: #123
See also: #456, #789
```

If you would like to contribute directly in the code of the documentation, keep
the lines width to 80 characters or less. You can attempt to build the docs
yourself to see if the formating is right:

```console
python3 -m pip install -r docs/requirements.txt
sphinx-build -b html docs/ docs/_build/
```

This will produce the html version of the documentation which you can navigate
by opening the local file `docs/_build/index.html`.

If you have any trouble building the docs, don't hesitate to open an issue to
request help.

## Sphinx / reStructuredText (reST)

The markup language used for the Mila Docs is
[reStructuredText](http://docutils.sourceforge.net/rst.html) and we follow the
[Python’s Style Guide for documenting](https://docs.python.org/devguide/documenting.html#style-guide).

Here are some of reST syntax useful to know (more can be found in
[Sphinx's reST Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html)):

### Inline markup

* one asterisk: `*text*` for emphasis (italics),
* two asterisks: `**text**` for strong emphasis (boldface), and
* backquotes: ` ``text`` ` for code samples, and
* external links: `` `Link text <http://target>`_ ``.

### Lists

```reST
* this is
* a list

  * with a nested list
  * and some subitems

* and here the parent list continues
```

### Sections

```reST
=================
This is a heading
=================
```

There are no heading levels assigned to certain characters as the structure is
determined from the succession of headings. However, the Python documentation is
suggesting the following convention:

    * `#` with overline, for parts
    * `*` with overline, for chapters
    * `=`, for sections
    * `-`, for subsections
    * `^`, for subsubsections
    * `"`, for paragraphs

### Note box

```reST
.. note:: This is a long
   long long note
```
