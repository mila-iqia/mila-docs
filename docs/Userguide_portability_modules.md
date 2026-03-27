### The module command

For a list of available modules, simply use:

```bash
$ module avail
```

Modules can be loaded using the `load` command:
```bash
module load <module>
```
To search for a module or a software, use the command `spider`:
```bash
module spider search_term
```
E.g.: by default, `python2` will refer to the os-shipped installation of `python2.7` and `python3` to `python3.10`.
If you want to use `python3.7` you can type:
```bash
module load python/3.7
```

### Available Software

Modules are divided in 5 main sections:

| Section            | Description                                                                                       |
| ------------------ | ------------------------------------------------------------------------------------------------- |
| Core               | Base interpreter and software (Python, go, etc...)                                                |
| Compiler           | Interpreter-dependent software (*see the note below*)                                             |
| Cuda               | Toolkits, cudnn and related libraries                                                             |
| Pytorch/Tensorflow | Pytorch/TF built with a specific Cuda/Cudnn version for Mila's GPUs (*see the related paragraph*) |

!!! note
    Modules which are nested (../../..) usually depend on other software/module
    loaded alongside the main module.  No need to load the dependent software,
    the complex naming scheme allows an automatic detection of the dependent
    module(s):
    i.e.: Loading ``cudnn/7.6/cuda/9.0/tensorrt/7.0`` will load ``cudnn/7.6`` and
    ``cuda/9.0`` alongside
    ``python/3.X`` is a particular dependency which can be served through
    ``python/3.X`` or ``anaconda/3`` and is not automatically loaded to let the
    user pick his favorite flavor.

### Default package location
<!-- todo: Move to extras -->

Python by default uses the user site package first and packages provided by
`module` last to not interfere with your installation.  If you want to skip
packages installed in your site-packages folder (in your /home directory), you
have to start Python with the `-s` flag.

To check which package is loaded at import, you can print `package.__file__`
to get the full path of the package.

*Example:*
```bash
$ module load pytorch/1.5.0
$ python -c 'import torch;print(torch.__file__)'
/home/mila/my_home/.local/lib/python3.7/site-packages/torch/__init__.py   <== package from your own site-package
```
Now with the `-s` flag:

```bash
$ module load pytorch/1.5.0
$ python -s -c 'import torch;print(torch.__file__)'
/cvmfs/ai.mila.quebec/apps/x86_64/debian/pytorch/python3.7-cuda10.1-cudnn7.6-v1.5.0/lib/python3.7/site-packages/torch/__init__.py'
```