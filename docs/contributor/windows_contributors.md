# Important note for Windows contributors

Due to the limited support for Windows in many of the dependencies and to ensure a consistent development environment, 
Windows users should always use wsl to operate in a Linux environment. 

## First time WSL setup

If this is the first time you're using wsl on Windows:

* First create a wsl linux environment with `wsl install`

* Make docker available from within the wsl environment:

  * In Docker Desktop, go to Settings - Resources - WSL Integration and check off the Linux distro used by wsl, e.g. `Ubuntu`

## Running commands through WSL

If you're a Windows contributor, always remember to first enter the linux environment before you run any commands CHAP CLI commands or Python testing: 

```bash
$ wsl
```

The only exception to this is that `git` and `docker` commands should be run through a regular Windows commandline (*not wsl*). 
