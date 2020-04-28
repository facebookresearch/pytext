# What is 'contrib' directory?

Code in 'contrib' directory is considered experimental. If it's proved to be useful, we may incorporate it into the core PyText structure. The code review turnaround time for changes in 'contrib' is expected to be shorter than changes in the core PyText structure.

'contrib' is organized by projects. We recommend the following directory structure to prevent file collisions:

```
  contrib/
    my_project/     - project name
      common/       - mimic the core PyText structure
      config/            
      data/
      ...
```
