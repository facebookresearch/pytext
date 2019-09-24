Hacking PyText
==============

Using your own classes in PyText
--------------------------------

Most people just want to create their own components and use them to load their
data, train models, etc. In this case, you just need to put all your `.py` files in a directory and include it with the option `--include <my directory>`. PyText will be able to find your code and import your classes. This works with PyText from `pip install` or from github sources.

`Example with Custom Data Source <datasource_tutorial.html>`_

Changing PyText
---------------

Why would you want to change PyText? Maybe you want to fix one of the `github issues <https://github.com/facebookresearch/pytext/issues>`_, or you want to experiment with your own changes that you can't simply include and you would like to see included in PyText's future releases. In this case you need to download the sources and submit them back to github. Since getting your changes ready and integrated can take some time, you might need to keep your sources up to date. Here's how to do it.


Installation
^^^^^^^^^^^^

First, make a copy of the PyText repo into your github account. For that (you need a github account), go to the `PyText repo <https://github.com/facebookresearch/pytext>`_ and click the Fork button at top-right of the page.

Once the fork is complete, clone your fork onto your computer by clicking the "Clone or download" button and copying the URL. Then, in a terminal, use the `git clone` command to clone the repo in the directory of your choice.

.. code-block:: console

  $ git clone https://github.com/<your_account>/pytext.git

To be able to update your github fork with the latest changes from Facebook's PyText sources, you need to add it as a "remote" with this command. (This can be done later.) The name "upstream" is what's typically used, but you can use any name you want for your remotes.

.. code-block:: console

  $ git remote add upstream https://github.com/facebookresearch/pytext.git

Now you should have 2 remotes: `origin` is your own github fork, and `upstream` is facebook's github repo.

Now you can install the PyText dependencies in a virtual environment. (This means the dependencies will be installed inside the directory `pytext_venv` under `pytext/`, not in your machine's system directory.) Notice the `(pytext_venv)` in the terminal prompt when it's active.

.. code-block:: console

  $ cd pytext
  $ source activation_venv
  (pytext_venv) $ ./install_deps

To exit the virtual environment:

.. code-block:: console

   (pytext_venv) $ deactivate


Writing Code
^^^^^^^^^^^^

After you've made some code changes, you need to create a branch to commit your code. Do not commit your code in your `master` branch! Give your branch a name that represents what your experiment is about. Then add your changes and commit them.

.. code-block:: console

  $ git checkout -b <my_experiment>
  $ git status -sb
  ... # list of files you changed
  $ git add <file1> <file2>
  $ git diff --cached  # see the code changes you added
  # ... maybe keep changing and run git add again
  $ git commit  # save your changes
  $ git show # optional, look at the code changes
  $ git push --set-upstream origin <my_experiment>  # send your branch to your github fork

At this point you should be able to see your branch in your github repo and create a Pull Request to Facebook's github if you want to submit it for review and later be integrated.


Keeping Up-to-Date
^^^^^^^^^^^^^^^^^^

To resume development in an already cloned repo, you might need re-activate the virtual environment:

.. code-block:: console

  $ cd pytext
  $ source activation_venv

If you need to update your github repo with the latest changes in the Facebook upstream repo, fetch the changes with this command, merge your master with those changes, and push the changes to your github forks. In order to do that, you can't have any pending changes, so make sure you commit your current work to a branch.

.. code-block:: console

  $ git fetch upstream
  $ git checkout master
  $ git merge upstream/master
  $ git push

Important: never commit changes in your master. Doing this would prevent further updates. Instead, always commit changes to a branch. (See below for more on this.)

Finally, you might need to rebase your branches to the latest master. Check out the branch, rebase it, and (optionally) push it again to your github fork.

.. code-block:: console

  $ git checkout <my_experiment>
  $ git rebase master
  $ git push  # optional


Modifying your Pull Request
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Many times you will need to modify your code and submit your pull request again. Maybe you found a bug that you need to fix, or you want to integrate some feedback you got in the pull request, or after you rebased your branch you had to solve a conflict.

If you're going to change your pull request, it's always a good idea to start by rebasing your branch on the lastest upstream/master (see above.)

After making your changes, amend to your existing commit rather than creating a new commit on top of it. This is to ensure your changes are in a single clean commit that does not contain your failed experiments. At this point, you will have a branch `<my_experiment>`, and the branch you pushed to your github forked `origin/<my_experiment>`. Then you will need to force the push to replace the github branch with your changes. The pull request will be automatically updated upstream.

.. code-block:: console

  $ git commit --amend
  $ git push --force


Addendum
--------

One commit or multiple commits?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For most contributions, you will want to keep your pull request as a single, clean commit. It's better to amend the same commit rather than keeping the entire history of intermediate edits.

If your change is more involved, it might be better to create multiple commits, as long as each commit does one thing and is self contained.

Code Quality
^^^^^^^^^^^^

In order to get your pull request integrated with PyText, it needs to pass the tests and be reviewed. The pull requests will automatically run the circleci tests, and they must be all green for your pull request to be accepted. These tests include building the documentation, run the unit tests under python 3.6 and 3.7, and run the linter `black` to verify code formatting. You can run the linter yourself after installing it with `pip install black`.

If all the tests are green, people will start reviewing your changes. (You too can review `other pull requests <https://github.com/facebookresearch/pytext/pulls>`_ and make comments and suggestions.) If reviewers ask questions or make suggestions, try your best to answer them with comments or code changes.

A very common reason to reject a pull request is lack of unit testing. Make sure your code is covered by unit tests (add your own tests) to make sure they work now and also in the future when other people make changes to your code!

Creating Documentation
^^^^^^^^^^^^^^^^^^^^^^

Whether you want to add documentation for your feature in code, or just change the existing the documentation, you will need to test it locally. First install extra dependencies needed to build the documentation:

.. code-block:: console

  $ pip install --upgrade -r docs_requirements.txt
  $ pip install --upgrade -r pytext/docs/requirements.txt

Then you can build the documentation

.. code-block:: console

  $ cd pytext/docs
  $ make html

Finally you can look at the documentation produced with a URL like this `file:///<path_to_pytext_sources>/pytext/docs/build/html/hacking_pytext.html`


Useful git alias
^^^^^^^^^^^^^^^^

One of the most useful command for git is one where you print the commits and branches like a tree. This is a complex command most useful when stored as an alias, so we're giving it here.

.. code-block:: console

  $ git config --global alias.lg "log --pretty=tformat:'%C(yellow)%h %Cgreen(%ad)%Cred%d %Creset%s %C(bold blue)<%cn>%Creset' --decorate --date=short --date=local --graph --all"

  $ # try it
  $ git lg
