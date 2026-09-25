# Git workflow

Our branching strategy, and the daily routine that goes with it.

Read the first two sections before your next task. The rest is here for the
moment something goes wrong.

---

## Strategy: short-lived feature branches, squashed into `main`

One branch per task. It starts from `main`, lives for a few days at most, and
disappears when its pull request is squashed into a single commit on `main`.

`main` is protected. Nobody pushes to it directly, including the repository
owner. Every change arrives through a pull request with one approving review,
and nobody approves their own work.

Because every branch is squashed on merge, **the history inside your branch does
not matter**. Commit as messily and as often as you like. Only the final squashed
commit message ends up on `main`. This matters more than it sounds, and the rest
of this document depends on it.

Branch names are the task number, then a short description, lowercase, dashes:

```
t16-verify-dataset-figures
t19-mlflow
```

---

## The daily routine

Four commands, every time you sit down to work:

```
cd <your netlift folder>
git checkout main
git pull
git checkout -b t16-verify-dataset-figures
```

The third line is the one people skip. Skipping it means you branch from a stale
`main`, and you find out days later when your pull request conflicts with work
that was merged while you were not looking.

If you are continuing a branch you already started, the last line is:

```
git checkout t16-verify-dataset-figures
```

---

## While you are working

**A commit is a save point. Uncommitted work is the only work git cannot get
back for you.**

So commit whenever you have something that is a little better than it was. Not
when it is finished — when it is *saved-worthy*:

```
git add .
git commit -m "Load Criteo and print row count"
```

Do not save these up for one perfect commit at the end. Five rough commits and
one polished commit look identical on `main` after the squash, and the five
rough ones mean an editor crash costs you twenty minutes instead of a day.

**Push your branch at the end of every session:**

```
git push
```

The first time on a new branch:

```
git push -u origin t16-verify-dataset-figures
```

A commit on your laptop protects you from your mistakes. A pushed branch
protects you from your laptop.

---

## When `main` moves while you are working

Someone else's task merges. Your branch is now based on an older `main`. This is
normal and expected, and it is not a problem until you leave it for a week.

Bring their work into your branch:

```
git add .
git commit -m "wip"          # commit first, always
git fetch origin
git merge origin/main
```

If git reports conflicts, it will list the files. Open each one, look for the
markers:

```
<<<<<<< HEAD
your version
=======
their version
>>>>>>> origin/main
```

Keep what should survive, delete the three marker lines, then:

```
git add .
git commit
git push
```

If a conflict is in a file you did not touch and do not understand, stop and
post it in `#help`. Do not guess. A wrongly resolved conflict silently deletes
someone else's work, and it will not show up until much later.

> Some teams rebase instead of merging here. We do not, deliberately. Rebasing
> rewrites your commits and needs a force push, and it replays conflicts once per
> commit instead of once in total. Since every branch is squashed on merge, the
> tidier history that rebasing buys is thrown away anyway. Merge is the safe
> option and it costs us nothing.

---

## You need to pull, but you have unsaved changes

Git will refuse, to protect you. Two ways out.

**Usually: just commit.** You are on your own branch, the history will be
squashed, and there is no reason not to.

```
git add .
git commit -m "wip"
```

**When you genuinely do not want a commit yet**, put the changes aside:

```
git stash
git pull
git stash pop
```

`git stash` is a shelf, not a drawer. Things left there get forgotten and are
invisible in every normal command. Check the shelf now and then:

```
git stash list
```

---

## You made changes on the wrong branch

Common, and completely recoverable. Nothing is committed yet, so the changes are
loose and can be carried across:

```
git stash
git checkout -b t16-verify-dataset-figures
git stash pop
```

If you already committed to the wrong branch, do not try to fix it alone. Post
in `#help`. It is a two-minute fix for someone who has done it before and an
hour of panic otherwise.

---

## The review cycle

```
git push
```

Open the pull request. Put `Closes #<issue number>` in the description, and
request a review from the partner named in your issue.

If the reviewer asks for changes, you do not open a new pull request. Commit on
the same branch and push:

```
git add .
git commit -m "Address review: add ipykernel to the checks"
git push
```

The pull request updates itself. Then press the ↻ next to the reviewer's name so
they are notified there is something new.

---

## After your pull request merges

```
git checkout main
git pull
git branch -d t16-verify-dataset-figures
```

The remote branch deletes itself on merge. The last line removes your local copy.
Git refuses `-d` if the branch has anything unmerged, which is a safety net, not
an obstacle — if it refuses, ask in `#help` before reaching for `-D`.

---

## Commands that delete work

These do not warn you, and what they remove was never committed, so there is
nothing to recover.

```
git checkout .          discards every unsaved change in the working tree
git restore .           the same thing, newer spelling
git reset --hard        discards unsaved changes and moves the branch back
git clean -fd           deletes untracked files and folders permanently
```

Do not run any of them to "fix" a confusing state. Commit first, then ask. A
commit you do not want is trivially removed later. A file you never committed is
gone.

---

## If you think you lost something

Anything that was ever committed is almost certainly still there. Git keeps a log
of everywhere your branch has pointed:

```
git reflog
```

You will see a list of recent positions with short hashes. Find the one from
before things went wrong and look at it:

```
git show <hash>
```

Bring it back on a branch of its own, so nothing else is disturbed:

```
git checkout -b recovered <hash>
```

Post in `#help` before doing this if you are unsure. Recovery is easy while the
reflog still holds the entry and harder once it expires.

---

## Never committed

```
data/raw/  data/interim/  data/processed/     datasets
.env                                          your local settings
.venv/                                        your virtual environment
mlruns/  mlflow.db  mlartifacts/              MLflow tracking store
```

`.gitignore` already handles all of these. If `git status` ever shows one of
them, do not commit it and do not edit `.gitignore` to hide it — say so in
`#help`, because it usually means a file landed somewhere it should not have.

The Criteo dataset is hundreds of megabytes and GitHub rejects any single file
over 100 MB. Committing it breaks the push and the history needs surgery to
recover, so this list is not a style preference.

---

## Quick reference

```
Start work         git checkout main
                   git pull
                   git checkout -b tNN-short-description

Save               git add .
                   git commit -m "what changed"

Back up            git push

Catch up on main   git fetch origin
                   git merge origin/main

Set aside          git stash  /  git stash pop

Where am I         git status
                   git branch
                   git log --oneline -10

Lost something     git reflog
```
