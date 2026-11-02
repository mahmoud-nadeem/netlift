# Working agreement

How the NetLift team works from 13 October 2026 to the final submission on 4 December 2026. It
applies to every member, the team leader included. The plan it serves is
[`project-plan.md`](project-plan.md).

## Why this exists

Milestone 1 was delivered, but too much of it was spent waiting: for a review, for an answer, for
someone else's piece to land. The daily routines we set up on Discord were not used, and there were
no fixed times when the team was expected to be working or reachable. With seven weeks left and
fixed DEPI deadlines, waiting is the one cost the schedule cannot absorb. Every rule below exists
to remove a reason to wait.

---

## 1. Where things happen

| What | Where |
|---|---|
| Tasks, owners, partners, due dates | GitHub Issues |
| The work itself, and its review | GitHub Pull Requests |
| Decisions that others depend on | `docs/decisions/`, through a pull request |
| Daily standup, review requests, blockers, reminders | Team WhatsApp group |
| Meetings and pairing | Discord voice: `Team Meeting`, `Pair Room` (screen sharing) |
| Repository activity feed | Discord `#github` |

The WhatsApp group is for short messages. Anything that has to be remembered goes into an issue or
pull-request comment, where it stays attached to the work. Discord is kept for voice, because
reviewing code together needs screen sharing, and for the `#github` channel, which shows repository
activity as it happens. Its other text channels were removed.

---

## 2. The weekly rhythm

**Daily standup, by 22:00, Saturday to Thursday.** Three lines in the WhatsApp group, with issue
numbers:

```
Done:    T38 response model trained on Hillstrom (#NN)
Next:    T38 open the pull request
Blocked: nothing
```

It takes one minute. A day with no standup gets a check-in from the team leader the next morning.

**Two meetings a week, 30 minutes each, in the Discord `Team Meeting` voice channel.**

| Meeting | When | Agenda |
|---|---|---|
| Plan the week | Saturday 21:00 | The leader shares the GitHub status first: what is late, what is waiting for review. Each member says which issues will be in a pull request by Thursday. |
| Unblock | Tuesday 21:00 | Anything that has waited more than 24 hours is resolved in the meeting, or reassigned. |

The times are confirmed at the first meeting and then stay fixed. If you cannot attend, post your
update in the group before the meeting starts.

**Twelve hours per member per week, in blocks you announce.** At the Saturday meeting each member
posts their working blocks for the coming week in the WhatsApp group: at least three different
days, no block shorter than 90 minutes, twelve hours in total. For example:

```
Sat 19:00-22:00 · Mon 20:00-23:00 · Wed 19:00-22:00 · Thu 16:00-19:00
```

The blocks tell the rest of the team when you can be reached for a review, a question or a pairing
session, and they tell the leader when to check in. If a block moves, say so in the group. If a
week will be shorter for you, say so on Saturday so the work can be moved before it is late, not
after.

---

## 3. Rules that remove waiting

1. **Build against the contract, not the person.** The three interfaces frozen in T36 (feature
   matrix and folds, metric signature, scoring response) exist so that nobody waits for anybody.
   If your input is not ready, work against a stub or fixture of the agreed shape and swap the real
   one in when it lands.
2. **Reviews within 24 hours.** When a pull request asks for your review, review it within 24 hours.
   If you cannot, say so in the group inside that window, and another member reviews it instead.
   Any member can approve; the partner is the first choice, not the only one. The reviewer checks
   each number against its source, and re-runs the script when the result depends on one. A pull
   request updated after review goes back to the reviewer the same day.
3. **Stuck for an hour, ask.** Post in the group: the issue number, what you tried, what you need.
   **Stuck for a day**, the team leader decides the same day: unblock it, simplify it, or move it.
4. **Make work visible early.** Open a draft pull request within 48 hours of starting a task, and
   push at the end of every working session. Work that is not pushed does not exist for anyone else.
5. **Small tasks.** Every task fits in three working days. If it does not, split it before starting.
6. **The internal deadline is the day before DEPI's.** The deadline day itself is for review,
   merging and uploading.
7. **Less paperwork.** A document is written only if DEPI requires it or it prevents a real error.
   Nothing already written is deleted. (Decided 16 October.)
8. **Block only what breaks.** A review requests changes only for something that breaks a result
   or a deliverable; everything else is a non-blocking comment.

---

## 4. When a task slips

**Say it early.** If you will miss a due date, say so in the group before the date passes, with a
new date. That is normal, and the plan can absorb it when it is known in advance.

**A silent slip is moved.** If a task has no commit, pull request or message for 48 hours after its
due date, the team leader moves it to its partner, and the original owner becomes the partner.
This is not a penalty; it keeps the milestone alive. Both are recorded in the issue history.

DEPI's final evaluation asks each team to show what every member built and was responsible for.
The GitHub record (issue assignees, pull-request authors, reviewers) is that evidence, for everyone.

---

## 5. Definition of done

A task is done when all of these hold:

- Its pull request is merged into `main`, with CI green and approval from someone other than the
  author.
- The pull-request description starts with `Closes #N` for the issue it resolves.
- Documentation is updated wherever the change alters behaviour, a number or an interface.
- Every number in a document comes from a script or file in the repository, not from memory.

---

## 6. Decisions

A decision other people depend on (a library, an interface, a threshold, a scope cut) gets a
numbered record in `docs/decisions/`, in the format of
[`0001-uplift-library.md`](decisions/0001-uplift-library.md): context, options considered, decision,
consequences. An agreement in the WhatsApp group is not a decision until it is recorded there.

---

## 7. The team leader

The team leader runs both meetings, shares the GitHub status every Saturday, decides on anything
blocked for a day within that day, reviews when nobody else can, and keeps the plan current. The
leader is held to every rule above.
