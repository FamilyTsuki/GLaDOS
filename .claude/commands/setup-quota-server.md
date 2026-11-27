---
description: Install the weekly Claude-quota burn (GLaDOS dataset generation) on this always-on server
---

Set up this machine so that every Wednesday from 00:00 to ~02:55 (Europe/Paris), leftover
weekly Claude quota is spent generating GLaDOS training examples with Opus, and the
results are pushed to the `dataset-auto` branch of origin. The weekly window resets on
Wednesday at 03:00 Paris time.

How it works (read these files first):
- `training/generate_dataset.py --backend claude`: calls `claude -p` with settings,
  plugins, hooks, tools and MCP disabled, writes JSONL examples, stops on deadline or
  when Claude reports the usage limit.
- `training/quota/burn_weekly_quota.py --force --minutes N`: runs N parallel generators
  into `training/parts/claude_<stamp>_w*.jsonl`.
- `training/quota/server_run.sh`: pulls `origin/myGLaDOS`, runs the burn, force-adds
  the new parts (the `parts/` folder is gitignored) and pushes `HEAD:dataset-auto`.
- `training/quota/glados-quota-burn-server.{service,timer}`: systemd user units.

Do these steps in order. Explain each step briefly in French to the user, who is a
beginner. Ask before anything that needs sudo.

1. Check prerequisites: `python3 --version` (3.10+), `git`, `systemctl --user` works,
   `claude --version`. If `claude` is missing, install it with
   `curl -fsSL https://claude.ai/install.sh | bash`.
2. Check the repo: this command must run inside a clone of
   https://github.com/FamilyTsuki/GLaDOS. Create and switch to a local branch
   `dataset-auto` based on `origin/myGLaDOS` if not already on it.
3. Auth for Claude: headless servers cannot do the browser login reliably. Ask the
   user to run `claude setup-token` (on this server or on their PC) and paste the
   token. Write it to `~/.config/glados-quota.env` as
   `CLAUDE_CODE_OAUTH_TOKEN=<token>` with mode 600. Never echo the token back, never
   commit it.
4. Auth for git push: check `git push --dry-run origin HEAD:dataset-auto`. If it fails,
   help the user add an SSH key to GitHub (`ssh-keygen -t ed25519`, then show the
   public key and tell them to add it in GitHub > Settings > SSH keys, or as a deploy
   key with write access on the repo) and switch origin to
   `git@github.com:FamilyTsuki/GLaDOS.git`.
5. Smoke test, about 1 minute of quota:
   `set -a; . ~/.config/glados-quota.env; set +a; python3 training/quota/burn_weekly_quota.py --force --minutes 1 --workers 1 --model opus`
   then check that `training/parts/claude_*_w0.jsonl` has more lines than
   `training/seeds.jsonl` and that the examples look like GLaDOS (cold "vous", short,
   sarcastic, useful). Delete that test file afterwards so it is not pushed.
6. Install the units: copy `training/quota/glados-quota-burn-server.service` and
   `.timer` to `~/.config/systemd/user/`. If the clone is not at `~/GLaDOS`, edit
   `Environment=REPO=` in the copied service. If `claude` is not in `~/.local/bin`,
   add its directory to `Environment=PATH=`. Then
   `systemctl --user daemon-reload && systemctl --user enable --now glados-quota-burn-server.timer`.
7. Make user timers run without an open session: `sudo loginctl enable-linger $USER`
   (ask first).
8. Show `systemctl --user list-timers glados-quota-burn-server.timer` and confirm the
   next run is a Wednesday at 00:00 Paris time. Check the server's timezone handling
   with `systemd-analyze calendar "Wed *-*-* 00:00:00 Europe/Paris"`.
9. Remind the user to disable the timer on their PC so both do not run at once:
   `systemctl --user disable --now glados-quota-burn.timer`.

Tuning: set `BURN_MINUTES`, `BURN_WORKERS` or `BURN_MODEL` in
`~/.config/glados-quota.env`. Logs: `journalctl --user -u glados-quota-burn-server`.
