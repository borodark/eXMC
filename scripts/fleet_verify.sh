#!/usr/bin/env bash
#
# Run the eXMC verification on ONE host and report what actually happened.
#
#   ssh <host> 'bash -s' < scripts/fleet_verify.sh
#   ssh <host> 'bash -s' -- --pdb < scripts/fleet_verify.sh     # also posteriordb
#
# It takes no host list on purpose. Every host this project runs on is private
# infrastructure and this repository has a public remote, so the addresses live
# in the operator's ssh config and the driver loop lives in the operator's
# shell. What is worth version-controlling is the part that kept going wrong:
# the environment, the gates, and not truncating the output.
#
# WHY EACH PIECE IS HERE. All four of these produced a wrong or unreadable
# report at least once, and two of them cost a re-run of a suite measured in
# tens of minutes.
#
#   1. PATH. A `bash -s` session gets none of a login shell's environment.
#      On the Jetson `mix` and `cargo` are asdf/rustup installs that are not
#      on the default PATH, so a run died at `mix: command not found` after
#      pulling and reported nothing useful.
#
#   2. epmd. It lives inside the erlang install and is NOT an asdf shim, so a
#      non-interactive shell cannot find it and the BEAM cannot auto-start it.
#      Two DistributedTest cases then fail with `econnrefused` and
#      `:nodistribution` -- twice mistaken for a code regression, on two
#      different hosts, three days apart.
#
#   3. NO `set -e` around the completion marker. `mix test` exits non-zero
#      whenever anything fails, so a script that ends with
#      `echo "### DONE: $?"` under `set -e` aborts before the echo. A waiting
#      loop then blocks forever on a marker that cannot arrive, while the
#      suite has in fact finished. A completion marker guarded by `set -e` is
#      not a completion marker.
#
#   4. NO `tail`, `head` or grep filter on the suite output. The failure blocks
#      print BEFORE the summary line, so any tail drops exactly the part that
#      says which tests failed. That happened four times in one week here; each
#      time the fix was to re-run the whole suite to recover names that had
#      already been printed once.
#
# Exit status is the suite's, so a caller can gate on it. Read the output for
# the failure names -- the summary line alone tells you a count, not a fact.
#
# ONE THING THIS SCRIPT CANNOT GUARD: the caller's own timeout. A
# `timeout 3000 ssh ...` around a suite that takes 62 minutes produces a log
# with failure blocks and no summary line -- which reads as "some failures"
# when it means "cut off part-way". The non-vacuity check below catches that
# shape when the script reaches it, and an outer kill means it never does.
# Budget the caller's timeout from the SLOWEST host, not the fastest: measured
# 2026-09-10, roughly 20 min on the Keplers and 62 on the Jetson.

pdb=0
[ "${1:-}" = "--pdb" ] && pdb=1

# --- 1. environment ----------------------------------------------------------

export PATH="$HOME/.asdf/shims:$HOME/.asdf/bin:$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# The erts bin directory, for epmd. Searched rather than hardcoded because the
# OTP version differs across the fleet and pinning one path here would silently
# stop working on the next upgrade.
erts_bin=$(dirname "$(find "$HOME/.asdf/installs/erlang" -name epmd -type f 2>/dev/null | head -1)" 2>/dev/null)
[ -n "$erts_bin" ] && [ -d "$erts_bin" ] && export PATH="$erts_bin:$PATH"

cd "$HOME/exmc_oss" || { echo "FATAL: no ~/exmc_oss on $(hostname)"; exit 2; }

# --- 2. safety: verify the checkout by its REMOTE, never by its directory ----
#
# At least one host carries a second checkout whose working tree holds tracked
# credentials. Directory names are not a safety property; the remote URL is.
url=$(git config --get remote.origin.url)
case "$url" in
  *repos/exmc.git) : ;;
  *) echo "FATAL: refusing to run, unexpected origin: $url"; exit 2 ;;
esac

echo "### HOST     $(hostname)  $(uname -s)/$(uname -m)"
echo "### BEFORE   $(git log --oneline -1)"
dirty=$(git status --porcelain -uno)
[ -n "$dirty" ] && { echo "### DIRTY (tracked changes present):"; echo "$dirty"; }

git fetch -q origin || { echo "FATAL: fetch failed"; exit 2; }
git merge --ff-only origin/gate1/reconcile-core || { echo "FATAL: not a fast-forward; resolve by hand"; exit 2; }

echo "### HEAD     $(git log --oneline -1)"
echo "### NX_VULKAN $(grep -o '"nx_vulkan": {:git[^}]*}' mix.lock | grep -oE '[0-9a-f]{40}' | head -1)"
echo "### GLSLANG  $(command -v glslangValidator || echo MISSING)"

# epmd, or two distributed tests fail for reasons unrelated to the code.
if command -v epmd >/dev/null 2>&1; then
  epmd -daemon 2>/dev/null
  echo "### EPMD     $(command -v epmd) ($(epmd -names 2>&1 | head -1))"
else
  echo "### EPMD     MISSING -- DistributedTest will fail with :nodistribution"
fi

mix deps.get </dev/null 2>&1 | tail -3
MIX_ENV=test mix compile </dev/null 2>&1 | grep -iE "^\*\* |error:"

# --- 3. the suite, unfiltered -----------------------------------------------

echo "### SUITE START"

# tee, so the output is BOTH printed in full and available to gate on. Piping
# straight into a filter is what lost the failure names four times; not keeping
# a copy is what made the non-vacuity check below impossible to write.
suite_log=$(mktemp)
EXMC_COMPILER=vulkan mix test </dev/null 2>&1 | tee "$suite_log"
suite_status=${PIPESTATUS[0]}
echo "### SUITE EXIT $suite_status"

# NON-VACUITY, and it gates. A suite that printed no summary line did not run
# to completion -- it segfaulted, or died in compilation -- and its exit status
# then describes a crash rather than a result.
#
# This is not hypothetical. A bare `mix test` on one host printed nothing and
# returned within seconds, having dumped a 524 MB core; the harness of the day
# caught it only because it grepped for a summary line and found none. A caller
# that counts "0 failures" lines reads that silence as success.
if grep -qE "[0-9]+ tests?, [0-9]+ failures?" "$suite_log"; then
  echo "### SUMMARY  $(grep -E "[0-9]+ tests?, [0-9]+ failures?" "$suite_log" | tail -1)"
else
  echo "### FATAL: no ExUnit summary line -- the suite did not finish."
  echo "###        Check for a core file; exit status $suite_status describes a crash, not a result."
  rm -f "$suite_log"
  exit 3
fi
rm -f "$suite_log"

echo "### NOTE: read the failure blocks above the summary line, not the count alone"

# --- 4. posteriordb, opt-in --------------------------------------------------

if [ "$pdb" = "1" ]; then
  n_fixtures=$(ls benchmark/posteriordb/posteriordb_processed 2>/dev/null | wc -l)
  echo "### PDB FIXTURES $n_fixtures"

  # Committed since 702fb780f. Before that they were gitignored as
  # "regenerable", and a fleet run found the directory absent on every host:
  # the harness listed zero posteriors and reported success by having nothing
  # to fail. Gate on the count so that cannot recur silently.
  if [ "$n_fixtures" -lt 2 ]; then
    echo "FATAL: posteriordb fixtures missing -- the run would pass by doing nothing"
    exit 2
  fi

  echo "### PDB START"
  EXMC_COMPILER=vulkan mix run benchmark/posteriordb/validate_posteriordb.exs \
    --mode validate --compiler vulkan --tier fast </dev/null 2>&1
  echo "### PDB EXIT $?"
fi

exit $suite_status
