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
# Exit status is the suite's, so a caller can gate on it -- or 4 when the suite
# is green and the consumer-path smoke (section 4) failed. Read the output for
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

# The working branch is `main`. `gate1/reconcile-core` was retired on
# 2026-09-13, merged into main by fast-forward. Every fleet checkout had it
# checked out, so a host still on it (or on anything else) is moved to main
# here. A host's local `main` may be far behind (mac-247's was 145 commits), which
# the fast-forward below takes care of. Tracked changes stop the checkout, and
# that is the right outcome: the DIRTY block above has already printed them.
#
# FLEET_BRANCH names another origin branch to verify, for a change that should
# pass the fleet BEFORE it lands on main (a lock bump, say). Default main. The
# host is left on that branch; its next ordinary run moves it back to main.
target=${FLEET_BRANCH:-main}
branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "$target" ]; then
  echo "### BRANCH   $branch -> $target"
  if git show-ref --verify --quiet "refs/heads/$target"; then
    git checkout -q "$target" || { echo "FATAL: cannot check out $target; resolve by hand"; exit 2; }
  else
    git checkout -q -b "$target" "origin/$target" || { echo "FATAL: cannot create $target; resolve by hand"; exit 2; }
  fi
fi
git merge --ff-only "origin/$target" || { echo "FATAL: not a fast-forward; resolve by hand"; exit 2; }

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

# PREBUILT NIF, when one is provably CURRENT -- mirrors nx_vulkan's
# scripts/fleet_verify.sh, keyed on THIS repo's lock rather than their HEAD.
#
# The Jetson compiles the nx_vulkan crate in ~12 min incremental, ~47 clean;
# super-io cross-builds it in ~2. From an nx_vulkan checkout on super-io:
#
#   REF=<lock sha> DEST_DIR='$HOME/exmc_oss/deps/nx_vulkan' sh scripts/deploy_jetson_nif.sh
#
# ships the .so into this checkout's deps dir with a provenance marker beside
# it: line 1 the commit it was built from, line 2 its sha256 as it landed.
# (That script refuses unless deps/nx_vulkan is already at the built sha, so
# `mix deps.get` at the lock comes first.)
#
# Use it only when the marker names the LOCK sha AND its hash matches the file
# on disk. Both halves are load-bearing. `mix deps.get` on a lock bump checks
# out the new commit but leaves priv/native alone -- TESTED by the nx_vulkan
# session -- so a stale .so and its stale marker survive together, and only
# the sha comparison sends that run to a native build. A native build
# overwrites the .so and leaves the old marker beside it, and only the hash
# comparison catches that. Anything else builds natively: slow and correct.
nxv_lock=$(grep -o '"nx_vulkan": {:git[^}]*}' mix.lock | grep -oE '[0-9a-f]{40}' | head -1)
# Rustler 0.38 installs and loads `priv/native/<crate>.so`, with NO `lib` prefix
# (exmc bumped 2026-09-13, nx_vulkan lock ca1e0c8). A `lib<crate>.so` left from
# 0.36 is never loaded again, but it is still a file with the old name, and a
# checksum of it measures something the VM does not run. So stale lib-prefixed
# NIFs, their markers and deploy_jetson_nif.sh's `.prev` backups of them are
# deleted here, before anything hashes or builds. The glob covers nx_vulkan's old
# libnx_vulkan_native.so too; the unprefixed `.so.prev` a deploy keeps is left.
rm -f deps/nx_vulkan/priv/native/lib*.so deps/nx_vulkan/priv/native/lib*.so.provenance \
  deps/nx_vulkan/priv/native/lib*.so.prev _build/*/lib/exmc/priv/native/lib*.so
nxv_so=deps/nx_vulkan/priv/native/nx_vulkan_vulkano.so
nxv_prov=$nxv_so.provenance
nxv_hash() { { sha256sum "$nxv_so" 2>/dev/null || sha256 -q "$nxv_so" 2>/dev/null; } | cut -d' ' -f1; }

prebuilt=0
if [ -f "$nxv_prov" ] && [ -f "$nxv_so" ]; then
  p_sha=$(sed -n 1p "$nxv_prov")
  p_hash=$(sed -n 2p "$nxv_prov")
  a_hash=$(nxv_hash)
  [ "$p_sha" = "$nxv_lock" ] && [ "$p_hash" = "$a_hash" ] && prebuilt=1
  echo "### PREBUILT marker=${p_sha:0:7} lock=${nxv_lock:0:7} hash_match=$([ "$p_hash" = "$a_hash" ] && echo yes || echo no) using=$prebuilt"
else
  echo "### PREBUILT none"
fi

# The skip is compile_env (see config/config.exs), baked into
# Nx.Vulkan.NativeV. A box switching between prebuilt and native would fail to
# boot against the other mode's compiled value, and neither `rm` of the beam
# nor `touch` of the source recovers it (nx_vulkan's fleet_verify.sh records
# both failing on the fleet). Wiping the app's ebin and manifests when the
# mode changes does, and costs only the Elixir side.
mode_marker=_build/test/.nxv_prebuilt_mode
last_mode=$(cat "$mode_marker" 2>/dev/null || echo unknown)
if [ "$last_mode" != "$prebuilt" ]; then
  echo "### BUILDMODE $last_mode -> $prebuilt (clean nx_vulkan Elixir recompile)"
  rm -rf _build/test/lib/nx_vulkan/ebin _build/test/lib/nx_vulkan/.mix
fi

if [ "$prebuilt" = "1" ]; then
  export NXV_SKIP_NIF_BUILD=1
fi
MIX_ENV=test mix compile </dev/null 2>&1 | grep -iE "^\*\* |error:"
mkdir -p _build/test && echo "$prebuilt" > "$mode_marker"

# Proof the skip held: a compile that rebuilt the crate anyway changes the
# hash. The suite below would then be valid -- a native build of the lock --
# but this run would be reporting a prebuilt it did not use.
if [ "$prebuilt" = "1" ] && [ "$(nxv_hash)" != "$p_hash" ]; then
  echo "### PREBUILT IGNORED: the compile replaced $nxv_so -- the skip did not reach Rustler"
fi

# --- 3. the device, pinned --------------------------------------------------
#
# WHICH GPU EACH HOST MUST RUN ON, keyed by `hostname -s`: a uuid prefix (the
# identity nx_vulkan's selector matches; an index is not one) and a substring of
# the device name, for a reader and as a second check. Hostnames and GPU
# models, never addresses -- see the header for why addresses stay out.
#
# Without a pin, nx_vulkan picks by device type (discrete < integrated < virtual
# < CPU), and two hosts here enumerate llvmpipe beside the real card (super-io,
# the NUC). If the real driver fails to start -- a kernel module that did not
# load, a driver upgrade -- llvmpipe is the only device left and is selected
# silently, and the suite reports a CPU rasteriser's count as the GPU's. On a
# two-card host the winner is whichever enumerates first, which a reseat moves.
#
# NXV_DEVICE is read by the NIF and takes precedence over everything; a selector
# that matches nothing is `{:error, :vulkan_init_failed, ...}` listing the
# devices, not a fallback (MEASURED on super-io, 2026-09-13). The probe below
# resolves it in a process of its own before the suite, because the NIF's own
# banner goes to stderr and interleaves with ExUnit's dots mid-string
# (nx_vulkan's fleet_verify.sh records a false failure from parsing it).
#
# A new host fails here until it has a row, which is the point: the row is the
# claim docs/ARMS.md makes about it. NXV_SKIP_DEVICE_PIN=1 runs a host on
# whatever it picks -- deliberately, and the log says so.
expected_device() {
  case "$1" in
    super-io)           echo "f7e146ef RTX 3060 Ti" ;;
    mac)                echo "c3fcb5dd GT 650M" ;;         # mac-247
    free-macpro-nvidia) echo "91f659e1 GT 750M" ;;         # mac-248
    nuc)                echo "86801619 HD Graphics 520" ;;
    jake-desktop)       echo "a220528a Tegra X1" ;;        # Jetson
    *)                  echo "" ;;
  esac
}

host_short=$(hostname -s)
pin=$(expected_device "$host_short")
if [ "${NXV_SKIP_DEVICE_PIN:-0}" = "1" ]; then
  echo "### DEVICE   pin SKIPPED (NXV_SKIP_DEVICE_PIN=1) -- this run is on whatever nx_vulkan picks"
  unset NXV_DEVICE
elif [ -z "$pin" ]; then
  echo "FATAL: no device pin for host '$host_short'."
  echo "       Add it to expected_device() in scripts/fleet_verify.sh (uuid prefix + name),"
  echo "       or rerun with NXV_SKIP_DEVICE_PIN=1 to run unpinned on purpose."
  exit 2
else
  export NXV_DEVICE="uuid:${pin%% *}"
fi

device_probe=$(MIX_ENV=test mix run --no-start --no-compile </dev/null 2>/dev/null -e '
  case Nx.Vulkan.NativeV.device_info() do
    {:ok, i, by} ->
      IO.puts("DEVICEINFO kind=#{i.kind} uuid=#{i.uuid} pci=#{i.pci || "none"} driver=#{i.driver} f64=#{i.supports_f64} selected_by=#{by} name=#{i.name}")
    other ->
      IO.puts("DEVICEINFO UNRESOLVED #{inspect(other)}")
  end' | grep '^DEVICEINFO' | tail -1)
echo "### DEVICE   ${device_probe:-DEVICEINFO NONE (the probe printed nothing)}"

device_fail=""
case "$device_probe" in
  "")                        device_fail="the probe printed nothing" ;;
  *UNRESOLVED*)              device_fail="${device_probe#DEVICEINFO }" ;;
  *kind=Cpu*)                device_fail="a CPU (software) Vulkan device" ;;
esac
if [ -z "$device_fail" ] && [ -n "${NXV_DEVICE:-}" ]; then
  case "$device_probe" in
    *"uuid=${pin%% *}"*"name="*"${pin#* }"*) : ;;
    *) device_fail="expected uuid ${pin%% *}... '${pin#* }', got ${device_probe#DEVICEINFO }" ;;
  esac
fi
if [ -n "$device_fail" ]; then
  echo "FATAL: wrong or missing GPU on $host_short: $device_fail"
  echo "       The suite was NOT run; a count from this box would describe another device."
  exit 2
fi

# --- 4. the consumer path ---------------------------------------------------
#
# `mix run`, not `mix test`: the path a release, a consumer project and every
# bench take. On 2026-09-13 the suite was green on every FreeBSD host while this
# path could not sample on the GPU at all (`:crypto` undeclared; `mix test`
# loads it regardless). scripts/vulkan_smoke.exs samples two models through the
# chain shader and checks synthesis, chain-dispatch count and loose moments.
#
# It does not stop the suite -- a broken consumer path and a failing suite are
# both worth knowing in one run -- but it gates the exit status: a smoke failure
# with a green suite exits 4.
echo "### SMOKE START"
EXMC_COMPILER=vulkan MIX_ENV=test mix run --no-compile scripts/vulkan_smoke.exs </dev/null 2>&1 |
  grep -vE '^\[nx_vulkan_vulkano\]'
smoke_status=${PIPESTATUS[0]}
echo "### SMOKE EXIT $smoke_status"

# --- 5. the suite, unfiltered -----------------------------------------------

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

# --- 6. posteriordb, opt-in --------------------------------------------------

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

# The suite's status first, since its failures say more; a green suite over a
# broken consumer path is not green.
if [ "$suite_status" = "0" ] && [ "$smoke_status" != "0" ]; then
  echo "### EXIT 4: suite green, consumer-path smoke FAILED -- see ### SMOKE above"
  exit 4
fi
exit $suite_status
