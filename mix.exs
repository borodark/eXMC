defmodule Exmc.MixProject do
  use Mix.Project

  @version "0.4.0"
  @source_url "https://github.com/borodark/eXMC"

  def project do
    [
      app: :exmc,
      version: @version,
      elixir: "~> 1.18",
      compilers: [:yecc, :leex | Mix.compilers()],
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "Probabilistic programming for the BEAM. NUTS/HMC, ADVI, SMC, Pathfinder. " <>
          "Inspired by PyMC. Beats PyMC on 4 of 7 benchmarks.",
      package: package(),
      docs: docs(),
      source_url: @source_url,
      homepage_url: "http://dataalienist.com"
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp package do
    [
      licenses: ["Apache-2.0", "Commercial"],
      links: %{
        "GitHub" => @source_url,
        "Website" => "http://dataalienist.com"
      },
      files: ~w(lib mix.exs README.md LICENSE_COMMUNITY.md CHANGELOG.md FOREWORD.md)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "FOREWORD.md",
        "DECISIONS.md",
        "docs/SCHEDULER_PINNING.md",
        "docs/WARM_START_NUTS.md",
        "docs/STATE_SPACE_MODELS.md"
      ],
      groups_for_extras: [
        Guides: ~r/docs\//,
        Architecture: ~r/DECISIONS|FOREWORD/
      ],
      groups_for_modules: [
        "Model Building": [Exmc.Builder, Exmc.DSL, Exmc.IR, Exmc.Node],
        Distributions: ~r/Exmc\.Dist\./,
        Inference: [
          Exmc.NUTS.Sampler,
          Exmc.MCLMC,
          Exmc.MAMS,
          Exmc.ADVI,
          Exmc.SMC,
          Exmc.Pathfinder,
          Exmc.SBI
        ],
        "Inference internals": [Exmc.MCLMC.Integrator, Exmc.MCLMC.Tuning],
        "Simulation-based inference": [
          Exmc.SBI.ABC,
          Exmc.SBI.ABCSMC,
          Exmc.SBI.Engine,
          Exmc.SBI.Prior,
          Exmc.SBI.Simulator
        ],
        Compiler: [Exmc.Compiler, Exmc.PointMap, Exmc.Transform],
        Diagnostics: [Exmc.Diagnostics, Exmc.ModelComparison, Exmc.Predictive]
      ],
      source_ref: "v#{@version}"
    ]
  end

  # FreeBSD is a Vulkan-only platform for this project, and that is a fact about
  # the toolchain rather than a preference. The `xla` archive ships
  # x86_64/aarch64 darwin and linux-gnu targets and nothing else, so on FreeBSD
  # `XLA.download_precompiled!/1` raises
  #
  #     no precompiled XLA archive available for this target: amd64-freebsd15.0-cpu
  #
  # and `mix compile` dies in the dependency before reaching a single module of
  # this library. Measured on mac-247, 2026-08-16, against a fresh clone of
  # 0.3.1. The only escape hatch upstream offers is XLA_BUILD=true — a Bazel
  # build of XLA from source, on a platform XLA has never been validated on.
  #
  # So exla is not merely optional here, it is *absent*: declaring it at all is
  # what breaks the build. 0.3.1 already made exla `runtime: false` so a broken
  # exla could not abort the VM at boot, and said in the CHANGELOG that
  # "optional has to mean optional at runtime too, not merely at resolution
  # time". This is the other half of that sentence — the resolution-time half,
  # which that fix assumed was already fine.
  #
  # Nothing else changes: `nx` is pure Elixir, `nx_vulkan` builds from source,
  # and `Exmc.JIT.auto_detect/0` already prefers EXLA where it exists and falls
  # through to Vulkan where it does not. With exla off the dependency list on
  # FreeBSD, auto-detect lands on Vulkan by itself — no config, no override.
  @freebsd? match?({:unix, :freebsd}, :os.type())

  defp deps do
    [
      nx_dep()
    ] ++ exla_deps() ++ rest_of_deps()
  end

  defp exla_deps, do: if(@freebsd?, do: [], else: [exla_dep()])

  defp rest_of_deps do
    [
      # EMLX (Apple Metal / MLX) is postponed until real Apple hardware is
      # available to develop and test against — see `Exmc.JIT` moduledoc.
      # Cross-platform GPU compute via Vulkan (FreeBSD + Linux non-CUDA + macOS via MoltenVK).
      # Local git server, pinned. Override with
      # `NX_VULKAN_PATH=/path/to/nx_vulkan mix deps.get` for local iteration.
      nx_vulkan_dep(),
      {:rustler, "~> 0.36", runtime: false},
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:propcheck, "~> 1.4", only: :test, runtime: false}
    ]
  end

  # Default to the pinned git source (the LOCAL server — see below, not
  # GitHub) so a fresh clone of this repo gets a working nx_vulkan without
  # needing a sibling checkout. Power users iterating on nx_vulkan locally
  # can override:
  #
  #     NX_VULKAN_PATH=/path/to/nx_vulkan mix deps.get
  #
  # This will eventually become `{:nx_vulkan, "~> x.x"}` from hex.pm.
  defp nx_vulkan_dep do
    nx_vulkan_dep(System.get_env("NX_VULKAN_PATH"))
  end

  # LOCAL git server, following `main` — not the public GitHub mirror, which
  # lags it. 192.168.0.249 rather than localhost so the same mix.exs resolves
  # from the FreeBSD Keplers and the Jetson, which reach the server over the
  # network.
  #
  # This tracked a pinned sha until 2026-08-28. The argument for the pin was
  # that a branch lets every `deps.get` pull a different backend — but that is
  # what `mix.lock` is for. With a branch, the lock still records one concrete
  # sha, `deps.get` still resolves to it, and only `mix deps.update nx_vulkan`
  # moves it. What actually changed is where a bump shows up: mix.lock alone,
  # rather than mix.exs and mix.lock together. Read the lock, not this file,
  # to learn which backend a given checkout builds against.
  #
  # To bump: push nx_vulkan to its origin first, then run
  # `mix deps.update nx_vulkan` here and commit the mix.lock change. The sha
  # the lock lands on MUST exist on the server above — a lock pointing at an
  # unpushed commit breaks `mix deps.get` for every other host, and the
  # Keplers are the ones that would find out.
  #
  # NX_VULKAN_REF still overrides, for bisecting a backend regression without
  # editing this file:
  #
  #     NX_VULKAN_REF=<sha> mix deps.get
  @nx_vulkan_git "git@192.168.0.249:/home/git/repos/nx_vulkan.git"
  @nx_vulkan_branch "main"

  defp nx_vulkan_dep(nil) do
    base = [git: System.get_env("NX_VULKAN_GIT", @nx_vulkan_git), optional: true]

    pin =
      case System.get_env("NX_VULKAN_REF") do
        nil -> [branch: @nx_vulkan_branch]
        ref -> [ref: ref]
      end

    {:nx_vulkan, base ++ pin}
  end

  defp nx_vulkan_dep(path),
    do: {:nx_vulkan, path: path, optional: true}

  # nx + exla: default to the hex 0.13 release. For local iteration against the
  # elixir-nx/nx monorepo (e.g. an unreleased EXLA fix), point at the checkout:
  #
  #     NX_PATH=/path/to/nx mix deps.get   # expects $NX_PATH/nx and $NX_PATH/exla
  #
  # The path deps `override: true` so exla's own `{:nx, path: "../nx"}` resolves
  # to the same tree.
  defp nx_dep, do: nx_dep(System.get_env("NX_PATH"))
  defp nx_dep(nil), do: {:nx, "~> 0.13"}
  defp nx_dep(path), do: {:nx, path: Path.join(path, "nx"), override: true}

  # `runtime: false` keeps exla off the boot path — it stays on the code path
  # and every module loads, but nothing starts it automatically. `Exmc.JIT`
  # starts it on first use and treats a failed start as "backend unavailable".
  #
  # Without this, an exla that is *present but broken* aborts the VM before a
  # single test runs. That is not hypothetical: the CUDA build of exla links
  # `libnvshmem_host.so.3`, and on a host without it `EXLA.Application.start/2`
  # fails, `mix test` never reaches ExUnit, and the whole suite is unrunnable
  # over a dependency this project advertises as optional. Optional has to mean
  # optional at runtime too, not merely at resolution time.
  #
  # This costs consumers nothing: `optional: true` means they declare exla in
  # their own deps to get it, which puts it in *their* application list and
  # starts it at boot as usual. The lazy start below is then a no-op.
  #
  # On a machine whose GPU stack cannot satisfy the CUDA build, compile the CPU
  # one. It has two traps that both report as something other than what they
  # are; the recipe and the reasoning are in docs/EXLA_CPU_BUILD.md.
  defp exla_dep, do: exla_dep(System.get_env("NX_PATH"))
  defp exla_dep(nil), do: {:exla, "~> 0.13", optional: true, runtime: false}

  defp exla_dep(path),
    do: {:exla, path: Path.join(path, "exla"), optional: true, runtime: false, override: true}
end
