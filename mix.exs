defmodule Exmc.MixProject do
  use Mix.Project

  @version "0.3.1"
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
        Inference: [Exmc.NUTS.Sampler, Exmc.ADVI, Exmc.SMC, Exmc.Pathfinder],
        Compiler: [Exmc.Compiler, Exmc.PointMap, Exmc.Transform],
        Diagnostics: [Exmc.Diagnostics, Exmc.ModelComparison, Exmc.Predictive]
      ],
      source_ref: "v#{@version}"
    ]
  end

  defp deps do
    [
      nx_dep(),
      exla_dep(),
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

  # Default to the GitHub source so a fresh clone of this repo gets a
  # working nx_vulkan without needing a sibling checkout. Power users
  # iterating on nx_vulkan locally can override:
  #
  #     NX_VULKAN_PATH=/path/to/nx_vulkan mix deps.get
  #
  # This will eventually become `{:nx_vulkan, "~> x.x"}` from hex.pm.
  defp nx_vulkan_dep do
    nx_vulkan_dep(System.get_env("NX_VULKAN_PATH"))
  end

  # LOCAL git server, pinned to a commit — not `branch: "main"`, and not the
  # public GitHub mirror, which lags it. A branch ref means every `deps.get`
  # can pull a different backend; a pinned ref makes the dependency a fact and
  # puts any bump in the diff. Override with NX_VULKAN_GIT/NX_VULKAN_REF.
  #
  # 192.168.0.249 rather than localhost so the same mix.exs resolves from the
  # FreeBSD Keplers, which reach the server over the network.
  # The ref MUST be a sha that exists on the server above — not merely one that
  # exists in someone's local nx_vulkan checkout. Pinning an unpushed commit
  # breaks `mix deps.get` for every other host, and the Keplers are the ones
  # that would find out.
  #
  # This attribute was previously declared with no value, so `ref:` below
  # resolved to `nil` and the pinning this comment describes was not actually
  # happening — mix fell back to the lockfile, and a fresh resolve would have
  # taken whatever the default branch pointed at. The compiler had been saying
  # so ("undefined module attribute @nx_vulkan_ref") in every build.
  #
  # To bump: push nx_vulkan to origin first, then set this to the new sha and
  # run `mix deps.update nx_vulkan` so mix.lock moves with it.
  @nx_vulkan_git "git@192.168.0.249:/home/git/repos/nx_vulkan.git"
  @nx_vulkan_ref "7067499ecdc2f4b6a2981e5be4860139bfb8c712"

  defp nx_vulkan_dep(nil) do
    {:nx_vulkan,
     git: System.get_env("NX_VULKAN_GIT", @nx_vulkan_git),
     ref: System.get_env("NX_VULKAN_REF", @nx_vulkan_ref),
     optional: true}
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

  defp exla_dep, do: exla_dep(System.get_env("NX_PATH"))
  defp exla_dep(nil), do: {:exla, "~> 0.13", optional: true}
  defp exla_dep(path), do: {:exla, path: Path.join(path, "exla"), optional: true, override: true}
end
