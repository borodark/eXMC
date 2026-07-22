defmodule Exmc.MixProject do
  use Mix.Project

  @version "0.2.0"
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
      # GitHub source until nx_vulkan reaches hex.pm. Override with
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

  defp nx_vulkan_dep(nil),
    do: {:nx_vulkan, github: "borodark/nx_vulkan", branch: "main", optional: true}

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
