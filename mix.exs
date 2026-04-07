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
      {:nx, "~> 0.10.0"},
      {:exla, "~> 0.10", optional: true},
      {:emlx, "~> 0.2", optional: true},
      {:rustler, "~> 0.36", runtime: false},
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false},
      {:propcheck, "~> 1.4", only: :test, runtime: false}
    ]
  end
end
