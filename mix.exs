defmodule Exmc.MixProject do
  use Mix.Project

  def project do
    [
      app: :exmc,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: "Probabilistic programming for the BEAM",
      package: package()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0", "Commercial"],
      links: %{}
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.10.0"},
      {:exla, "~> 0.10", optional: true},
      {:emlx, "~> 0.2", optional: true},
      {:rustler, "~> 0.36", runtime: false}
    ]
  end
end
