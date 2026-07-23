defmodule Exmc.PushFallbackTest do
  @moduledoc false
  use ExUnit.Case, async: false

  # A many-parameter observed model whose prior floats exceed the 128-byte
  # f64 push-constants block. detect_meta signals {:unsupported,
  # :push_too_large} on every backend; under Vulkan this previously crashed
  # at dispatch with {:error, :push_too_large}, now it degrades to per-op
  # sampling. The assertions hold on any backend, so the test is untagged.
  test "push_too_large model falls back to per-op instead of crashing" do
    code = """
    data { real y; }
    parameters {
      real a1; real a2; real a3; real a4; real a5;
      real a6; real a7; real a8; real a9; real a10;
    }
    model {
      a1 ~ normal(0, 1); a2 ~ normal(0, 1); a3 ~ normal(0, 1);
      a4 ~ normal(0, 1); a5 ~ normal(0, 1); a6 ~ normal(0, 1);
      a7 ~ normal(0, 1); a8 ~ normal(0, 1); a9 ~ normal(0, 1);
      a10 ~ normal(0, 1);
      y ~ normal(a1, 1);
    }
    """

    ir = Exmc.Stan.compile!(code, %{"y" => Nx.tensor(3.0)})

    assert Exmc.NUTS.ChainShaderCodegen.detect_meta(ir) == {:unsupported, :push_too_large}

    {trace, _stats} =
      Exmc.Sampler.sample(ir, %{}, num_warmup: 40, num_samples: 40, seed: 42)

    vals = trace["a1"] |> Nx.to_flat_list()
    assert length(vals) == 40
    assert Enum.all?(vals, &is_number/1)
  end
end
