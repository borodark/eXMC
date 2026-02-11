defmodule Exmc.NotebooksTest do
  use ExUnit.Case, async: true

  test "notebooks exist and are valid livemd" do
    notebooks_dir = Path.expand("../../notebooks", __DIR__)
    assert File.dir?(notebooks_dir)

    livemd_files = Path.wildcard(Path.join(notebooks_dir, "*.livemd"))
    assert length(livemd_files) >= 3

    for file <- livemd_files do
      content = File.read!(file)
      assert String.contains?(content, "# "), "#{file} missing heading"
      assert String.contains?(content, "```elixir"), "#{file} missing code cell"
    end
  end

  test "each notebook has Mix.install block" do
    notebooks_dir = Path.expand("../../notebooks", __DIR__)
    livemd_files = Path.wildcard(Path.join(notebooks_dir, "*.livemd"))

    for file <- livemd_files do
      content = File.read!(file)

      assert String.contains?(content, "Mix.install"),
             "#{Path.basename(file)} missing Mix.install"
    end
  end
end
