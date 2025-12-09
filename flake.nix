{
  description = "RISC-V cross-compiler";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        cross = import nixpkgs {
          crossSystem = {
            #config = "riscv64-unknown-linux-gnu";
            config = "riscv64-unknown-linux-musl";
          };
          overlays = [ (self: super: { gcc = self.gcc15; }) ];
          inherit system;
        };
      in
      {
        devShells.default =
          cross.mkShell {
            # RISC-V dependencies if you need them.
            nativeBuildInputs = with nixpkgs.legacyPackages.x86_64-linux;[ clang cmake gnuplot meson ninja pkg-config zig zls ];
            buildInputs = with cross.pkgsStatic; [ catch2 zlib ];
          };
      }
    );
}
