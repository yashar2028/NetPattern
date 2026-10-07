{
  description = "NetPattern sandbox environments (pinned by flake.lock) and the engine dev shell";

  inputs = {
    # Stable release; every PyTorch package used here is prebuilt in cache.nixos.org.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});

      engineProject = (builtins.fromTOML (builtins.readFile ./engine/pyproject.toml)).project;

      # Runtime dependencies of the engine, all from the pinned nixpkgs revision.
      # Keep in sync with `dependencies` in engine/pyproject.toml.
      engineDependencies =
        ps: with ps; [
          pydantic
          numpy
          pillow
          pydicom
          nibabel
          torch
          torchvision
          timm
          torchmetrics
        ];

      mkEngine =
        pkgs: ps:
        ps.buildPythonPackage {
          pname = engineProject.name;
          version = engineProject.version;
          pyproject = true;
          # Only the package sources: caches or tests in engine/ never change the environment hash.
          src = pkgs.lib.fileset.toSource {
            root = ./engine;
            fileset = pkgs.lib.fileset.unions [
              ./engine/pyproject.toml
              ./engine/src
            ];
          };
          build-system = [ ps.setuptools ];
          dependencies = engineDependencies ps;
          # Engine tests run in the engine-dev shell, against the same pinned packages.
          doCheck = false;
          pythonImportsCheck = [ "netpattern_engine" ];
        };
    in
    {
      packages = forAllSystems (
        pkgs:
        let
          python = pkgs.python3;
          engine = mkEngine pkgs python.pkgs;
          envCpu = python.withPackages (ps: [ engine ]);
        in
        {
          inherit engine;
          # The CPU sandbox environment: Python + PyTorch stack + NetPattern engine.
          env-cpu = envCpu;
          default = envCpu;
        }
      );

      devShells = forAllSystems (pkgs: {
        # Same pinned packages as env-cpu plus pytest; the engine is imported from source.
        engine-dev = pkgs.mkShell {
          packages = [
            (pkgs.python3.withPackages (ps: engineDependencies ps ++ [ ps.pytest ]))
          ];
        };
      });
    };
}
