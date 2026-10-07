{
  description = "NetPattern engine and its pinned sandbox environments";

  inputs = {
    # Stable release; the CPU PyTorch stack used here is prebuilt in cache.nixos.org.
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
  };

  outputs =
    { self, nixpkgs }:
    let
      lib = nixpkgs.lib;
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forAllSystems = f: lib.genAttrs systems f;

      project = (builtins.fromTOML (builtins.readFile ./pyproject.toml)).project;

      # What the engine always needs. Keep in sync with `dependencies` in pyproject.toml.
      baseDependencies =
        ps: with ps; [
          pydantic
          numpy
          pillow
          safetensors
          torch
          torchvision
          timm
          torchmetrics
        ];

      # Capability packs (PLAN D23): curated extras the platform adds automatically when a
      # pipeline needs them. Keep in sync with `optional-dependencies` in pyproject.toml.
      packDependencies = {
        medical = ps: with ps; [
          pydicom
          nibabel
        ];
        detection = ps: with ps; [ pycocotools ];
      };

      # Templates are Python package sets from the same pinned nixpkgs revision.
      pythonFor = {
        pytorch-cpu = system: nixpkgs.legacyPackages.${system}.python3;

        # Official PyTorch wheels with CUDA 12 bundled (unfree). Packages built on top of
        # torch are rebuilt against the wheels; their test suites are skipped.
        pytorch-cuda12 =
          system:
          let
            pkgs = import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
          in
          pkgs.python3.override {
            packageOverrides = final: prev: {
              torch = prev.torch-bin;
              torchvision = prev.torchvision-bin;
              timm = prev.timm.overridePythonAttrs { doCheck = false; };
              torchmetrics = prev.torchmetrics.overridePythonAttrs { doCheck = false; };
            };
          };
      };

      mkEngine =
        ps:
        ps.buildPythonPackage {
          pname = project.name;
          version = project.version;
          pyproject = true;
          # Only the package sources: caches or tests never change the environment hash.
          src = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./pyproject.toml
              ./src
            ];
          };
          build-system = [ ps.setuptools ];
          dependencies = baseDependencies ps;
          optional-dependencies = lib.mapAttrs (_: deps: deps ps) packDependencies;
          # Engine tests run in the engine-dev shell, against the same pinned packages.
          doCheck = false;
          pythonImportsCheck = [ "netpattern_engine" ];
        };

      # One sandbox environment: a template plus capability packs, all pinned by flake.lock.
      mkEnvironment =
        {
          system ? "x86_64-linux",
          template ? "pytorch-cpu",
          packs ? [ ],
        }:
        let
          python =
            if pythonFor ? ${template} then
              pythonFor.${template} system
            else
              throw "unknown environment template '${template}'";
          packDeps =
            ps:
            lib.concatMap (
              name:
              if packDependencies ? ${name} then
                packDependencies.${name} ps
              else
                throw "unknown capability pack '${name}'"
            ) packs;
          engine = mkEngine python.pkgs;
        in
        python.withPackages (ps: [ engine ] ++ packDeps ps);
    in
    {
      lib = {
        inherit mkEnvironment;
        templates = builtins.attrNames pythonFor;
        packs = builtins.attrNames packDependencies;
      };

      packages = forAllSystems (system: rec {
        env-cpu = mkEnvironment { inherit system; };
        env-cpu-all-packs = mkEnvironment {
          inherit system;
          packs = builtins.attrNames packDependencies;
        };
        env-cuda12 = mkEnvironment {
          inherit system;
          template = "pytorch-cuda12";
        };
        default = env-cpu;
      });

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          allPacks = ps: lib.concatMap (deps: deps ps) (builtins.attrValues packDependencies);
        in
        {
          # The CPU template with every pack plus pytest; the engine is imported from source.
          engine-dev = pkgs.mkShell {
            packages = [
              (pkgs.python3.withPackages (ps: baseDependencies ps ++ allPacks ps ++ [ ps.pytest ]))
            ];
          };
        }
      );
    };
}
