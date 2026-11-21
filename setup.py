import glob
import os
import shutil
import subprocess
import zipfile
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Limit parallel compilation jobs to avoid OOM on Colab/Kaggle
if "MAX_JOBS" not in os.environ:
    os.environ["MAX_JOBS"] = "2"
    print(f"[setup.py] Setting default MAX_JOBS={os.environ['MAX_JOBS']} to prevent CPU/RAM exhaustion")


def get_cutlass_include():
    """Locate NVIDIA CUTLASS headers or shallow-clone into third_party/cutlass."""
    third_party_cutlass = os.path.join(THIS_DIR, "third_party", "cutlass")
    for cand in [
        os.path.join(third_party_cutlass, "include"),
        third_party_cutlass,
        os.path.join(THIS_DIR, "thirdparty", "cutlass", "include"),
        os.path.join(THIS_DIR, "thirdparty", "cutlass"),
    ]:
        if os.path.isfile(os.path.join(cand, "cute", "tensor.hpp")):
            print(f"[setup.py] Found CUTLASS in third_party: {cand}")
            return os.path.abspath(cand)

    if os.path.isfile(os.path.join(THIS_DIR, ".gitmodules")):
        try:
            print("[setup.py] Initializing third_party/cutlass submodule...")
            subprocess.run(
                ["git", "submodule", "update", "--init", "--depth", "1", "third_party/cutlass"],
                cwd=THIS_DIR,
                check=False,
            )
            cand = os.path.join(third_party_cutlass, "include")
            if os.path.isfile(os.path.join(cand, "cute", "tensor.hpp")):
                return os.path.abspath(cand)
        except Exception as e:
            print(f"[setup.py] Notice: Submodule update skipped ({e})")

    env_path = os.environ.get("CUTLASS_PATH")
    if env_path:
        for cand in [os.path.join(env_path, "include"), env_path]:
            if os.path.isfile(os.path.join(cand, "cute", "tensor.hpp")):
                print(f"[setup.py] Found CUTLASS from CUTLASS_PATH: {cand}")
                return os.path.abspath(cand)

    for colab_cand in [
        "/content/Varlen-Attention/third_party/cutlass/include",
        "/content/cutlass/include",
        "/content/cutlass",
        "/kaggle/working/cutlass/include",
    ]:
        if os.path.isfile(os.path.join(colab_cand, "cute", "tensor.hpp")):
            print(f"[setup.py] Found CUTLASS in notebook path: {colab_cand}")
            return os.path.abspath(colab_cand)

    for cand in [
        os.path.join(THIS_DIR, "cutlass", "include"),
        os.path.join(THIS_DIR, "cutlass"),
        os.path.join(THIS_DIR, "..", "cutlass", "include"),
    ]:
        if os.path.isfile(os.path.join(cand, "cute", "tensor.hpp")):
            print(f"[setup.py] Found CUTLASS in local path: {cand}")
            return os.path.abspath(cand)

    target_dir = os.path.join(THIS_DIR, "third_party", "cutlass")
    print(f"[setup.py] CUTLASS (<cute/tensor.hpp>) not found. Shallow cloning into {target_dir}...")
    try:
        if os.path.exists(target_dir) and not os.path.isfile(os.path.join(target_dir, "include", "cute", "tensor.hpp")):
            shutil.rmtree(target_dir, ignore_errors=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", target_dir],
            check=True,
        )
        for cand in [os.path.join(target_dir, "include"), target_dir]:
            if os.path.isfile(os.path.join(cand, "cute", "tensor.hpp")):
                print(f"[setup.py] Successfully cloned CUTLASS: {cand}")
                return os.path.abspath(cand)
    except Exception as e:
        print(f"[setup.py] Warning: Automatic git clone of CUTLASS failed: {e}")

    return os.path.abspath(os.path.join(THIS_DIR, "include"))


def get_cuda_arch_flags():
    """Target Turing SM75 (Tesla T4 / RTX 20-Series) with PTX forward compatibility."""
    if os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []
    return [
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_75,code=compute_75",
    ]


def get_existing_prebuilt():
    """Find pre-compiled CUDA extension binary in varlen_transformer/ or wheels/."""
    pkg_dir = os.path.join(THIS_DIR, "varlen_transformer")
    if os.path.isdir(pkg_dir):
        for fname in os.listdir(pkg_dir):
            if fname.startswith("_C") and fname.endswith(".so"):
                return os.path.join(pkg_dir, fname)

    wheel_candidates = (
        glob.glob(os.path.join(THIS_DIR, "wheels", "*.whl"))
        + glob.glob(os.path.join(THIS_DIR, "*.whl"))
        + glob.glob(os.path.join(THIS_DIR, "dist", "*.whl"))
    )
    for whl in wheel_candidates:
        try:
            with zipfile.ZipFile(whl, "r") as z:
                for member in z.namelist():
                    if member.startswith("varlen_transformer/_C") and member.endswith(".so"):
                        print(f"[setup.py] Extracting prebuilt extension {member} from {whl}...")
                        z.extract(member, THIS_DIR)
                        extracted = os.path.join(THIS_DIR, member)
                        if os.path.isfile(extracted):
                            return extracted
        except Exception as e:
            print(f"[setup.py] Notice: Could not inspect wheel {whl} ({e})")
    return None


force_rebuild = os.environ.get("FORCE_REBUILD", os.environ.get("REBUILD", "0")) in ("1", "true", "True")
existing_prebuilt = get_existing_prebuilt()

if force_rebuild:
    print("[setup.py] FORCE_REBUILD=1 requested: Recompiling CUDA extension from source...")
    should_compile = True
elif existing_prebuilt is not None:
    print(f"[setup.py] Prebuilt extension found: {os.path.basename(existing_prebuilt)}. Skipping compilation.")
    print("[setup.py] (Set FORCE_REBUILD=1 to recompile from source).")
    should_compile = False
else:
    print("[setup.py] No prebuilt extension found. Compiling from source...")
    should_compile = True

ext_modules = []
cmdclass = {}

if should_compile:
    cutlass_include = get_cutlass_include()
    include_dirs = [
        os.path.join(THIS_DIR, "include"),
        cutlass_include,
    ]

    cxx_std = os.environ.get("CXX_STANDARD", "c++20")

    nvcc_flags = [
        "-O3",
        f"-std={cxx_std}",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ] + get_cuda_arch_flags()

    sources = [
        os.path.join(THIS_DIR, "src", s)
        for s in [
            "binding.cpp",
            "mha_fwd.cu",
            "mha_bwd.cu",
            "mha_decode.cu",
        ]
    ]

    ext_modules = [
        CUDAExtension(
            name="varlen_transformer._C",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                "cxx": ["-O3", f"-std={cxx_std}"],
                "nvcc": nvcc_flags,
            },
        ),
    ]

    class Cpp20WithCpp17FallbackBuildExtension(BuildExtension):
        """Compile with C++20 by default; fall back to C++17 only on legacy PyTorch (<2.1)."""

        def build_extensions(self):
            try:
                super().build_extensions()
            except Exception as exc:
                major_minor = tuple(
                    int(x) for x in torch.__version__.split("+")[0].split(".")[:2] if x.isdigit()
                )
                if os.environ.get("CXX_STANDARD") == "c++17" or major_minor >= (2, 1):
                    raise
                print(f"[setup.py] C++20 build failed ({exc}). Falling back to C++17...")
                for ext in self.extensions:
                    for key in ("cxx", "nvcc"):
                        if key in ext.extra_compile_args:
                            ext.extra_compile_args[key] = [
                                "-std=c++17" if flag == "-std=c++20" else flag
                                for flag in ext.extra_compile_args[key]
                            ]
                if os.path.isdir(self.build_temp):
                    shutil.rmtree(self.build_temp, ignore_errors=True)
                super().build_extensions()

    cmdclass = {"build_ext": Cpp20WithCpp17FallbackBuildExtension}

setup(
    name="varlen_transformer",
    version="1.0.0",
    description="High-Performance Variable-Length FlashAttention & Fused RoPE CUDA Kernels (SM75 / CuTe)",
    author="CBG6682",
    packages=find_packages(include=["varlen_transformer", "varlen_transformer.*"]),
    package_data={"varlen_transformer": ["*.so"]},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "ninja",
    ],
)
