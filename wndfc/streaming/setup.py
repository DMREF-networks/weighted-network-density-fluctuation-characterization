from __future__ import annotations
import os
import sys
import glob
from pathlib import Path

import numpy as np
from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    print("ERROR: Cython is required. pip install cython", file=sys.stderr)
    raise

HERE = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent.parent

# 2D sources: the vendored tree at wndfc/c_voro/2d/src/ (canonical chr1shr/voro
# layout), else the VOROPP_2D_SRC override.
EXCLUDE_3D = {"v_base_wl.cc", "cmd_line.cc", "voro++.cc"}
EXCLUDE_2D = {"v_base_wl_2d.cc", "cq_test.cc", "voro++_2d.cc",
              "container_2d_old.cc", "ctr_quad_2d.cc",
              "quad_march.cc", "v_connect.cc"}


def _find_voropp_src() -> tuple[list[str], list[str]]:
    env = os.environ.get("VOROPP_SRC")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [
        PROJ_ROOT / "wndfc" / "c_voro" / "src",
        HERE / "voro_src",
        Path("/usr/local/include/voro++"),
        Path("/usr/include/voro++"),
    ]
    for c in candidates:
        if (c / "voro++.hh").is_file():
            cc_files = sorted(
                p for p in glob.glob(str(c / "*.cc"))
                if Path(p).name not in EXCLUDE_3D)
            if cc_files:
                print(f"[setup] using vendored voro++ (3D) at {c} "
                      f"({len(cc_files)} .cc files)")
                return [str(c), str(HERE)], cc_files
            print(f"[setup] using system voro++ at {c}")
            return [str(c), str(HERE)], []
    print("[setup] ERROR: voro++ (3D) source not found.", file=sys.stderr)
    print("[setup]   export VOROPP_SRC=/path/to/voro/src", file=sys.stderr)
    sys.exit(1)


def _find_voropp_2d_src() -> tuple[list[str], list[str]] | None:
    """Locate 2D voro++ source. Returns (include_dirs, cc_files) or None
    if 2D source is unavailable (in which case the 2D extension is skipped
    silently and the 3D-only build still succeeds)."""
    env = os.environ.get("VOROPP_2D_SRC")
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates += [
        PROJ_ROOT / "wndfc" / "c_voro" / "2d" / "src",
        HERE / "voro_src_2d",
        HERE / "voro_src" / "2d" / "src",
    ]
    for c in candidates:
        if (c / "voro++_2d.hh").is_file():
            cc_files = sorted(
                p for p in glob.glob(str(c / "*.cc"))
                if Path(p).name not in EXCLUDE_2D)
            if cc_files:
                print(f"[setup] using vendored voro++ (2D) at {c} "
                      f"({len(cc_files)} .cc files)")
                return [str(c), str(HERE)], cc_files
    return None


include_dirs, voropp_sources = _find_voropp_src()
include_dirs.append(np.get_include())

# MinGW-w64 on Windows needs MS_WIN64 or Cython's pointer-size check
# (SIZEOF_VOID_P) fails at compile time. Linux/macOS builds are unaffected.
_EXTRA = ["-O3", "-std=c++11", "-Wno-unused-result"]
if sys.platform == "win32" and any("mingw" in a for a in sys.argv):
    _EXTRA.append("-DMS_WIN64")
elif sys.platform == "win32":
    _EXTRA = ["/O2", "/std:c++14"]  # MSVC rejects the GCC-style flags above

sources = [str(HERE / "_streaming.pyx")] + voropp_sources
libraries = [] if voropp_sources else ["voro++"]

ext_3d = Extension(
    "_streaming",
    sources=sources,
    include_dirs=include_dirs,
    libraries=libraries,
    language="c++",
    extra_compile_args=_EXTRA,
)

extensions = [ext_3d]

# Optional 2D extension (separate .so to avoid symbol collisions on common.cc
# etc., which exist with identical names in both 3D and 2D source trees).
_2d_info = _find_voropp_2d_src()
if _2d_info is not None:
    inc_2d, cc_2d = _2d_info
    inc_2d = list(inc_2d) + [np.get_include()]
    ext_2d = Extension(
        "_streaming_2d",
        sources=[str(HERE / "_streaming_2d.pyx")] + cc_2d,
        include_dirs=inc_2d,
        language="c++",
        extra_compile_args=_EXTRA,
    )
    extensions.append(ext_2d)
else:
    print("[setup] 2D voro++ source not found; skipping _streaming_2d. "
          "Set VOROPP_2D_SRC or place sources at "
          "wndfc/c_voro/2d/src/.", file=sys.stderr)

setup(
    name="wndfc_streaming",
    ext_modules=cythonize(extensions, language_level=3),
    zip_safe=False,
)
