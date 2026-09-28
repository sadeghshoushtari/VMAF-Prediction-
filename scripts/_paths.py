"""Path resolution for every script in this repository.

Locations are derived from this file's own position, so a fresh clone works with no editing:

    <repo>/data      inputs
    <repo>/out       outputs (gitignored, so pulls stay clean)

Both can be overridden with the environment variables VMAF_DATA and VMAF_OUT.

The analysis scripts need nothing beyond these and the packages in requirements.txt.
The measurement scripts additionally need external binaries and the raw .y4m sources; those are
resolved by `tool()` below, which reads an environment variable and fails with an explicit message
naming what is missing rather than dying somewhere deeper.
"""
import os
from pathlib import Path

ROOT   = Path(__file__).resolve().parent.parent
DATA   = Path(os.environ.get('VMAF_DATA', ROOT / 'data'))
OUT    = Path(os.environ.get('VMAF_OUT',  ROOT / 'out'))
COMPET = DATA / 'competitors'
OUT.mkdir(parents=True, exist_ok=True)


def data(name):
    """An input file under <repo>/data, checked so a missing file names itself."""
    p = DATA / name
    if p.exists():
        return str(p)
    p2 = COMPET / name
    if p2.exists():
        return str(p2)
    raise FileNotFoundError(
        f'{name} not found under {DATA} or {COMPET}. '
        f'Set VMAF_DATA if the data lives elsewhere.')


def out(name):
    """An output path under <repo>/out."""
    return str(OUT / name)


_TOOL_ENV = {
    'ffmpeg':  ('VMAF_FFMPEG',  'ffmpeg built with libvmaf'),
    'ffprobe': ('VMAF_FFPROBE', 'ffprobe from the same build as ffmpeg'),
    'svt':     ('VMAF_SVT',     'SvtAv1EncApp (SVT-AV1 encoder)'),
    'vca':     ('VMAF_VCA',     'vca (Video Complexity Analyzer, github.com/cd-athena/VCA)'),
    'sources': ('VMAF_SOURCES', 'directory holding the raw .y4m sources '
                                '(fetch with scripts/download_sources.sh)'),
}


def tool(kind):
    """Resolve an external binary, or the raw-source directory.

    Only the MEASUREMENT scripts need these; the analysis scripts run from the repository alone.
    """
    env, what = _TOOL_ENV[kind]
    v = os.environ.get(env)
    if v and Path(v).exists():
        return v
    if kind in ('ffmpeg', 'ffprobe', 'vca'):
        from shutil import which
        w = which(kind)
        if w:
            return w
    raise FileNotFoundError(
        f'{what} not found. Set {env} to its location.\n'
        f'  This script performs MEASUREMENT and needs the encoder, the scorer and the raw\n'
        f'  sources. The analysis scripts in this repository need none of them.')
