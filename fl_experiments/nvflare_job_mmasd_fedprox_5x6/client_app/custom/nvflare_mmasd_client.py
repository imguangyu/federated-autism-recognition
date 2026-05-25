#!/usr/bin/env python
"""
NVFLARE client for FreqMixFormer on MMASD+.

Each FL client corresponds to one theme (theme1, theme2, theme3) and
uses the existing FreqMixFormer model and Feeder implementation.

This script is meant to be launched by NVIDIA FLARE (or the FL simulator)
as the client-side training loop.
"""

from nvflare_job_mmasd_fedprox.client_app.custom.nvflare_mmasd_client import *  # noqa: F401,F403


