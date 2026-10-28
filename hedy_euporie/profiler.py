"""Optional in-kernel, interactive-session measurements; no nbconvert needed.

The timer includes time spent thinking/editing between cells. Training timing is
captured separately with mark(); neither is Slurm allocation/billing time.
"""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import resource
import socket
import time

class SessionProfiler:
    def __init__(self, label, *, backend='torch', enabled=None):
        self.label = label
        self.backend = backend
        self.enabled = (os.environ.get('HEDY_PROFILE', '1') == '1') if enabled is None else enabled
        self.start = time.perf_counter()
        self.started_utc = dt.datetime.now(dt.timezone.utc).isoformat()
        self.marks = {}
        self._torch = None
        if backend == 'torch' and self.enabled:
            try:
                import torch
                self._torch = torch
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(0)
            except Exception as exc:
                print('Profiling warning: CUDA counters unavailable:', exc)
        print(f'Interactive session: {label} | profiling {"on" if self.enabled else "off"}')

    def mark(self, name):
        if self.enabled:
            self.marks[name] = round(time.perf_counter()-self.start, 3)
        return self.marks.get(name)

    def report(self):
        if not self.enabled:
            print('Profiling disabled (HEDY_PROFILE=0).')
            return None
        torch = self._torch
        cuda_metrics = None
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
            cuda_metrics = {
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_total_mib': round(torch.cuda.get_device_properties(0).total_memory/2**20,2),
                'torch_peak_allocated_mib': round(torch.cuda.max_memory_allocated(0)/2**20,2),
                'torch_peak_reserved_mib': round(torch.cuda.max_memory_reserved(0)/2**20,2),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            }
        jax_metrics = None
        if self.backend == 'jax':
            try:
                import jax
                jax_metrics = {'backend': jax.default_backend(), 'devices': [str(x) for x in jax.devices()]}
            except Exception as exc:
                jax_metrics = {'error': str(exc)}
        now = dt.datetime.now(dt.timezone.utc)
        data = {
            'label': self.label,
            'backend': self.backend,
            'status': 'manual_notebook_checkpoint',
            'started_utc': self.started_utc,
            'finished_utc': now.isoformat(),
            'interactive_elapsed_seconds': round(time.perf_counter()-self.start, 3),
            'timestamps_seconds': dict(self.marks),
            'kernel_max_rss_mib': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, 2),
            'cuda': cuda_metrics,
            'jax': jax_metrics,
            'hostname': socket.gethostname(),
            'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
            'slurm_step_id': os.environ.get('SLURM_STEP_ID', os.environ.get('SLURM_STEPID')),
            'slurm_reservation': os.environ.get('SLURM_RESERVATION_NAME'),
            'slurm_job_gres': os.environ.get('SLURM_JOB_GRES'),
            'notes': [
                'Elapsed is interactive time since the setup cell, INCLUDING human idle time.',
                'Kernel peak RSS is a process high-water mark, not total node RAM.',
                'PyTorch peak allocated/reserved covers the current kernel since setup, not total device VRAM.',
                'JAX device memory peaks are not measured; do not infer them from PyTorch.',
                'Slurm allocation GPU-hours require sacct; MIG billing follows site policy.',
            ],
        }
        outdir = Path(os.environ.get('HEDY_PROFILE_DIR', 'reports/interactive'))
        outdir.mkdir(parents=True, exist_ok=True)
        safe_label = ''.join(c if c.isalnum() or c in '-_' else '_' for c in self.label)
        timestamp = now.strftime('%Y%m%dT%H%M%SZ')
        out = outdir / f'{safe_label}_{data["slurm_job_id"] or "local"}_{timestamp}.json'
        out.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        print('Interactive measurement saved:', out.resolve())
        print('Session elapsed (including user idle):',data['interactive_elapsed_seconds'],'s')
        print('Peak kernel RSS:',data['kernel_max_rss_mib'],'MiB')
        if cuda_metrics:
            print('PyTorch CUDA peak allocated/reserved:',cuda_metrics['torch_peak_allocated_mib'], '/',cuda_metrics['torch_peak_reserved_mib'],'MiB')
        return data

def begin_session(label, *, backend='torch'):
    return SessionProfiler(label, backend=backend)
