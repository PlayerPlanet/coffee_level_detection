"""Small utility to visualize the coffeeCNN model graph.

This script attempts to use torchviz.make_dot to produce a PDF/SVG
rendering of the network. If torchviz or Graphviz is not available it
will fall back to saving a TorchScript trace graph as plain text.

Usage:
	python -m coffee_level_detection.training.tests.plot_model

Options (env/args):
	--out FILE    Output basename (default: coffeeCNN_graph)
	--H INT       Image height used when constructing model (default 480)
	--W INT       Image width used when constructing model (default 320)
	--num-classes INT  Number of output classes (default 11)

This file is intentionally defensive so it can run in CI or developer
machines that may not have graphviz installed.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path

try:
	import torch
except Exception as e:  # pragma: no cover - environment dependent
	print("ERROR: PyTorch is required to run this script. Please install torch.")
	raise

from coffee_level_detection.training.coffee import coffeeCNN


def build_and_plot(out_base: str = "coffeeCNN_graph", H: int = 480, W: int = 320, num_classes: int = 11):
	"""Instantiate the model, run a dummy forward pass and try to visualize it.

	The function tries these steps in order:
	1. Use torchviz.make_dot to render a PDF (requires torchviz and graphviz).
	2. If torchviz/graphviz fails, trace the model with torch.jit.trace and
	   save the TorchScript graph text to <out_base>.torchscript.txt.
	"""

	device = torch.device("cpu")
	model = coffeeCNN(num_classes=num_classes, H=H, W=W).to(device)
	model.eval()

	dummy = torch.randn(1, 3, H, W, device=device)

	# Try torchviz first
	try:
		from torchviz import make_dot

		with torch.no_grad():
			out = model(dummy)

		dot = make_dot(out, params=dict(model.named_parameters()))

		# Try to render to PDF first, then SVG as fallback
		pdf_path = f"{out_base}.pdf"
		svg_path = f"{out_base}.svg"

		try:
			dot.format = "pdf"
			rendered = dot.render(out_base, cleanup=True)
			print(f"Saved graph via torchviz to: {rendered}")
			return
		except Exception:
			try:
				dot.format = "svg"
				rendered = dot.render(out_base, cleanup=True)
				print(f"Saved graph via torchviz to: {rendered}")
				return
			except Exception as e:  # pragma: no cover - environment dependent
				print(f"torchviz rendering failed: {e}")
				# fallthrough to TorchScript fallback

	except Exception as e:  # pragma: no cover - environment dependent
		print(f"torchviz not available or failed: {e}")

	# Fallback: TorchScript trace and save textual graph
	try:
		traced = torch.jit.trace(model, dummy)
		ts_text = str(traced.graph)
		out_file = f"{out_base}.torchscript.txt"
		Path(out_file).write_text(ts_text, encoding="utf-8")
		print(f"Saved TorchScript graph text to: {out_file}")
	except Exception as e:  # pragma: no cover - environment dependent
		print(f"Fallback TorchScript tracing failed: {e}")
		# As a last resort, save a tiny model summary
		try:
			summary_file = f"{out_base}.model_summary.txt"
			with open(summary_file, "w", encoding="utf-8") as fh:
				fh.write(repr(model))
			print(f"Saved model repr to: {summary_file}")
		except Exception as ee:
			print(f"Unable to save any model graph: {ee}")


def parse_args(argv=None):
	p = argparse.ArgumentParser(description="Visualize coffeeCNN model graph")
	p.add_argument("--out", type=str, default="coffeeCNN_graph", help="Output basename")
	p.add_argument("--H", type=int, default=480, help="Image height (pixels)")
	p.add_argument("--W", type=int, default=320, help="Image width (pixels)")
	p.add_argument("--num-classes", type=int, default=11, help="Number of output classes")
	return p.parse_args(argv)


if __name__ == "__main__":
	args = parse_args()
	try:
		build_and_plot(out_base=args.out, H=args.H, W=args.W, num_classes=args.num_classes)
	except Exception as exc:  # pragma: no cover - developer machine/runtime dependent
		print(f"Unexpected error while plotting model: {exc}")
		sys.exit(2)

