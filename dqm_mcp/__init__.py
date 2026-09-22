"""
MCP server for the DQM vision bench.

Exposes the existing pipeline — resolve a shift-layout plot, fetch it from the
online DQM GUI, attach the shifter instructions, ask a vision model, structure
the answer — as MCP tools that any MCP client can call.

Only ``server.py`` depends on MCP. Every other module here is plain Python and
can be used from a notebook or lifted into its own package later.
"""

__version__ = "0.1.0"
