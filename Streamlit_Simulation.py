"""Entrypoint for the existing Streamlit deployment."""
import runpy

runpy.run_module("forest_fire.app", run_name="__main__")
