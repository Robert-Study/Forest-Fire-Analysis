import importlib.util
from pathlib import Path
import unittest


@unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Streamlit is not installed; CI installs it.')
class StreamlitStartupTests(unittest.TestCase):
    def test_app_initialises_without_exceptions(self):
        from streamlit.testing.v1 import AppTest
        script = Path(__file__).resolve().parents[1] / 'Streamlit_Simulation.py'
        app = AppTest.from_file(str(script), default_timeout=20).run()
        self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])
        for preset in ('preset_anim', 'preset_decay', 'preset_counts', 'preset_perim'):
            app.button(key=preset).click().run()
            self.assertEqual(len(app.exception), 0, [item.message for item in app.exception])


if __name__ == '__main__':
    unittest.main()
