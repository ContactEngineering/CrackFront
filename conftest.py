import base64
import inspect
import io

import pytest
import pytest_html


def pytest_addoption(parser):
    parser.addoption(
        "--report-plots",
        action="store_true",
        default=False,
        help="Generate plots for the HTML test report (use together with "
        "--html=report.html --self-contained-html).",
    )


class PlotReporter:
    def __init__(self, enabled, extras, request):
        self.enabled = enabled
        self._extras = extras
        self._request = request
        self._source_attached = False

    def attach(self, fig, name="plot"):
        if not self.enabled:
            return
        self._attach_source()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        content = base64.b64encode(buf.getvalue()).decode("ascii")
        self._extras.append(pytest_html.extras.png(content, name=name))

        import matplotlib.pyplot as plt
        plt.close(fig)

    def _attach_source(self):
        if self._source_attached:
            return
        self._source_attached = True
        source = inspect.getsource(self._request.node.function)
        self._extras.append(pytest_html.extras.text(source, name="source"))


@pytest.fixture
def plot_reporter(request, extras):
    return PlotReporter(request.config.getoption("--report-plots"), extras, request)
