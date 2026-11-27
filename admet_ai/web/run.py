"""Runs the development web interface for ADMET-AI (Flask)."""

import random
import string
from datetime import timedelta
from threading import Thread

import matplotlib
from tap import tapify

from admet_ai.admet_info import load_admet_info
from admet_ai.drugbank import load_drugbank
from admet_ai.web.app import app
from admet_ai.web.app.models import load_admet_model
from admet_ai.web.app.storage import cleanup_storage
from admet_ai.web.app.views import configure_request_controls

matplotlib.use("Agg")


def setup_web(
    secret_key: str = "".join(random.choices(string.ascii_letters + string.digits, k=20)),
    session_lifetime: int = 5 * 60,
    heartbeat_frequency: int = 60,
    max_molecules: int | None = None,
    max_visible_molecules: int = 25,
    rate_limit_requests: int | None = 10,
    rate_limit_window: int = 60,
    max_queue_size: int | None = 10,
    queue_timeout: float | None = 120.0,
) -> None:
    """Sets up the ADMET-AI website.

    :param secret_key: Secret key for Flask app.
    :param session_lifetime: Session lifetime in seconds.
    :param heartbeat_frequency: Frequency of client heartbeat in seconds.
    :param max_molecules: Maximum number of molecules to allow predictions for, or None for no limit.
    :param max_visible_molecules: Maximum number of molecules to display.
    :param rate_limit_requests: Maximum prediction requests per client within rate_limit_window, or None to disable.
    :param rate_limit_window: Sliding rate limit window in seconds.
    :param max_queue_size: Maximum number of queued prediction requests waiting on the model, or None for unbounded.
    :param queue_timeout: Maximum time in seconds a request may wait in the prediction queue, or None for no timeout.
    """
    # Set up Flask app variables
    app.secret_key = secret_key
    app.permanent_session_lifetime = timedelta(seconds=session_lifetime)
    app.config["SESSION_LIFETIME"] = session_lifetime
    app.config["HEARTBEAT_FREQUENCY"] = heartbeat_frequency
    app.config["MAX_MOLECULES"] = max_molecules
    app.config["MAX_VISIBLE_MOLECULES"] = max_visible_molecules
    app.config["RATE_LIMIT_REQUESTS"] = rate_limit_requests
    app.config["RATE_LIMIT_WINDOW"] = rate_limit_window
    app.config["MAX_QUEUE_SIZE"] = max_queue_size
    app.config["QUEUE_TIMEOUT"] = queue_timeout
    configure_request_controls()

    # Load ADMET info, DrugBank, and models into memory
    with app.app_context():
        print("Loading ADMET INFO")
        load_admet_info()

        print("Loading DrugBank")
        load_drugbank()

        print("Loading models")
        load_admet_model()

    # Set up garbage collection
    Thread(target=cleanup_storage, daemon=True).start()


def admet_web(host: str = "127.0.0.1", port: int = 5000) -> None:
    """Runs the ADMET-AI website locally.

    :param host: Host to run the website on.
    :param port: Port to run the website on.
    """
    # Set up web app with command line arguments
    tapify(setup_web)

    # Run web app
    app.run(host=host, port=port)
