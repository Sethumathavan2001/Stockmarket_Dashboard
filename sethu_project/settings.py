import os
from pathlib import Path

# joblib (used by scikit-learn) probes the CPU with "wmic", which newer Windows lacks, and prints a
# traceback. A limit below the core count skips that probe. Used by the ML pages (prediction, patterns).
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) - 1)))

BASE_DIR = Path(__file__).resolve().parent.parent

# Secrets such as ANTHROPIC_API_KEY live in a local .env file next to manage.py (gitignored):
#   ANTHROPIC_API_KEY=sk-ant-...
# Real environment variables win over the file.
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _name, _sep, _value = _line.strip().partition("=")
        if _sep and _name and not _name.startswith("#"):
            os.environ.setdefault(_name.strip(), _value.strip().strip('"').strip("'"))

SECRET_KEY = 'replace-this-with-a-secure-key'
DEBUG = True
ALLOWED_HOSTS = []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'market',
    "prediction",
    "patterns",
    "paper_trading",
    "ai_prediction",
    "news",
]
# Local development settings note: update SECRET_KEY and DEBUG for production

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": (
        "rest_framework.renderers.JSONRenderer",
        "rest_framework.renderers.BrowsableAPIRenderer",
    ),
    "DEFAULT_PERMISSION_CLASSES": (
        # tighten this for production, e.g. IsAuthenticated
        "rest_framework.permissions.AllowAny",
    ),
}
YFINANCE_TICKER_MAP = {
    # "RELIANCE": "RELIANCE.NS",
    # "TCS": "TCS.NS",
}
MARKET_HOLIDAYS = [
    # "2026-01-26", "2026-08-15", ...
]
ENABLE_MARKET_CONTEXT = False
YFINANCE_DEFAULT_SUFFIX = ".NS"
ML_MODELS_DIR = BASE_DIR / "ml_models"

# AI Prediction page (Claude). Key: ANTHROPIC_API_KEY in the environment or .env
AI_PREDICTION_MODEL = os.environ.get("AI_PREDICTION_MODEL", "claude-opus-5-5")
AI_PREDICTION_EFFORT = os.environ.get("AI_PREDICTION_EFFORT", "high")   # low | medium | high | xhigh | max
ROOT_URLCONF = 'sethu_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'sethu_project.wsgi.application'

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.locmem.LocMemCache"
    }
}
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "loggers": {
        "prediction": {"handlers": ["console"], "level": "INFO"},
    },
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
