# Configuration settings for the application

# Application settings
DEBUG = True
SECRET_KEY = 'your_secret_key'
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

# Database settings
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Other settings
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_L10N = True
USE_TZ = True