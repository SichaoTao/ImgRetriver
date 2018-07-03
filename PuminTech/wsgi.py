"""
WSGI config for PuminTech project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/2.0/howto/deployment/wsgi/
"""

import os
import sys

from django.core.wsgi import get_wsgi_application

# sys.path.append('/home/PMtech/PuminTech')

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
os.environ['DJANGO_SETTINGS_MODULE'] = 'PuminTech.settings'


DEBUG = True

application = get_wsgi_application()
