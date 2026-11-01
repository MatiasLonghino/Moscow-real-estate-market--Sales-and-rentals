"""Procedimientos del análisis de propiedades usadas.

El único alias histórico admite los objetos Joblib previos a la reorganización.
Las nuevas importaciones y serializaciones usan siempre el paquete scripts.
"""
import sys
from . import preparacion_train_test

sys.modules['preparacion_train_test'] = preparacion_train_test
