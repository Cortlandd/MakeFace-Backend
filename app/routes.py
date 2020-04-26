from app import app
import os
import requests
from flask import Flask, jsonify

@app.route('/')
def index():
    return 'Make Face Backend'